"""Seldon: a tabular in-context-learning foundation model.

The column and row stages interleave, one column block then one row block per
layer. A cell is embedded from learned Fourier features of its feature group.
Each layer runs a column block (a column's rows attend to its context rows
through inducing points), then a row block (a row's cells and readout tokens
attend to one another). The last layer's readout tokens, concatenated and
projected, are the row; the ICL stage attends every row over the labelled
context rows.

Every parameter belongs to one of the stages: ``col_embedder`` holds the cell
and label embeddings and the column blocks, ``row_interactor`` the readout
tokens, the row blocks, the final norm and the projection, ``icl_predictor``
the ICL.

A query row depends on the context rows only through each column block's
induced set and each ICL block's context state, so ``encode_context`` computes
those once and ``predict_query`` runs any number of query rows against them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from neuralk_oss.model.col import ColEmbedding
from neuralk_oss.model.config import FF_FACTOR, BlockConfig, SeldonConfig
from neuralk_oss.model.icl import ICLearning
from neuralk_oss.model.row import RowInteraction

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


class Context(TypedDict):
    """What ``predict_query`` needs of the context rows, as ``encode_context`` returns it."""

    mean: Tensor
    """Each feature's context mean, ``(B, 1, H)``."""
    induced: list[Tensor]
    """Each column block's induced set, ``(B, H, num_inds, E)``."""
    icl: list[Tensor]
    """Each ICL block's context state, ``(B, train_size, D)``."""


class Seldon(nn.Module):
    """A tabular in-context-learning foundation model.

    Parameters
    ----------
    config : SeldonConfig
        Every architecture setting (see ``neuralk_oss.model.config``).

    Attributes
    ----------
    VERSION : str
        The architecture's version, stored in a checkpoint's config.
    """

    VERSION = "0.2"

    def __init__(self, config: SeldonConfig) -> None:
        super().__init__()
        self.config = config
        self.col_embedder = ColEmbedding(config, self.block_config(config, "col"))
        self.row_interactor = RowInteraction(config, self.block_config(config, "row"))
        self.icl_predictor = ICLearning(config, self.block_config(config, "icl"))

    @staticmethod
    def block_config(config: SeldonConfig, stage: Literal["col", "row", "icl"]) -> BlockConfig:
        """The settings of every attention block of ``stage``."""
        if stage == "col":
            group, width = config.col, config.embed_dim
        elif stage == "row":
            group, width = config.row, config.embed_dim
        elif stage == "icl":
            group, width = config.icl, config.icl_width
        else:
            raise ValueError(f"unknown stage {stage!r}")
        return BlockConfig(
            d_model=width,
            nhead=group.nhead,
            dim_feedforward=FF_FACTOR * width,
            use_rmsnorm=config.use_rmsnorm,
            bias_free_ln=group.bias_free_ln,
            bias=not config.bias_free_linear,
        )

    @property
    def max_classes(self) -> int:
        return self.config.max_classes

    @property
    def num_quantiles(self) -> int:
        return self.config.num_quantiles

    def encode_context(
        self, X: Tensor, y: Tensor, categorical_mask: Tensor | None = None
    ) -> Context:
        """The context rows through every layer: what ``predict_query`` reads of them.

        Parameters
        ----------
        X : Tensor
            Features of the context rows, ``(B, T, H)``; a NaN cell takes its
            feature's mean over them.
        y : Tensor
            Their labels, ``(B, T)``: class indices below ``max_classes``, or
            standardized targets for a regressor.
        categorical_mask : Tensor, optional
            Which columns are categorical, ``(B, H)``; none by default.
        """
        col, row = self.col_embedder, self.row_interactor
        col.check_classes(y)
        categorical_mask = self._categorical_mask(X, categorical_mask)
        mean = col.context_mean(X)
        state = row.state(col.embed(X, mean, categorical_mask, y))
        induced = []
        for i, block in enumerate(col.tf_col.blocks):
            cells = self._cells(state)
            summary = block.induce(cells)
            induced.append(summary)
            state = self._after_column_block(i, state, block.expand(cells, summary))
        R = row(state)
        return Context(mean=mean, induced=induced, icl=self.icl_predictor.encode_context(R, y))

    def predict_query(
        self, X: Tensor, context: Context, categorical_mask: Tensor | None = None
    ) -> Tensor:
        """Raw logits (``max_classes`` wide) or quantiles (``num_quantiles`` wide) of the query rows ``X`` ``(B, T, H)``: ``(B, T, out_dim)``.

        Each query row is computed alone, so the rows may come in any chunks.
        """
        col, row = self.col_embedder, self.row_interactor
        categorical_mask = self._categorical_mask(X, categorical_mask)
        state = row.state(col.embed(X, context["mean"], categorical_mask))
        for i, (block, summary) in enumerate(
            zip(col.tf_col.blocks, context["induced"], strict=True)
        ):
            state = self._after_column_block(i, state, block.expand(self._cells(state), summary))
        return self.icl_predictor(row(state), context["icl"])

    def forward(
        self,
        X: Tensor,
        y: Tensor,
        train_size: int,
        *,
        categorical_mask: Tensor | None = None,
    ) -> Tensor:
        """Outputs of the query rows, the rows of ``X`` ``(B, T, H)`` after the first ``train_size``: ``(B, T - train_size, out_dim)``.

        ``y`` holds the labels of the first ``train_size`` rows, ``(B, train_size)``.
        """
        context = self.encode_context(X[:, :train_size], y[:, :train_size], categorical_mask)
        return self.predict_query(X[:, train_size:], context, categorical_mask)

    @staticmethod
    def _categorical_mask(X: Tensor, categorical_mask: Tensor | None) -> Tensor:
        B, _, H = X.shape
        if categorical_mask is None:
            return X.new_zeros((B, H), dtype=torch.bool)
        return categorical_mask.to(X.device).expand(B, H)

    def _cells(self, state: Tensor) -> Tensor:
        """The cells of the state ``(B, T, K+H, E)``, per column: ``(B, H, T, E)``."""
        return state[:, :, self.row_interactor.num_cls :].transpose(1, 2)

    def _after_column_block(self, i: int, state: Tensor, cols: Tensor) -> Tensor:
        """The state after layer ``i``'s column block, whose output is ``cols`` ``(B, H, T, E)``, and its row block but the last."""
        readout = state[:, :, : self.row_interactor.num_cls]
        state = torch.cat([readout.to(cols.dtype), cols.transpose(1, 2)], dim=2)
        if i < len(self.col_embedder.tf_col.blocks) - 1:
            state = self.row_interactor.block(i, state)
        return state

    @classmethod
    def from_pretrained(cls, path: str | os.PathLike, device: str | torch.device = "cpu") -> Seldon:
        """The model saved in the directory ``path`` (``config.json`` and ``model.safetensors``), loaded strictly, in eval mode."""
        path = Path(path)
        config = json.loads((path / CONFIG_NAME).read_text())
        version = config.pop("model_version", cls.VERSION)
        if version != cls.VERSION:
            raise ValueError(f"checkpoint of Seldon {version}, this is Seldon {cls.VERSION}")
        model = cls(SeldonConfig.model_validate(config))
        model.load_state_dict(load_file(path / WEIGHTS_NAME, device=str(device)), strict=True)
        return model.to(device).eval()

    def save_pretrained(self, path: str | os.PathLike) -> None:
        """Save the model in the directory ``path`` as ``from_pretrained`` reads it."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {"model_version": self.VERSION, **self.config.model_dump()}
        (path / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
        state = {key: value.detach().contiguous() for key, value in self.state_dict().items()}
        save_file(state, path / WEIGHTS_NAME, metadata={"format": "pt"})
