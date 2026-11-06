# Copyright (c) 2026 Neuralk-AI
# SPDX-License-Identifier: Apache-2.0

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
from typing import Any, Literal, TypedDict

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from sdm._memory import split_size
from torch import Tensor, nn

from neuralk_oss.model.col import ColEmbedding
from neuralk_oss.model.config import FEATURE_GROUP_SIZE, FF_FACTOR, BlockConfig, SeldonConfig
from neuralk_oss.model.icl import ICLearning
from neuralk_oss.model.row import RowInteraction

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
# The released checkpoints by name: the Hugging Face repository holding each at
# its root, next to the other checkpoints of its architecture (see
# ``from_pretrained``). Every load reads the root config.json, the request the
# Hub counts as a download.
HUB_CHECKPOINTS = {
    "seldon-classifier": "Neuralk-AI/seldon",
    "seldon-regressor": "Neuralk-AI/seldon",
    "seldon-mini-classifier": "Neuralk-AI/seldon-mini",
    "seldon-mini-regressor": "Neuralk-AI/seldon-mini",
}
Task = Literal["classification", "regression"]


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
        B, T, H = X.shape
        device = X.device.type
        dtype = (
            torch.get_autocast_dtype(device)
            if torch.is_autocast_enabled(device)
            else col.num_lin.weight.dtype
        )
        # One state, each stage written into it in place, a slice at a time: a
        # slice's output keeps the state's dtype, and reads only its own rows (or columns).
        # Allocated first, so that the slices are sized to the memory left beside it.
        state = X.new_empty((B, T, row.num_cls + H, self.config.embed_dim), dtype=dtype)
        embed_step, col_step, row_step = self._context_steps(X)
        mean = col.context_mean(X)
        for t in range(0, max(T, 1), embed_step):
            cells = col.embed(
                X[:, t : t + embed_step], mean, categorical_mask, y[:, t : t + embed_step]
            )
            state[:, t : t + embed_step] = row.state(cells)
        induced = []
        blocks = col.tf_col.blocks
        for i, block in enumerate(blocks):
            # A column block reads all context rows of a column, the inducing
            # points over them: the slices are of columns.
            cells, summaries = self._cells(state), []
            for h in range(0, max(H, 1), col_step):
                column = cells[:, h : h + col_step]
                # One copy of the transposed view, where each norm would make its own.
                dense = column.contiguous()
                summaries.append(block.induce(dense))
                column[:] = block.expand(dense, summaries[-1])
            induced.append(torch.cat(summaries, dim=1))
            if i < len(blocks) - 1:
                for t in range(0, max(T, 1), row_step):
                    state[:, t : t + row_step] = row.block(i, state[:, t : t + row_step])
        R = torch.cat(
            [row(state[:, t : t + row_step]) for t in range(0, max(T, 1), row_step)], dim=1
        )
        return Context(mean=mean, induced=induced, icl=self.icl_predictor.encode_context(R, y))

    def _context_steps(self, X: Tensor) -> tuple[int, int, int]:
        """Rows per slice of the cell embedding, columns per slice of a column block, rows per slice of a row block, for the context ``X`` ``(B, T, H)``.

        As sdm slices Kumo Tabular's stages: each slice within sdm's chunk
        memory limit (a share of the GPU), by the same per-item estimates, and
        within the memory still free; everything at once off CUDA.
        """
        B, T, H = X.shape
        E, K = self.config.embed_dim, self.row_interactor.num_cls
        cuda = X.device.type
        size = (
            torch.get_autocast_dtype(cuda).itemsize
            if torch.is_autocast_enabled(cuda)
            else X.element_size()
        )
        cell_size = 4 if self.col_embedder.fp32_cells else size
        cell = 2 * FEATURE_GROUP_SIZE * self.config.num_frequencies * cell_size + 2 * E * size
        # sdm's peak of a transformer block per sequence: factor x length x size x width.
        # shortcut: sdm's private sizing (sdm._memory), pinned at 0.2.1; recheck on an sdm upgrade.
        factor = 15 if size <= 2 else 8
        return (
            _split(T, B * H * cell, X.device),
            _split(H, B * factor * T * size * E, X.device),
            _split(T, B * factor * (K + H) * size * E, X.device),
        )

    def predict_query(
        self,
        X: Tensor,
        context: Context,
        categorical_mask: Tensor | None = None,
        chunk_size: int | None = None,
    ) -> Tensor:
        """Raw logits (``max_classes`` wide) or quantiles (``num_quantiles`` wide) of the query rows ``X`` ``(B, T, H)``: ``(B, T, out_dim)``.

        Each query row is computed alone: the column and row stages run
        ``chunk_size`` rows at a time (all at once by default), and the ICL
        stage computes each block's context keys and values once for them all.
        """
        categorical_mask = self._categorical_mask(X, categorical_mask)
        step = chunk_size or max(X.shape[1], 1)
        R = torch.cat(
            [
                self._represent(X[:, t : t + step], context, categorical_mask)
                for t in range(0, max(X.shape[1], 1), step)
            ],
            dim=1,
        )
        return self.icl_predictor(R, context["icl"], chunk_size)

    def _represent(self, X: Tensor, context: Context, categorical_mask: Tensor) -> Tensor:
        """The row representations ``(B, T, icl_width)`` of the query rows ``X``, through the column and row stages."""
        col, row = self.col_embedder, self.row_interactor
        state = row.state(col.embed(X, context["mean"], categorical_mask))
        blocks = col.tf_col.blocks
        for i, (block, summary) in enumerate(zip(blocks, context["induced"], strict=True)):
            # In place, as encode_context: the column block's output keeps the state's dtype.
            cells = self._cells(state)
            cells[:] = block.expand(cells.contiguous(), summary)
            if i < len(blocks) - 1:
                state = row.block(i, state)
        return row(state)

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

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | os.PathLike,
        *,
        task: Task | None = None,
        subfolder: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        device: str | torch.device = "cpu",
    ) -> Seldon:
        """A saved model, loaded strictly, in eval mode.

        A checkpoint directory holds one model, ``config.json`` and
        ``model.safetensors`` (as ``save_pretrained`` writes it), or several
        sharing an architecture: ``config.json`` then names them under
        ``checkpoints``, each with its head, and the weights of checkpoint
        ``name`` are ``<name>.safetensors``.

        Parameters
        ----------
        name_or_path : str or PathLike
            A local directory; else a checkpoint name of ``HUB_CHECKPOINTS``,
            such as ``"seldon-mini-classifier"`` or ``"seldon-regressor"``;
            else a Hugging Face repository id (``"org/repo"``). A checkpoint
            on the Hub is downloaded once into the Hugging Face cache, with
            only its own weights.
        task : {"classification", "regression"}, optional
            The checkpoint's task: picks it in a directory of several, and
            checks it otherwise.
        subfolder : str, optional
            The checkpoint's directory inside the local directory or the
            repository.
        revision : str, optional
            A branch, tag or commit of the repository; its default branch by
            default.
        token : str, optional
            A Hugging Face token, for a private repository; the logged-in
            one (``hf auth login``) by default.
        device : str or torch.device, default="cpu"
            Device of the weights.
        """
        config, weights = resolve_checkpoint(
            name_or_path, task=task, subfolder=subfolder, revision=revision, token=token
        )
        version = config.pop("model_version", cls.VERSION)
        if version != cls.VERSION:
            raise ValueError(f"checkpoint of Seldon {version}, this is Seldon {cls.VERSION}")
        # Built without storage, then given the checkpoint's tensors: no
        # random initialization to overwrite, and the weights load onto device.
        with torch.device("meta"):
            model = cls(SeldonConfig.model_validate(config))
        state = load_file(weights, device=str(device))
        model.load_state_dict(state, strict=True, assign=True)
        return model.eval()

    def save_pretrained(self, path: str | os.PathLike) -> None:
        """Save the model in the directory ``path`` as ``from_pretrained`` reads it."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config = {"model_version": self.VERSION, **self.config.model_dump()}
        (path / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")
        state = {key: value.detach().contiguous() for key, value in self.state_dict().items()}
        save_file(state, path / WEIGHTS_NAME, metadata={"format": "pt"})


def _split(num_items: int, item_bytes: int, device: torch.device) -> int:
    """sdm's ``split_size``, in balanced chunks that also fit the memory free on a CUDA device, the allocator's cache included."""
    step = split_size(num_items, item_bytes, device)
    if device.type != "cuda" or torch.is_grad_enabled() or item_bytes == 0:
        return step
    free = torch.cuda.mem_get_info(device)[0]
    free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    if step * item_bytes <= free:
        return step
    chunks = -(-num_items // max(free // item_bytes, 1))
    return -(-num_items // chunks)


def resolve_checkpoint(
    name_or_path: str | os.PathLike,
    *,
    task: Task | None = None,
    subfolder: str | None = None,
    revision: str | None = None,
    token: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """A checkpoint's config (its ``head`` set) and the local path of its weights, downloaded from the Hub when ``name_or_path`` is not a local directory.

    A name without ``/`` is a checkpoint of ``HUB_CHECKPOINTS``, at the root of its repository.
    """
    name = str(name_or_path)
    checkpoint = None
    local = Path(name_or_path) / (subfolder or "")
    if local.is_dir():

        def fetch(file: str) -> Path:
            return local / file

    else:
        if "/" in name:
            repo_id = name
        elif subfolder is None and name in HUB_CHECKPOINTS:
            repo_id, checkpoint = HUB_CHECKPOINTS[name], name
        else:
            raise ValueError(
                f"{name!r} is neither a local directory, a checkpoint name ({', '.join(HUB_CHECKPOINTS)}) "
                "nor a Hugging Face repository id"
            )

        def fetch(file: str) -> Path:
            return Path(
                hf_hub_download(repo_id, file, subfolder=subfolder, revision=revision, token=token)
            )

    config = json.loads(fetch(CONFIG_NAME).read_text())
    heads = config.pop("checkpoints", None)
    if heads is None:
        if checkpoint is not None:
            raise FileNotFoundError(f"{repo_id} holds a single checkpoint, not {checkpoint!r}")
        if task is not None and config["head"]["task"] != task:
            raise ValueError(
                f"the checkpoint at {name_or_path} is a {config['head']['task']} model, not {task}"
            )
        return config, fetch(WEIGHTS_NAME)
    if checkpoint is None:
        matches = [n for n, head in heads.items() if task is None or head["task"] == task]
        if len(matches) != 1:
            found = ", ".join(f"{n} ({head['task']})" for n, head in heads.items())
            if task is None:
                problem = "pass the task, classification or regression"
            else:
                problem = f"{'several' if matches else 'no'} {task} checkpoints match"
            raise ValueError(f"{name_or_path} holds {found}: {problem}")
        checkpoint = matches[0]
    elif checkpoint not in heads:
        raise FileNotFoundError(f"{repo_id} holds no checkpoint {checkpoint!r}: {', '.join(heads)}")
    if task is not None and heads[checkpoint]["task"] != task:
        raise ValueError(f"{checkpoint} is a {heads[checkpoint]['task']} model, not {task}")
    config["head"] = heads[checkpoint]
    return config, fetch(f"{checkpoint}.safetensors")
