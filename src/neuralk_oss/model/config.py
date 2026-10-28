"""Seldon's architecture, validated once from a checkpoint's ``config.json``.

The file holds every value; these models hold none, so a missing key, an extra
key or an out-of-range value fails before the model is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Width of every feedforward network, as a multiple of its stage's width.
FF_FACTOR = 2
# Features a cell is embedded with: its own, then ``FEATURE_GROUP_SIZE - 1``
# others of its row (``ColEmbedding._group_index``).
FEATURE_GROUP_SIZE = 3


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ClassificationHead(_Config):
    """Predicts up to ``max_classes`` classes; tables with more go through error-correcting output codes."""

    task: Literal["classification"]
    max_classes: int = Field(gt=0)


class RegressionHead(_Config):
    """Predicts ``num_quantiles`` quantiles of a continuous target."""

    task: Literal["regression"]
    num_quantiles: int = Field(gt=0)


HeadConfig = Annotated[ClassificationHead | RegressionHead, Field(discriminator="task")]


class ColConfig(_Config):
    """The column blocks: each column's rows attend to its context rows through inducing points."""

    num_blocks: int = Field(gt=0)
    nhead: int = Field(gt=0)
    bias_free_ln: bool


class RowConfig(_Config):
    """The row blocks: a row's readout tokens and cells attend to one another."""

    num_blocks: int = Field(gt=0)
    nhead: int = Field(gt=0)
    num_cls: int = Field(gt=0)
    """Readout tokens, whose outputs are concatenated into the row's representation."""
    bias_free_ln: bool


class ICLConfig(_Config):
    """In-context learning: attention from every row to the labelled context rows."""

    num_blocks: int = Field(gt=0)
    nhead: int = Field(gt=0)
    bias_free_ln: bool


class SeldonConfig(_Config):
    """Every architecture setting of a Seldon model.

    ``col.num_blocks`` and ``row.num_blocks`` are the number of layers, one
    column block then one row block each, and must be equal.

    Parameters
    ----------
    use_rmsnorm : bool
        RMSNorm for every block, readout and head norm, else LayerNorm, without
        bias under its stage's ``bias_free_ln``. The per-head norm of q and k
        is an RMSNorm either way.
    num_inducing_points : int
        Inducing points of each column block.
    num_frequencies : int
        Learned Fourier frequencies per feature of a cell's group.
    icl_channels : int | None
        Width of the ICL stage, which the row stage projects its readout to.
        None is ``embed_dim * row.num_cls``, with no projection.
    num_icl_key_value_heads_for_query : int | None
        Key/value heads the query rows attend with in the ICL stage, each
        shared by ``icl.nhead // num_icl_key_value_heads_for_query`` query
        heads. None gives them every head, as the context rows have.
    bias_free_linear : bool
        No bias on the attention projections and feedforward linears of every
        block; the query gates, the cell embedding, the projection and the
        head keep theirs.
    fp32_cells : bool
        Embed the cells in fp32 under autocast.
    fp32_head : bool
        Run the ICL stage's final norm and head in fp32 under autocast.
    """

    embed_dim: int = Field(gt=0)
    use_rmsnorm: bool
    head: HeadConfig
    col: ColConfig
    row: RowConfig
    icl: ICLConfig
    num_inducing_points: int = Field(gt=0)
    num_frequencies: int = Field(gt=0)
    icl_channels: int | None = Field(gt=0)
    num_icl_key_value_heads_for_query: int | None = Field(gt=0)
    bias_free_linear: bool = False
    fp32_cells: bool = False
    fp32_head: bool = False

    @model_validator(mode="after")
    def _check(self) -> SeldonConfig:
        for name, width, nhead in (
            ("col", self.embed_dim, self.col.nhead),
            ("row", self.embed_dim, self.row.nhead),
            ("icl", self.icl_width, self.icl.nhead),
        ):
            if width % nhead:
                raise ValueError(f"{name}.nhead={nhead} does not divide the stage width {width}")
        if self.col.num_blocks != self.row.num_blocks:
            raise ValueError(
                "a layer is one column and one row block: "
                f"col.num_blocks={self.col.num_blocks}, row.num_blocks={self.row.num_blocks}"
            )
        if (self.embed_dim // self.row.nhead) % 2:
            raise ValueError("the row attention's head width must be even (RoPE)")
        kv = self.num_icl_key_value_heads_for_query
        if kv is not None and self.icl.nhead % kv:
            raise ValueError(
                f"num_icl_key_value_heads_for_query={kv} does not divide icl.nhead={self.icl.nhead}"
            )
        return self

    @property
    def icl_width(self) -> int:
        """Width of the ICL stage: ``icl_channels``, else the readout tokens' outputs concatenated."""
        return self.icl_channels or self.embed_dim * self.row.num_cls

    @property
    def max_classes(self) -> int:
        """Width of the classification head; 0 for a regressor."""
        return self.head.max_classes if self.head.task == "classification" else 0

    @property
    def num_quantiles(self) -> int:
        """Number of predicted quantiles; 0 for a classifier."""
        return self.head.num_quantiles if self.head.task == "regression" else 0


@dataclass(frozen=True)
class BlockConfig:
    """The settings every attention block of one stage shares.

    ``Seldon.block_config`` builds one per stage from the ``SeldonConfig``.

    Parameters
    ----------
    d_model : int
        Model dimension, divisible by ``nhead``.
    nhead : int
        Number of attention heads.
    dim_feedforward : int
        Hidden width of each block's feedforward network.
    use_rmsnorm : bool
        If True, the norms are RMSNorm instead of LayerNorm.
    bias_free_ln : bool
        If True, a LayerNorm has no bias.
    bias : bool
        If False, the attention projections and the feedforward linears have
        no bias. The norms follow ``bias_free_ln`` alone.
    """

    d_model: int
    nhead: int
    dim_feedforward: int
    use_rmsnorm: bool
    bias_free_ln: bool
    bias: bool
