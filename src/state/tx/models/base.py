"""扰动预测模型的公共 Lightning 基类与基因表达解码器。

该层集中处理所有具体模型共享的职责：超参数保存、原始计数归一化、可选的基因空间
解码、训练/验证/预测协议以及优化器配置。具体模型只需实现网络结构与前向传播。
"""

import logging
import math
import typing as tp
from abc import ABC, abstractmethod

import torch
from lightning.pytorch import LightningModule
from torch import nn

from .utils import get_loss_fn

logger = logging.getLogger(__name__)


class LatentToGeneDecoder(nn.Module):
    """
    将细胞隐向量还原到基因表达空间的多层感知机。

    This takes concat([cell embedding]) as the input, and predicts
    counts over all genes as output.

    解码器可与主扰动模型分开训练；末尾 ReLU 保证预测计数非负。开启残差模式时，
    每两个相邻 block 构成一组跳连，因此相加两端的维度必须兼容。

    Args:
        latent_dim: Dimension of latent space
        gene_dim: Dimension of gene space (number of HVGs)
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        residual_decoder: If True, adds residual connections between every other layer block
    """

    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: list[int] = [512, 1024],
        dropout: float = 0.1,
        residual_decoder=False,
    ):
        super().__init__()

        self.residual_decoder = residual_decoder

        if residual_decoder:
            # 保留独立 block，前向传播时才能显式插入跨 block 的残差连接。
            self.blocks = nn.ModuleList()
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                block = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
                )
                self.blocks.append(block)
                input_dim = hidden_dim

            # Final output layer
            self.final_layer = nn.Sequential(nn.Linear(input_dim, gene_dim), nn.ReLU())
        else:
            # 非残差路径可直接展平为 Sequential，结构更简单、执行开销也更小。
            layers = []
            input_dim = latent_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim

            # Final output layer
            layers.append(nn.Linear(input_dim, gene_dim))
            # Make sure outputs are non-negative
            layers.append(nn.ReLU())

            self.decoder = nn.Sequential(*layers)

    def gene_dim(self):
        # return the output dimension of the last layer
        if self.residual_decoder:
            return self.final_layer[0].out_features
        else:
            for module in reversed(self.decoder):
                if isinstance(module, nn.Linear):
                    return module.out_features
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x: Latent embeddings of shape [batch_size, latent_dim]

        Returns:
            Gene expression predictions of shape [batch_size, gene_dim]
        """
        if self.residual_decoder:
            # Apply blocks with residual connections between every other block
            block_outputs = []
            current = x

            for i, block in enumerate(self.blocks):
                output = block(current)

                # Add residual connection from every other previous block
                # Pattern: blocks 1, 3, 5, ... get residual from blocks 0, 2, 4, ...
                if i >= 1 and i % 2 == 1:  # Odd-indexed blocks (1, 3, 5, ...)
                    residual_idx = i - 1  # Previous even-indexed block
                    output = output + block_outputs[residual_idx]

                block_outputs.append(output)
                current = output

            return self.final_layer(current)
        else:
            return self.decoder(x)


class PerturbationModel(ABC, LightningModule):
    """
    可同时服务于原始计数和预计算嵌入的扰动模型抽象基类。

    ``output_space`` 是理解数据流的关键：``embedding`` 直接输出隐表示；``gene``
    输出选定基因空间；``all`` 则面向完整基因计数。子类应实现 ``_build_networks``，
    并遵守 batch 字典中的约定键，以复用本类的训练和推理生命周期。

    Args:
        input_dim: Dimension of input features (genes or embeddings)
        hidden_dim: Hidden dimension for neural network layers
        output_dim: Dimension of output (gene space or embedding space)
        pert_dim: Dimension of perturbation embeddings
        dropout: Dropout rate
        lr: Learning rate for optimizer
        loss_fn: Loss function ('mse' or custom nn.Module)
        output_space: 'gene', 'all', or 'embedding'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        dropout: float = 0.1,
        lr: float = 3e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        control_pert: str = "non-targeting",
        embed_key: str | None = None,
        output_space: str = "gene",
        gene_names: list[str] | None = None,
        batch_size: int = 64,
        gene_dim: int = 5000,
        hvg_dim: int = 2001,
        decoder_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.decoder_cfg = decoder_cfg
        self.save_hyperparameters()
        self.gene_decoder_bool = kwargs.get("gene_decoder_bool", True)

        # 架构维度在基类统一保存，防止不同子模型对输入、扰动和输出维度理解不一致。
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pert_dim = pert_dim
        self.batch_dim = batch_dim
        self.gene_dim = gene_dim
        self.hvg_dim = hvg_dim

        if kwargs.get("batch_encoder", False):
            self.batch_dim = batch_dim
        else:
            self.batch_dim = None

        self.residual_decoder = kwargs.get("residual_decoder", False)

        self.embed_key = embed_key
        self.output_space = output_space
        if self.output_space not in {"embedding", "gene", "all"}:
            raise ValueError(
                f"Unsupported output_space '{self.output_space}'. Expected one of 'embedding', 'gene', or 'all'."
            )
        self.batch_size = batch_size
        self.control_pert = control_pert

        self.log1p_from_raw_counts = bool(kwargs.get("log1p_from_raw_counts", False))
        self.counts_target_sum = float(kwargs.get("counts_target_sum", 10_000.0))
        if self.log1p_from_raw_counts and (not math.isfinite(self.counts_target_sum) or self.counts_target_sum <= 0):
            raise ValueError("counts_target_sum must be positive and finite")

        # 训练相关配置由 Lightning checkpoint 一并持久化，恢复时可重建相同数据语义。
        self.gene_names = gene_names  # store the gene names that this model output for gene expression space
        self.dropout = dropout
        self.lr = lr
        self.loss_fn = get_loss_fn(loss_fn)

        if self.output_space == "embedding":
            self.gene_decoder_bool = False
            self.decoder_cfg = None
            # keep hyperparameters metadata consistent with the actual model state
            try:
                if hasattr(self, "hparams"):
                    self.hparams["gene_decoder_bool"] = False  # type: ignore[index]
                    self.hparams["decoder_cfg"] = None  # type: ignore[index]
            except Exception:
                pass

        self._build_decoder()

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        """只搬运张量值，保留字符串、样本标识等元数据在 CPU 上。"""
        return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    def _cp_log1p(self, value: torch.Tensor) -> torch.Tensor:
        """逐细胞执行 CP 归一化后取 ``log1p``。

        每行先缩放到相同的总计数 ``counts_target_sum``，消除测序深度差异。全零细胞的
        缩放系数显式设为零，既保持其生物学含义，也避免除零产生 NaN。
        """
        counts = value.float().clamp_min(0)
        totals = counts.sum(dim=-1, keepdim=True)
        scale = torch.where(
            totals > 0,
            self.counts_target_sum / totals,
            torch.zeros_like(totals),
        )
        return torch.log1p(counts * scale)

    def _normalize_count_keys(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """复制 batch，并把其中属于计数空间的张量转换到 log1p(CP) 空间。

        不原地修改输入，避免同一个 batch 被辅助损失或回调再次读取时看到意外数据。
        当输入本身是外部嵌入时，只转换真实扰动计数，不触碰嵌入张量。
        """
        if not self.log1p_from_raw_counts:
            return batch

        keys = ["pert_cell_counts"]
        if self.embed_key in (None, "X_hvg"):
            keys.extend(("ctrl_cell_emb", "pert_cell_emb"))

        normalized = dict(batch)
        for key in keys:
            value = normalized.get(key)
            if isinstance(value, torch.Tensor):
                normalized[key] = self._cp_log1p(value)
        return normalized

    @abstractmethod
    def _build_networks(self):
        """Build the core neural network components."""

    def _build_decoder(self):
        """Create self.gene_decoder from self.decoder_cfg (or leave None)."""
        if self.gene_decoder_bool == False:
            self.gene_decoder = None
            return
        if self.decoder_cfg is None:
            self.gene_decoder = None
            return
        self.gene_decoder = LatentToGeneDecoder(**self.decoder_cfg)

    def on_load_checkpoint(self, checkpoint: dict[str, tp.Any]) -> None:
        """
        Lightning calls this *before* the checkpoint's state_dict is loaded.
        Re-create the decoder using the exact hyper-parameters saved in the ckpt,
        so that parameter shapes match and load_state_dict succeeds.
        """
        # Check if decoder_cfg was already set externally (e.g., by training script for output_space mismatch)
        decoder_already_configured = (
            hasattr(self, "_decoder_externally_configured") and self._decoder_externally_configured
        )

        if self.gene_decoder_bool == False:
            self.gene_decoder = None
            return

        # When finetuning with the pretrained VCI decoder, keep the existing
        # FinetuneVCICountsDecoder instance. Overwriting it with a freshly
        # constructed LatentToGeneDecoder would make the checkpoint weights
        # incompatible and surface load_state_dict errors.
        finetune_decoder_active = False
        hparams = getattr(self, "hparams", None)
        if hparams is not None:
            if hasattr(hparams, "get"):
                finetune_decoder_active = bool(hparams.get("finetune_vci_decoder", False))
            else:
                finetune_decoder_active = bool(getattr(hparams, "finetune_vci_decoder", False))
        if not finetune_decoder_active:
            finetune_decoder_active = bool(getattr(self, "finetune_vci_decoder", False))

        if finetune_decoder_active:
            # Preserve decoder_cfg for completeness but avoid rebuilding the module.
            if "decoder_cfg" in checkpoint.get("hyper_parameters", {}):
                self.decoder_cfg = checkpoint["hyper_parameters"]["decoder_cfg"]
            logger.info("Finetune VCI decoder active; keeping existing decoder during checkpoint load")
            return

        if not decoder_already_configured and "decoder_cfg" in checkpoint["hyper_parameters"]:
            self.decoder_cfg = checkpoint["hyper_parameters"]["decoder_cfg"]
            self.gene_decoder = LatentToGeneDecoder(**self.decoder_cfg)
            logger.info(f"Loaded decoder from checkpoint decoder_cfg: {self.decoder_cfg}")
        elif not decoder_already_configured:
            # Only fall back to old logic if no decoder_cfg was saved and not externally configured
            self.decoder_cfg = None
            self._build_decoder()
            logger.info(f"DEBUG: output_space: {self.output_space}")
            if self.gene_decoder is None:
                gene_dim = self.hvg_dim if self.output_space == "gene" else self.gene_dim
                logger.info(f"DEBUG: gene_dim: {gene_dim}")
                if (self.embed_key and self.embed_key != "X_hvg" and self.output_space == "gene") or (
                    self.embed_key and self.output_space == "all"
                ):  # we should be able to decode from hvg to all
                    logger.info("DEBUG: Creating gene_decoder, checking conditions...")
                    if gene_dim > 10000:
                        hidden_dims = [1024, 512, 256]
                    else:
                        if "DMSO_TF" in self.control_pert:
                            if self.residual_decoder:
                                hidden_dims = [2058, 2058, 2058, 2058, 2058]
                            else:
                                hidden_dims = [4096, 2048, 2048]
                        elif "PBS" in self.control_pert:
                            hidden_dims = [2048, 1024, 1024]
                        else:
                            hidden_dims = [1024, 1024, 512]  # make this config

                    self.gene_decoder = LatentToGeneDecoder(
                        latent_dim=self.output_dim,
                        gene_dim=gene_dim,
                        hidden_dims=hidden_dims,
                        dropout=self.dropout,
                        residual_decoder=self.residual_decoder,
                    )
                    logger.info(f"Initialized gene decoder for embedding {self.embed_key} to gene space")
        else:
            logger.info("Decoder was already configured externally, skipping checkpoint decoder configuration")

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step logic for both main model and decoder."""
        batch = self._normalize_count_keys(batch)
        # Get model predictions (in latent space)
        pred = self(batch)

        # Compute main model loss
        main_loss = self.loss_fn(pred, batch["pert_cell_emb"])
        self.log("train_loss", main_loss)

        # Process decoder if available
        decoder_loss = None
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            # Train decoder to map latent predictions to gene space
            with torch.no_grad():
                latent_preds = pred.detach()  # Detach to prevent gradient flow back to main model

            pert_cell_counts_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["pert_cell_counts"]
            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets)

            # Log decoder loss
            self.log("decoder_loss", decoder_loss)

            total_loss = main_loss + decoder_loss
        else:
            total_loss = main_loss

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step logic."""
        batch = self._normalize_count_keys(batch)
        pred = self(batch)
        loss = self.loss_fn(pred, batch["pert_cell_emb"])

        # TODO: remove unused
        # is_control = self.control_pert in batch["pert_name"]
        self.log("val_loss", loss)

        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        batch = self._normalize_count_keys(batch)
        latent_output = self(batch)
        target = batch[self.embed_key]
        loss = self.loss_fn(latent_output, target)

        output_dict = {
            "preds": latent_output,  # The distribution's sample
            "pert_cell_emb": batch.get("pert_cell_emb", None),  # The target gene expression or embedding
            "pert_cell_counts": batch.get("pert_cell_counts", None),  # the true, raw gene expression
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds
            decoder_loss = self.loss_fn(pert_cell_counts_preds, batch["pert_cell_counts"])
            self.log("test_decoder_loss", decoder_loss, prog_bar=True)

        self.log("test_loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, **kwargs):
        """
        Typically used for final inference. We'll replicate old logic:
         returning 'preds', 'X', 'pert_name', etc.
        """
        batch = self._normalize_count_keys(batch)
        latent_output = self.forward(batch)
        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }

        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds

        return output_dict

    def decode_to_gene_space(self, latent_embeds: torch.Tensor, basal_expr: None) -> torch.Tensor:
        """
        Decode latent embeddings to gene expression space.

        Args:
            latent_embeds: Embeddings in latent space

        Returns:
            Gene expression predictions or None if decoder is not available
        """
        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_embeds)
            if basal_expr is not None:
                # Add basal expression if provided
                pert_cell_counts_preds += basal_expr
            return pert_cell_counts_preds
        return None

    def configure_optimizers(self):
        """
        Configure a single optimizer for both the main model and the gene decoder.
        """
        # Use a single optimizer for all parameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
