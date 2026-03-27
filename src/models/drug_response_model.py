"""Drug response prediction model with configurable fusion and modality dropout."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import FusionHead


class DrugResponseModel(nn.Module):
    """Full drug response prediction model combining drug/cell encoders with fusion head."""

    def __init__(
        self,
        drug_encoder: nn.Module,
        cell_encoder: nn.Module,
        latent_dim: int = 512,
        rank_dim: int = 128,
        hidden_dim: int = 1024,
        dropout: float = 0.25,
        force_cell_blind: bool = False,
        fusion_type: str = "concat",
        modality_dropout_drug: float = 0.0,
        modality_dropout_cell: float = 0.0,
        modality_dropout_schedule: str = "warmup_decay",
        modality_dropout_final_scale: float = 0.25,
        bounded_output: str = "none",
        output_center: float = 0.0,
        output_scale: float = 10.0,
        output_tau: float = 1.0,
        cell_feature_noise_std: float | None = None,
    ):
        super().__init__()

        self.drug_encoder = drug_encoder
        self.cell_encoder = cell_encoder
        self.force_cell_blind = force_cell_blind
        self.fusion_type = str(fusion_type).lower()

        self.modality_dropout_drug = max(0.0, float(modality_dropout_drug))
        self.modality_dropout_cell = max(0.0, float(modality_dropout_cell))
        self.modality_dropout_schedule = str(modality_dropout_schedule)
        self.modality_dropout_final_scale = float(modality_dropout_final_scale)
        self.current_epoch = 1
        self.total_epochs = 1

        drug_dim = drug_encoder.embed_dim
        cell_dim = cell_encoder.embed_dim

        if cell_feature_noise_std is None:
            cell_feature_noise_std = (
                0.0 if self.fusion_type == "xattn_dcn_residual" else 0.01
            )
        self.cell_feature_noise_std = max(0.0, float(cell_feature_noise_std))

        self.head = FusionHead(
            drug_dim=drug_dim,
            cell_dim=cell_dim,
            latent_dim=latent_dim,
            rank_dim=rank_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            fusion_type=self.fusion_type,
            bounded_output=bounded_output,
            output_center=output_center,
            output_scale=output_scale,
            output_tau=output_tau,
        )

    def set_training_progress(self, epoch: int, total_epochs: int) -> None:
        self.current_epoch = max(1, int(epoch))
        self.total_epochs = max(1, int(total_epochs))

    def set_output_bounds(self, center: float, scale: float, tau: float) -> None:
        self.head.set_output_bounds(center=center, scale=scale, tau=tau)

    def _maybe_add_cell_feature_noise(
        self, cell_features: torch.Tensor
    ) -> torch.Tensor:
        """Apply training-only Gaussian noise to cell features."""
        if (
            self.training
            and self.cell_feature_noise_std > 0.0
            and torch.is_floating_point(cell_features)
        ):
            noise = torch.randn_like(cell_features) * self.cell_feature_noise_std
            return cell_features + noise
        return cell_features

    def _effective_modality_dropout(self, base_probability: float) -> float:
        base = float(max(0.0, min(0.999, base_probability)))
        if base <= 0.0:
            return 0.0
        if self.modality_dropout_schedule == "constant":
            return base
        if self.modality_dropout_schedule == "warmup_decay":
            if self.total_epochs <= 1:
                return base
            progress = min(
                1.0, max(0.0, (float(self.current_epoch) - 1.0) / (self.total_epochs - 1.0))
            )
            final_scale = max(0.0, min(1.0, self.modality_dropout_final_scale))
            scale = 1.0 - ((1.0 - final_scale) * progress)
            return base * scale
        return base

    def _apply_modality_dropout(
        self,
        drug_emb: torch.Tensor,
        cell_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return drug_emb, cell_emb

        p_drug = self._effective_modality_dropout(self.modality_dropout_drug)
        p_cell = self._effective_modality_dropout(self.modality_dropout_cell)
        if p_drug <= 0.0 and p_cell <= 0.0:
            return drug_emb, cell_emb

        batch_size = drug_emb.shape[0]
        keep_drug = torch.rand(batch_size, device=drug_emb.device) >= p_drug
        keep_cell = torch.rand(batch_size, device=cell_emb.device) >= p_cell

        both_dropped = (~keep_drug) & (~keep_cell)
        if both_dropped.any():
            fallback = torch.rand(int(both_dropped.sum().item()), device=drug_emb.device)
            keep_drug[both_dropped] = fallback >= 0.5
            keep_cell[both_dropped] = fallback < 0.5

        drug_keep_prob = max(1e-6, 1.0 - p_drug)
        cell_keep_prob = max(1e-6, 1.0 - p_cell)
        drug_scale = 1.0 / drug_keep_prob
        cell_scale = 1.0 / cell_keep_prob

        keep_drug = keep_drug.to(drug_emb.dtype).unsqueeze(-1)
        keep_cell = keep_cell.to(cell_emb.dtype).unsqueeze(-1)

        dropped_drug = drug_emb * keep_drug * drug_scale
        dropped_cell = cell_emb * keep_cell * cell_scale
        return dropped_drug, dropped_cell

    def forward(
        self,
        drug_emb: torch.Tensor,
        cell_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict drug response from drug embeddings and cell features."""
        cell_features = self._maybe_add_cell_feature_noise(cell_features)
        cell_emb = self.cell_encoder(cell_features)
        if self.force_cell_blind:
            cell_emb = torch.zeros_like(cell_emb)

        drug_emb, cell_emb = self._apply_modality_dropout(drug_emb, cell_emb)
        return self.head(drug_emb, cell_emb)

    def predict_from_smiles(
        self,
        smiles: list[str],
        cell_features: torch.Tensor,
    ) -> torch.Tensor:
        """Predict drug response directly from SMILES strings."""
        drug_emb = self.drug_encoder.encode_smiles(smiles).to(cell_features.device)
        return self.forward(drug_emb, cell_features)

    def encode_drugs(self, smiles: list[str], batch_size: int = 32) -> torch.Tensor:
        """Encode drugs in batches for caching."""
        return self.drug_encoder.encode_batch(smiles, batch_size)
