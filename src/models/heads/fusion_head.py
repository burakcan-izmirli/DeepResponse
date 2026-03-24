"""Fusion head for drug-cell response prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class _CrossInteractionLayer(nn.Module):
    """DCN-style cross-interaction layer."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply one cross-interaction step."""
        cross = x0 * self.proj(x)
        return self.norm(x + self.dropout(cross))


class _BidirectionalCrossAttentionBlock(nn.Module):
    """Pre-norm bidirectional cross-attention with FFN residual."""

    def __init__(self, token_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm_d_1 = nn.LayerNorm(token_dim)
        self.norm_c_1 = nn.LayerNorm(token_dim)

        self.d_to_c = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.c_to_d = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.drop_attn = nn.Dropout(dropout)

        self.norm_d_2 = nn.LayerNorm(token_dim)
        self.norm_c_2 = nn.LayerNorm(token_dim)

        ff_hidden = token_dim * 4
        self.ff_d = nn.Sequential(
            nn.Linear(token_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, token_dim),
        )
        self.ff_c = nn.Sequential(
            nn.Linear(token_dim, ff_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, token_dim),
        )
        self.drop_ff = nn.Dropout(dropout)

    def forward(
        self, d_tokens: torch.Tensor, c_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run bidirectional cross-attention and FFN on drug/cell token sequences."""
        d_q = self.norm_d_1(d_tokens)
        c_q = self.norm_c_1(c_tokens)

        d_ctx, _ = self.d_to_c(d_q, c_q, c_q, need_weights=False)
        c_ctx, _ = self.c_to_d(c_q, d_q, d_q, need_weights=False)

        d_tokens = d_tokens + self.drop_attn(d_ctx)
        c_tokens = c_tokens + self.drop_attn(c_ctx)

        d_tokens = d_tokens + self.drop_ff(self.ff_d(self.norm_d_2(d_tokens)))
        c_tokens = c_tokens + self.drop_ff(self.ff_c(self.norm_c_2(c_tokens)))
        return d_tokens, c_tokens


class FusionHead(nn.Module):
    """Configurable fusion head with concat, FiLM-bilinear, or cross-attention variants."""

    def __init__(
        self,
        drug_dim: int,
        cell_dim: int,
        latent_dim: int = 512,
        rank_dim: int = 128,
        hidden_dim: int = 1024,
        dropout: float = 0.25,
        fusion_type: str = "concat",
        bounded_output: str = "none",
        output_center: float = 0.0,
        output_scale: float = 10.0,
        output_tau: float = 1.0,
    ):
        super().__init__()

        self.fusion_type = fusion_type.lower()
        self.bounded_output = bounded_output

        self.register_buffer(
            "_output_center", torch.tensor(float(output_center), dtype=torch.float32)
        )
        self.register_buffer(
            "_output_scale", torch.tensor(float(output_scale), dtype=torch.float32)
        )
        self.register_buffer(
            "_output_tau", torch.tensor(float(output_tau), dtype=torch.float32)
        )

        hidden_mid = max(1, hidden_dim // 2)

        if self.fusion_type == "concat":
            input_dim = drug_dim + cell_dim
            self.predictor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_mid),
                nn.BatchNorm1d(hidden_mid),
                nn.ReLU(),
                nn.Dropout(dropout * 0.8),
                nn.Linear(hidden_mid, 1),
            )

        elif self.fusion_type == "film_bilinear":
            self.drug_proj = nn.Sequential(
                nn.Linear(drug_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )
            self.cell_proj = nn.Sequential(
                nn.Linear(cell_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )

            self.drug_to_cell_gamma = nn.Linear(latent_dim, latent_dim)
            self.drug_to_cell_beta = nn.Linear(latent_dim, latent_dim)
            self.cell_to_drug_gamma = nn.Linear(latent_dim, latent_dim)
            self.cell_to_drug_beta = nn.Linear(latent_dim, latent_dim)

            self.drug_rank = nn.Linear(latent_dim, rank_dim)
            self.cell_rank = nn.Linear(latent_dim, rank_dim)

            fusion_input_dim = (latent_dim * 2) + rank_dim
            self.predictor = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_mid),
                nn.LayerNorm(hidden_mid),
                nn.GELU(),
                nn.Dropout(dropout * 0.8),
                nn.Linear(hidden_mid, 1),
            )

        elif self.fusion_type == "xattn_dcn_residual":
            raw_dim = max(drug_dim, cell_dim)
            self.raw_drug_align = nn.Linear(drug_dim, raw_dim)
            self.raw_cell_align = nn.Linear(cell_dim, raw_dim)

            token_dim = latent_dim
            num_tokens = rank_dim // 8
            num_heads = 8
            xattn_layers = 3
            cross_layers = 4

            self.num_tokens = num_tokens
            self.token_dim = token_dim

            self.drug_token_proj = nn.Linear(raw_dim, num_tokens * token_dim)
            self.cell_token_proj = nn.Linear(raw_dim, num_tokens * token_dim)
            self.drug_token_pos = nn.Parameter(torch.zeros(1, num_tokens, token_dim))
            self.cell_token_pos = nn.Parameter(torch.zeros(1, num_tokens, token_dim))

            self.xattn_blocks = nn.ModuleList(
                [
                    _BidirectionalCrossAttentionBlock(
                        token_dim=token_dim,
                        num_heads=num_heads,
                        dropout=dropout * 0.5,
                    )
                    for _ in range(xattn_layers)
                ]
            )

            attn_feat_dim = token_dim * 8
            branch_dim = hidden_mid
            self.attn_out_proj = nn.Sequential(
                nn.Linear(attn_feat_dim, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )

            dcn_dim = raw_dim * 4
            self.cross_layers = nn.ModuleList(
                [
                    _CrossInteractionLayer(dcn_dim, dropout=dropout * 0.35)
                    for _ in range(cross_layers)
                ]
            )
            self.dcn_out_proj = nn.Sequential(
                nn.Linear(dcn_dim, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )

            self.skip_out_proj = nn.Sequential(
                nn.Linear(raw_dim * 2, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout * 0.25),
            )

            self.branch_gate_logits = nn.Parameter(
                torch.tensor([0.0, 0.0, 0.5], dtype=torch.float32)
            )

            fusion_dim = branch_dim * 4
            self.predictor = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_mid),
                nn.GELU(),
                nn.Dropout(dropout * 0.6),
                nn.Linear(hidden_mid, 1),
            )

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        self._fusion_op = {
            "concat": self._forward_concat,
            "film_bilinear": self._forward_film_bilinear,
            "xattn_dcn_residual": self._forward_xattn_dcn_residual,
        }[self.fusion_type]

        self._bound_op = {
            "none": lambda x: x,
            "tanh": self._tanh_bound,
        }[self.bounded_output]

    def set_output_bounds(self, center: float, scale: float, tau: float) -> None:
        """Update bounded output parameters in-place."""
        self._output_center.fill_(float(center))
        self._output_scale.fill_(max(float(scale), 1e-6))
        self._output_tau.fill_(max(float(tau), 1e-6))

    def _tanh_bound(self, raw_output: torch.Tensor) -> torch.Tensor:
        """Clamp predictions via tanh bounding."""
        return self._output_center + self._output_scale * torch.tanh(raw_output / self._output_tau)

    def _forward_concat(self, drug_emb: torch.Tensor, cell_emb: torch.Tensor) -> torch.Tensor:
        """Simple concatenation fusion."""
        combined_features = torch.cat([drug_emb, cell_emb], dim=-1)
        return self.predictor(combined_features)

    def _forward_film_bilinear(
        self, drug_emb: torch.Tensor, cell_emb: torch.Tensor
    ) -> torch.Tensor:
        """FiLM conditioning with low-rank bilinear interaction."""
        drug_lat = self.drug_proj(drug_emb)
        cell_lat = self.cell_proj(cell_emb)

        gamma_c = torch.tanh(self.drug_to_cell_gamma(drug_lat))
        beta_c = self.drug_to_cell_beta(drug_lat)
        cell_t = (1.0 + gamma_c) * cell_lat + beta_c

        gamma_d = torch.tanh(self.cell_to_drug_gamma(cell_lat))
        beta_d = self.cell_to_drug_beta(cell_lat)
        drug_t = (1.0 + gamma_d) * drug_lat + beta_d

        u = self.drug_rank(drug_t)
        v = self.cell_rank(cell_t)
        bilin = u * v

        fusion = torch.cat([drug_t, cell_t, bilin], dim=-1)
        return self.predictor(fusion)

    def _forward_xattn_dcn_residual(
        self, drug_emb: torch.Tensor, cell_emb: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attention + DCN with gated residual branches."""
        d_raw = self.raw_drug_align(drug_emb)
        c_raw = self.raw_cell_align(cell_emb)

        bsz = d_raw.shape[0]

        d_tokens = self.drug_token_proj(d_raw).view(
            bsz, self.num_tokens, self.token_dim
        )
        c_tokens = self.cell_token_proj(c_raw).view(
            bsz, self.num_tokens, self.token_dim
        )
        d_tokens = d_tokens + self.drug_token_pos
        c_tokens = c_tokens + self.cell_token_pos

        for blk in self.xattn_blocks:
            d_tokens, c_tokens = blk(d_tokens, c_tokens)

        d_mean = d_tokens.mean(dim=1)
        c_mean = c_tokens.mean(dim=1)
        d_max = d_tokens.max(dim=1).values
        c_max = c_tokens.max(dim=1).values

        d_pool = torch.cat([d_mean, d_max], dim=-1)
        c_pool = torch.cat([c_mean, c_max], dim=-1)

        x_attn = torch.cat(
            [d_pool, c_pool, d_pool * c_pool, (d_pool - c_pool).abs()], dim=-1
        )
        attn_branch = self.attn_out_proj(x_attn)

        x0 = torch.cat([d_raw, c_raw, d_raw * c_raw, (d_raw - c_raw).abs()], dim=-1)
        x = x0
        for layer in self.cross_layers:
            x = layer(x0, x)
        dcn_branch = self.dcn_out_proj(x)

        skip_branch = self.skip_out_proj(torch.cat([d_raw, c_raw], dim=-1))

        w = torch.softmax(self.branch_gate_logits, dim=0).to(attn_branch.dtype)
        gated_sum = (w[0] * attn_branch) + (w[1] * dcn_branch) + (w[2] * skip_branch)

        fusion = torch.cat([attn_branch, dcn_branch, skip_branch, gated_sum], dim=-1)
        return self.predictor(fusion)

    def forward(
        self,
        drug_emb: torch.Tensor,
        cell_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse drug and cell embeddings into a scalar response prediction."""
        return self._bound_op(self._fusion_op(drug_emb, cell_emb))
