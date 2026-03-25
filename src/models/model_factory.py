"""Factory functions for creating model components."""

import logging

from config.constants import DIR_PRETRAINED_SELFORMER, ENCODER_POOLING_CHOICES
from src.models.encoders import CellLineEncoder, DrugEncoder
from src.models.drug_response_model import DrugResponseModel


def create_drug_encoder(
    trainable_layers: int = 0,
    pooling: str = "mean",
) -> DrugEncoder:
    """Create drug encoder."""
    if pooling not in ENCODER_POOLING_CHOICES:
        raise ValueError(f"Unknown pooling: {pooling}. Allowed: {sorted(ENCODER_POOLING_CHOICES)}")

    logging.info("Creating drug encoder: trainable_layers=%s, pooling=%s", trainable_layers, pooling)
    return DrugEncoder(
        model_path=str(DIR_PRETRAINED_SELFORMER),
        pooling=pooling,
        trainable_layers=trainable_layers,
    )


def create_cell_encoder(
    cell_input_shape: tuple,
    embed_dim: int = 256,
    dropout: float = 0.2,
) -> CellLineEncoder:
    """Create cell encoder."""
    in_channels, n_genes = cell_input_shape
    logging.info("Creating cell encoder: in_channels=%s, n_genes=%s, embed_dim=%s", in_channels, n_genes, embed_dim)
    return CellLineEncoder(
        n_genes=n_genes,
        in_channels=in_channels,
        embed_dim=embed_dim,
        dropout=dropout,
    )


def create_model(
    cell_input_shape: tuple,
    device: str,
    hidden_dim: int = 1024,
    cell_embed_dim: int = 256,
    trainable_layers: int = 0,
    pooling: str = "mean",
    latent_dim: int = 512,
    rank_dim: int = 128,
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
) -> DrugResponseModel:
    """Create drug response model."""
    logging.info(
        "Creating drug response model: fusion=%s, hidden=%s, latent=%s, rank=%s, "
        "dropout=%.2f, device=%s",
        fusion_type, hidden_dim, latent_dim, rank_dim, dropout, device,
    )

    drug_encoder = create_drug_encoder(
        trainable_layers=trainable_layers,
        pooling=pooling,
    )
    cell_encoder = create_cell_encoder(
        cell_input_shape=cell_input_shape,
        embed_dim=cell_embed_dim,
        dropout=dropout,
    )

    model = DrugResponseModel(
        drug_encoder=drug_encoder,
        cell_encoder=cell_encoder,
        latent_dim=latent_dim,
        rank_dim=rank_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        force_cell_blind=force_cell_blind,
        fusion_type=fusion_type,
        modality_dropout_drug=modality_dropout_drug,
        modality_dropout_cell=modality_dropout_cell,
        modality_dropout_schedule=modality_dropout_schedule,
        modality_dropout_final_scale=modality_dropout_final_scale,
        bounded_output=bounded_output,
        output_center=output_center,
        output_scale=output_scale,
        output_tau=output_tau,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model created: %s total params, %s trainable", f"{total_params:,}", f"{trainable_params:,}")
    return model
