"""SELFormer-based drug encoder."""

from __future__ import annotations

import logging

import selfies as sf
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast


class DrugEncoder(nn.Module):
    """SELFormer-based drug molecule encoder."""

    def __init__(
        self,
        model_path: str,
        pooling: str = "mean",
        max_length: int = 512,
        trainable_layers: int = 0,
        embed_dim: int = 256,
        projection_hidden: int = 512,
        projection_dropout: float = 0.1,
    ):
        super().__init__()
        self.pooling = pooling
        self.max_length = max_length
        self.embed_dim = embed_dim

        logging.info("Loading SELFormer from: %s", model_path)

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)

        config = RobertaConfig.from_pretrained(model_path)
        config.output_hidden_states = True
        self.backbone = RobertaModel.from_pretrained(model_path, config=config)

        self._configure_trainable_layers(trainable_layers)

        base_dim = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(base_dim, projection_hidden),
            nn.LayerNorm(projection_hidden),
            nn.GELU(),
            nn.Dropout(projection_dropout),
            nn.Linear(projection_hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _configure_trainable_layers(self, trainable_layers: int):
        """Freeze/unfreeze backbone layers."""
        if trainable_layers == 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logging.info("DrugEncoder: all backbone layers frozen")
        elif trainable_layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = True
            logging.info("DrugEncoder: all backbone layers trainable")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False
            num_layers = len(self.backbone.encoder.layer)
            for i in range(num_layers - trainable_layers, num_layers):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = True
            if self.backbone.pooler is not None:
                for param in self.backbone.pooler.parameters():
                    param.requires_grad = True
            logging.info("DrugEncoder: last %d backbone layers trainable", trainable_layers)

    def smiles_to_selfies(self, smiles_list: list[str]) -> list[str]:
        """Convert SMILES to SELFIES."""
        result = []
        for smi in smiles_list:
            try:
                sel = sf.encoder(smi)
                result.append(sel if sel is not None else "[C]")
            except Exception:
                result.append("[C]")
        return result

    def tokenize(self, selfies_list: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize SELFIES strings and move to model device."""
        device = next(self.parameters()).device
        tokens = self.tokenizer(
            selfies_list,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(device) for k, v in tokens.items()}

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool sequence hidden states into a single vector."""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if self.pooling == "max":
            hidden_states = hidden_states.masked_fill(~attention_mask.bool().unsqueeze(-1), float("-inf"))
            return hidden_states.max(dim=1).values
        raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Encode token ids into fixed-size embeddings."""
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        return self.projection(pooled)

    def encode_smiles(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode SMILES strings into fixed-size embeddings."""
        selfies = self.smiles_to_selfies(smiles_list)
        tokens = self.tokenize(selfies)
        return self.forward(tokens["input_ids"], tokens["attention_mask"])

    @torch.no_grad()
    def encode_batch(self, smiles_list: list[str], batch_size: int = 32) -> torch.Tensor:
        """Encode a large list of SMILES in batches."""
        embeddings = []
        for i in range(0, len(smiles_list), batch_size):
            embeddings.append(self.encode_smiles(smiles_list[i : i + batch_size]).cpu())
        return torch.cat(embeddings, dim=0)
