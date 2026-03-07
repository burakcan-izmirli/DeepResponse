"""Repurposing prediction manager for fold-level inference artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from dataset_creator.common.common_dataset_creator import BaseDatasetCreator

clean_string = BaseDatasetCreator.clean_string
normalize_smiles_text = BaseDatasetCreator.normalize_smiles_text


def _stable_synthetic_drug_id(smiles: str) -> int:
    """Generate a stable negative id for custom compounds."""
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()
    return -((int(digest[:15], 16) % 2_000_000_000) + 1)


@dataclass
class _InferenceRow:
    drug_name: str
    smiles: str
    drug_id: int
    cell_line_name: str
    display_cell_line: str
    cell_features: np.ndarray


class _CustomPredictionDataset(Dataset):
    def __init__(self, rows: List[_InferenceRow]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        return {
            "smiles": row.smiles,
            "cell_features": torch.tensor(row.cell_features, dtype=torch.float32),
        }


def _collate_inference_batch(batch: List[Dict]) -> Dict:
    return {
        "smiles": [item["smiles"] for item in batch],
        "cell_features": torch.stack([item["cell_features"] for item in batch]),
    }


class PredictionManager:
    """Build and export repurposing-candidate predictions after each fold."""

    def __init__(
        self,
        data_source: str,
        device: str,
        config_path: str = "config/repurposing_candidates.json",
        shared_axis_path: str = "dataset_creator/common/curation/l1000_genes.txt",
        predictions_filename: str = "repurposing_candidates_predictions.csv",
    ) -> None:
        self.data_source = str(data_source).lower()
        self.device = torch.device(device)
        self.config_path = Path(config_path)
        self.shared_axis_path = Path(shared_axis_path)
        self.predictions_filename = str(predictions_filename)
        self.mapping_csv_path = Path(
            "dataset_creator/common/curation/reference_cell_line_list.csv"
        )
        processed_dir = Path("dataset_creator") / self.data_source / "processed"
        self.cell_features_path = processed_dir / "cell_line_features.npz"
        self.drug_response_path = processed_dir / "drug_response_features.csv"
        self.enabled = True

        self._targets: List[Dict] = []
        self._ach_lookup: Dict[str, str] = {}
        self._ach_to_display: Dict[str, str] = {}
        self._drug_smiles_lookup: Dict[str, str] = {}
        self._cell_feature_lookup: Dict[str, np.ndarray] = {}
        self._shared_axis: List[str] = []
        self._source_axis_index: Dict[str, int] = {}

        self._load_config()
        self._load_shared_axis()
        self._load_cell_features()
        self._load_drug_smiles_lookup()

        if not self._targets:
            self.enabled = False
            logging.info("Repurposing predictions disabled: no valid targets found.")
        if not self._cell_feature_lookup:
            self.enabled = False
            logging.info("Repurposing predictions disabled: no usable cell feature lookup.")

    def _load_config(self) -> None:
        if not self.config_path.exists():
            self.enabled = False
            logging.info(
                "Repurposing config not found at %s; skipping.",
                self.config_path,
            )
            return

        with self.config_path.open(encoding="utf-8") as handle:
            config = json.load(handle)

        if self.mapping_csv_path.exists():
            with self.mapping_csv_path.open(
                newline="", encoding="utf-8", errors="ignore"
            ) as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    depmap_id = (row.get("depmap_id") or "").strip()
                    display = (row.get("cell_line_name") or "").strip()
                    if depmap_id and display:
                        self._ach_lookup[clean_string(display)] = depmap_id
                        self._ach_to_display[depmap_id] = display
        else:
            logging.warning("Cell-line mapping CSV not found: %s", self.mapping_csv_path)

        targets = []
        for entry in config.get("repurposing_targets", []):
            drug_name = str(entry.get("drug_name", "")).strip()
            if not drug_name:
                continue
            prepared_cells = []
            for cell in entry.get("cell_lines", []):
                display = str(cell.get("display_name") or cell.get("name") or "").strip()
                if not display:
                    continue
                ach_id = str(cell.get("ach_id") or "").strip()
                if not ach_id:
                    ach_id = self._ach_lookup.get(clean_string(display), "")
                if ach_id:
                    self._ach_to_display[ach_id] = display
                prepared_cells.append({"display_name": display, "ach_id": ach_id or None})
            if not prepared_cells:
                continue
            targets.append(
                {
                    "drug_name": drug_name,
                    "drug_smiles": normalize_smiles_text(entry.get("drug_smiles")),
                    "cell_lines": prepared_cells,
                }
            )
        self._targets = targets

    def _load_shared_axis(self) -> None:
        if not self.shared_axis_path.exists():
            logging.warning("Shared L1000 axis file not found: %s", self.shared_axis_path)
            return
        if self.shared_axis_path.suffix.lower() == ".txt":
            axis_df = pd.read_csv(self.shared_axis_path, header=None)
        else:
            axis_df = pd.read_csv(self.shared_axis_path)
        if axis_df.empty:
            return
        first_col = axis_df.columns[0]
        cleaned = [
            gene
            for value in axis_df[first_col].astype(str).tolist()
            if (gene := clean_string(value))
        ]
        # Keep order, drop duplicates.
        self._shared_axis = list(dict.fromkeys(cleaned))

    def _load_cell_features(self) -> None:
        if not self.cell_features_path.exists():
            logging.warning("Cell feature lookup not found: %s", self.cell_features_path)
            return

        features_npz = np.load(self.cell_features_path, allow_pickle=True)
        if "__gene_axis__" in features_npz.files:
            self._source_axis_index = {
                gene: idx
                for idx, gene in enumerate(
                    clean_string(value)
                    for value in features_npz["__gene_axis__"].tolist()
                )
                if gene
            }

        for key in features_npz.files:
            if key.startswith("__"):
                continue
            arr = np.asarray(features_npz[key], dtype=np.float32)
            arr = self._to_channels_first(arr)
            if arr is None:
                continue
            arr = self._project_to_shared_axis(arr)
            self._cell_feature_lookup[str(key)] = arr

        logging.info(
            "Repurposing prediction cell lookup ready: source=%s cells=%d genes=%d",
            self.data_source,
            len(self._cell_feature_lookup),
            (
                next(iter(self._cell_feature_lookup.values())).shape[1]
                if self._cell_feature_lookup
                else 0
            ),
        )

    @staticmethod
    def _to_channels_first(arr: np.ndarray) -> Optional[np.ndarray]:
        if arr.ndim != 2:
            return None
        if arr.shape[0] == 4:
            return arr.astype(np.float32, copy=False)
        if arr.shape[1] == 4:
            return arr.T.astype(np.float32, copy=False)
        return None

    def _project_to_shared_axis(self, arr: np.ndarray) -> np.ndarray:
        if not self._shared_axis:
            return arr.astype(np.float32, copy=False)

        target_genes = len(self._shared_axis)
        projected = np.zeros((arr.shape[0], target_genes), dtype=np.float32)

        if self._source_axis_index:
            for target_idx, gene in enumerate(self._shared_axis):
                source_idx = self._source_axis_index.get(gene)
                if source_idx is None:
                    continue
                if 0 <= source_idx < arr.shape[1]:
                    projected[:, target_idx] = arr[:, source_idx]
            return projected

        if arr.shape[1] == target_genes:
            return arr.astype(np.float32, copy=False)

        width = min(arr.shape[1], target_genes)
        projected[:, :width] = arr[:, :width]
        return projected

    def _load_drug_smiles_lookup(self) -> None:
        if not self.drug_response_path.exists():
            logging.warning(
                "Drug response feature file not found: %s",
                self.drug_response_path,
            )
            return

        df = pd.read_csv(self.drug_response_path, usecols=["drug_name", "smiles"])
        for row in df.itertuples(index=False):
            drug_name = clean_string(getattr(row, "drug_name", ""))
            smiles = normalize_smiles_text(getattr(row, "smiles", None))
            if not drug_name or not smiles:
                continue
            self._drug_smiles_lookup.setdefault(drug_name, smiles)

    def _build_inference_rows(self) -> Tuple[List[_InferenceRow], List[str]]:
        rows: List[_InferenceRow] = []
        skipped: List[str] = []
        for target in self._targets:
            drug_name = target["drug_name"]
            drug_key = clean_string(drug_name)
            default_smiles = normalize_smiles_text(target.get("drug_smiles"))
            for cell in target["cell_lines"]:
                display = str(cell.get("display_name") or "").strip()
                ach_id = str(cell.get("ach_id") or "").strip()
                if not ach_id:
                    skipped.append(f"{drug_name} x {display} (missing ACH id)")
                    continue
                cell_features = self._cell_feature_lookup.get(ach_id)
                if cell_features is None:
                    skipped.append(f"{drug_name} x {display} ({ach_id} not in lookup)")
                    continue
                smiles = default_smiles or self._drug_smiles_lookup.get(drug_key)
                if not smiles:
                    skipped.append(f"{drug_name} x {display} (missing SMILES)")
                    continue
                rows.append(
                    _InferenceRow(
                        drug_name=drug_name,
                        smiles=smiles,
                        drug_id=_stable_synthetic_drug_id(smiles),
                        cell_line_name=ach_id,
                        display_cell_line=display or self._ach_to_display.get(ach_id, ach_id),
                        cell_features=cell_features,
                    )
                )
        return rows, skipped

    @torch.no_grad()
    def _predict_rows(self, model, rows: List[_InferenceRow]) -> np.ndarray:
        dataset = _CustomPredictionDataset(rows)
        batch_size = max(1, min(64, len(rows)))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_inference_batch,
        )

        outputs: List[np.ndarray] = []
        was_training = bool(model.training)
        model.eval()
        for batch in loader:
            cell_features = batch["cell_features"].to(self.device)
            smiles = batch["smiles"]
            predictions = model(smiles, cell_features)
            outputs.append(predictions.detach().cpu().numpy().reshape(-1))
        if was_training:
            model.train()
        return np.concatenate(outputs, axis=0) if outputs else np.asarray([], dtype=np.float32)

    def log_predictions(self, model, fold_idx: int, output_dir: Path) -> None:
        if not self.enabled:
            return

        rows, skipped = self._build_inference_rows()
        if not rows:
            logging.warning(
                "No valid repurposing prediction samples for fold %d. Skipped=%d",
                fold_idx,
                len(skipped),
            )
            if skipped:
                logging.info("Repurposing prediction skips: %s", "; ".join(skipped))
            return

        predictions = self._predict_rows(model, rows)
        n = min(len(rows), len(predictions))
        if n == 0:
            logging.warning(
                "Repurposing prediction inference produced no outputs for fold %d.",
                fold_idx,
            )
            return

        records = []
        for idx in range(n):
            row = rows[idx]
            records.append(
                {
                    "fold_idx": int(fold_idx),
                    "drug_name": row.drug_name,
                    "drug_smiles": row.smiles,
                    "drug_id": int(row.drug_id),
                    "cell_line_name": row.cell_line_name,
                    "display_cell_line": row.display_cell_line,
                    "predicted_pic50": float(predictions[idx]),
                }
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.predictions_filename
        pd.DataFrame.from_records(records).to_csv(output_path, index=False)
        logging.info(
            "Saved repurposing predictions for fold %d: %s (%d rows)",
            fold_idx,
            output_path,
            len(records),
        )

        if skipped:
            logging.info(
                "Repurposing prediction skips (fold %d): %s",
                fold_idx,
                "; ".join(skipped),
            )
