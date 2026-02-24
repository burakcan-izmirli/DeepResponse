"""DepMap dataset creator."""

import gc
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dataset_creator.common import BaseDatasetCreator

logger = logging.getLogger(__name__)


class DepMapDatasetCreator(BaseDatasetCreator):
    """DepMap dataset creator."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        super().__init__(base_dir)
        self.gene_expression_path = (
            self.raw_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
        )
        self.crispr_path = self.raw_dir / "CRISPRGeneDependency.csv"
        self.cnv_path = self.raw_dir / "OmicsCNGene.csv"
        self.methylation_path = (
            self.raw_dir / "Methylation_(1kb_upstream_TSS)_subsetted.csv"
        )
        self.drug_response_path = self.raw_dir / "sanger-dose-response.csv"

        with self.l1000_genes_path.open(
            "r", encoding="utf-8", errors="ignore"
        ) as handle:
            self.target_genes = [line.strip() for line in handle if line.strip()]
        if len(self.target_genes) != 978:
            raise ValueError(
                f"Expected 978 L1000 genes in {self.l1000_genes_path}, found {len(self.target_genes)}."
            )

        self.baseline_cell_lines = self.load_reference_cell_lines()
        self.allowed_gdsc_ids = self.load_reference_drug_ids("gdsc_id")

    @staticmethod
    def _coerce_numeric(
        df: pd.DataFrame, cell_line_column: str = "cell_line_name"
    ) -> pd.DataFrame:
        """Coerce gene columns to numeric values."""
        gene_data = df.drop(columns=[cell_line_column]).copy()
        gene_data = gene_data.apply(pd.to_numeric, errors="coerce")
        cleaned_df = pd.DataFrame(gene_data, index=df.index, columns=gene_data.columns)
        cleaned_df.insert(0, cell_line_column, df[cell_line_column])
        return cleaned_df

    @staticmethod
    def _strip_gene_annotations(df: pd.DataFrame) -> pd.DataFrame:
        """Remove parenthetical annotations from gene columns."""
        df = df.copy()
        df.columns = [re.sub(r"\s*\(.*?\)", "", col) for col in df.columns]
        return df

    @staticmethod
    def _extract_gene_prefix(col: str) -> str:
        """Extract the gene name prefix before underscore."""
        return col.split("_")[0] if "_" in col else col

    @staticmethod
    def _normalize_gene_name(name: str) -> str:
        """Normalize gene names for cross-omics alignment."""
        name = str(name)
        name = re.sub(r"\s*\(.*?\)", "", name)
        if "_" in name:
            name = name.split("_")[0]
        return BaseDatasetCreator.clean_string(name)

    def _load_raw_omics_table(self, dataset_name: str) -> pd.DataFrame:
        """Load and normalize a raw omics table."""
        raw_paths = {
            "gene_expression": self.gene_expression_path,
            "crispr": self.crispr_path,
            "copy_number_variation": self.cnv_path,
            "methylation": self.methylation_path,
        }
        df = pd.read_csv(raw_paths[dataset_name], low_memory=False)
        if "cell_line_name" not in df.columns:
            first_col = df.columns[0]
            if first_col in {"", "Unnamed: 0"}:
                df = df.rename(columns={first_col: "cell_line_name"})
            elif "depmap_id" in df.columns:
                df = df.rename(columns={"depmap_id": "cell_line_name"})
            elif "ModelID" in df.columns:
                df = df.rename(columns={"ModelID": "cell_line_name"})
            elif "DepMap_ID" in df.columns:
                df = df.rename(columns={"DepMap_ID": "cell_line_name"})
            else:
                raise ValueError(
                    f"{raw_paths[dataset_name]} is missing a cell line identifier column. "
                    f"Columns: {sorted(df.columns)}"
                )

        if dataset_name == "methylation":
            df.columns = [
                self._extract_gene_prefix(col) if col != "cell_line_name" else col
                for col in df.columns
            ]

        df = self._strip_gene_annotations(df)
        return self._coerce_numeric(df, cell_line_column="cell_line_name")

    def _build_cell_line_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """Build per-cell features on a fixed L1000 axis with zero-imputation (channels-first)."""
        target_genes = list(self.target_genes)
        target_gene_keys = [self._normalize_gene_name(gene) for gene in target_genes]

        all_cell_lines = set()
        omic_cell_matrices: Dict[str, pd.DataFrame] = {}
        for omic_name in self.OMICS_ORDER:
            df = self._load_raw_omics_table(omic_name)
            df = df.dropna(subset=["cell_line_name"]).copy()
            df["cell_line_name"] = df["cell_line_name"].astype(str)
            all_cell_lines.update(df["cell_line_name"].unique())

            gene_cols = [col for col in df.columns if col != "cell_line_name"]
            normalized_cols = [self._normalize_gene_name(col) for col in gene_cols]

            matrix = df.copy()
            matrix.columns = ["cell_line_name"] + normalized_cols
            matrix = matrix.groupby("cell_line_name", as_index=False).mean(
                numeric_only=True
            )
            matrix = matrix.set_index("cell_line_name")
            matrix = matrix.groupby(matrix.columns, axis=1).mean()
            matrix = matrix.reindex(columns=target_gene_keys, fill_value=0.0)
            matrix = matrix.fillna(0.0).astype(np.float32)
            omic_cell_matrices[omic_name] = matrix
            del df
            gc.collect()

        logger.info("Total unique cell lines across all omics: %d", len(all_cell_lines))

        if self.baseline_cell_lines:
            before = len(all_cell_lines)
            all_cell_lines = all_cell_lines.intersection(self.baseline_cell_lines)
            logger.info(
                "Restricted cell lines from %d to %d using baseline list.",
                before,
                len(all_cell_lines),
            )

        cell_lines = sorted(all_cell_lines)
        cell_frames = []
        n_genes = len(target_genes)
        n_channels = len(self.OMICS_ORDER)
        for cell_line in cell_lines:
            feature_array = np.zeros((n_channels, n_genes), dtype=np.float32)
            for channel_idx, omic_name in enumerate(self.OMICS_ORDER):
                matrix = omic_cell_matrices[omic_name]
                if cell_line in matrix.index:
                    feature_array[channel_idx, :] = matrix.loc[cell_line].to_numpy(
                        dtype=np.float32,
                        copy=False,
                    )
            cell_frames.append(
                {"cell_line_name": cell_line, "cell_line_features": feature_array}
            )

        return pd.DataFrame(cell_frames), target_genes

    def _load_drug_response_data(self) -> pd.DataFrame:
        """Load IC50 drug response data with baseline filters."""
        ic50_df_raw = pd.read_csv(self.drug_response_path)

        # Filter by allowed GDSC drug IDs
        ic50_df_raw["DRUG_ID"] = pd.to_numeric(ic50_df_raw["DRUG_ID"], errors="coerce")
        initial_unique_drugs = ic50_df_raw["DRUG_ID"].nunique(dropna=True)

        ic50_df_raw = ic50_df_raw[
            ic50_df_raw["DRUG_ID"].isin(self.allowed_gdsc_ids)
        ].copy()
        filtered_unique_drugs = ic50_df_raw["DRUG_ID"].nunique(dropna=True)

        logger.info(
            f"Filtered IC50 table from {initial_unique_drugs} to {filtered_unique_drugs} GDSC drugs."
        )

        # Filter by baseline cell lines
        if self.baseline_cell_lines:
            initial_cell_lines = ic50_df_raw["ARXSPAN_ID"].nunique(dropna=True)
            ic50_df_raw = ic50_df_raw[
                ic50_df_raw["ARXSPAN_ID"].isin(self.baseline_cell_lines)
            ].copy()
            filtered_cell_lines = ic50_df_raw["ARXSPAN_ID"].nunique(dropna=True)
            logger.info(
                f"Aligned IC50 cell lines from {initial_cell_lines} to {filtered_cell_lines}."
            )

        # Rename columns
        ic50_df_raw = ic50_df_raw[
            ["DATASET", "DRUG_ID", "DRUG_NAME", "ARXSPAN_ID", "IC50_PUBLISHED"]
        ]
        ic50_df_raw = ic50_df_raw.rename(
            columns={
                "DATASET": "dataset",
                "DRUG_ID": "drug_id",
                "DRUG_NAME": "drug_name",
                "ARXSPAN_ID": "cell_line_name",
                "IC50_PUBLISHED": "ic50",
            }
        )
        ic50_df_raw["drug_name"] = (
            ic50_df_raw["drug_name"]
            .astype(str)
            .str.replace(r"[^0-9A-Za-z]+", "", regex=True)
            .str.lower()
        )
        ic50_df_raw["smiles"] = np.nan

        ic50_df_raw = self._dedupe_by_dataset_preference(
            ic50_df_raw,
            ["cell_line_name", "drug_id"],
            "dataset",
        )

        return ic50_df_raw

    def _add_smiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMILES strings using the shared reference and vocabulary mapping."""
        vocab_tuple = self.load_drug_vocabulary()
        canonical_smiles_map = self.load_canonical_smiles_map()
        _, reference_smiles_map = self._filter_reference_drugs()

        df = self._harmonize_reference_smiles(df, reference_smiles_map)
        if vocab_tuple and vocab_tuple[0] is not None:
            df = self.expand_drug_synonyms(df, vocab_tuple)
            df = self._harmonize_reference_smiles(df, reference_smiles_map)

        combined_dict = vocab_tuple[1] if vocab_tuple else None
        df = self._apply_canonical_smiles(
            df, canonical_smiles_map, combined_dict, reference_smiles_map
        )
        df = self._apply_pubchem_smiles_fallback(df)
        return df

    def create_dataset(self) -> pd.DataFrame:
        """Create the DepMap dataset from raw files."""
        cell_features, gene_axis = self._build_cell_line_features()
        cell_features = self.filter_cell_lines_by_missing_values(cell_features)
        self.gene_axis = gene_axis

        ic50_df = self._load_drug_response_data()
        cell_features_lookup = dict(
            zip(cell_features["cell_line_name"], cell_features["cell_line_features"])
        )

        ic50_df["cell_line_features"] = ic50_df["cell_line_name"].map(
            cell_features_lookup
        )

        final_df = ic50_df.dropna(subset=["cell_line_features"]).reset_index(drop=True)

        final_df = self._add_smiles(final_df)
        final_df["_molecule_identity"] = final_df["smiles"].apply(
            self.normalize_smiles_text
        )
        final_df["_molecule_identity"] = final_df["_molecule_identity"].fillna(
            final_df["drug_name"].astype(str)
        )
        final_df = self._filter_to_preferred_dataset(
            final_df,
            ["cell_line_name", "_molecule_identity"],
            "dataset",
            preferred_value="GDSC2",
        )
        final_df = final_df.drop(columns=["_molecule_identity"])

        final_df["pic50"] = -np.log10(final_df["ic50"] * 1e-6)
        final_df = final_df.drop(columns=["ic50"])
        final_df = self._aggregate_molecule_cell_pic50(
            final_df, variance_threshold_log=1.0
        )

        final_df = final_df.dropna(
            subset=["cell_line_features", "drug_name", "drug_id"]
        )

        final_df = self._reorder_columns(final_df, self.DATASET_COLUMN_ORDER)

        return final_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    try:
        DepMapDatasetCreator().create_and_save_dataset()
    except FileNotFoundError as exc:
        logger.error("Unable to create DepMap dataset: %s", exc)
    logger.info("DepMap dataset artifacts stored in processed/")
