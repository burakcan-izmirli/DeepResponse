"""DepMap dataset creator."""
import logging
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from dataset.common import BaseDatasetCreator

logger = logging.getLogger(__name__)


class DepMapDatasetCreator(BaseDatasetCreator):
    """DepMap dataset creator."""
    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        super().__init__(base_dir)
        self.gene_expression_path = self.raw_dir / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
        self.crispr_path = self.raw_dir / "CRISPRGeneDependency.csv"
        self.cnv_path = self.raw_dir / "OmicsCNGene.csv"
        self.methylation_path = self.raw_dir / "Methylation_(1kb_upstream_TSS)_subsetted.csv"
        
        self.baseline_cell_lines = self.load_reference_cell_lines()
        self.allowed_gdsc_ids = self.load_reference_drug_ids("gdsc_id")
    
    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, cell_line_column: str = 'cell_line_name') -> pd.DataFrame:
        """Coerce gene columns to numeric values."""
        gene_data = df.drop(columns=[cell_line_column]).copy()
        gene_data = gene_data.apply(pd.to_numeric, errors='coerce')
        cleaned_df = pd.DataFrame(gene_data, index=df.index, columns=gene_data.columns)
        cleaned_df.insert(0, cell_line_column, df[cell_line_column])
        return cleaned_df
    
    @staticmethod
    def _strip_gene_annotations(df: pd.DataFrame) -> pd.DataFrame:
        """Remove parenthetical annotations from gene columns."""
        df = df.copy()
        df.columns = [re.sub(r'\s*\(.*?\)', '', col) for col in df.columns]
        return df
    
    @staticmethod
    def _extract_gene_prefix(col: str) -> str:
        """Extract the gene name prefix before underscore."""
        return col.split('_')[0] if '_' in col else col

    @staticmethod
    def _normalize_gene_name(name: str) -> str:
        """Normalize gene names for cross-omics alignment."""
        name = str(name)
        name = re.sub(r'\s*\(.*?\)', '', name)
        if '_' in name:
            name = name.split('_')[0]
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
        """Build per-cell feature arrays and a shared gene axis."""
        omics_data = {name: self._load_raw_omics_table(name) for name in self.OMICS_ORDER}

        all_cell_lines = set()
        for df in omics_data.values():
            all_cell_lines.update(df['cell_line_name'].dropna().unique())

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

        def merge_cell_line_data(cell_line_name: str) -> pd.DataFrame:
            gene_dfs = []
            for omic in self.OMICS_ORDER:
                omic_df = omics_data[omic]
                cell_data = omic_df[omic_df['cell_line_name'] == cell_line_name]
                if cell_data.empty:
                    continue
                cell_data = cell_data.set_index('cell_line_name').T.reset_index()
                cell_data.columns = ['gene_name', omic]
                gene_dfs.append(cell_data)

            if not gene_dfs:
                return pd.DataFrame()

            merged = gene_dfs[0]
            for df in gene_dfs[1:]:
                merged = pd.merge(merged, df, on='gene_name', how='outer')

            for omic in self.OMICS_ORDER:
                if omic not in merged.columns:
                    merged[omic] = np.nan
            merged = merged[['gene_name'] + self.OMICS_ORDER]
            merged['gene_name'] = merged['gene_name'].apply(self._normalize_gene_name)

            if merged['gene_name'].duplicated().any():
                merged = merged.groupby('gene_name', as_index=False).mean(numeric_only=True)

            return merged

        cell_frames = []
        gene_axis = []
        seen = set()
        for cell_line in cell_lines:
            cell_df = merge_cell_line_data(cell_line)
            cell_frames.append((cell_line, cell_df))
            if cell_df.empty:
                continue
            for gene in cell_df['gene_name'].tolist():
                if gene and gene not in seen:
                    seen.add(gene)
                    gene_axis.append(gene)

        if not gene_axis:
            raise ValueError("No gene features available to build DepMap gene axis.")

        cell_features = []
        for cell_line, cell_df in cell_frames:
            if cell_df.empty:
                arr = np.full((len(gene_axis), len(self.OMICS_ORDER)), np.nan, dtype=np.float32)
            else:
                aligned = cell_df.set_index('gene_name').reindex(gene_axis)
                aligned = aligned[self.OMICS_ORDER]
                arr = aligned.to_numpy(dtype=np.float32)
            cell_features.append({'cell_line_name': cell_line, 'cell_line_features': arr})

        return pd.DataFrame(cell_features), gene_axis
    
    def _load_drug_response_data(self) -> pd.DataFrame:
        """Load IC50 drug response data with baseline filters."""
        ic50_df_raw = pd.read_csv(self.raw_dir / "sanger-dose-response.csv")
        
        # Filter by allowed GDSC drug IDs
        ic50_df_raw['DRUG_ID'] = pd.to_numeric(ic50_df_raw['DRUG_ID'], errors='coerce')
        initial_unique_drugs = ic50_df_raw['DRUG_ID'].nunique(dropna=True)
        
        ic50_df_raw = ic50_df_raw[ic50_df_raw['DRUG_ID'].isin(self.allowed_gdsc_ids)].copy()
        filtered_unique_drugs = ic50_df_raw['DRUG_ID'].nunique(dropna=True)
        
        logger.info(f"Filtered IC50 table from {initial_unique_drugs} to {filtered_unique_drugs} GDSC drugs.")
        
        # Filter by baseline cell lines
        if self.baseline_cell_lines:
            initial_cell_lines = ic50_df_raw['ARXSPAN_ID'].nunique(dropna=True)
            ic50_df_raw = ic50_df_raw[ic50_df_raw['ARXSPAN_ID'].isin(self.baseline_cell_lines)].copy()
            filtered_cell_lines = ic50_df_raw['ARXSPAN_ID'].nunique(dropna=True)
            logger.info(f"Aligned IC50 cell lines from {initial_cell_lines} to {filtered_cell_lines}.")
        
        # Rename columns
        ic50_df_raw = ic50_df_raw[['DATASET', 'DRUG_ID', 'DRUG_NAME', 'ARXSPAN_ID', 'IC50_PUBLISHED']]
        ic50_df_raw = ic50_df_raw.rename(columns={
            'DATASET': 'dataset',
            'DRUG_ID': 'drug_id',
            'DRUG_NAME': 'drug_name',
            'ARXSPAN_ID': 'cell_line_name',
            'IC50_PUBLISHED': 'ic50'
        })
        ic50_df_raw['drug_name'] = ic50_df_raw['drug_name'].astype(str).apply(self.clean_string)
        ic50_df_raw['smiles'] = np.nan

        ic50_df_raw = self._dedupe_by_dataset_preference(ic50_df_raw, ["cell_line_name", "drug_id"], "dataset",)
        
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
        df = self._apply_canonical_smiles(df, canonical_smiles_map, combined_dict, reference_smiles_map)
        df = self._apply_pubchem_smiles_fallback(df)
        return df
    
    def create_dataset(self) -> pd.DataFrame:
        """Create the DepMap dataset from raw files."""
        cell_features, gene_axis = self._build_cell_line_features()
        cell_features, gene_axis = self.filter_genes_by_missing_values(cell_features, gene_axis)
        cell_features, gene_axis = self.filter_gene_axis_by_names(cell_features, gene_axis)
        cell_features = self.filter_cell_lines_by_missing_values(cell_features)
        self.gene_axis = gene_axis

        ic50_df = self._load_drug_response_data()
        cell_features_lookup = dict(zip(cell_features['cell_line_name'], cell_features['cell_line_features']))
        
        ic50_df['cell_line_features'] = ic50_df['cell_line_name'].map(cell_features_lookup)
        
        final_df = ic50_df.dropna(subset=['cell_line_features']).reset_index(drop=True)

        final_df = self._add_smiles(final_df)

        final_df['pic50'] = -np.log10(final_df['ic50'] * 1e-6)
        final_df = final_df.drop(columns=['ic50'])
        
        final_df = final_df.dropna(subset=['cell_line_features', 'drug_name', 'drug_id'])

        final_df = self._reorder_columns(final_df, self.DATASET_COLUMN_ORDER)
        
        return final_df
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    try:
        DepMapDatasetCreator().create_and_save_dataset()
    except FileNotFoundError as exc:
        logger.error("Unable to create DepMap dataset: %s", exc)
    logger.info("DepMap dataset artifacts stored in processed/")
