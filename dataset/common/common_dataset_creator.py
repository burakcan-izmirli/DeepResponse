"""Common dataset creator."""
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseDatasetCreator(ABC):
    """Abstract base class for dataset creators."""
    EPSILON = 1e-9
    DATASET_COLUMN_ORDER = ["drug_name", "smiles", "cell_line_name", "cell_line_features", "pic50"]

    def __init__(self, base_dir: Path):
        """Initialize the base dataset creator."""
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"

        # Shared curated reference assets.
        self.reference_dir = self.base_dir.parent / "common" / "curation"
        self.reference_drug_list_path = self.reference_dir / "reference_drug_list.csv"
        self.reference_cell_line_list_path = self.reference_dir / "reference_cell_line_list.csv"
        self.drug_vocabulary_path = self.reference_dir / "drug_vocabulary.csv"
        self.census_gene_names_path = self.reference_dir / "census_gene_names.csv"
        self.cross_domain_gene_axis_path = self.reference_dir / "ccle_gdsc_gene_intersection.csv"

        # Standard output paths
        self.drug_response_features_path = self.processed_dir / "drug_response_features.csv"
        self.cell_line_features_path = self.processed_dir / "cell_line_features.npz"

    @staticmethod
    def clean_string(s: object) -> str:
        """Clean a string by removing non-alphanumeric characters and whitespace."""
        s = str(s)
        s = re.sub(r'\W+', ' ', s)
        s = s.lower()
        s = re.sub(r'\s+', '', s)
        return s

    def load_gene_name_set(self) -> Optional[Set[str]]:
        """Load the cancer gene census list and its synonyms."""
        genes_df = pd.read_csv(self.census_gene_names_path)
        if 'Gene Symbol' in genes_df.columns:
            genes_df.rename(columns={'Gene Symbol': 'gene_name'}, inplace=True)

        genes_df['synonyms'] = genes_df.get('Synonyms', pd.Series([], dtype=str)).apply(
            lambda x: x.split(',') if pd.notna(x) else []
        )

        all_gene_names = set(genes_df['gene_name'].tolist())
        for synonyms_list in genes_df['synonyms']:
            all_gene_names.update(synonyms_list)

        return {self.clean_string(name) for name in all_gene_names}

    def filter_by_gene_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter nested gene tables to the cancer gene census list."""
        gene_set = self.load_gene_name_set()

        def filter_nested(nested_df):
            if isinstance(nested_df, pd.DataFrame):
                nested_df = nested_df.copy()
                nested_df['gene_name'] = nested_df['gene_name'].apply(self.clean_string)
                return nested_df[nested_df['gene_name'].isin(gene_set)].reset_index(drop=True)
            return nested_df

        filtered_df = df.copy()
        filtered_df['cell_line_features'] = df['cell_line_features'].apply(filter_nested)
        return filtered_df

    @staticmethod
    def filter_genes_by_missing_values(cell_line_features_df: pd.DataFrame, gene_axis: List[str], 
                                       threshold: float = 0.6) -> Tuple[pd.DataFrame, List[str]]:
        """Filter genes by missing-value rate and keep the axis aligned."""
        features_list = cell_line_features_df['cell_line_features'].tolist()
        if not features_list or not isinstance(features_list[0], np.ndarray):
            return cell_line_features_df, gene_axis
        stacked = np.stack(features_list, axis=0)
        missing_rate = np.isnan(stacked).mean(axis=(0, 2))
        keep_mask = missing_rate <= threshold
        kept = int(keep_mask.sum())
        if kept == 0 or kept == stacked.shape[1]:
            return cell_line_features_df, gene_axis

        filtered_df = cell_line_features_df.copy()
        filtered_df['cell_line_features'] = filtered_df['cell_line_features'].apply(
            lambda arr: arr[keep_mask, :] if isinstance(arr, np.ndarray) else arr
        )
        filtered_axis = [g for g, keep in zip(gene_axis, keep_mask) if keep]
        return filtered_df, filtered_axis

    @staticmethod
    def _read_header_line(path: Path) -> str:
        """Return the first line of a text file."""
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return handle.readline().rstrip("\n")

    @staticmethod
    def _read_header_fields(path: Path, delimiter: str = "\t") -> List[str]:
        """Return header fields split by delimiter."""
        return BaseDatasetCreator._read_header_line(path).split(delimiter)

    @staticmethod
    def _infer_delimiter(path: Path) -> str:
        """Return delimiter for delimited text based on file suffix."""
        return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","

    @staticmethod
    def _merge_grouped_aggregates(sum_df: Optional[pd.DataFrame], count_df: Optional[pd.DataFrame],
                                  grouped_sum: pd.DataFrame, grouped_cnt: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return merged grouped aggregates for chunked aggregation."""
        if sum_df is None:
            return grouped_sum, grouped_cnt
        return sum_df.add(grouped_sum, fill_value=0), count_df.add(grouped_cnt, fill_value=0)

    @staticmethod
    def _matrix_to_lookup(matrix: pd.DataFrame, cell_list: Iterable[str]) -> Dict[str, np.ndarray]:
        """Convert a gene-by-cell matrix to a cell->vector lookup."""
        return {
            cell: matrix[cell].to_numpy(dtype=np.float32, copy=False)
            for cell in matrix.columns
            if cell in cell_list
        }

    @staticmethod
    def _ensure_cnv_cell(depmap_id: str, cell_list: Optional[Iterable[str]], cell_to_col: Dict[str, int], 
                         mat: Optional[np.ndarray], n_genes: int) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Ensure a CNV matrix column exists for a given cell id."""
        if cell_list is not None:
            return cell_to_col.get(depmap_id), mat
        if depmap_id in cell_to_col:
            return cell_to_col[depmap_id], mat
        new_idx = len(cell_to_col)
        cell_to_col[depmap_id] = new_idx
        if mat is None:
            mat = np.full((n_genes, 1), np.nan, dtype=np.float32)
        else:
            mat = np.concatenate(
                [mat, np.full((n_genes, 1), np.nan, dtype=np.float32)],
                axis=1
            )
        return new_idx, mat

    @staticmethod
    def filter_cell_lines_by_missing_values(df: pd.DataFrame, missing_threshold: float = 0.75) -> pd.DataFrame:
        """Filter out cell lines with too many missing values per channel."""
        def is_valid(cell_features):
            if not isinstance(cell_features, np.ndarray) or cell_features.size == 0:
                return False
            missing_per_channel = np.isnan(cell_features).mean(axis=0)
            return not (missing_per_channel > missing_threshold).any()

        mask = df['cell_line_features'].apply(is_valid)
        filtered_df = df[mask].reset_index(drop=True)
        if filtered_df.empty:
            logger.warning("Cell-line filtering removed all entries; skipping filter.")
            return df.reset_index(drop=True)
        return filtered_df

    def load_drug_vocabulary(self) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, str]]]:
        """Load drug vocabulary for synonym expansion."""
        vocabulary = pd.read_csv(self.drug_vocabulary_path)[['Common name', 'Synonyms']]
        vocabulary = vocabulary.dropna(subset=['Common name'])
        vocabulary['Common name'] = vocabulary['Common name'].apply(self.clean_string)
        vocabulary['Synonyms'] = vocabulary['Synonyms'].apply(
            lambda x: [self.clean_string(syn) for syn in x.split('|')] if pd.notna(x) else []
        )

        combined_dict = {}
        for common_name, synonyms in zip(vocabulary['Common name'], vocabulary['Synonyms']):
            combined_dict[common_name] = common_name
            for synonym in synonyms:
                combined_dict[synonym] = common_name

        return vocabulary, combined_dict

    @staticmethod
    def expand_drug_synonyms(df: pd.DataFrame, vocab_tuple: Tuple[pd.DataFrame, Dict[str, str]]) -> pd.DataFrame:
        """Expand dataset rows with synonyms based on vocabulary mapping."""
        vocabulary, combined_dict = vocab_tuple
        new_rows = []
        for _, row in df.iterrows():
            drug_name_raw = row['drug_name']
            drug_name = BaseDatasetCreator.clean_string(drug_name_raw)
            found_common_name = combined_dict.get(drug_name)
            if found_common_name is None:
                continue
            syn_list = vocabulary[vocabulary['Common name'] == found_common_name]['Synonyms'].values
            if len(syn_list) == 0:
                continue
            for synonym in syn_list[0]:
                syn_clean = BaseDatasetCreator.clean_string(synonym)
                new_row = row.copy()
                new_row['drug_name'] = syn_clean
                new_rows.append(new_row)
        if not new_rows:
            return df
        return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    @staticmethod
    def normalize_smiles_text(smiles_value: object) -> Optional[str]:
        """Normalize a SMILES string and handle missing values."""
        if pd.isna(smiles_value):
            return None
        text = str(smiles_value).strip()
        if not text or text.lower() == 'nan':
            return None
        # Some datasets contain duplicate SMILES separated by commas
        if ',' in text:
            parts = [p.strip() for p in text.split(',') if p.strip()]
            if not parts:
                return None
            text = parts[0]
        return text

    @staticmethod
    def _first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
        """Return the first existing path from candidates."""
        return next((p for p in candidates if p and p.exists()), None)

    @staticmethod
    def parse_gtf_attributes(attr: str) -> Dict[str, str]:
        """Parse a GTF attribute string into a dict."""
        parts = [p.strip() for p in str(attr).split(';') if p.strip()]
        data: Dict[str, str] = {}
        for part in parts:
            if " " not in part:
                continue
            key, value = part.split(" ", 1)
            data[key] = value.replace('"', '')
        return data
    
    def load_canonical_smiles_map(self) -> Optional[Dict[str, str]]:
        """Load canonical SMILES mapping from the shared drug vocabulary."""
        df = pd.read_csv(self.drug_vocabulary_path, usecols=['cmpdname', 'canonicalsmiles'])
        df = df.dropna(subset=['cmpdname', 'canonicalsmiles']).copy()
        df['drug_name'] = df['cmpdname'].astype(str).apply(BaseDatasetCreator.clean_string)
        df['smiles'] = df['canonicalsmiles'].astype(str).str.strip()
        df = df[df['drug_name'] != '']
        df = df.drop_duplicates(subset=['drug_name'])
        return dict(zip(df['drug_name'], df['smiles']))

    def _filter_reference_drugs(self) -> Tuple[Set[str], Dict[str, str]]:
        """Return reference drug names and SMILES from the curation list."""
        ref = pd.read_csv(self.reference_drug_list_path, dtype=str, keep_default_na=False)
        if 'pubchem_id' not in ref.columns:
            raise ValueError(f"Missing pubchem_id in {self.reference_drug_list_path}")
        pubchem_ids = pd.to_numeric(ref['pubchem_id'], errors='coerce').dropna().astype('Int64')
        pubchem_ids = set(pubchem_ids.astype(int).astype(str))
        if not pubchem_ids:
            return set(), {}

        vocab = pd.read_csv(
            self.drug_vocabulary_path,
            usecols=['cid', 'cmpdname', 'canonicalsmiles', 'Common name']
        ).rename(columns={'Common name': 'common_name'})
        vocab['cid'] = pd.to_numeric(vocab['cid'], errors='coerce').astype('Int64')
        vocab = vocab[vocab['cid'].astype(str).isin(pubchem_ids)].copy()

        allowed = set()
        smiles_map = {}
        for row in vocab.itertuples(index=False):
            for name in (row.cmpdname, row.common_name):
                if not isinstance(name, str) or not name.strip():
                    continue
                cleaned = BaseDatasetCreator.clean_string(name)
                if not cleaned:
                    continue
                allowed.add(cleaned)
                smiles = row.canonicalsmiles
                if isinstance(smiles, str) and smiles.strip():
                    smiles_map.setdefault(cleaned, smiles.strip())
        return allowed, smiles_map

    @staticmethod
    def _harmonize_reference_smiles(df: pd.DataFrame, reference_smiles_map: Dict[str, str]) -> pd.DataFrame:
        """Fill missing SMILES using a reference mapping."""
        if not reference_smiles_map or df.empty:
            return df
        before_missing = int(df['smiles'].isna().sum())
        mapped = df['drug_name'].map(reference_smiles_map)
        replaced_rows = int(mapped.notna().sum())
        df['smiles'] = mapped.fillna(df['smiles'])
        after_missing = int(df['smiles'].isna().sum())
        logger.info(
            "Harmonized SMILES with reference vocabulary for %d rows (missing SMILES %d -> %d).",
            replaced_rows,
            before_missing,
            after_missing
        )
        return df

    @staticmethod
    def _apply_canonical_smiles(dataset: pd.DataFrame, canonical_smiles_map: Dict[str, str], 
                                combined_dict: Dict[str, str], reference_smiles_map: Dict[str, str]) -> pd.DataFrame:
        """Apply canonical SMILES for non-reference drug names."""
        if not canonical_smiles_map or not combined_dict or dataset.empty:
            return dataset
        ref_drugs = set(reference_smiles_map.keys()) if reference_smiles_map else set()
        common_name = dataset['drug_name'].map(lambda n: combined_dict.get(n, n))
        canonical = common_name.map(canonical_smiles_map)
        mask = canonical.notna() & ~dataset['drug_name'].isin(ref_drugs)
        replaced = int(mask.sum())
        if replaced:
            dataset.loc[mask, 'smiles'] = canonical[mask]
            logger.info(
                "Replaced %d rows with canonicalsmiles (drug_vocabulary.csv) for non-reference drug names.",
                replaced
            )
        return dataset

    @staticmethod
    def strip_cell_line_name_column(nested_df: object) -> object:
        """Remove cell_line_name column from nested DataFrames."""
        if isinstance(nested_df, pd.DataFrame):
            return nested_df.loc[:, ~nested_df.columns.str.contains('cell_line_name')]
        return nested_df
    
    @staticmethod
    def dataframe_to_array(nested_df: object) -> object:
        """Convert nested DataFrame to numpy array (drop gene_name column)."""
        if isinstance(nested_df, pd.DataFrame):
            if 'gene_name' in nested_df.columns:
                nested_df = nested_df.drop(columns=['gene_name'])
            arr = nested_df.to_numpy(dtype=np.float32)
            return arr.astype(np.float32, copy=True)
        return nested_df
    
    @staticmethod
    def ensure_float32(features: object) -> Optional[np.ndarray]:
        """Convert nested feature arrays to float32 numpy arrays (or return None)."""
        if features is None:
            return None
        arr = np.asarray(features)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    def save_cell_line_features(self, dataset_df: pd.DataFrame, output_path: Path) -> Path:
        """Save cell line features to a .npz lookup."""
        feature_df = (
            dataset_df[["cell_line_name", "cell_line_features"]]
            .dropna(subset=["cell_line_features"])
            .drop_duplicates(subset=["cell_line_name"])
        )
        lookup = {}
        for row in feature_df.itertuples(index=False):
            cell_id = row.cell_line_name
            features = row.cell_line_features
            arr = self.ensure_float32(features)
            if arr is None:
                continue
            lookup[cell_id] = arr
        if not lookup:
            raise ValueError("No cell line features available to save.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **lookup)
        return output_path

    def save_drug_response_features(self, dataset_df: pd.DataFrame, output_path: Path) -> Path:
        """Save drug response features to a CSV."""
        required_cols = ["cell_line_name", "drug_name", "smiles", "pic50"]
        missing = [col for col in required_cols if col not in dataset_df.columns]
        if missing:
            raise ValueError(f"Cannot create drug response records; missing columns: {missing}")
        records_df = dataset_df[required_cols].dropna()
        records_df = self._reorder_columns(records_df, self.DATASET_COLUMN_ORDER)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records_df.to_csv(output_path, index=False)
        return output_path

    def save_dataset(self, dataset_df: pd.DataFrame, write_records: bool = True, write_lookup: bool = True) -> Path:
        """Save dataset to disk (CSV records + NPZ lookup)."""
        dataset_df = dataset_df.copy()
        dataset_df = self._reorder_columns(dataset_df, self.DATASET_COLUMN_ORDER)
        if "cell_line_features" in dataset_df.columns:
            dataset_df["cell_line_features"] = dataset_df["cell_line_features"].apply(self.ensure_float32)

        self.drug_response_features_path.parent.mkdir(parents=True, exist_ok=True)

        if write_records:
            self.save_drug_response_features(dataset_df, self.drug_response_features_path)

        if write_lookup and "cell_line_features" in dataset_df.columns:
            self.save_cell_line_features(dataset_df, self.cell_line_features_path)

        return self.drug_response_features_path

    def create_and_save_dataset(self) -> pd.DataFrame:
        """Create dataset from raw inputs and save standard artifacts."""
        dataset_df = self.create_dataset()
        self.save_dataset(dataset_df)
        return dataset_df

    @abstractmethod
    def create_dataset(self) -> pd.DataFrame:
        """Create the dataset from raw files with standard columns."""
        pass

    @staticmethod
    def _reorder_columns(df: pd.DataFrame, preferred_order: List[str]) -> pd.DataFrame:
        """Reorder DataFrame columns with a preferred prefix order."""
        ordered = [col for col in preferred_order if col in df.columns]
        remaining = [col for col in df.columns if col not in ordered]
        return df[ordered + remaining]

