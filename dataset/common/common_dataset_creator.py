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
    OMICS_ORDER = ["gene_expression", "crispr", "copy_number_variation", "methylation"]

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
        self.gencode_gtf_path = self.reference_dir / "gencode_v19_genes.gtf"

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

    def load_reference_drug_ids(self, column: str) -> Set[int]:
        """Return reference drug ids from the curated list."""
        ref = pd.read_csv(self.reference_drug_list_path, dtype=str, keep_default_na=False)
        if column not in ref.columns:
            raise ValueError(f"Missing {column} in {self.reference_drug_list_path}")
        ids = pd.to_numeric(ref[column], errors="coerce").dropna().astype("Int64")
        return set(ids.astype(int).tolist())

    def load_reference_cell_lines(self) -> Set[str]:
        """Return reference cell line ids from the curated list."""
        ref = pd.read_csv(self.reference_cell_line_list_path, dtype=str, keep_default_na=False)
        if "depmap_id" not in ref.columns:
            raise ValueError(f"Missing depmap_id in {self.reference_cell_line_list_path}")
        return set(ref["depmap_id"].dropna().astype(str).unique())

    def load_cross_domain_gene_axis(self) -> List[str]:
        """Return cleaned gene axis from the CCLE/GDSC intersection file."""
        gene_df = pd.read_csv(self.cross_domain_gene_axis_path)
        gene_col = gene_df.columns[0]
        genes = gene_df[gene_col].astype(str).apply(self.clean_string).tolist()
        return [g for g in genes if g]

    def save_gene_axis(self, gene_axis: List[str], output_path: Path) -> None:
        """Save a gene axis to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"gene_name": gene_axis}).to_csv(output_path, index=False)

    def load_gene_axis(self, axis_path: Path) -> List[str]:
        """Load a gene axis from disk."""
        gene_df = pd.read_csv(axis_path)
        gene_col = gene_df.columns[0]
        genes = gene_df[gene_col].astype(str).apply(self.clean_string).tolist()
        return [g for g in genes if g]

    def apply_cross_domain_intersection(self, cell_line_features_df: pd.DataFrame,
                                        gene_axis: List[str],
                                        other_axis_path: Path) -> Tuple[pd.DataFrame, List[str]]:
        """Apply post-filter intersection axis to cell line features."""
        if not other_axis_path.exists():
            msg = (
                f"Missing cross-domain axis at {other_axis_path}. "
                "Build the other dataset first to create its gene_axis.csv, then rerun."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)
        other_axis = self.load_gene_axis(other_axis_path)
        other_set = set(other_axis)
        intersection_set = set(gene_axis).intersection(other_set)
        if not intersection_set:
            raise ValueError(f"No overlapping genes between axes: {other_axis_path}")

        if self.cross_domain_gene_axis_path.exists():
            canonical = [g for g in self.load_cross_domain_gene_axis() if g in intersection_set]
            if not canonical:
                raise ValueError(
                    f"Cross-domain axis at {self.cross_domain_gene_axis_path} has no overlap with current axes."
                )
        else:
            canonical = [g for g in gene_axis if g in intersection_set]

        index_map = {g: i for i, g in enumerate(gene_axis)}
        indices = [index_map[g] for g in canonical if g in index_map]

        filtered_df = cell_line_features_df.copy()
        filtered_df['cell_line_features'] = filtered_df['cell_line_features'].apply(
            lambda arr: arr[indices, :] if isinstance(arr, np.ndarray) else arr
        )
        self.save_gene_axis(canonical, self.cross_domain_gene_axis_path)
        return filtered_df, canonical

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

    def filter_gene_axis_by_names(self, cell_line_features_df: pd.DataFrame,
                                  gene_axis: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """Filter array features to the cancer gene census list using gene_axis."""
        gene_set = self.load_gene_name_set()
        keep_mask = [gene in gene_set for gene in gene_axis]
        kept = sum(keep_mask)
        filtered_df = cell_line_features_df.copy()
        filtered_df['cell_line_features'] = filtered_df['cell_line_features'].apply(
            lambda arr: arr[keep_mask, :] if isinstance(arr, np.ndarray) else arr)
        filtered_axis = [gene for gene, keep in zip(gene_axis, keep_mask) if keep]
        return filtered_df, filtered_axis

    @staticmethod
    def filter_genes_by_missing_values(cell_line_features_df: pd.DataFrame, gene_axis: List[str],
                                       threshold: float = 0.6) -> Tuple[pd.DataFrame, List[str]]:
        """Filter genes by missing-value rate per channel and keep the axis aligned."""
        features_list = cell_line_features_df['cell_line_features'].tolist()
        if not features_list:
            return cell_line_features_df, gene_axis
        if not isinstance(features_list[0], np.ndarray):
            return cell_line_features_df, gene_axis
        stacked = np.stack(features_list, axis=0)
        missing_rate = np.isnan(stacked).mean(axis=0)
        keep_mask = (missing_rate <= threshold).all(axis=1)
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

    def _load_clean_table(self, path: Path) -> pd.DataFrame:
        """Read a delimited file and normalize column names."""
        sep = self._infer_delimiter(path)
        df = pd.read_csv(path, sep=sep, low_memory=False)
        df.columns = [self.clean_string(c) for c in df.columns]
        return df

    @staticmethod
    def _require_column(columns: Iterable[str], candidates: Set[str], path: Path, label: str) -> str:
        """Pick a required column name from candidates."""
        col = next((c for c in columns if c in candidates), None)
        if col is None:
            raise ValueError(f"{path} must include {label} (found: {sorted(columns)}).")
        return col

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
    def filter_cell_lines_by_missing_values(df: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
        """Filter out cell lines with too many missing values per channel."""
        def is_valid(cell_features):
            if isinstance(cell_features, np.ndarray):
                if cell_features.size == 0:
                    return False
                missing_per_channel = np.isnan(cell_features).mean(axis=0)
                return not (missing_per_channel > threshold).any()
            return False

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
        pubchem_ids = self.load_reference_drug_ids("pubchem_id")

        vocab = pd.read_csv(self.drug_vocabulary_path, usecols=['cid', 'cmpdname', 'canonicalsmiles', 'Common name']
        ).rename(columns={'Common name': 'common_name'})
        vocab['cid'] = pd.to_numeric(vocab['cid'], errors='coerce').astype('Int64')
        vocab = vocab[vocab['cid'].isin(pubchem_ids)].copy()

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
        if df.empty:
            raise ValueError("Cannot harmonize SMILES: dataset is empty.")
        if not reference_smiles_map:
            raise ValueError("Reference SMILES map is empty; check reference drug list/vocabulary.")
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

    def _apply_pubchem_smiles_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing SMILES using PubChem ids from the reference drug list."""
        if df.empty or 'drug_id' not in df.columns or 'smiles' not in df.columns:
            return df
        missing_mask = df['smiles'].isna()
        if not missing_mask.any():
            return df

        ref = pd.read_csv(self.reference_drug_list_path, usecols=['gdsc_id', 'pubchem_id'])
        ref['gdsc_id'] = pd.to_numeric(ref['gdsc_id'], errors='coerce')
        ref['pubchem_id'] = pd.to_numeric(ref['pubchem_id'], errors='coerce')
        ref = ref.dropna(subset=['gdsc_id', 'pubchem_id']).drop_duplicates(subset=['gdsc_id'])

        vocab = pd.read_csv(self.drug_vocabulary_path, usecols=['cid', 'canonicalsmiles'])
        vocab['cid'] = pd.to_numeric(vocab['cid'], errors='coerce')
        vocab = vocab.dropna(subset=['cid', 'canonicalsmiles']).drop_duplicates(subset=['cid'])

        merged = ref.merge(vocab, left_on='pubchem_id', right_on='cid', how='left')
        merged = merged.dropna(subset=['canonicalsmiles']).drop_duplicates(subset=['gdsc_id'])
        if merged.empty:
            return df

        map_dict = dict(zip(merged['gdsc_id'], merged['canonicalsmiles']))
        drug_ids = pd.to_numeric(df['drug_id'], errors='coerce')
        fallback = drug_ids.map(map_dict)
        replaced = int(fallback[missing_mask].notna().sum())
        if replaced:
            df.loc[missing_mask, 'smiles'] = fallback[missing_mask]
            logger.info("Filled %d missing SMILES values via PubChem fallback.", replaced)
        return df

    @staticmethod
    def _dedupe_by_dataset_preference(df: pd.DataFrame, group_cols: List[str],
                                      dataset_col: Optional[str],
                                      preferred_value: str = "GDSC2") -> pd.DataFrame:
        """Deduplicate by group_cols, preferring rows with dataset_col == preferred_value."""
        df = df.copy()
        df["_dataset_rank"] = (df[dataset_col].astype(str).str.upper() != preferred_value).astype(int)
        df = df.sort_values(group_cols + ["_dataset_rank"])
        df = df.drop_duplicates(subset=group_cols, keep="first").drop(columns=["_dataset_rank"])
        return df

    @staticmethod
    def _apply_canonical_smiles(dataset: pd.DataFrame, canonical_smiles_map: Dict[str, str], 
                                combined_dict: Dict[str, str], reference_smiles_map: Dict[str, str]) -> pd.DataFrame:
        """Apply canonical SMILES for non-reference drug names."""
        if dataset.empty:
            raise ValueError("Cannot apply canonical SMILES: dataset is empty.")
        if not canonical_smiles_map:
            raise ValueError("Canonical SMILES map is empty; check drug_vocabulary.csv.")
        if not combined_dict:
            raise ValueError("Drug synonym mapping is empty; check drug_vocabulary.csv.")
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

    def zscore_cell_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score cell_line_features per gene/channel across cells."""
        features_list = df['cell_line_features'].tolist()
        if not features_list or not isinstance(features_list[0], np.ndarray):
            return df
        stacked = np.stack(features_list, axis=0).astype(np.float32, copy=False)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        std = np.where(std == 0, 1.0, std)
        normalized = (stacked - mean) / std
        df = df.copy()
        df['cell_line_features'] = [normalized[i] for i in range(normalized.shape[0])]
        return df
    
    @staticmethod
    def ensure_float32(features: object) -> Optional[np.ndarray]:
        """Convert nested feature arrays to float32 numpy arrays (or return None)."""
        if features is None:
            return None
        arr = np.asarray(features)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    def save_cell_line_features(self, dataset_df: pd.DataFrame, output_path: Path,
                                gene_axis: Optional[List[str]] = None) -> Path:
        """Save cell line features (and gene axis when provided) to a .npz lookup."""
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
        if gene_axis:
            lookup["__gene_axis__"] = np.asarray(gene_axis, dtype=str)
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
            gene_axis = getattr(self, "gene_axis", None)
            self.save_cell_line_features(dataset_df, self.cell_line_features_path, gene_axis=gene_axis)

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
