"""CCLE dataset creator."""
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.common import BaseDatasetCreator

warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

class CCLEDatasetCreator(BaseDatasetCreator):
    """CCLE dataset creator with multi-omics features."""

    def __init__(self, ic50_min: float = 1e-6, ic50_max: float = 1e6):
        """Initialize the CCLE dataset creator."""
        base_dir = Path(__file__).resolve().parent
        super().__init__(base_dir)
        self.ic50_min = ic50_min
        self.ic50_max = ic50_max
        # CCLE-specific path configurations
        self.expression_path = self.raw_dir / "CCLE_RNAseq_rsem_genes_tpm_20180929.txt"
        self.methylation_path = self.raw_dir / "CCLE_RRBS_TSS1kb_20181022.txt"
        self.cnv_path = self.raw_dir / "CCLE_ABSOLUTE_combined_20181227.csv"
        self.drug_response_path = self.raw_dir / "prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv"
        self.cell_map_path = self.raw_dir / "Cell_lines_annotations_20181226.txt"
        self.crispr_path = self.raw_dir / "CRISPRGeneDependency.csv"
        self.baseline_cell_lines = self.load_reference_cell_lines()

    def load_cell_line_map(self):
        """Map CCLE_ID to depmap_id for consistent joins."""
        ann = pd.read_csv(self.cell_map_path, sep='\t')
        ann['CCLE_ID'] = ann['CCLE_ID'].astype(str).str.upper()
        ann['depMapID'] = ann['depMapID'].astype(str)
        return dict(zip(ann['CCLE_ID'], ann['depMapID']))

    def load_ensembl_to_symbol_map(self):
        """Load Ensembl (no version) -> gene symbol map from a GTF."""
        mapping = {}
        for _chrom, _start, _end, _strand, attrs in self._iter_gtf_gene_rows():
            gene_id = attrs.get("gene_id", "")
            if not gene_id:
                continue
            gene_id = gene_id.split(".", 1)[0]
            symbol = attrs.get("gene_name", "")
            mapping[gene_id.upper()] = str(symbol).upper()
        return mapping

    def load_ccle_rrbs_gene_matrix(self, gene_axis, cell_map, cell_list=None, chunksize=2000):
        """Aggregate CCLE RRBS loci to gene-level methylation values."""
        gene_axis = list(gene_axis)
        gene_set = set(gene_axis)

        header = self._read_header_fields(self.methylation_path)
        if len(header) < 4:
            raise ValueError(f"Unexpected methylation header format in {self.methylation_path}")
        col_to_cell = self._resolve_cell_columns(header[3:], cell_map, cell_list)
        if not col_to_cell:
            raise ValueError(
                "No CCLE methylation columns could be mapped to cell lines via Cell_lines_annotations. "
                f"Check {self.cell_map_path} and methylation column naming in {self.methylation_path}."
            )

        sum_df = None
        count_df = None
        for grouped_sum, grouped_cnt in self._iter_rrbs_gene_chunks(col_to_cell, gene_set, chunksize):
            sum_df, count_df = self._merge_grouped_aggregates(sum_df, count_df, grouped_sum, grouped_cnt)

        if sum_df is None or count_df is None:
            raise ValueError("Failed to compute gene-level methylation matrix; no matching loci found.")

        return self._finalize_rrbs_matrix(sum_df, count_df, gene_axis, col_to_cell)

    def load_ccle_absolute_gene_cnv(self, gene_axis, cell_list=None, chunksize=200_000):
        """Convert CCLE ABSOLUTE segments to gene-level CNV values."""
        gene_axis = list(gene_axis)
        gene_set = set(gene_axis)

        cnv_path = self.cnv_path
        usecols = ["Chromosome", "Start", "End", "Modal_Total_CN", "depMapID"]
        gene_coords = self._load_gene_coords(gene_set)

        if not gene_coords:
            raise ValueError("No gene coordinates matched gene axis in GTF; cannot create CNV channel.")

        genes_by_chr = {}
        gene_index = {g: i for i, g in enumerate(gene_axis)}
        for gene, (chrom, mid) in gene_coords.items():
            genes_by_chr.setdefault(chrom, []).append((gene, mid))

        cell_list, cell_to_col, mat = self._init_cnv_matrix(gene_axis, cell_list)

        for chunk in self._iter_cnv_chunks(cnv_path, cell_list, usecols, chunksize):
            for chrom, sub in chunk.groupby("Chromosome"):
                gene_list = genes_by_chr.get(str(chrom))
                if not gene_list:
                    continue
                starts = sub["Start"].to_numpy(dtype=np.int64, copy=False)
                ends = sub["End"].to_numpy(dtype=np.int64, copy=False)
                vals = sub["Modal_Total_CN"].to_numpy(dtype=np.float32, copy=False)
                depmap_ids = sub["depMapID"].astype(str).to_numpy()

                for gene, mid in gene_list:
                    mask = (starts <= mid) & (ends >= mid)
                    if not mask.any():
                        continue
                    gi = gene_index.get(gene)
                    if gi is None:
                        continue
                    for depmap_id, v in zip(depmap_ids[mask], vals[mask]):
                        col, mat = self._ensure_cnv_cell(depmap_id, cell_list, cell_to_col, mat, len(gene_axis))
                        if col is None:
                            continue
                        mat[gi, col] = v

        if mat is None or not cell_to_col:
            raise ValueError("Failed to derive gene-level CNV values from ABSOLUTE segments.")

        cols_sorted = sorted(cell_to_col.items(), key=lambda kv: kv[1])
        col_names = [k for k, _ in cols_sorted]
        return pd.DataFrame(mat, index=gene_axis, columns=col_names, dtype=np.float32)

    def _iter_rrbs_gene_chunks(self, col_to_cell, gene_set, chunksize):
        usecols = ["locus_id"] + sorted(col_to_cell.keys())
        for chunk in pd.read_csv(
            self.methylation_path,
            sep="\t",
            usecols=usecols,
            low_memory=False,
            chunksize=chunksize,
        ):
            genes = chunk["locus_id"].astype(str).str.split("_", n=1, expand=True)[0].apply(BaseDatasetCreator.clean_string)
            chunk = chunk.drop(columns=["locus_id"])
            chunk.insert(0, "gene_name", genes)
            chunk = chunk[chunk["gene_name"].isin(gene_set)]
            if chunk.empty:
                continue
            values = chunk.drop(columns=["gene_name"]).apply(pd.to_numeric, errors="coerce")
            key = chunk["gene_name"]
            yield values.groupby(key).sum(min_count=1), values.groupby(key).count()

    @staticmethod
    def _finalize_rrbs_matrix(sum_df, count_df, gene_axis, col_to_cell):
        mean_df = sum_df.divide(count_df.where(count_df != 0))
        mean_df = mean_df.reindex(gene_axis)
        mean_df = mean_df.rename(columns=col_to_cell)
        if mean_df.columns.duplicated().any():
            mean_df = mean_df.groupby(level=0, axis=1).mean()
        return mean_df.astype(np.float32)

    @staticmethod
    def _init_cnv_matrix(gene_axis, cell_list):
        if cell_list is None:
            return None, {}, None
        cell_list = sorted(set(cell_list))
        cell_to_col = {c: i for i, c in enumerate(cell_list)}
        mat = np.full((len(gene_axis), len(cell_list)), np.nan, dtype=np.float32)
        return cell_list, cell_to_col, mat

    def _iter_cnv_chunks(self, cnv_path, cell_list, usecols, chunksize):
        for chunk in pd.read_csv(cnv_path, usecols=usecols, low_memory=False, chunksize=chunksize):
            chunk = chunk.copy()
            chunk["depMapID"] = chunk["depMapID"].astype(str)
            if cell_list is not None:
                chunk = chunk[chunk["depMapID"].isin(cell_list)]
            if chunk.empty:
                continue
            chunk["Chromosome"] = chunk["Chromosome"].astype(str).str.replace("chr", "", regex=False).str.upper()
            chunk["Modal_Total_CN"] = pd.to_numeric(chunk["Modal_Total_CN"], errors="coerce")
            chunk["Start"] = pd.to_numeric(chunk["Start"], errors="coerce")
            chunk["End"] = pd.to_numeric(chunk["End"], errors="coerce")
            chunk = chunk.dropna(subset=["Chromosome", "Start", "End", "Modal_Total_CN", "depMapID"])
            if chunk.empty:
                continue
            yield chunk

    @staticmethod
    def clean_crispr_gene_columns(columns):
        cleaned = []
        for col in columns:
            if col.lower().startswith('unnamed'):
                cleaned.append('cell_line_name')
                continue
            cleaned.append(col.split(' (')[0])
        return cleaned

    def _iter_gtf_gene_rows(self):
        """Yield (chrom, start, end, strand, attrs) tuples for gene features in the GTF."""
        with self.gencode_gtf_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                chrom, _source, feature, start, end, _score, strand, _frame, attrs = parts
                if feature != "gene":
                    continue
                attrs_dict = self.parse_gtf_attributes(attrs)
                yield chrom, start, end, strand, attrs_dict

    def _load_gene_coords(self, gene_set):
        """Load gene midpoint coordinates for genes in gene_set."""
        gene_coords = {}
        for chrom, start, end, _strand, attrs in self._iter_gtf_gene_rows():
            gene_name = attrs.get("gene_name", "")
            if not gene_name:
                continue
            gene_clean = BaseDatasetCreator.clean_string(gene_name)
            if gene_clean not in gene_set or gene_clean in gene_coords:
                continue
            try:
                s = int(start)
                e = int(end)
            except ValueError:
                continue
            midpoint = (s + e) // 2
            chrom_norm = str(chrom).replace("chr", "").upper()
            gene_coords[gene_clean] = (chrom_norm, midpoint)
            if len(gene_coords) == len(gene_set):
                break
        return gene_coords

    @staticmethod
    def _resolve_cell_columns(cell_cols, cell_map, cell_list=None):
        col_to_cell = {}
        for col in cell_cols:
            depmap_id = cell_map.get(str(col).upper())
            if depmap_id is None:
                continue
            if cell_list is not None and depmap_id not in cell_list:
                continue
            col_to_cell[col] = depmap_id
        return col_to_cell

    def _filter_reference_cells(self, expr_df, cell_map):
        expr_cols_upper = {c.upper() for c in expr_df.columns}
        cell_list = {depmap_id for ccle_id, depmap_id in cell_map.items() if ccle_id.upper() in expr_cols_upper}
        if self.baseline_cell_lines:
            cell_list = cell_list.intersection(self.baseline_cell_lines)
        return cell_list

    def _filter_expression_cells(self, expr_df, cell_map, cell_list):
        expr_col_by_cell = {}
        keep_cols = []
        for col in expr_df.columns:
            dep = cell_map.get(str(col).upper())
            if dep in cell_list:
                expr_col_by_cell[dep] = col
                keep_cols.append(col)
        return expr_df[keep_cols], expr_col_by_cell

    @staticmethod
    def _build_expression_lookup(expr_df, expr_col_by_cell, gene_axis):
        expr_df = expr_df.reindex(gene_axis)
        lookup = {}
        for depmap_id, col in expr_col_by_cell.items():
            if col not in expr_df.columns:
                continue
            lookup[depmap_id] = expr_df[col].to_numpy(dtype=np.float32, copy=False)
        return lookup

    @staticmethod
    def _build_crispr_lookup(crispr_raw, gene_axis):
        gene_cols = [c for c in crispr_raw.columns if c != 'ModelID']
        crispr = crispr_raw.drop_duplicates(subset=['ModelID'], keep='first').copy()
        crispr = crispr.rename(columns={col: BaseDatasetCreator.clean_string(col) for col in gene_cols})
        crispr = crispr.loc[:, ~crispr.columns.duplicated()].set_index('ModelID')
        crispr = crispr.apply(pd.to_numeric, errors='coerce')
        crispr = crispr.reindex(columns=gene_axis)
        values = crispr.to_numpy(dtype=np.float32)
        return {model_id: values[i] for i, model_id in enumerate(crispr.index.astype(str))}

    def _build_cell_feature_array(self, cell_line_depmap, gene_axis, expr_lookup, crispr_lookup, cnv_lookup, meth_lookup):
        gene_count = len(gene_axis)
        expr_arr = expr_lookup.get(cell_line_depmap)
        if expr_arr is None:
            expr_arr = np.full(gene_count, np.nan, dtype=np.float32)

        crispr_arr = crispr_lookup.get(cell_line_depmap)
        if crispr_arr is None:
            crispr_arr = np.full(gene_count, np.nan, dtype=np.float32)

        cnv_arr = cnv_lookup.get(cell_line_depmap)
        if cnv_arr is None:
            cnv_arr = np.full(gene_count, np.nan, dtype=np.float32)

        meth_arr = meth_lookup.get(cell_line_depmap)
        if meth_arr is None:
            meth_arr = np.full(gene_count, np.nan, dtype=np.float32)

        arr = np.stack([expr_arr, crispr_arr, cnv_arr, meth_arr], axis=1)
        if np.isnan(arr).all():
            return None
        return arr

    def _build_cell_features_df(self, cell_lines, gene_axis, expr_lookup, crispr_lookup, cnv_lookup, meth_lookup):
        cell_features = []
        for cell_line in tqdm(cell_lines):
            arr = self._build_cell_feature_array(
                cell_line, gene_axis, expr_lookup, crispr_lookup, cnv_lookup, meth_lookup
            )
            if arr is None:
                continue
            cell_features.append({'cell_line_name': cell_line, 'cell_line_features': arr})
        return pd.DataFrame(cell_features)

    def _load_crispr_data(self):
        crispr_raw = pd.read_csv(self.crispr_path)
        crispr_raw.columns = self.clean_crispr_gene_columns(crispr_raw.columns)
        crispr_raw.rename(columns={'cell_line_name': 'ModelID'}, inplace=True)
        crispr_raw['ModelID'] = crispr_raw['ModelID'].astype(str)
        return crispr_raw

    @staticmethod
    def _filter_eligible_cells(expr_cells, crispr_cells, meth_cells, cnv_cells, cells_with_drug_response):
        all_ccle_cells = expr_cells.union(crispr_cells, meth_cells, cnv_cells)
        eligible_cells = all_ccle_cells.intersection(cells_with_drug_response)
        logger.info(
            "CCLE cells by omics: expr=%d, crispr=%d, cnv=%d, meth=%d, union=%d, with_drug_response=%d",
            len(expr_cells), len(crispr_cells), len(cnv_cells), len(meth_cells),
            len(all_ccle_cells), len(eligible_cells)
        )
        return eligible_cells

    def load_gene_expression_data(self):
        """Load expression table and return genes, expression, and cell map."""
        expr_df = pd.read_csv(self.expression_path, sep='\t')
        expr_df = expr_df.rename(columns={'gene_id': 'gene_name'})
        expr_df['gene_name'] = expr_df['gene_name'].apply(lambda x: str(x).split('.')[0].upper())

        # Map Ensembl IDs to symbols to align with census filtering
        ens2sym = self.load_ensembl_to_symbol_map()
        expr_df['gene_symbol'] = expr_df['gene_name'].apply(lambda x: ens2sym.get(x, x))
        expr_df.drop(columns=['gene_name'], inplace=True)
        expr_df.rename(columns={'gene_symbol': 'gene_name'}, inplace=True)

        # Drop non-numeric columns and cast before aggregation
        non_numeric_cols = [c for c in ['transcript_ids'] if c in expr_df.columns]
        expr_df.drop(columns=non_numeric_cols, inplace=True, errors='ignore')
        numeric_cols = [c for c in expr_df.columns if c != 'gene_name']
        expr_df[numeric_cols] = expr_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        expr_df['gene_name'] = expr_df['gene_name'].apply(BaseDatasetCreator.clean_string)
        expr_df = expr_df[expr_df['gene_name'] != '']

        # Deduplicate after cleaning to ensure unique index for fast lookup.
        expr_df = expr_df.groupby('gene_name', as_index=False)[numeric_cols].mean()

        cell_map = self.load_cell_line_map()
        return expr_df['gene_name'].tolist(), expr_df, cell_map

    def create_drug_cell_dataframe(self, drug_list=None, drug_vocab=None, cell_list=None):
        """Create drug-cell pairs using PRISM repurposing response file."""
        drug_cell_pairs_raw = pd.read_csv(self.drug_response_path)
        drug_cell_pairs_raw['drug_name'] = drug_cell_pairs_raw['name'].apply(BaseDatasetCreator.clean_string)
        combined_dict = drug_vocab[1] if drug_vocab else None
        if combined_dict:
            drug_cell_pairs_raw['drug_name'] = drug_cell_pairs_raw['drug_name'].apply(
                lambda n: combined_dict.get(n, n)
            )
        if drug_list:
            drug_list_clean = {BaseDatasetCreator.clean_string(name) for name in drug_list}
            drug_cell_pairs_raw = drug_cell_pairs_raw[drug_cell_pairs_raw['drug_name'].isin(drug_list_clean)]
        drug_cell_pairs_raw['cell_line_name'] = drug_cell_pairs_raw['depmap_id'].astype(str)
        drug_cell_pairs_raw['smiles'] = drug_cell_pairs_raw.get('smiles')
        if 'smiles' in drug_cell_pairs_raw.columns:
            drug_cell_pairs_raw['smiles'] = drug_cell_pairs_raw['smiles'].apply(self.normalize_smiles_text)

        ic50_um = pd.to_numeric(drug_cell_pairs_raw.get('ic50'), errors='coerce')
        is_finite = np.isfinite(ic50_um.to_numpy())
        valid = is_finite & (ic50_um > 0)
        if self.ic50_min is not None:
            valid &= (ic50_um >= float(self.ic50_min))
        if self.ic50_max is not None:
            valid &= (ic50_um <= float(self.ic50_max))
        dropped = int((~valid).sum())
        if dropped:
            logger.info("Dropping %d PRISM rows with invalid/out-of-range IC50.", dropped)
        drug_cell_pairs_raw = drug_cell_pairs_raw.loc[valid].copy()

        ic50_um = pd.to_numeric(drug_cell_pairs_raw['ic50'], errors='coerce')
        drug_cell_pairs_raw['pic50'] = -np.log10(ic50_um * 1e-6)

        if cell_list is not None:
            cell_list = set(cell_list)
            drug_cell_pairs_raw = drug_cell_pairs_raw[drug_cell_pairs_raw['cell_line_name'].isin(cell_list)]
        drug_cell_pairs_raw = drug_cell_pairs_raw[['drug_name', 'cell_line_name', 'pic50', 'smiles']]
        drug_cell_pairs_raw.dropna(subset=['drug_name', 'cell_line_name', 'pic50'], inplace=True)

        if not drug_cell_pairs_raw.empty:
            before = len(drug_cell_pairs_raw)
            drug_cell_pairs_raw = (
                drug_cell_pairs_raw
                .sort_values(['drug_name', 'cell_line_name'])
                .groupby(['drug_name', 'cell_line_name'], as_index=False)
                .agg(
                    pic50=('pic50', 'median'),
                    smiles=('smiles', 'first'),
                )
            )
            after = len(drug_cell_pairs_raw)
            if after != before:
                logger.info(
                    "Aggregated PRISM replicates from %d to %d rows (median pic50 per drug/cell).",
                    before,
                    after
                )
        return drug_cell_pairs_raw

    def create_dataset(self):
        """Create the CCLE dataset from raw files."""
        vocab_tuple = self.load_drug_vocabulary()
        canonical_smiles_map = self.load_canonical_smiles_map()
        drug_list, reference_smiles_map = self._filter_reference_drugs()
        gene_axis = self.load_cross_domain_gene_axis()

        _, expr_df, cell_map = self.load_gene_expression_data()
        expr_df = expr_df.set_index('gene_name')
        expr_df = expr_df[~expr_df.index.duplicated(keep='first')]
        cell_list = self._filter_reference_cells(expr_df, cell_map)

        drug_cell_smiles_df = self.create_drug_cell_dataframe(
            drug_list=drug_list,
            drug_vocab=vocab_tuple,
            cell_list=cell_list,
        )
        drug_cell_smiles_df = self._harmonize_reference_smiles(drug_cell_smiles_df, reference_smiles_map)
        cell_list = cell_list.intersection(set(drug_cell_smiles_df['cell_line_name'].unique()))

        expr_df, expr_col_by_cell = self._filter_expression_cells(expr_df, cell_map, cell_list)
        expr_lookup = self._build_expression_lookup(expr_df, expr_col_by_cell, gene_axis)

        crispr_raw = self._load_crispr_data()
        crispr_lookup = self._build_crispr_lookup(crispr_raw, gene_axis)

        meth_matrix = self.load_ccle_rrbs_gene_matrix(gene_axis, cell_map, cell_list=cell_list)
        cnv_matrix = self.load_ccle_absolute_gene_cnv(gene_axis, cell_list=cell_list)

        cells_with_expression = set(expr_col_by_cell.keys())
        cells_with_crispr = set(crispr_lookup.keys())
        cells_with_meth = set(meth_matrix.columns.astype(str))
        cells_with_cnv = set(cnv_matrix.columns.astype(str))
        cells_with_drug_response = set(drug_cell_smiles_df['cell_line_name'].unique())
        eligible_cells = self._filter_eligible_cells(
            cells_with_expression,
            cells_with_crispr,
            cells_with_meth,
            cells_with_cnv,
            cells_with_drug_response,
        )
        
        drug_cell_smiles_df = drug_cell_smiles_df[drug_cell_smiles_df['cell_line_name'].isin(eligible_cells)].copy()
        cell_list = set(drug_cell_smiles_df['cell_line_name'].unique())

        meth_lookup = self._matrix_to_lookup(meth_matrix, cell_list)
        cnv_lookup = self._matrix_to_lookup(cnv_matrix, cell_list)
        cell_line_features_df = self._build_cell_features_df(
            drug_cell_smiles_df['cell_line_name'].unique(),
            gene_axis,
            expr_lookup,
            crispr_lookup,
            cnv_lookup,
            meth_lookup,
        )
        logger.info("Cell features before filtering: %d cells", len(cell_line_features_df))
        cell_line_features_df, gene_axis = self.filter_genes_by_missing_values(
            cell_line_features_df,
            gene_axis
        )

        cell_line_features_df = self.filter_cell_lines_by_missing_values(cell_line_features_df)
        logger.info("Cell features after filtering: %d cells", len(cell_line_features_df))

        drug_cell_smiles_df = drug_cell_smiles_df[
            drug_cell_smiles_df['cell_line_name'].isin(cell_line_features_df['cell_line_name'])
        ].copy()

        dataset = pd.merge(drug_cell_smiles_df, cell_line_features_df, how='inner')
        dataset = dataset[['drug_name', 'cell_line_name', 'pic50', 'smiles', 'cell_line_features']]

        if vocab_tuple[0] is not None:
            dataset = self.expand_drug_synonyms(dataset, vocab_tuple)
            dataset = self._harmonize_reference_smiles(dataset, reference_smiles_map)

        # For CCLE-only drug aliases not present in the reference set, fall back to
        # canonicalsmiles from the shared drugbank vocabulary via the same synonym->common mapping.
        combined_dict = vocab_tuple[1] if vocab_tuple else None
        dataset = self._apply_canonical_smiles(dataset, canonical_smiles_map, combined_dict, reference_smiles_map)
        dataset = self._apply_pubchem_smiles_fallback(dataset)

        dataset = dataset.dropna(subset=['smiles'])

        dataset = dataset.dropna(subset=['cell_line_features', 'drug_name', 'pic50'])

        return dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    try:
        CCLEDatasetCreator().create_and_save_dataset()
    except FileNotFoundError as exc:
        logger.error("Unable to create CCLE dataset: %s", exc)
    logger.info("CCLE dataset artifacts stored in processed/")
