"""GDSC dataset creator."""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from dataset.common import BaseDatasetCreator

logger = logging.getLogger(__name__)


class GDSCDatasetCreator(BaseDatasetCreator):
    """GDSC dataset creator with multi-omics features."""

    def __init__(self):
        """Initialize the GDSC dataset creator."""
        base_dir = Path(__file__).resolve().parent
        super().__init__(base_dir)

        # GDSC raw input paths
        self.drug_response_path = self.raw_dir / "gdsc_dose_response.csv"
        self.drug_smiles_path = self.raw_dir / "gdsc_drug_smiles.csv"
        self.cell_line_map_path = self.raw_dir / "gdsc_cell_lines.csv"
        self.expression_path = self.raw_dir / "gdsc_expression.csv"
        self.cnv_summary_path = self.raw_dir / "cnv_summary_20250207.csv"
        self.methylation_matrix_path = self.raw_dir / "F2_METH_CELL_DATA.txt"
        self.methylation_sample_map_path = self.raw_dir / "methSampleId_2_cosmicIds.csv"
        self.project_score_fitness_path = self.raw_dir / "Project_score_combined_Sanger_v2_Broad_21Q2_fitness_scores_scaled_bayesian_factors_20250624.tsv"
        self.model_list_download_path = self.raw_dir / "model_list_20250630.csv"

        self.baseline_gdsc_ids = self.load_reference_drug_ids("gdsc_id")
        self.baseline_cell_lines = self.load_reference_cell_lines()

    def load_cell_line_map(self, cosmic_col: str = "cosmic_id", depmap_col: str = "depmap_id") -> Dict[str, str]:
        """Load COSMIC->DepMap mapping."""
        df = self._load_clean_table(self.cell_line_map_path)
        df = df.dropna(subset=[cosmic_col, depmap_col]).copy()
        cosmic_series = df[cosmic_col].astype(str)
        depmap_series = df[depmap_col].astype(str)
        return dict(zip(cosmic_series, depmap_series))

    def _load_methylation_sample_map(self) -> Dict[str, str]:
        """Load Sentrix methylation sample mapping to COSMIC IDs."""
        df = self._load_clean_table(self.methylation_sample_map_path)
        sentrix_id_col = self._require_column(df.columns, {"sentrixid", "sentrix_id"}, 
                                              self.methylation_sample_map_path, "sentrix_id")
        sentrix_pos_col = self._require_column(df.columns, {"sentrixposition", "sentrix_position"}, 
                                               self.methylation_sample_map_path, "sentrix_position")
        cosmic_col = self._require_column(df.columns, {"cosmicid", "cosmic_id"}, 
                                          self.methylation_sample_map_path, "cosmic_id")

        sentrix_id = df[sentrix_id_col].astype(str).str.replace(".0", "", regex=False).str.strip()
        sentrix_pos = df[sentrix_pos_col].astype(str).str.strip()
        key = sentrix_id + "_" + sentrix_pos

        mapping = {
            str(key_val): str(int(float(cosmic_id)))
            for key_val, cosmic_id in zip(key, df[cosmic_col].astype(str))
            if key_val.strip() and cosmic_id.strip() and cosmic_id != "nan"
        }

        if not mapping:
            raise ValueError("Sample mapping produced no usable Sentrix->COSMIC mappings (cosmic_id all missing).")
        return mapping

    def _load_gene_tss_index(self, gene_axis: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Create chromosome->TSS index arrays for genes in gene_axis."""
        wanted = set(gene_axis)
        gene_to_tss = {}

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
                attrs_dict = BaseDatasetCreator.parse_gtf_attributes(attrs)
                gene_name = attrs_dict.get("gene_name")
                if not gene_name:
                    continue
                gene_clean = BaseDatasetCreator.clean_string(gene_name)
                if gene_clean not in wanted or gene_clean in gene_to_tss:
                    continue
                try:
                    s = int(start)
                    e = int(end)
                except ValueError:
                    continue
                tss = s if strand == "+" else e
                chrom_norm = str(chrom).replace("chr", "").upper()
                gene_to_tss[gene_clean] = (chrom_norm, tss)
                if len(gene_to_tss) == len(wanted):
                    break

        axis_index = {gene: i for i, gene in enumerate(gene_axis)}
        by_chr: Dict[str, List[Tuple[int, int]]] = {}
        for gene, (chrom, tss) in gene_to_tss.items():
            idx = axis_index.get(gene)
            if idx is None:
                continue
            by_chr.setdefault(chrom, []).append((tss, idx))

        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for chrom, entries in by_chr.items():
            entries.sort(key=lambda x: x[0])
            tss_arr = np.asarray([t for t, _ in entries], dtype=np.int64)
            idx_arr = np.asarray([i for _, i in entries], dtype=np.int32)
            out[str(chrom)] = (tss_arr, idx_arr)
        return out

    def _load_methylation_lookup_from_f2(self, gene_axis: List[str], depmap_ids: Set[str], cosmic_to_depmap: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Create gene-level methylation vectors (aligned to gene_axis) from the GDSC F2 matrix."""
        sentrix_to_cosmic = self._load_methylation_sample_map()
        tss_by_chr = self._load_gene_tss_index(gene_axis)

        cosmic_to_depmap = {cosmic: depmap for cosmic, depmap in cosmic_to_depmap.items() if depmap in depmap_ids}
        wanted_cosmic = set(cosmic_to_depmap.keys())
        if not wanted_cosmic:
            raise ValueError("No COSMIC IDs available for requested depmap_ids; cannot map methylation columns.")

        header_cols = self._read_header_fields(self.methylation_matrix_path)
        sample_cols = header_cols[1:]

        selected_indices: List[int] = []
        selected_depmap_ids: List[str] = []
        seen_depmap: Set[str] = set()
        for idx, sample in enumerate(sample_cols):
            cosmic = sentrix_to_cosmic.get(sample)
            if cosmic is None or cosmic not in wanted_cosmic:
                continue
            depmap_id = cosmic_to_depmap.get(cosmic)
            if depmap_id is None or depmap_id in seen_depmap:
                continue
            seen_depmap.add(depmap_id)
            selected_indices.append(idx)
            selected_depmap_ids.append(depmap_id)

        if not selected_indices:
            raise ValueError("No methylation columns mapped to requested depmap_ids.")

        promoter_window = 1500
        sum_mat = np.zeros((len(selected_depmap_ids), len(gene_axis)), dtype=np.float64)
        count_mat = np.zeros((len(selected_depmap_ids), len(gene_axis)), dtype=np.int32)

        with self.methylation_matrix_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in tqdm(handle, desc="GDSC methylation"):
                if not line or not line.startswith("chr"):
                    continue
                parts = line.rstrip("\n\r").split("\t")
                if len(parts) < 2:
                    continue
                region = parts[0]
                try:
                    chrom, coords = region.split(":")
                    start_str, end_str = coords.split("-")
                    start = int(start_str)
                    end = int(end_str)
                except Exception:
                    continue

                chrom_key = str(chrom).replace("chr", "").upper()
                tss_tuple = tss_by_chr.get(chrom_key)
                if not tss_tuple:
                    continue
                tss_arr, idx_arr = tss_tuple

                region_start = start - promoter_window
                region_end = end + promoter_window
                left = np.searchsorted(tss_arr, region_start, side="left")
                right = np.searchsorted(tss_arr, region_end, side="right")
                if left >= right:
                    continue

                try:
                    values = [float(parts[i + 1]) for i in selected_indices]
                except Exception:
                    continue

                values_arr = np.asarray(values, dtype=np.float64)
                if np.all(np.isnan(values_arr)):
                    continue
                genes_idx = idx_arr[left:right]
                sum_mat[:, genes_idx] += np.nan_to_num(values_arr)[:, None]
                count_mat[:, genes_idx] += (~np.isnan(values_arr))[:, None]

        with np.errstate(invalid="ignore", divide="ignore"):
            out_mat = sum_mat / count_mat
        out_mat[count_mat == 0] = np.nan

        return {depmap_id: out_mat[i].astype(np.float32, copy=False) for i, depmap_id in enumerate(selected_depmap_ids)}

    def _load_crispr_lookup_from_project_score(self, gene_axis: List[str], depmap_ids: Set[str]) -> Dict[str, np.ndarray]:
        """Load CRISPR dependency vectors from Project Score fitness scores."""
        source_path = self.project_score_fitness_path
        model_list = self._load_clean_table(self.model_list_download_path)
        model_id_col = self._require_column(model_list.columns, {"modelid", "model_id"}, self.model_list_download_path, "model_id")
        broad_col = self._require_column(model_list.columns, {"broadid", "broad_id"}, self.model_list_download_path, "broad_id")
        model_list = model_list.dropna(subset=[model_id_col, broad_col]).copy()
        model_list[model_id_col] = model_list[model_id_col].astype(str)
        model_list[broad_col] = model_list[broad_col].astype(str)
        model_to_depmap = dict(zip(model_list[model_id_col], model_list[broad_col]))

        wanted = set(depmap_ids)

        sep = "\t"
        preview = pd.read_csv(source_path, sep=sep, nrows=6, low_memory=False)
        first_col = preview.columns[0]
        if "model_id" not in preview[first_col].astype(str).str.lower().tolist():
            raise ValueError(f"Unrecognized Project Score format in {source_path} (missing 'model_id' metadata row).")

        if len(preview.columns) < 5:
            raise ValueError(f"Project Score file {source_path} has too few columns.")
        gene_id_col = preview.columns[0]
        symbol_col = preview.columns[1]
        model_cols = list(preview.columns[3:])

        model_id_row = preview.loc[preview[gene_id_col].astype(str).str.lower() == "model_id"].iloc[0]

        depmap_to_model_cols: Dict[str, List[str]] = {}
        for col in model_cols:
            model_id = model_id_row.get(col)
            if pd.isna(model_id):
                continue
            depmap_id = model_to_depmap.get(str(model_id))
            if depmap_id is None:
                continue
            depmap_id = str(depmap_id)
            if depmap_id not in wanted:
                continue
            depmap_to_model_cols.setdefault(depmap_id, []).append(col)

        if not depmap_to_model_cols:
            raise ValueError("Project Score file did not yield any matching depmap_ids. " 
                             "Check that model_list_*.csv BROAD_IDs match the cells in your baseline.")

        depmap_ids_order = sorted(depmap_to_model_cols.keys())
        all_cols: List[str] = sorted({c for cols in depmap_to_model_cols.values() for c in cols})
        col_index = {c: i for i, c in enumerate(all_cols)}
        groups = [[col_index[c] for c in depmap_to_model_cols[depmap_id]] for depmap_id in depmap_ids_order]

        gene_to_idx = {g: i for i, g in enumerate(gene_axis)}
        mat = np.full((len(depmap_ids_order), len(gene_axis)), np.nan, dtype=np.float32)
        seen: Set[str] = set()

        meta_labels = {"model_id", "source", "qc_pass", "gene_id", "model_name"}

        for chunk in pd.read_csv(source_path, sep=sep, chunksize=2000, low_memory=False):
            chunk = chunk[~chunk[gene_id_col].astype(str).str.lower().isin(meta_labels)].copy()
            if chunk.empty:
                continue
            chunk["_gene_clean"] = chunk[symbol_col].astype(str).apply(BaseDatasetCreator.clean_string)
            chunk = chunk[chunk["_gene_clean"].isin(gene_to_idx)]
            if chunk.empty:
                continue
            chunk = chunk.drop_duplicates(subset=["_gene_clean"])
            chunk = chunk[~chunk["_gene_clean"].isin(seen)]
            if chunk.empty:
                continue

            values = chunk[all_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=False)

            agg = np.full((values.shape[0], len(depmap_ids_order)), np.nan, dtype=np.float32)
            for i, idxs in enumerate(groups):
                if len(idxs) == 1:
                    agg[:, i] = values[:, idxs[0]]
                else:
                    with np.errstate(all="ignore"):
                        agg[:, i] = np.nanmean(values[:, idxs], axis=1)

            genes = chunk["_gene_clean"].to_numpy(dtype=str, copy=False)
            for row_idx, g in enumerate(genes):
                gi = gene_to_idx[g]
                mat[:, gi] = agg[row_idx]
                seen.add(g)
            if len(seen) == len(gene_axis):
                break

        return {depmap_id: mat[i].astype(np.float32, copy=False) for i, depmap_id in enumerate(depmap_ids_order)}

    def _load_cnv_lookup(self, gene_axis: List[str], depmap_ids: Set[str], 
                         cosmic_to_depmap: Dict[str, str]) -> Dict[str, np.ndarray]:
        """Create gene-level CNV vectors from the GDSC CNV summary."""
        model_list = self._load_clean_table(self.model_list_download_path)
        model_id_col = self._require_column(model_list.columns, {"modelid", "model_id"}, self.model_list_download_path, "model_id")
        cosmic_col = self._require_column(model_list.columns, {"cosmicid", "cosmic_id"}, self.model_list_download_path, "cosmic_id")
        model_list = model_list.dropna(subset=[model_id_col, cosmic_col]).copy()
        model_list[model_id_col] = model_list[model_id_col].astype(str)
        model_list[cosmic_col] = model_list[cosmic_col].astype(str)
        model_to_cosmic = dict(zip(model_list[model_id_col], model_list[cosmic_col]))

        cnv = self._load_clean_table(self.cnv_summary_path)
        model_col = self._require_column(cnv.columns, {"modelid", "model_id"}, 
                                         self.cnv_summary_path, "model_id")
        gene_col = self._require_column(cnv.columns, {"genename", "gene_name", "hgncsymbol", "symbol"}, 
                                        self.cnv_summary_path, "gene_name")
        value_col = self._require_column(cnv.columns, {"totalcopynumber", "total_copy_number"}, 
                                         self.cnv_summary_path, "total_copy_number")

        cnv = cnv.dropna(subset=[model_col, gene_col, value_col]).copy()
        cnv[model_col] = cnv[model_col].astype(str)
        cnv[gene_col] = cnv[gene_col].astype(str).apply(BaseDatasetCreator.clean_string)
        cnv[value_col] = pd.to_numeric(cnv[value_col], errors="coerce")
        cnv = cnv.dropna(subset=[value_col])

        axis_index = {g: i for i, g in enumerate(gene_axis)}
        lookup = {depmap_id: np.full(len(gene_axis), np.nan, dtype=np.float32) for depmap_id in depmap_ids}

        for row in cnv.itertuples(index=False):
            model_id = getattr(row, model_col)
            gene = getattr(row, gene_col)
            value = getattr(row, value_col)
            cosmic_id = model_to_cosmic.get(model_id)
            if cosmic_id is None:
                continue
            depmap_id = cosmic_to_depmap.get(cosmic_id)
            if depmap_id is None or depmap_id not in lookup:
                continue
            idx = axis_index.get(gene)
            if idx is None:
                continue
            lookup[depmap_id][idx] = value
        return lookup

    def _resolve_response_columns(self, resp: pd.DataFrame) -> Tuple[str, str, str, Optional[str]]:
        """Resolve required response columns."""
        cosmic_col = self._require_column(resp.columns, {"cosmicid", "cosmic_id"}, self.drug_response_path, "cosmic_id")
        drug_id_col = self._require_column(resp.columns, {"gdscid", "gdsc_id", "drugid", "drug_id"}, self.drug_response_path, "drug_id")
        ic50_col = self._require_column(resp.columns, {"ic50um", "ic50_um", "ic50published", "ic50_published", "ic50"}, self.drug_response_path, "ic50")
        dataset_col = "dataset" if "dataset" in resp.columns else None
        return cosmic_col, drug_id_col, ic50_col, dataset_col

    def _normalize_response_table(self, resp: pd.DataFrame, cosmic_col: str, drug_id_col: str, ic50_col: str, cosmic_to_depmap: Dict[str, str]) -> pd.DataFrame:
        """Normalize response values and map cell_line_name."""
        resp[cosmic_col] = resp[cosmic_col].astype(str)
        resp[drug_id_col] = pd.to_numeric(resp[drug_id_col], errors="coerce").astype("Int64")
        resp[ic50_col] = pd.to_numeric(resp[ic50_col], errors="coerce")
        resp = resp.dropna(subset=[cosmic_col, drug_id_col, ic50_col])
        resp = resp[resp[ic50_col] > 0].copy()
        resp["cell_line_name"] = resp[cosmic_col].map(cosmic_to_depmap).fillna(resp[cosmic_col])
        return resp

    def _resolve_drug_columns(self, drugs: pd.DataFrame) -> Tuple[str, str, str]:
        """Resolve required drug columns."""
        drug_smiles_id_col = self._require_column(drugs.columns, {"gdscid", "gdsc_id", "drugid", "drug_id"}, self.drug_smiles_path, "drug_id")
        drug_name_col = self._require_column(drugs.columns, {"drugname", "drug_name", "name"}, self.drug_smiles_path, "drug_name")
        smiles_col = self._require_column(drugs.columns, {"smiles", "canonicalsmiles", "canonical_smiles"}, self.drug_smiles_path, "smiles")
        return drug_smiles_id_col, drug_name_col, smiles_col

    def _normalize_drug_table(self, drugs: pd.DataFrame, drug_smiles_id_col: str, drug_name_col: str, smiles_col: str) -> pd.DataFrame:
        """Normalize drug table values."""
        drugs[drug_smiles_id_col] = pd.to_numeric(drugs[drug_smiles_id_col], errors="coerce").astype("Int64")
        drugs = drugs.dropna(subset=[drug_smiles_id_col, drug_name_col]).copy()
        drugs["drug_name"] = drugs[drug_name_col].astype(str).apply(BaseDatasetCreator.clean_string)
        drugs["smiles"] = drugs[smiles_col].astype(str)
        return drugs.drop_duplicates(subset=[drug_smiles_id_col])

    def _apply_baseline_filters(self, merged: pd.DataFrame, drug_id_col: str) -> pd.DataFrame:
        """Apply baseline filters."""
        if self.baseline_gdsc_ids:
            merged = merged[merged[drug_id_col].isin(self.baseline_gdsc_ids)].copy()
        if self.baseline_cell_lines:
            merged = merged[merged["cell_line_name"].isin(self.baseline_cell_lines)].copy()
        return merged

    @staticmethod
    def _finalize_response_records(merged: pd.DataFrame, ic50_col: str) -> pd.DataFrame:
        """Compute pic50 and select output columns."""
        merged["pic50"] = -np.log10(merged[ic50_col] * 1e-6)
        records = merged[["cell_line_name", "drug_name", "smiles", "pic50"]].dropna().copy()
        if records.empty:
            raise ValueError("No usable GDSC records after filtering.")
        return records

    def _load_drug_response_records(self, cosmic_to_depmap: Dict[str, str]) -> pd.DataFrame:
        """Load and filter drug response records with SMILES and pic50."""
        resp = self._load_clean_table(self.drug_response_path)
        cosmic_col, drug_id_col, ic50_col, dataset_col = self._resolve_response_columns(resp)
        resp = self._normalize_response_table(resp, cosmic_col, drug_id_col, ic50_col, cosmic_to_depmap)

        drugs = self._load_clean_table(self.drug_smiles_path)
        drug_smiles_id_col, drug_name_col, smiles_col = self._resolve_drug_columns(drugs)
        drugs = self._normalize_drug_table(drugs, drug_smiles_id_col, drug_name_col, smiles_col)

        merged = resp.merge(drugs[[drug_smiles_id_col, "drug_name", "smiles"]],
                            left_on=drug_id_col,
                            right_on=drug_smiles_id_col,
                            how="inner",)

        merged = self._apply_baseline_filters(merged, drug_id_col)
        merged = self._dedupe_by_dataset_preference(merged, ["cell_line_name", drug_id_col], dataset_col)
        merged["drug_id"] = pd.to_numeric(merged[drug_id_col], errors="coerce")
        merged = self._apply_pubchem_smiles_fallback(merged)
        return self._finalize_response_records(merged, ic50_col)

    def _load_expression_matrix(self, gene_axis: List[str], records: pd.DataFrame, cosmic_to_depmap: Dict[str, str]) -> pd.DataFrame:
        """Load expression matrix aligned to gene_axis and depmap_id columns."""
        expr_sep = self._infer_delimiter(self.expression_path)
        header_cells = self._read_header_fields(self.expression_path, delimiter=expr_sep)
        gene_col = header_cells[0]
        available_cell_cols = set(header_cells[1:])

        wanted_depmap = sorted(set(records["cell_line_name"].astype(str)))
        depmap_to_cosmic = {depmap: cosmic for cosmic, depmap in cosmic_to_depmap.items() if depmap}
        wanted_cosmic = sorted({depmap_to_cosmic.get(x) for x in wanted_depmap if depmap_to_cosmic.get(x) is not None})

        depmap_hits = len(set(wanted_depmap).intersection(available_cell_cols))
        cosmic_hits = len(set(wanted_cosmic).intersection(available_cell_cols))
        use_cosmic = cosmic_hits > depmap_hits
        if depmap_hits == 0 and cosmic_hits == 0:
            raise ValueError(
                f"{self.expression_path} columns do not match either DepMap IDs or COSMIC IDs from dose-response. "
                "Provide gdsc_cell_lines.csv mapping or rename expression columns to match."
            )

        selected_cols = wanted_cosmic if use_cosmic else wanted_depmap
        selected_cols = [c for c in selected_cols if c in available_cell_cols]
        if not selected_cols:
            raise ValueError("No expression columns selected after matching cell ids.")

        expr = pd.read_csv(
            self.expression_path,
            sep=expr_sep,
            usecols=[gene_col] + selected_cols,
            low_memory=False,
        )
        expr[gene_col] = expr[gene_col].astype(str).apply(BaseDatasetCreator.clean_string)
        expr = expr.dropna(subset=[gene_col]).drop_duplicates(subset=[gene_col])
        expr = expr.set_index(gene_col)
        expr = expr.reindex(gene_axis)

        if use_cosmic:
            cosmic_to_depmap_effective = {depmap_to_cosmic.get(d): d for d in wanted_depmap if depmap_to_cosmic.get(d)}
            expr = expr.rename(columns=cosmic_to_depmap_effective)
        return expr

    def _build_cell_line_features(self, expr: pd.DataFrame, gene_axis: List[str], depmap_ids: Set[str],
                                  crispr_lookup: Dict[str, np.ndarray], cnv_lookup: Dict[str, np.ndarray],
                                  methylation_lookup: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Assemble multi-omics tensors per cell line."""
        cells_with_expression = set(expr.columns.astype(str))
        cells_with_crispr = set(crispr_lookup.keys())
        cells_with_cnv = set(cnv_lookup.keys())
        cells_with_meth = set(methylation_lookup.keys())

        all_cells = cells_with_expression.union(cells_with_crispr).union(cells_with_cnv).union(cells_with_meth)
        eligible_cells = all_cells.intersection(depmap_ids)
        if not eligible_cells:
            raise ValueError("No GDSC cells with omics + drug response after alignment.")

        cell_features = []
        for depmap_id in sorted(eligible_cells):
            if depmap_id in expr.columns:
                gene_expression = pd.to_numeric(expr[depmap_id], errors="coerce").to_numpy(dtype=np.float32, copy=False)
            else:
                gene_expression = np.full(len(gene_axis), np.nan, dtype=np.float32)

            crispr = crispr_lookup.get(depmap_id)
            if crispr is None:
                crispr = np.full(len(gene_axis), np.nan, dtype=np.float32)

            cnv = cnv_lookup.get(depmap_id)
            if cnv is None:
                cnv = np.full(len(gene_axis), np.nan, dtype=np.float32)

            meth = methylation_lookup.get(depmap_id)
            if meth is None:
                meth = np.full(len(gene_axis), np.nan, dtype=np.float32)

            tensor = np.stack([gene_expression, crispr, cnv, meth], axis=1).astype(np.float32, copy=False)
            if np.isnan(tensor).all():
                continue
            cell_features.append({
                "cell_line_name": depmap_id,
                "cell_line_features": tensor,
            })
        return pd.DataFrame(cell_features)

    def create_dataset(self) -> pd.DataFrame:
        """Create the GDSC dataset from raw files."""
        cosmic_to_depmap = self.load_cell_line_map()
        records = self._load_drug_response_records(cosmic_to_depmap)
        gene_axis = self.load_cross_domain_gene_axis()
        expr = self._load_expression_matrix(gene_axis, records, cosmic_to_depmap)
        depmap_ids_requested = set(records["cell_line_name"].astype(str))
        if not cosmic_to_depmap:
            raise ValueError("Building GDSC features requires gdsc_cell_lines.csv mapping COSMIC_ID to DepMap IDs "
                             f"at {self.cell_line_map_path}.")

        cnv_lookup = self._load_cnv_lookup(gene_axis, depmap_ids_requested, cosmic_to_depmap)
        crispr_lookup = self._load_crispr_lookup_from_project_score(gene_axis, depmap_ids_requested)
        methylation_lookup = self._load_methylation_lookup_from_f2(gene_axis, depmap_ids_requested, cosmic_to_depmap,)

        cell_line_features_df = self._build_cell_line_features(expr, gene_axis, depmap_ids_requested, crispr_lookup,
                                                               cnv_lookup, methylation_lookup,)
        cell_line_features_df, gene_axis = self.filter_genes_by_missing_values(cell_line_features_df, gene_axis,)
        cell_line_features_df = self.filter_cell_lines_by_missing_values(cell_line_features_df,)

        dataset = records.merge(cell_line_features_df, on="cell_line_name", how="inner")
        dataset = dataset[["drug_name", "smiles", "cell_line_name", "cell_line_features", "pic50"]]
        dataset = dataset.dropna(subset=["drug_name", "cell_line_name", "pic50", "cell_line_features"])
        return dataset

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    try:
        GDSCDatasetCreator().create_and_save_dataset()
    except FileNotFoundError as exc:
        logger.error("Unable to create GDSC dataset: %s", exc)
    logger.info("GDSC dataset artifacts stored in processed/")
