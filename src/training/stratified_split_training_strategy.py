"""Stratified split training strategy for cell, drug, and drug-cell splits."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable as IterableABC

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.constants import (
    EARLY_STOP_MIN_DELTA,
    GRAD_CLIP_NORM,
    ONECYCLE_DIV_FACTOR_STRATIFIED,
)
from src.evaluation import evaluate_test_metrics
from src.models import create_model
from src.repurposing.inference_engine import RepurposingInferenceEngine
from src.training.base_training_strategy import (
    BaseTrainingStrategy,
    pairwise_ranking_loss,
)


class StratifiedSplitTrainingStrategy(BaseTrainingStrategy):
    """Training strategy for stratified splits."""

    onecycle_div_factor = ONECYCLE_DIV_FACTOR_STRATIFIED

    def train_and_evaluate_model(
        self,
        strategy_creator,
        dataset_iterator,
        comet_logger,
    ) -> list[dict[str, float | str]]:
        """Train and evaluate model across stratified folds."""
        logging.info("Run isolation ID: %s", self._run_id)
        self.prediction_manager = RepurposingInferenceEngine(
            data_source=str(strategy_creator.data_source),
            device=self.device,
        )

        all_fold_results: list[dict[str, float | str]] = []

        for fold_idx, data_fold in enumerate(self._iter_data_folds(dataset_iterator), start=1):
            all_fold_results.append(
                self._train_and_evaluate_single_fold(
                    strategy_creator=strategy_creator,
                    data_fold=data_fold,
                    fold_idx=fold_idx,
                    comet_logger=comet_logger,
                )
            )

        self._log_final_cv_results(all_fold_results, comet_logger)
        self._save_artifacts(strategy_creator, all_fold_results)
        return all_fold_results

    def _train_and_evaluate_single_fold(
        self,
        strategy_creator,
        data_fold,
        fold_idx: int,
        comet_logger,
    ) -> dict[str, float | str]:
        """Run one stratified fold end-to-end and return test metrics."""
        logging.info("%s", "=" * 60)
        logging.info("Starting CV Fold %d", fold_idx)
        logging.info("%s", "=" * 60)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        (
            dims,
            train_data,
            valid_data,
            test_data,
            y_test_df,
            fold_metadata,
        ) = self._unpack_fold_data(data_fold, fold_idx)
        _, cell_input_shape = dims
        split_type = str(strategy_creator.split_type)
        ranking_weight = float(strategy_creator.ranking_weight)

        model = create_model(
            cell_input_shape=cell_input_shape,
            hidden_dim=strategy_creator.hidden_dim,
            cell_embed_dim=strategy_creator.cell_embed_dim,
            trainable_layers=strategy_creator.trainable_encoder_layers,
            pooling=strategy_creator.encoder_pooling,
            latent_dim=strategy_creator.latent_dim,
            rank_dim=strategy_creator.rank_dim,
            dropout=strategy_creator.dropout,
            force_cell_blind=strategy_creator.force_cell_blind,
            fusion_type=strategy_creator.fusion_type,
            modality_dropout_drug=strategy_creator.modality_dropout_drug,
            modality_dropout_cell=strategy_creator.modality_dropout_cell,
            modality_dropout_schedule=strategy_creator.modality_dropout_schedule,
            modality_dropout_final_scale=strategy_creator.modality_dropout_final_scale,
            bounded_output=strategy_creator.bounded_output,
            output_center=strategy_creator.bounded_output_center,
            output_scale=strategy_creator.bounded_output_scale,
            output_tau=strategy_creator.bounded_output_tau,
            device=self.device,
        )

        train_loader = self._get_dataloader(train_data)
        val_loader = self._get_dataloader(valid_data)
        test_loader = self._get_dataloader(test_data)
        self._enable_cached_drug_embeddings(
            strategy_creator,
            model,
            train_loader,
            val_loader,
            test_loader,
        )
        optimizer = self._create_optimizer(
            model=model,
            lr=strategy_creator.learning_rate,
            weight_decay=strategy_creator.weight_decay,
        )
        scheduler, scheduler_mode = self._create_scheduler(
            strategy_creator, optimizer, len(train_loader)
        )

        checkpoint_path = self._get_checkpoint_path(strategy_creator, fold_idx)
        train_summary = self._train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_mode=scheduler_mode,
            n_epochs=strategy_creator.epoch,
            checkpoint_path=checkpoint_path,
            fold_idx=fold_idx,
            comet_logger=comet_logger,
            use_amp=bool(strategy_creator.use_amp),
            patience=int(strategy_creator.patience),
            checkpoint_metric=str(strategy_creator.checkpoint_metric),
            use_ranking=self._ranking_enabled(split_type, ranking_weight),
            ranking_weight=ranking_weight,
            ranking_group_mode=str(strategy_creator.ranking_group_mode),
            split_type=split_type,
            bounded_output=str(strategy_creator.bounded_output),
            bounded_output_mode=str(strategy_creator.bounded_output_mode),
            bounded_output_center=float(strategy_creator.bounded_output_center),
            bounded_output_scale=float(strategy_creator.bounded_output_scale),
            bounded_output_tau=float(strategy_creator.bounded_output_tau),
            bounded_output_std_factor=float(
                strategy_creator.bounded_output_std_factor
            ),
            bounded_output_min_scale=float(
                strategy_creator.bounded_output_min_scale
            ),
            unfreeze_epoch=int(strategy_creator.unfreeze_epoch),
            unfreeze_layers=int(strategy_creator.unfreeze_layers),
            unfreeze_lr_factor=float(strategy_creator.unfreeze_lr_factor),
        )

        checkpoint = self._load_best_checkpoint(model, checkpoint_path)
        if checkpoint is not None:
            logging.info("Loaded best model from epoch %s", checkpoint.get("epoch", "n/a"))

        y_pred = self._predict(model, test_loader)
        y_pred = self._apply_residual_target_inverse(y_pred, fold_metadata)
        if fold_metadata.get("residual_target"):
            logging.info(
                "Applied residual-target inverse transform (fold=%d).",
                fold_idx,
            )

        y_true = self._flatten_targets(y_test_df)
        test_metrics = evaluate_test_metrics(
            y_true,
            y_pred,
            comet_logger,
            split_type=split_type,
            trainable_encoder_layers=int(strategy_creator.trainable_encoder_layers),
            data_source=str(strategy_creator.data_source),
            fold_idx=fold_idx,
            output_dir=self._get_fold_results_dir(strategy_creator, fold_idx),
        )
        self._attach_best_summary_to_test_metrics(test_metrics, train_summary)

        try:
            self.prediction_manager.log_predictions(
                model=model,
                fold_idx=fold_idx,
                output_dir=self._get_fold_results_dir(strategy_creator, fold_idx),
            )
        except Exception as exc:
            logging.warning(
                "Repurposing prediction export failed for fold %d (%s).",
                fold_idx,
                exc,
            )

        return test_metrics

    @staticmethod
    def _unpack_fold_data(data_fold, fold_idx: int):
        """Validate and unpack stratified fold tuple."""
        if isinstance(data_fold, (tuple, list)) and len(data_fold) >= 5:
            dims, train_data, valid_data, test_data, y_test_df = data_fold[:5]
            metadata_raw = data_fold[5] if len(data_fold) > 5 else {}
            fold_metadata = metadata_raw if isinstance(metadata_raw, dict) else {}
            return dims, train_data, valid_data, test_data, y_test_df, fold_metadata
        raise ValueError(
            f"Unexpected data fold format at fold {fold_idx}: {type(data_fold)}"
        )

    @staticmethod
    def _iter_data_folds(dataset_input):
        """Yield stratified folds from dataset iterator."""
        if isinstance(dataset_input, tuple):
            yield dataset_input
            return

        if isinstance(dataset_input, IterableABC):
            yielded = False
            for fold in dataset_input:
                yielded = True
                yield fold
            if not yielded:
                raise ValueError("Dataset iterator yielded no folds.")
            return

        raise ValueError(f"Unexpected dataset_input format: {type(dataset_input)}")

    @staticmethod
    def _ranking_enabled(split_type: str, ranking_weight: float) -> bool:
        """Return whether ranking loss is active for this split."""
        return (
            split_type in {"cell_stratified", "drug_stratified", "drug_cell_stratified"}
            and ranking_weight > 0
        )

    def _load_best_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str,
    ) -> dict | None:
        """Load best-checkpoint weights into model when file exists."""
        if not os.path.exists(checkpoint_path):
            return None
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    @staticmethod
    def _apply_residual_target_inverse(
        predictions: np.ndarray,
        fold_metadata: dict,
    ) -> np.ndarray:
        """Inverse-transform residual target predictions when configured."""
        if not fold_metadata.get("residual_target"):
            return predictions
        target_mean = float(
            fold_metadata.get(
                "target_mean",
                fold_metadata.get("fallback_global_mean", 0.0),
            )
        )
        return predictions + target_mean

    @staticmethod
    def _flatten_targets(y_test_df) -> np.ndarray:
        """Convert test targets to a flattened numpy vector."""
        if hasattr(y_test_df, "values"):
            return y_test_df.values.flatten()
        return np.asarray(y_test_df).flatten()

    @staticmethod
    def _resolve_ranking_group_mode(split_type: str, ranking_group_mode: str) -> str:
        mode = str(ranking_group_mode or "auto").lower()
        default_mode = "drug" if split_type == "cell_stratified" else "cell"
        if mode == "auto":
            return default_mode
        if mode in {"cell", "drug"}:
            return mode
        return default_mode

    def _resolve_batch_group_ids(
        self,
        split_type: str,
        ranking_group_mode: str,
        smiles: list[str],
        group_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if ranking_group_mode == "drug":
            return self._group_ids_from_smiles(smiles, device=self.device)
        if ranking_group_mode == "cell":
            if group_ids is not None:
                return group_ids
            if split_type == "cell_stratified":
                return self._group_ids_from_smiles(smiles, device=self.device)
            return None
        return group_ids

    def _train_fold(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scheduler_mode: str,
        n_epochs: int,
        checkpoint_path: str,
        fold_idx: int,
        comet_logger,
        use_amp: bool,
        patience: int,
        checkpoint_metric: str,
        use_ranking: bool,
        ranking_weight: float,
        ranking_group_mode: str,
        split_type: str,
        bounded_output: str,
        bounded_output_mode: str,
        bounded_output_center: float,
        bounded_output_scale: float,
        bounded_output_tau: float,
        bounded_output_std_factor: float,
        bounded_output_min_scale: float,
        unfreeze_epoch: int,
        unfreeze_layers: int,
        unfreeze_lr_factor: float,
    ) -> dict[str, float | str]:
        """Train one stratified fold with configurable validation monitor metric."""
        criterion_train = nn.SmoothL1Loss(reduction="none")
        criterion_eval = nn.SmoothL1Loss()
        monitor_metric, monitor_mode, early_stopping, best_state = (
            self._build_checkpoint_policy(
                checkpoint_metric=checkpoint_metric,
                patience=patience,
                min_delta=EARLY_STOP_MIN_DELTA,
            )
        )
        effective_group_mode = self._resolve_ranking_group_mode(
            split_type=split_type,
            ranking_group_mode=ranking_group_mode,
        )
        logging.info(
            "Checkpoint/Early-stop policy (stratified strategy): monitor=%s mode=%s",
            monitor_metric,
            monitor_mode,
        )

        self._configure_bounded_output(
            model=model,
            train_loader=train_loader,
            bounded_output=bounded_output,
            bounded_output_mode=bounded_output_mode,
            bounded_output_center=bounded_output_center,
            bounded_output_scale=bounded_output_scale,
            bounded_output_tau=bounded_output_tau,
            bounded_output_std_factor=bounded_output_std_factor,
            bounded_output_min_scale=bounded_output_min_scale,
        )

        if use_ranking:
            logging.info(
                "Ranking regularization: ENABLED (weight=%.4f, group_mode=%s)",
                ranking_weight,
                effective_group_mode,
            )
        else:
            logging.info("Ranking regularization: DISABLED")

        amp_enabled = bool(use_amp and self.device == "cuda")
        amp_device = "cuda" if self.device == "cuda" else "cpu"
        scaler = torch.amp.GradScaler(amp_device, enabled=amp_enabled)

        for epoch in range(1, n_epochs + 1):
            if hasattr(model, "set_training_progress"):
                model.set_training_progress(epoch=epoch, total_epochs=n_epochs)
            self._staged_unfreeze(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                unfreeze_epoch=unfreeze_epoch,
                unfreeze_layers=unfreeze_layers,
                unfreeze_lr_factor=unfreeze_lr_factor,
            )
            start_time = time.time()
            model.train()

            train_loss_total = 0.0
            huber_total = 0.0
            rank_total = 0.0
            n_batches = 0
            train_preds_list = []
            train_targets_list = []

            for batch in train_loader:
                targets = self._batch_targets(batch)
                smiles = batch["smiles"]

                sample_weight = self._resolve_sample_weight(batch)

                group_ids = self._resolve_group_ids(batch)
                rank_group_ids = self._resolve_batch_group_ids(
                    split_type=split_type,
                    ranking_group_mode=effective_group_mode,
                    smiles=smiles,
                    group_ids=group_ids,
                )

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(amp_device, enabled=amp_enabled):
                    predictions = self._forward_batch(model, batch)
                    huber_loss = self._compute_weighted_huber(
                        criterion_train,
                        predictions,
                        targets,
                        sample_weight,
                    )

                    rank_loss = torch.tensor(0.0, device=self.device)
                    if use_ranking and rank_group_ids is not None:
                        rank_loss, _ = pairwise_ranking_loss(
                            targets.squeeze(-1),
                            predictions.squeeze(-1),
                            rank_group_ids,
                        )

                    loss = huber_loss + (ranking_weight * rank_loss)

                optimizer_stepped = self._optimizer_step(
                    loss=loss,
                    optimizer=optimizer,
                    scaler=scaler,
                    model=model,
                    grad_clip_norm=GRAD_CLIP_NORM,
                )

                if scheduler_mode == "batch" and optimizer_stepped:
                    scheduler.step()

                train_loss_total += float(loss.item())
                huber_total += float(huber_loss.item())
                rank_total += float(rank_loss.item())
                n_batches += 1

                train_preds_list.append(self._tensor_to_numpy(predictions))
                train_targets_list.append(self._tensor_to_numpy(targets))

            train_loss = train_loss_total / max(1, n_batches)
            train_huber = huber_total / max(1, n_batches)
            train_rank = rank_total / max(1, n_batches)

            train_metrics, train_mse = self._compute_train_epoch_metrics(
                train_preds_list=train_preds_list,
                train_targets_list=train_targets_list,
            )

            val_loss, val_metrics = self._evaluate(model, val_loader, criterion_eval)

            if scheduler_mode == "epoch":
                scheduler.step()

            elapsed = time.time() - start_time
            current_lr = max(group.get("lr", 0.0) for group in optimizer.param_groups)
            val_mse = float(val_metrics.get("mse", float("nan")))
            fold_number = fold_idx
            logging.info(
                "[Fold %d | Epoch %d/%d] Time: %.1fs | LR: %.2e | "
                "Train {Loss: %.4f, MSE: %.4f, R2: %.4f, PCC: %.4f} | "
                "Val {Loss: %.4f, MSE: %.4f, R2: %.4f, PCC: %.4f}",
                fold_number,
                epoch,
                n_epochs,
                elapsed,
                current_lr,
                train_loss,
                train_mse,
                train_metrics["r2"],
                train_metrics["pcc"],
                val_loss,
                val_mse,
                val_metrics["r2"],
                val_metrics["pcc"],
            )

            if comet_logger:
                metrics_payload = {
                    f"fold_{fold_number}/train_loss": train_loss,
                    f"fold_{fold_number}/train_mse": train_mse,
                    f"fold_{fold_number}/train_huber": train_huber,
                    f"fold_{fold_number}/train_r2": train_metrics["r2"],
                    f"fold_{fold_number}/train_pcc": train_metrics["pcc"],
                    f"fold_{fold_number}/val_loss": val_loss,
                    f"fold_{fold_number}/val_mse": val_mse,
                    f"fold_{fold_number}/val_r2": val_metrics["r2"],
                    f"fold_{fold_number}/val_pcc": val_metrics["pcc"],
                    f"fold_{fold_number}/lr": current_lr,
                }
                if use_ranking:
                    metrics_payload[f"fold_{fold_number}/train_rank"] = train_rank
                comet_logger.log_metrics(metrics_payload, epoch=epoch)

            monitor_score, improved = self._update_best_checkpoint(
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                val_loss=float(val_loss),
                val_metrics=val_metrics,
                best_state=best_state,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                checkpoint_path=checkpoint_path,
            )
            if improved:
                self._log_best_model_saved(monitor_metric, best_state)

            if early_stopping(monitor_score, epoch):
                self._log_early_stopping(epoch, early_stopping, monitor_metric)
                break

        return self._best_state_to_summary(monitor_metric, best_state)
