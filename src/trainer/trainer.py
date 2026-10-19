from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if not torch.isfinite(batch["loss"]).all():
            raise FloatingPointError(
                f"Non-finite loss for audio files: {batch['audio_path']}"
            )

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item(), n=len(batch["text"]))

        if "logits" in outputs and (not self.is_train or metric_funcs):
            if self.is_train:
                token_ids = self.model.greedy_decode(
                    batch["spectrogram"], batch["spectrogram_length"]
                )
            else:
                token_ids = self.model.greedy_decode_features(
                    batch["encoder_features"], batch["logit_lengths"]
                )
            batch["predictions"] = [
                self.text_encoder.decode(ids).strip() for ids in token_ids
            ]

        for met in metric_funcs:
            metrics.update(met.name, met(**batch), n=len(batch["text"]))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        self.log_audio(
            **batch, sample_rate=self.config.datasets[mode].get("target_sr", 16000)
        )

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_audio(
        self,
        audio: torch.Tensor,
        audio_length: torch.Tensor,
        audio_path: list[str],
        sample_rate: int = 16000,
        **batch,
    ) -> None:
        length = int(audio_length[0])
        self.writer.add_audio(
            f"audio_{Path(audio_path[0]).stem}",
            audio[0, :, :length],
            sample_rate=sample_rate,
        )

    def log_predictions(
        self,
        text,
        audio_path,
        log_probs=None,
        log_probs_length=None,
        predictions: list[str] | None = None,
        examples_to_log=10,
        **batch,
    ):
        if predictions is None:
            argmax_inds = log_probs.cpu().argmax(-1).numpy()
            argmax_inds = [
                inds[: int(ind_len)]
                for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
            ]
            raw_predictions = [self.text_encoder.decode(inds) for inds in argmax_inds]
            predictions = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]
        else:
            raw_predictions = [None] * len(predictions)
        tuples = list(zip(predictions, text, raw_predictions, audio_path))

        rows = {}
        for pred, target, raw_pred, audio_path in tuples[:examples_to_log]:
            target = self.text_encoder.normalize_text(target)
            if raw_pred is None:
                target = target.strip()
            wer = calc_wer(target, pred) * 100
            cer = calc_cer(target, pred) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "predictions": pred,
                "wer": wer,
                "cer": cer,
            }
            if raw_pred is not None:
                rows[Path(audio_path).name]["raw prediction"] = raw_pred
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
