import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from transformers import DataCollator, EvalPrediction, PreTrainedModel, Trainer, TrainingArguments
from transformers.trainer_utils import PredictionOutput
from transformers.training_args import is_tpu_available


if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.getLogger(__name__)

UD_HEAD_LABELS = [
    "_",
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
]


@dataclass
class UDTrainingArguments(TrainingArguments):
    """
    Extends TrainingArguments for Universal Dependencies (UD) dependency parsing.
    TrainingArguments is the subset of the arguments we use in our example scripts
    **which relate to the training loop itself**.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    decode_mode: str = field(default="greedy", metadata={"help": "Whether to use mst decoding or greedy decoding"})
    store_best_model: bool = field(default=False, metadata={"help": "Whether to store best model during training."})
    metric_score: Optional[str] = field(
        default=None, metadata={"help": "Metric used to determine best model during training."}
    )


class Metric(object):
    def add(self, gold, prediction):
        raise NotImplementedError

    def get_metric(self) -> Dict[str, float]:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def unpack(*tensors: torch.Tensor):
        return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


class ParsingMetric(Metric):
    """
    based on allennlp.training.metrics.AttachmentScores
    Computes labeled and unlabeled attachment scores for a dependency parse. Note that the input
    to this metric is the sampled predictions, not the distribution itself.
    """

    def __init__(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._total_words = 0.0

    def add(
        self,
        gold_indices: torch.Tensor,
        gold_labels: torch.Tensor,
        predicted_indices: torch.Tensor,
        predicted_labels: torch.Tensor,
    ):
        """
        Parameters
        ----------
        predicted_indices : ``torch.Tensor``, required.
            A tensor of head index predictions of shape (batch_size, timesteps).
        predicted_labels : ``torch.Tensor``, required.
            A tensor of arc label predictions of shape (batch_size, timesteps).
        gold_indices : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_indices``.
        gold_labels : ``torch.Tensor``, required.
            A tensor of the same shape as ``predicted_labels``.
        """
        unwrapped = self.unpack(predicted_indices, predicted_labels, gold_indices, gold_labels)
        predicted_indices, predicted_labels, gold_indices, gold_labels = unwrapped

        predicted_indices = predicted_indices.long()
        predicted_labels = predicted_labels.long()
        gold_indices = gold_indices.long()
        gold_labels = gold_labels.long()

        correct_indices = predicted_indices.eq(gold_indices).long()
        correct_labels = predicted_labels.eq(gold_labels).long()
        correct_labels_and_indices = correct_indices * correct_labels

        self._unlabeled_correct += correct_indices.sum().item()
        self._labeled_correct += correct_labels_and_indices.sum().item()
        self._total_words += correct_indices.numel()

    def get_metric(self):
        unlabeled_attachment_score = 0.0
        labeled_attachment_score = 0.0
        if self._total_words > 0.0:
            unlabeled_attachment_score = self._unlabeled_correct / self._total_words
            labeled_attachment_score = self._labeled_correct / self._total_words
        return {
            "uas": unlabeled_attachment_score * 100,
            "las": labeled_attachment_score * 100,
        }

    def reset(self):
        self._labeled_correct = 0.0
        self._unlabeled_correct = 0.0
        self._total_words = 0.0


class DependencyParsingTrainer(Trainer):
    args: UDTrainingArguments

    def __init__(
        self,
        model: PreTrainedModel,
        args: UDTrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        do_save_full_model: bool = True,
        do_save_adapters: bool = False,
        do_save_adapter_fusion: bool = False,
        adapter_names: Optional[List[List[str]]] = None,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            compute_metrics,
            prediction_loss_only,
            do_save_full_model,
            do_save_adapters,
            do_save_adapter_fusion,
            adapter_names,
            tb_writer,
            optimizers,
        )
        # for finding the best model.
        # assumes higher is better
        self.best_score = 0.0
        # torch.autograd.set_detect_anomaly(True)

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self._prediction_loop(eval_dataloader, description="Evaluation")

        if self.args.store_best_model:
            self.store_best_model(output)

        self._log(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    def store_best_model(self, output):

        if self.args.metric_score not in output.metrics:
            raise Exception(
                "Metric %s not in output.\nThe following output was generated: %s",
                str(self.args.metric_score),
                str(output),
            )

        if output.metrics[self.args.metric_score] > self.best_score:
            self.best_score = output.metrics[self.args.metric_score]
            # Save model checkpoint
            self.save_model(os.path.join(self.args.output_dir, "best_model"))
            with open(os.path.join(self.args.output_dir, "best_model", "output.txt"), "w") as f:
                f.write(str(output.metrics))

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
                Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

                Works both with or without labels.
                """

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        logger.info("  Decode mode = %s", self.args.decode_mode)
        eval_losses: List[float] = []
        model.eval()

        metric = ParsingMetric()

        if is_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description):

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                step_eval_loss, rel_preds, arc_preds = model(**inputs, adapter_names=self.adapter_names)

                eval_losses += [step_eval_loss.mean().item()]

            mask = inputs["labels_arcs"].ne(self.model.config.pad_token_id)
            predictions_arcs = torch.argmax(arc_preds, dim=-1)[mask]

            labels_arcs = inputs["labels_arcs"][mask]

            predictions_rels, labels_rels = rel_preds[mask], inputs["labels_rels"][mask]
            predictions_rels = predictions_rels[torch.arange(len(labels_arcs)), labels_arcs]
            predictions_rels = torch.argmax(predictions_rels, dim=-1)

            metric.add(labels_arcs, labels_rels, predictions_arcs, predictions_rels)

        results = metric.get_metric()
        results[f"{description}_loss"] = np.mean(eval_losses)

        # Add predictions_rels to output, even though we are only interested in the metrics
        return PredictionOutput(predictions=predictions_rels, label_ids=None, metrics=results)
