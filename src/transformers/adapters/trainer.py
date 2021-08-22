import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data.dataset import Dataset

from transformers import PreTrainedModel, Trainer, __version__
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import is_fairscale_available

from ..configuration_utils import PretrainedConfig
from ..data.data_collator import DataCollator
from ..file_utils import CONFIG_NAME, WEIGHTS_NAME, is_sagemaker_mp_enabled, logger
from ..modeling_utils import PreTrainedModel
from ..optimization import Adafactor, AdamW
from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..trainer_callback import TrainerCallback, TrainerControl, TrainerState
from ..trainer_pt_utils import get_parameter_names
from ..trainer_utils import EvalPrediction, PredictionOutput, ShardedDDPOption
from ..training_args import TrainingArguments


if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.optim import OSS

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


class AdapterTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        do_save_full_model: Optional[bool] = None,
        do_save_adapters: Optional[bool] = None,
        do_save_adapter_fusion: Optional[bool] = None,
        adapter_names: Optional[List[List[str]]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        # TODO allow additional callbacks
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=[AdapterTrainerCallback(self)],
            optimizers=optimizers,
        )

        if adapter_names is not None:
            self.model.set_active_adapters(adapter_names)
        # Set the defaults for loading/ saving model & adapters
        if isinstance(self.model, PreTrainedModel):
            model_freezed = getattr(self.model.base_model, "model_freezed", False)
        else:
            model_freezed = False
        if model_freezed and self.model.active_adapters:
            self.do_save_full_model = False
            self.do_save_adapters = True
            self.do_save_adapter_fusion = True
        else:
            self.do_save_full_model = True
            self.do_save_adapters = False
            self.do_save_adapter_fusion = False
        # override with explicit setting
        if do_save_full_model is not None:
            self.do_save_full_model = do_save_full_model
        if do_save_adapters is not None:
            self.do_save_adapters = do_save_adapters
        if do_save_adapter_fusion is not None:
            self.do_save_adapter_fusion = do_save_adapter_fusion

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if hasattr(self.model, "config") and hasattr(self.model.config, "adapter_fusion_models"):
                no_decay = [f"adapter_fusion_layer.{n}.value" for n in self.model.config.adapter_fusion_models]
                decay_parameters = [name for name in decay_parameters if name not in no_decay]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)


class AdapterTrainerCallback(TrainerCallback):
    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.trainer.do_save_adapters:
            self.trainer.model.save_all_adapters(args.output_dir)
        if self.trainer.do_save_adapter_fusion:
            self.trainer.model.save_all_adapter_fusions(args.output_dir)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.pop("model")
        if args.load_best_model_at_end and state.best_model_checkpoint is not None:
            if self.trainer.do_save_full_model:
                logger.info(f"Loading best model from {state.best_model_checkpoint} (score: {state.best_metric}).")

                best_model_path = os.path.join(state.best_model_checkpoint, WEIGHTS_NAME)
                if os.path.exists(best_model_path):
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = torch.load(best_model_path, map_location="cpu")
                    # If the model is on the GPU, it still works!
                    self.trainer._load_state_dict_in_model(state_dict)
                else:
                    logger.warn(
                        f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                        "on multiple nodes, you should activate `--save_on_each_node`."
                    )
            if self.trainer.do_save_adapters:
                # ToDo enable logger
                # logger.info(
                #     f"Loading best adapter(s) from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
                # )
                # attempt to re-load all adapters from checkpoint
                for adapter in model.config.adapters.adapters:
                    adapter_dir = os.path.join(state.best_model_checkpoint, adapter)
                    if os.path.exists(adapter_dir):
                        model.load_adapter(adapter_dir)
            if self.trainer.do_save_adapter_fusion:
                logger.info(
                    f"Loading best adapter fusion(s) from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
                )
                # attempt to re-load all adapter fusions from checkpoint
                fusion_models = getattr(self.model.config, "adapter_fusion_models", [])
                for fusion in fusion_models:
                    fusion_dir = os.path.join(self.state.best_model_checkpoint, fusion)
                    if os.path.exists(fusion_dir):
                        self.model.load_adapter_fusion(fusion_dir)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # apply adapter fusion weight regularization on the value matrix
        model = kwargs.pop("model")
        if hasattr(model.config, "adapter_fusion") and model.config.adapter_fusion["regularization"]:
            fusion_reg_loss = model.base_model.get_fusion_regularization_loss()
            fusion_reg_loss.backward()

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.resume_from_checkpoint is not None:
            if os.path.isfile(os.path.join(args.resume_from_checkpoint, WEIGHTS_NAME)):
                logger.info(f"Loading model from {args.resume_from_checkpoint}).")
            elif self.do_save_full_model:
                raise ValueError(f"Can't find a valid checkpoint at {args.resume_from_checkpoint}")

            if os.path.isfile(os.path.join(args.resume_from_checkpoint, CONFIG_NAME)):
                config = PretrainedConfig.from_json_file(os.path.join(args.resume_from_checkpoint, CONFIG_NAME))
                checkpoint_version = config.transformers_version
                if checkpoint_version is not None and checkpoint_version != __version__:
                    logger.warn(
                        f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                        f"Transformers but your current version is {__version__}. This is not recommended and could "
                        "yield to errors or unwanted behaviors."
                    )

            if args.deepspeed:
                # will be resumed in deepspeed_init
                pass
            else:
                if self.do_save_full_model:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
                    # If the model is on the GPU, it still works!
                    self._load_state_dict_in_model(state_dict)
                if self.do_save_adapters:
                    adapter_loaded = False
                    if os.path.isdir(args.resume_from_checkpoint):
                        for file_name in os.listdir(args.resume_from_checkpoint):
                            if os.path.isdir(os.path.join(args.resume_from_checkpoint, file_name)):
                                if "," in file_name:
                                    self.model.load_adapter_fusion(
                                        os.path.join(args.resume_from_checkpoint, file_name)
                                    )
                                    adapter_loaded = True
                                else:
                                    self.model.load_adapter(
                                        os.path.join(os.path.join(args.resume_from_checkpoint, file_name))
                                    )
                                    adapter_loaded = True

                    if not adapter_loaded:
                        raise Exception("Can't find a valid checkpoint at {}".format(args.resume_from_checkpoint))


class AdapterSeq2SeqTrainer(AdapterTrainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        self._max_length = max_length
        self._num_beams = num_beams
        return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in :obj:`evaluate()`.

        Args:
            test_dataset (:obj:`Dataset`):
                Dataset to run the predictions on. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. Has to implement the method :obj:`__len__`
            ignore_keys (:obj:`List[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is ``"eval"`` (default)
            max_length (:obj:`int`, `optional`):
                The maximum target length to use when predicting with the generate method.
            num_beams (:obj:`int`, `optional`):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.

        .. note::

            If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
            padding in a token classification task) the predictions will be padded (on the right) to allow for
            concatenation into one array. The padding index is -100.

        Returns: `NamedTuple` A namedtuple with the following keys:

            - predictions (:obj:`np.ndarray`): The predictions on :obj:`test_dataset`.
            - label_ids (:obj:`np.ndarray`, `optional`): The labels (if the dataset contained some).
            - metrics (:obj:`Dict[str, float]`, `optional`): The potential dictionary of metrics (if the dataset
              contained labels).
        """
        self._max_length = max_length
        self._num_beams = num_beams
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, generated_tokens, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is None:
            raise ValueError(
                f"Tensor need to be padded to `max_length={max_length}` but no tokenizer was passed when creating "
                "this `Trainer`. Make sure to create your `Trainer` with the appropriate tokenizer."
            )
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = (
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor
