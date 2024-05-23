import random
import tempfile

import datasets
import torch
from transformers import WhisperForAudioClassification, WhisperForConditionalGeneration, WhisperForCausalLM, \
    T5ForQuestionAnswering, WhisperConfig, T5Config, T5ForConditionalGeneration, T5ForSequenceClassification, \
    TrainingArguments
from adapters import init, get_adapter_info, SeqBnConfig, WhisperAdapterModel, AdapterTrainer


def make_config(config_class, **kwargs):
    return staticmethod(lambda: config_class(**kwargs))


def try_Whisper():
    config = make_config(
        WhisperConfig,
        d_model=16,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=4,
        decoder_ffn_dim=4,
        vocab_size=51865,
    )
    model = WhisperForCausalLM._from_config(WhisperConfig())
    print(model)
    init(model)
    print(model)
    model.eval()

    # fusion between a and b should be possible whereas fusion between a and c should fail
    model.add_adapter("a", config=SeqBnConfig(reduction_factor=16))
    model.add_adapter("b", config=SeqBnConfig(reduction_factor=2))
    model.add_adapter("c", config="double_seq_bn")
    # correct fusion
    # model.add_adapter_fusion(["a", "b"])

    model.add_adapter_fusion(["a", "c"])

    print(model.adapters_config.fusions)


def try_T5():
    model = T5ForSequenceClassification._from_config(T5Config())
    init(model)
    model.eval()
    model.add_adapter("a", config=SeqBnConfig(reduction_factor=16))
    model.add_adapter("b", config=SeqBnConfig(reduction_factor=2))
    model.add_adapter("c", config="double_seq_bn")
    # model.add_adapter_fusion(["a", "b"])
    model.add_adapter_fusion(["a", "c"])
    print(model.adapters_config.fusions)


def try_Whisper_training():
    # setup dataset
    train_dataset = datasets.load_from_disk(
        r"C:\Users\timoi\PycharmProjects\adapters\tests\fixtures\sample_generation\common_voice_encoded")["train"]
    print(train_dataset[0])

    training_args = TrainingArguments(
        output_dir="./examples",
        do_train=True,
        learning_rate=1.0,
        max_steps=8,
        no_cuda=True,
        per_device_train_batch_size=2,
        remove_unused_columns=False,
    )

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    init(model)
    model.add_adapter("a", config=SeqBnConfig(reduction_factor=16))
    model.train_adapter("a")

    # evaluate
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()


def try_Whisper_generation():
    from datasets import load_dataset
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    # Select an audio file and read it:
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio_sample = ds[0]["audio"]
    waveform = audio_sample["array"]
    sampling_rate = audio_sample["sampling_rate"]

    # Load the Whisper model in Hugging Face format:
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    # Use the model and processor to transcribe the audio:
    input_features = processor(
        waveform, sampling_rate=sampling_rate, return_tensors="pt"
    ).input_features

    # Generate token ids
    predicted_ids = model.generate(input_features)

    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription)


def try_Whisper_saving_loading():
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    init(model)
    model.add_adapter("a", config="seq_bn")
    print(model)

    with tempfile.TemporaryDirectory() as temp_dir:
        model.save_pretrained(temp_dir)
        model = WhisperForConditionalGeneration.from_pretrained(temp_dir)

        print(model)


def try_Whisper_generation():
    model = WhisperAdapterModel.from_pretrained("openai/whisper-tiny.en")
    model.add_seq2seq_lm_head("a")
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from datasets import load_dataset, Audio
    # model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features
    print(f"input_features: {input_features.shape}")

    # transcribe audio to ids
    generated_ids = model.generate(input_features)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(transcription)


def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long).view(shape).contiguous()


def get_sample(shape, config, **kwargs):
    total_dims = 1
    for dim in shape:
        total_dims *= dim
    values = []
    for _ in range(total_dims):
        values.append(random.random())
    input_features = torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()
    in_data = {"input_features": input_features}
    with_labels = kwargs.pop("with_labels", False)
    num_labels = kwargs.pop("num_labels", None)
    if with_labels:
        if num_labels is not None:
            in_data["labels"] = torch.tensor(data=[random.randint(0, num_labels - 1) for _ in range(shape[0])])
        else:
            in_data["labels"] = ids_tensor((shape[:-1]), config.vocab_size)
    if config and config.is_encoder_decoder:
        in_data["decoder_input_ids"] = ids_tensor((shape[:-1]), config.vocab_size)

    return in_data


def try_Whisper_classification():
    model = WhisperAdapterModel.from_pretrained("openai/whisper-tiny.en")
    model.add_audio_classification_head("a", num_labels=2)

    sample = get_sample((3, 80, 3000), model.config, num_labels=2, with_labels=True)

    outputs = model(**sample)
    print(outputs)


try_Whisper_classification()
