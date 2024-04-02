from transformers.models.whisper.modeling_whisper import WhisperModel, WhisperPreTrainedModel

from ...wrappers import init


class WhisperAdapterModel(WhisperModel):
    def __init__(self, config):
        super().__init__(config)

        self.whisper = WhisperModel(config)
        init(self.whisper)
