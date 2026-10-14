
from ..utils import DummyObject, requires_backends


class GraniteSpeechFeatureExtractor(metaclass=DummyObject):
    _backends = ["torchaudio"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torchaudio"])


class GraniteSpeechProcessor(metaclass=DummyObject):
    _backends = ["torchaudio"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torchaudio"])


class MusicgenMelodyFeatureExtractor(metaclass=DummyObject):
    _backends = ["torchaudio"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torchaudio"])


class MusicgenMelodyProcessor(metaclass=DummyObject):
    _backends = ["torchaudio"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["torchaudio"])
