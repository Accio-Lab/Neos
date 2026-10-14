
from ..utils import DummyObject, requires_backends


class PreTrainedTokenizerFast(metaclass=DummyObject):
    _backends = ["tokenizers"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["tokenizers"])
