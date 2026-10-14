
from ..utils import DummyObject, requires_backends


class ImageProcessingMixin(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class BaseImageProcessor(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])


class ImageFeatureExtractionMixin(metaclass=DummyObject):
    _backends = ["vision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["vision"])
