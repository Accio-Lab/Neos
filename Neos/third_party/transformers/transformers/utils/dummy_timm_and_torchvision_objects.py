
from ..utils import DummyObject, requires_backends


class TimmWrapperImageProcessor(metaclass=DummyObject):
    _backends = ["timm", "torchvision"]

    def __init__(self, *args, **kwargs):
        requires_backends(self, ["timm", "torchvision"])
