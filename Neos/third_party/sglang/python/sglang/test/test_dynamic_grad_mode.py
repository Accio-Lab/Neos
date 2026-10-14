import unittest

import torch

from sglang.srt.utils import DynamicGradMode
from sglang.test.test_utils import CustomTestCase


class TestDynamicGradMode(CustomTestCase):
    def test_inference(self):
        
        DynamicGradMode.set_inference_mode(True)

        @DynamicGradMode()
        def create_tensor_x():
            return torch.empty(0)

        X = create_tensor_x()
        self.assertTrue(not X.requires_grad and X.is_inference())

    def test_no_grad(self):
        
        DynamicGradMode.set_inference_mode(False)

        @DynamicGradMode()
        def create_tensor_y():
            return torch.empty(0)

        Y = create_tensor_y()
        self.assertTrue(not Y.requires_grad and not Y.is_inference())

    def test_nested_inference(self):
        
        DynamicGradMode.set_inference_mode(False)

        @DynamicGradMode()
        def create_tensor_z():
            with torch.inference_mode():
                return torch.empty(0)

        Z = create_tensor_z()
        self.assertTrue(not Z.requires_grad and Z.is_inference())

    def test_nested_no_grad(self):
        
        DynamicGradMode.set_inference_mode(True)

        @DynamicGradMode()
        def create_tensor_w():
            with torch.no_grad():
                return torch.empty(0)

        W = create_tensor_w()
        self.assertTrue(not W.requires_grad and W.is_inference())


if __name__ == "__main__":
    unittest.main(verbosity=2)
