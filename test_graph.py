from typing import Any
import torch
from torch.library import impl
import torch_npu
import torchair
from torch_npu.testing.testcase import TestCase, run_tests
from torchair import register_fx_node_ge_converter
from torchair.ge import Tensor
from torch_npu.op_plugin.meta._meta_registrations import m
import numpy as np


@impl(m, "fused_add_topk_div")
def fused_add_topk_div(x, add_num,group_num, group_topk, n, k, activate_type, is_norm, scale):
    a = x.shape[0] 
    output_shape = (a, k)

    y = torch.empty(output_shape, dtype = torch.float32, device = "meta")
    indices = torch.empty(output_shape, dtype = torch.int32, device = "meta")
    return y, indices

@register_fx_node_ge_converter(torch.ops.npu.fused_add_topk_div.default)        
def convert_fused_add_topk_div(x: Tensor, add_num: Tensor, group_num: int, 
                               group_topk: int, n: int, k: int, activate_type: int, 
                               is_norm:bool, scale: float, y: Tensor = None, 
                               indices: Tensor = None, meta_outputs: Any = None):
    return torchair.ge.custom_op(
        "FusedAddTopkDiv",
        inputs={
            "x": x,
            "add_num": add_num,
        },
        attrs={
            "group_num": torchair.ge.attr.Int(group_num),
            "group_topk": torchair.ge.attr.Int(group_topk),
            "n": torchair.ge.attr.Int(n),
            "k": torchair.ge.attr.Int(k),
            "activate_type": torchair.ge.attr.Int(activate_type),
            "is_norm": torchair.ge.attr.Bool(is_norm),
            "scale": torchair.ge.attr.Float(scale)
        },
        outputs=['y', 'indices']
    )

class TestTorchCompileCustomAdd(TestCase):
    def test_fused_add_topk_div(self):
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        #config.mode = "reduce-overhead"
        npu_backend = torchair.get_npu_backend(compiler_config=config)

        torch.manual_seed(42)
        input_x = torch.rand([16, 256], dtype = torch.float32)
        input_add_num = torch.rand([256], dtype = torch.float32)
        
        input_x = input_x.npu()
        input_add_num = input_add_num.npu()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, add_num, group_num, group_topk, n, k, activate_type, is_norm, scale):
                return torch_npu.fused_add_topk_div(x, add_num, group_num, group_topk, n, k, activate_type, is_norm, scale)

        mod = torch.compile(Module().npu(), backend=npu_backend)
        output_y, output_indices = mod(x=input_x, add_num=input_add_num, group_num = 8,group_topk = 4,n = 2,k = 8,activate_type=0,is_norm = True,scale = 1.0)
        #torch.npu.synchronize()
        print("---------out1 = ", output_y.cpu().numpy())
        print("---------out2 = ", output_indices.cpu().numpy())


if __name__ == "__main__":
    run_tests()




