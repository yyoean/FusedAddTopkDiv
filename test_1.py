# 必须先导torch_npu再导torchair
import torch
import torch_npu
import torchair

# (可选)若涉及集合通信算子入图，可调用patch方法
from torchair import patch_for_hcom
patch_for_hcom()

# 定义模型Model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        return torch.add(x, y)

# 实例化模型model
model = Model()

# 从TorchAir框架获取NPU提供的默认backend
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)

# 使用TorchAir的backend去调用compile接口编译模型
opt_model = torch.compile(model, backend=npu_backend)

# 使用编译后的model去执行
x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
opt_model(x, y)
print(opt_model(x, y))

