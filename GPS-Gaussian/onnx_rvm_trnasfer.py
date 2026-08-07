import torch

rvm = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50").cuda().eval()
#                                                              ↑ 여기

class RVMExportWrapper(torch.nn.Module):
    def __init__(self, rvm, ds_ratio=0.4):
        super().__init__()
        self.rvm = rvm
        self.ds_ratio = ds_ratio
    def forward(self, src, r1, r2, r3, r4):
        return self.rvm(src, r1, r2, r3, r4, downsample_ratio=self.ds_ratio)

wrapped = RVMExportWrapper(rvm, ds_ratio=0.4).cuda().eval()
#                                            ↑ 여기 (선택, rvm이 이미 cuda면 자동 상속되지만 명시하는 게 안전)

dummy_src = torch.randn(1, 3, 1280, 720).cuda()      # ← 여기
dummy_r1 = torch.zeros(1, 16, 256, 144).cuda()       # ← 여기
dummy_r2 = torch.zeros(1, 32, 128, 72).cuda()        # ← 여기
dummy_r3 = torch.zeros(1, 64, 64, 36).cuda()         # ← 여기
dummy_r4 = torch.zeros(1, 128, 32, 18).cuda()        #  ← 여기

torch.onnx.export(
    wrapped,
    (dummy_src, dummy_r1, dummy_r2, dummy_r3, dummy_r4),
    "./onnx_models/rvm_resnet50_static.onnx",
    input_names=["src", "r1i", "r2i", "r3i", "r4i"],
    output_names=["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
    opset_version=16,
    dynamic_axes=None,
)