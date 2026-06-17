"""
trt_stereo_wrapper.py
=====================
IGEV++ / Selective-IGEV TRT 8.6.1 엔진 래퍼

입력: [-1, 1] 정규화된 이미지 텐서
출력: disparity 텐서 (B, 1, H, W)  -- 입력 해상도 기준 pixel disparity
"""

import numpy as np
import torch
import torch.nn.functional as F
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")


class TRTStereoWrapper:
    def __init__(self, engine_path: str):
        self.engine_path = engine_path

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        assert self.engine is not None, f"엔진 로드 실패: {engine_path}"
        self.context = self.engine.create_execution_context()

        self.input_names  = [self.engine.get_binding_name(i)
                             for i in range(self.engine.num_bindings)
                             if self.engine.binding_is_input(i)]
        self.output_names = [self.engine.get_binding_name(i)
                             for i in range(self.engine.num_bindings)
                             if not self.engine.binding_is_input(i)]

        idx = self.engine.get_binding_index(self.input_names[0])
        shape = tuple(self.engine.get_binding_shape(idx))
        self.engine_H = shape[2]
        self.engine_W = shape[3]

        self._alloc_buffers()
        self.stream = cuda.Stream()

        print(f"[TRTStereoWrapper] 로드 완료: {engine_path}  (TRT {trt.__version__})")
        print(f"  엔진 해상도: {self.engine_H}x{self.engine_W}")

    def _alloc_buffers(self):
        self.host_inputs  = []
        self.cuda_inputs  = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings     = []

        for name in self.input_names:
            idx   = self.engine.get_binding_index(name)
            shape = tuple(self.engine.get_binding_shape(idx))
            size  = int(np.prod(shape)) * np.dtype(np.float32).itemsize
            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype=np.float32)
            cuda_mem = cuda.mem_alloc(size)
            self.host_inputs.append(host_mem)
            self.cuda_inputs.append(cuda_mem)
            self.bindings.append(int(cuda_mem))

        for name in self.output_names:
            idx   = self.engine.get_binding_index(name)
            shape = tuple(self.engine.get_binding_shape(idx))
            size  = int(np.prod(shape)) * np.dtype(np.float32).itemsize
            host_mem = cuda.pagelocked_empty(int(np.prod(shape)), dtype=np.float32)
            cuda_mem = cuda.mem_alloc(size)
            self.host_outputs.append(host_mem)
            self.cuda_outputs.append(cuda_mem)
            self.bindings.append(int(cuda_mem))

        self.output_shape = tuple(
            self.engine.get_binding_shape(
                self.engine.get_binding_index(self.output_names[0])
            )
        )

    def infer(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        B, C, H, W = left.shape

        # [-1,1] → [0,255] + 엔진 해상도로 리사이즈
        left_255  = (left  + 1.0) / 2.0 * 255.0
        right_255 = (right + 1.0) / 2.0 * 255.0

        need_resize = (H != self.engine_H or W != self.engine_W)
        if need_resize:
            left_in  = F.interpolate(left_255,  size=(self.engine_H, self.engine_W), mode='bilinear', align_corners=False)
            right_in = F.interpolate(right_255, size=(self.engine_H, self.engine_W), mode='bilinear', align_corners=False)
        else:
            left_in  = left_255
            right_in = right_255

        np.copyto(self.host_inputs[0], left_in.contiguous().cpu().numpy().ravel())
        np.copyto(self.host_inputs[1], right_in.contiguous().cpu().numpy().ravel())

        for h_in, c_in in zip(self.host_inputs, self.cuda_inputs):
            cuda.memcpy_htod_async(c_in, h_in, self.stream)

        self.context.execute_async_v2(self.bindings, self.stream.handle)

        for h_out, c_out in zip(self.host_outputs, self.cuda_outputs):
            cuda.memcpy_dtoh_async(h_out, c_out, self.stream)

        self.stream.synchronize()

        disp = torch.from_numpy(
            self.host_outputs[0].reshape(self.output_shape)
        ).cuda()  # (1, 1, engine_H, engine_W)

        if need_resize:
            # 해상도 복원 + disparity 스케일 보정
            # 엔진은 engine_H 기준 pixel disparity 출력
            # GT는 원본 H 기준 → disparity × (H / engine_H)
            scale = H / self.engine_H
            disp = F.interpolate(disp, size=(H, W), mode='bilinear', align_corners=False)
            disp = disp * scale

        return disp  # (B, 1, H, W)

    def __del__(self):
        try:
            del self.context
            del self.engine
        except Exception:
            pass
