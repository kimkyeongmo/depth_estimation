import os
import sys
import copy
import time
import json
import queue
import shutil
import subprocess
import threading
from fractions import Fraction
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

# Windows DLL 검색 경로. 실제 폴더가 있을 때만 등록한다.
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    for dll_dir in (
        r"C:/Users/COM/miniconda3/envs/gps_gaussian/lib/site-packages/torch/lib",
        r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin",
    ):
        if os.path.isdir(dll_dir):
            os.add_dll_directory(dll_dir)

from pybind11 import CudaRuntime1

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tensorrt as trt

try:
    import PyNvVideoCodec as nvc
except ImportError:
    nvc = None
import warnings
import pyglet
from pyglet.gl import *
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'Video-Depth-Anything')))
try:
    from video_depth_anything.video_depth_stream import VideoDepthAnything as StreamingVDA
except ImportError:
    print("Error: Could not import StreamingVDA. Check your Video-Depth-Anything path.")
    exit()

from config.stereo_human_config import ConfigStereoHuman as config
from lib.network import VDAGaussianModel
from lib.GaussianRender import pts2render
from lib.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov
from lib.utils import depth2pc

cv2.setNumThreads(0)
torch.set_float32_matmul_precision('high')



def _subprocess_creationflags():
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        return subprocess.CREATE_NO_WINDOW
    return 0


def _windows_return_code_text(return_code):
    """Windows NTSTATUS 형태의 프로세스 종료 코드를 사람이 읽기 쉽게 표시한다."""
    unsigned = int(return_code) & 0xFFFFFFFF
    known = {
        0xC0000135: "STATUS_DLL_NOT_FOUND",
        0xC0000139: "STATUS_ENTRYPOINT_NOT_FOUND",
        0xC000007B: "STATUS_INVALID_IMAGE_FORMAT",
    }
    name = known.get(unsigned)
    if name:
        return f"{return_code} (0x{unsigned:08X}, {name})"
    return f"{return_code} (0x{unsigned:08X})"


def _ffmpeg_subprocess_env(ffmpeg_exe):
    """FFmpeg 자식 프로세스가 PyTorch/Conda DLL을 잘못 집어오는 것을 줄인다."""
    env = os.environ.copy()
    if os.name != "nt":
        return env

    exe_path = os.path.normcase(os.path.abspath(ffmpeg_exe))
    exe_dir = os.path.dirname(exe_path)
    conda_prefix = env.get("CONDA_PREFIX")

    current_parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    filtered_parts = []

    # imageio-ffmpeg나 독립 설치 FFmpeg를 실행할 때는 Conda/PyTorch DLL 경로가
    # 외부 FFmpeg의 DLL보다 먼저 선택되지 않도록 제거한다.
    external_ffmpeg = not (
        conda_prefix
        and exe_path.startswith(os.path.normcase(os.path.abspath(conda_prefix)) + os.sep)
    )

    blocked = []
    if external_ffmpeg and conda_prefix:
        blocked.extend([
            os.path.join(conda_prefix, "Library", "bin"),
            os.path.join(conda_prefix, "lib", "site-packages", "torch", "lib"),
        ])

    blocked_norm = [os.path.normcase(os.path.abspath(p)) for p in blocked]
    for part in current_parts:
        norm_part = os.path.normcase(os.path.abspath(part))
        if any(norm_part == b or norm_part.startswith(b + os.sep) for b in blocked_norm):
            continue
        if norm_part == exe_dir:
            continue
        filtered_parts.append(part)

    windir = env.get("WINDIR", r"C:\Windows")
    preferred = [
        os.path.dirname(os.path.abspath(ffmpeg_exe)),
        os.path.join(windir, "System32"),
        windir,
    ]
    env["PATH"] = os.pathsep.join(preferred + filtered_parts)
    return env


def _validate_ffmpeg_executable(candidate):
    """후보 FFmpeg가 실제로 실행되는지 확인한다."""
    try:
        result = subprocess.run(
            [candidate, "-hide_banner", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            env=_ffmpeg_subprocess_env(candidate),
            creationflags=_subprocess_creationflags(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        reason = _windows_return_code_text(result.returncode) if os.name == "nt" else str(result.returncode)
        if stderr:
            reason += f": {stderr}"
        return False, reason
    return True, ""


def resolve_ffmpeg_executable():
    """실제로 실행 가능한 FFmpeg를 찾는다.

    Windows Conda FFmpeg는 패키지 DLL 충돌로 0xC0000139가 날 수 있으므로,
    명시 경로 다음에는 self-contained imageio-ffmpeg 바이너리를 우선한다.
    """
    candidates = []

    env_path = os.environ.get("FFMPEG_BINARY")
    if env_path:
        candidates.append(("FFMPEG_BINARY", env_path))

    try:
        import imageio_ffmpeg
        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled:
            candidates.append(("imageio-ffmpeg", bundled))
    except (ImportError, RuntimeError):
        pass

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        candidates.append(("PATH", system_ffmpeg))

    seen = set()
    rejected = []
    for source, candidate in candidates:
        candidate = os.path.abspath(candidate)
        key = os.path.normcase(candidate)
        if key in seen or not os.path.isfile(candidate):
            continue
        seen.add(key)

        ok, reason = _validate_ffmpeg_executable(candidate)
        if ok:
            print(f"[FFMPEG] selected ({source}): {candidate}")
            return candidate

        rejected.append(f"- {source}: {candidate}\n  {reason}")
        print(f"[FFMPEG] 사용할 수 없는 후보 건너뜀: {candidate}\n  {reason}")

    detail = "\n".join(rejected) if rejected else "- 발견된 후보 없음"
    raise RuntimeError(
        "실행 가능한 ffmpeg.exe를 찾지 못했습니다.\n"
        "Windows에서는 우선 'python -m pip install -U imageio-ffmpeg'를 실행하세요.\n"
        "검사 결과:\n" + detail
    )


def resolve_ffprobe_executable(ffmpeg_exe):
    env_path = os.environ.get("FFPROBE_BINARY")
    if env_path and os.path.isfile(env_path):
        return env_path

    # 선택한 FFmpeg와 같은 폴더의 ffprobe를 우선한다.
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    ffprobe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
    sibling = os.path.join(ffmpeg_dir, ffprobe_name)
    if os.path.isfile(sibling):
        return sibling

    system_ffprobe = shutil.which("ffprobe")
    if system_ffprobe:
        ok, _ = _validate_ffmpeg_executable(system_ffprobe)
        if ok:
            return system_ffprobe
    return None

def _parse_fps(value, default=30.0):
    try:
        if value and value != "0/0":
            parsed = float(Fraction(value))
            if parsed > 0 and np.isfinite(parsed):
                return parsed
    except (ValueError, ZeroDivisionError):
        pass
    return default


def probe_video_metadata(video_path, ffmpeg_exe):
    """가능하면 ffprobe를 사용하고, 없으면 OpenCV FFmpeg 백엔드로 메타데이터만 읽는다."""
    ffprobe_exe = resolve_ffprobe_executable(ffmpeg_exe)
    if ffprobe_exe:
        command = [
            ffprobe_exe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames:stream_tags=rotate:stream_side_data=rotation:format=duration",
            "-of", "json",
            video_path,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
                env=_ffmpeg_subprocess_env(ffprobe_exe),
                creationflags=_subprocess_creationflags(),
            )
            metadata = json.loads(result.stdout)
            stream = metadata["streams"][0]
            fps = _parse_fps(stream.get("avg_frame_rate"))
            if fps == 30.0:
                fps = _parse_fps(stream.get("r_frame_rate"), default=fps)

            duration = float(metadata.get("format", {}).get("duration") or 0.0)
            nb_frames_text = stream.get("nb_frames")
            total_frames = int(nb_frames_text) if nb_frames_text and nb_frames_text.isdigit() else 0
            if total_frames <= 0 and duration > 0:
                total_frames = int(round(duration * fps))

            rotation = 0
            tags = stream.get("tags") or {}
            if "rotate" in tags:
                rotation = int(float(tags["rotate"]))
            for side_data in stream.get("side_data_list") or []:
                if "rotation" in side_data:
                    rotation = int(float(side_data["rotation"]))

            return {
                "width": int(stream["width"]),
                "height": int(stream["height"]),
                "fps": fps,
                "total_frames": total_frames,
                "rotation": rotation % 360,
                "probe_backend": "ffprobe",
            }
        except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError):
            print("[FFMPEG] ffprobe 메타데이터 조회 실패, OpenCV로 대체합니다.")

    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"입력 영상을 열 수 없습니다: {video_path}")
    metadata = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "rotation": 0,
        "probe_backend": f"OpenCV/{cap.getBackendName()}",
    }
    cap.release()
    if metadata["fps"] <= 0 or not np.isfinite(metadata["fps"]):
        metadata["fps"] = 30.0
    return metadata


class FFmpegVideoReader:
    """ffmpeg.exe를 직접 실행해 RGB24 raw frame을 파이프로 받는 파일 리더."""

    def __init__(self, video_path, ffmpeg_exe=None, use_nvdec=False):
        self.video_path = os.path.abspath(video_path)
        self.ffmpeg_exe = ffmpeg_exe or resolve_ffmpeg_executable()
        self.metadata = probe_video_metadata(self.video_path, self.ffmpeg_exe)
        self.width = self.metadata["width"]
        self.height = self.metadata["height"]
        self.fps = self.metadata["fps"]
        self.total_frames = self.metadata["total_frames"]
        self.rotation = self.metadata["rotation"]
        self.frame_bytes = self.width * self.height * 3
        self.use_nvdec = use_nvdec
        self.closed = False

        command = [
            self.ffmpeg_exe,
            "-hide_banner",
            "-loglevel", "error",
            "-noautorotate",
        ]
        if use_nvdec:
            # 디코딩은 NVDEC를 사용하지만 RGB24 파이프로 보내기 위해 CPU 다운로드는 남는다.
            command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

        command += ["-i", self.video_path, "-map", "0:v:0", "-an", "-sn", "-dn"]

        if use_nvdec:
            command += ["-vf", "hwdownload,format=nv12,format=rgb24"]

        command += [
            "-vsync", "0",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=_ffmpeg_subprocess_env(self.ffmpeg_exe),
            creationflags=_subprocess_creationflags(),
        )
        if self.process.stdout is None:
            raise RuntimeError("FFmpeg stdout 파이프를 생성하지 못했습니다.")

        print(f"[FFMPEG INPUT] executable: {self.ffmpeg_exe}")
        print(f"[FFMPEG INPUT] mode: {'NVDEC + CPU download' if use_nvdec else 'software decode'}")
        print(f"[FFMPEG INPUT] metadata backend: {self.metadata['probe_backend']}")
        print(f"[FFMPEG INPUT] opened: {self.video_path}")
        print(f"[FFMPEG INPUT] fps: {self.fps:.3f}")
        print(f"[FFMPEG INPUT] total frames: {self.total_frames}")
        print(f"[FFMPEG INPUT] resolution: {self.width} x {self.height}")

    def _read_exact(self, size):
        chunks = bytearray(size)
        view = memoryview(chunks)
        offset = 0
        while offset < size:
            count = self.process.stdout.readinto(view[offset:])
            if not count:
                break
            offset += count
        return chunks if offset == size else None

    def read(self):
        raw = self._read_exact(self.frame_bytes)
        if raw is None:
            return_code = self.process.poll()
            if return_code not in (None, 0):
                stderr_text = ""
                if self.process.stderr:
                    stderr_text = self.process.stderr.read().decode("utf-8", errors="replace").strip()
                code_text = _windows_return_code_text(return_code) if os.name == "nt" else str(return_code)
                raise RuntimeError(
                    f"FFmpeg 입력 디코더가 첫 프레임을 보내기 전에 종료됐습니다: {code_text}\n"
                    f"{stderr_text}"
                )
            return None

        frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        # -noautorotate를 사용했으므로 기존 코드와 동일하게 여기서 회전한다.
        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return np.ascontiguousarray(frame)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.stdout:
            self.process.stdout.close()
        try:
            return_code = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                return_code = self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                return_code = self.process.wait()

        stderr_text = ""
        if self.process.stderr:
            stderr_text = self.process.stderr.read().decode("utf-8", errors="replace").strip()
            self.process.stderr.close()
        if return_code not in (0, 255) and stderr_text:
            print(f"[FFMPEG INPUT] 종료 메시지: {stderr_text}")


def can_use_h264_nvenc(ffmpeg_exe):
    """현재 FFmpeg 빌드와 NVIDIA 드라이버에서 NVENC가 실제 동작하는지 검사한다.

    RTX 50 시리즈에서는 64x64 프레임이 NVENC 최소 크기보다 작아 테스트가
    실패할 수 있으므로 256x256 프레임으로 검사한다.
    """
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-loglevel", "error",
        "-f", "lavfi",
        "-i", "color=c=black:s=256x256:r=30",
        "-frames:v", "1",
        "-pix_fmt", "yuv420p",
        "-c:v", "h264_nvenc",
        "-f", "null",
        "-",
    ]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
            env=_ffmpeg_subprocess_env(ffmpeg_exe),
            creationflags=_subprocess_creationflags(),
        )
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[NVENC] 동작 검사 중 예외 발생: {exc}")
        return False

    if result.returncode == 0:
        print("[NVENC] h264_nvenc 사용 가능")
        return True

    reason = (
        _windows_return_code_text(result.returncode)
        if os.name == "nt"
        else str(result.returncode)
    )
    print(f"[NVENC] 동작 검사 실패: {reason}")
    stderr_text = (result.stderr or "").strip()
    if stderr_text:
        print(stderr_text)
    return False


class AsyncFFmpegWriter:
    """RGB 프레임을 별도 스레드에서 FFmpeg stdin으로 전달하는 비동기 MP4 writer."""

    def __init__(self, output_path, width, height, fps, ffmpeg_exe=None, prefer_nvenc=True, queue_size=4):
        self.output_path = os.path.abspath(output_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.ffmpeg_exe = ffmpeg_exe or resolve_ffmpeg_executable()
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.worker_error = None
        self.closed = False

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        use_nvenc = prefer_nvenc and can_use_h264_nvenc(self.ffmpeg_exe)
        self.encoder = "h264_nvenc" if use_nvenc else "libx264"

        command = [
            self.ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s:v", f"{self.width}x{self.height}",
            "-r", f"{self.fps:.6f}",
            "-i", "pipe:0",
            "-an",
        ]

        if use_nvenc:
            command += [
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-rc", "vbr",
                "-cq", "19",
                "-b:v", "0",
            ]
        else:
            command += [
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "18",
            ]

        command += [
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            self.output_path,
        ]

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=_ffmpeg_subprocess_env(self.ffmpeg_exe),
            creationflags=_subprocess_creationflags(),
        )
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg stdin 파이프를 생성하지 못했습니다.")

        self.worker = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker.start()
        print(f"[FFMPEG OUTPUT] encoder: {self.encoder}")
        print(f"[FFMPEG OUTPUT] resolution: {self.width} x {self.height}")
        print(f"[FFMPEG OUTPUT] file: {self.output_path}")

    def _writer_loop(self):
        try:
            while True:
                frame = self.frame_queue.get()
                try:
                    if frame is None:
                        return
                    self.process.stdin.write(frame.tobytes())
                finally:
                    self.frame_queue.task_done()
        except Exception as exc:
            self.worker_error = exc
        finally:
            try:
                self.process.stdin.close()
            except (BrokenPipeError, OSError):
                pass

    def write(self, frame_rgb):
        if self.closed:
            raise RuntimeError("이미 닫힌 FFmpeg writer입니다.")
        if self.worker_error:
            raise RuntimeError(f"FFmpeg writer thread 오류: {self.worker_error}")

        frame = np.asarray(frame_rgb, dtype=np.uint8)
        expected_shape = (self.height, self.width, 3)
        if frame.shape != expected_shape:
            raise ValueError(f"출력 프레임 크기 불일치: {frame.shape}, expected={expected_shape}")
        queued_frame = np.ascontiguousarray(frame).copy()

        while True:
            if self.worker_error:
                raise RuntimeError(f"FFmpeg writer thread 오류: {self.worker_error}")
            if self.process.poll() is not None:
                raise RuntimeError("FFmpeg 인코더 프로세스가 예기치 않게 종료됐습니다.")
            try:
                self.frame_queue.put(queued_frame, timeout=0.5)
                return
            except queue.Full:
                continue

    def close(self):
        if self.closed:
            return
        self.closed = True

        while self.worker.is_alive():
            try:
                self.frame_queue.put(None, timeout=0.5)
                break
            except queue.Full:
                if self.worker_error or self.process.poll() is not None:
                    break
        self.worker.join(timeout=30)
        if self.worker.is_alive():
            self.process.kill()
            raise RuntimeError("FFmpeg writer thread가 정상적으로 종료되지 않았습니다.")

        try:
            return_code = self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            return_code = self.process.wait()

        stderr_text = ""
        if self.process.stderr:
            stderr_text = self.process.stderr.read().decode("utf-8", errors="replace").strip()
            self.process.stderr.close()

        if self.worker_error:
            raise RuntimeError(f"FFmpeg writer thread 오류: {self.worker_error}\n{stderr_text}")
        if return_code != 0:
            code_text = _windows_return_code_text(return_code) if os.name == "nt" else str(return_code)
            raise RuntimeError(f"FFmpeg 인코딩 실패(return code={code_text}):\n{stderr_text}")



def _metadata_attr(metadata, *names, default=None):
    """PyNvVideoCodec 버전별 metadata 속성명 차이를 흡수한다."""
    for name in names:
        if hasattr(metadata, name):
            value = getattr(metadata, name)
            if value is not None:
                return value
    return default


def _metadata_fps(value, default=30.0):
    """Fraction, 문자열, 숫자형 frame rate를 float FPS로 변환한다."""
    if value is None:
        return default
    try:
        # Fraction 또는 numerator/denominator 속성을 제공하는 객체
        numerator = getattr(value, "numerator", None)
        denominator = getattr(value, "denominator", None)
        if numerator is not None and denominator not in (None, 0):
            fps = float(numerator) / float(denominator)
        else:
            # 일부 버전의 num/den 표기 지원
            num = getattr(value, "num", None)
            den = getattr(value, "den", None)
            if num is not None and den not in (None, 0):
                fps = float(num) / float(den)
            else:
                fps = float(Fraction(str(value)))
        if fps > 0 and np.isfinite(fps):
            return fps
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return default



def _is_stream_url(source):
    """rtsp://, srt:// 같은 네트워크 URL인지 판별한다."""
    return "://" in str(source)


def _safe_source_for_print(source):
    return str(source)


class PyNvCodecVideoReader:
    """NVDEC → CUDA RGBP → DLPack → PyTorch CUDA Tensor 리더.

    로컬 MP4 파일뿐 아니라 PyNvVideoCodec/FFmpeg demuxer가 지원하는
    rtsp:// MediaMTX URL도 같은 경로로 시도한다.

    read()는 CPU NumPy 배열이 아니라 [3, H, W] CUDA uint8 Tensor를 반환한다.
    PyNvVideoCodec 프레임과 PyTorch Tensor는 DLPack으로 같은 CUDA 메모리를 공유한다.
    """

    def __init__(self, video_path, gpu_id=0, buffer_size=8):
        if nvc is None:
            raise RuntimeError(
                "PyNvVideoCodec가 설치되어 있지 않습니다.\n"
                "gps_gaussian 환경에서 다음을 실행하세요:\n"
                "  python -m pip install PyNvVideoCodec"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA를 사용할 수 없어 NVDEC zero-copy 입력을 시작할 수 없습니다.")

        self.source = str(video_path)
        self.is_url = _is_stream_url(self.source)
        # 로컬 파일은 절대 경로로 고정하고, rtsp:// 같은 MediaMTX URL은 그대로 유지한다.
        self.video_path = self.source if self.is_url else os.path.abspath(self.source)
        self.gpu_id = int(gpu_id)
        self.buffer_size = max(2, int(buffer_size))
        self.closed = False
        self._last_decoded_frame = None
        self._logged_frame_info = False

        torch.cuda.set_device(self.gpu_id)
        # PyTorch primary CUDA context를 먼저 생성해 decoder와 같은 context를 사용하게 한다.
        torch.cuda.init()

        try:
            self.decoder = nvc.ThreadedDecoder(
                enc_file_path=self.video_path,
                buffer_size=self.buffer_size,
                gpu_id=self.gpu_id,
                use_device_memory=True,
                output_color_type=nvc.OutputColorType.RGBP,
            )
            self.decoder_name = "ThreadedDecoder"
        except Exception as exc:
            raise RuntimeError(
                "PyNvVideoCodec ThreadedDecoder 초기화에 실패했습니다. "
                "NVIDIA 드라이버, PyNvVideoCodec 설치 및 입력 코덱을 확인하세요.\n"
                f"원인: {exc}"
            ) from exc

        metadata = self.decoder.get_stream_metadata()
        self.width = int(_metadata_attr(metadata, "width", "Width", default=0) or 0)
        self.height = int(_metadata_attr(metadata, "height", "Height", default=0) or 0)
        self.total_frames = int(
            _metadata_attr(metadata, "num_frames", "numFrames", default=0) or 0
        )
        fps_value = _metadata_attr(
            metadata, "avg_frame_rate", "frameRate", "frame_rate", default=None
        )
        self.fps = _metadata_fps(fps_value, default=30.0)

        # 회전 메타데이터와 불완전한 프레임 수는 1회성 OpenCV metadata 조회로 보완한다.
        # 실제 프레임 데이터 경로에는 OpenCV/CPU 복사가 사용되지 않는다.
        self.rotation = 0
        if not self.is_url:
            cap = cv2.VideoCapture(self.video_path, cv2.CAP_FFMPEG)
            if cap.isOpened():
                cv_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                cv_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cv_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                cv_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if self.fps <= 0 and cv_fps > 0:
                    self.fps = cv_fps
                if self.total_frames <= 0 and cv_frames > 0:
                    self.total_frames = cv_frames
                if self.width <= 0:
                    self.width = cv_width
                if self.height <= 0:
                    self.height = cv_height
                if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
                    rotation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
                    if np.isfinite(rotation):
                        self.rotation = int(round(rotation)) % 360
                cap.release()

        if self.fps <= 0 or not np.isfinite(self.fps):
            self.fps = 30.0

        print(f"[PYNVC INPUT] decoder: {self.decoder_name}")
        print("[PYNVC INPUT] mode: NVDEC + CUDA device memory + DLPack zero-copy")
        print(f"[PYNVC INPUT] opened: {self.video_path}")
        print(f"[PYNVC INPUT] fps: {self.fps:.3f}")
        print(f"[PYNVC INPUT] total frames: {self.total_frames}")
        print(f"[PYNVC INPUT] resolution: {self.width} x {self.height}")
        print(f"[PYNVC INPUT] prefetch buffer: {self.buffer_size} frames")

    def read(self):
        frames = self.decoder.get_batch_frames(1)
        if not frames:
            return None

        decoded_frame = frames[0]
        # decoded_frame이 Tensor 수명 동안 device memory를 소유하도록 참조를 유지한다.
        self._last_decoded_frame = decoded_frame
        frame_cuda = torch.from_dlpack(decoded_frame)

        if not frame_cuda.is_cuda:
            raise RuntimeError(
                "PyNvVideoCodec가 host frame을 반환했습니다. "
                "use_device_memory=True 설정이 적용되지 않았습니다."
            )

        # RGBP는 [3, H, W]가 정상이다. 버전/설정 차이를 방어적으로 처리한다.
        if frame_cuda.ndim != 3:
            raise RuntimeError(
                f"예상하지 못한 decoded frame shape: {tuple(frame_cuda.shape)}"
            )
        if frame_cuda.shape[0] == 3:
            frame_chw = frame_cuda
        elif frame_cuda.shape[-1] == 3:
            frame_chw = frame_cuda.permute(2, 0, 1)
        else:
            raise RuntimeError(
                f"RGB 채널을 찾을 수 없는 decoded frame shape: {tuple(frame_cuda.shape)}"
            )

        if not self._logged_frame_info:
            print(
                "[PYNVC INPUT] first tensor: "
                f"shape={tuple(frame_chw.shape)}, dtype={frame_chw.dtype}, "
                f"device={frame_chw.device}"
            )
            self._logged_frame_info = True

        # no-auto-rotate와 동일한 동작을 GPU에서 수행한다.
        if self.rotation == 90:
            frame_chw = torch.rot90(frame_chw, k=-1, dims=(-2, -1))
        elif self.rotation == 180:
            frame_chw = torch.rot90(frame_chw, k=2, dims=(-2, -1))
        elif self.rotation == 270:
            frame_chw = torch.rot90(frame_chw, k=1, dims=(-2, -1))

        return frame_chw

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._last_decoded_frame = None
        # ThreadedDecoder는 별도 close API 없이 객체 수명으로 정리된다.
        self.decoder = None


class LatestFrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.stopped = False

    def update(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_id
            return self.frame.copy(), self.frame_id

    def stop(self):
        with self.lock:
            self.stopped = True

    def is_stopped(self):
        with self.lock:
            return self.stopped


def capture_latest_loop(stream_url, latest_buffer):
    """별도 thread에서 스트림 받아 LatestFrameBuffer에 최신 프레임 저장."""
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 내부 버퍼 최소화

    if not cap.isOpened():
        latest_buffer.stop()
        print(f"[STREAM] Cannot open: {stream_url}")
        return

    print(f"[STREAM] opened: {stream_url}")
    print(f"[STREAM] reported fps = {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"[STREAM] reported resolution = "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    read_count = 0
    fail_count = 0
    while not latest_buffer.is_stopped():
        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 100:
                print("[STREAM] too many read failures, stopping")
                break
            time.sleep(0.01)
            continue
        fail_count = 0
        latest_buffer.update(frame)
        read_count += 1
        if read_count % 100 == 0:
            print(f"[STREAM] read_count={read_count}")

    cap.release()
    latest_buffer.stop()
    print("[STREAM] capture thread stopped")
# ============================================================


class TRTWrapper:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                trt_dtype = self.engine.get_tensor_dtype(name)
                if trt_dtype == trt.DataType.HALF:
                    torch_dtype = torch.float16
                elif trt_dtype == trt.DataType.INT32:
                    torch_dtype = torch.int32
                elif hasattr(trt.DataType, 'BFLOAT16') and trt_dtype == trt.DataType.BFLOAT16:
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float32
                self.outputs[name] = torch.empty(shape, dtype=torch_dtype, device='cuda')
                self.context.set_tensor_address(name, self.outputs[name].data_ptr())

    def __call__(self, **inputs):
        keep_alive = []
        for name, tensor in inputs.items():
            tensor = tensor.contiguous()
            keep_alive.append(tensor)
            self.context.set_tensor_address(name, tensor.data_ptr())
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        res = {}
        for k, v in self.outputs.items():
            res[k] = v.to(torch.float32)
        return res


def pad_to_square_tensor(tensor):
    h, w = tensor.shape[2], tensor.shape[3]
    max_side = max(h, w)
    pad_t = (max_side - h) // 2
    pad_b = max_side - h - pad_t
    pad_l = (max_side - w) // 2
    pad_r = max_side - w - pad_l
    return F.pad(tensor, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0.0)



def run_ultimate_streaming_inference_filtered(
    vid_path, ckpt_path, engine_dir, ref_intr_path, out_dir,
    is_stream=False,
    max_process_frames=-1,
    save_output=True,
    show_window=True,
    prefer_nvenc=True,
    use_nvdec=False,
    use_pynvcodec=True,
    pynv_buffer_size=8,
    gpu_id=0,
    allow_opencv_stream_fallback=False,
):
    os.makedirs(out_dir, exist_ok=True)
    out_name = Path(vid_path).stem if not is_stream else "stream_output"
    # PyNvVideoCodec 파일 입력 + 저장 비활성화 상태에서는 외부 ffmpeg.exe가 필요 없다.
    ffmpeg_exe = None
    if save_output or (not is_stream and not use_pynvcodec):
        ffmpeg_exe = resolve_ffmpeg_executable()
    if show_window:
        CudaRuntime1.init_window(3840, 1080)

    cfg_wrapper = config()
    cfg_wrapper.load("config/stage2.yaml")
    cfg = cfg_wrapper.get_cfg()
    cfg.defrost()
    cfg.dataset.src_res = 504
    cfg.freeze()

    base_model = VDAGaussianModel(cfg, with_gs_render=True)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    base_model.load_state_dict(ckpt['network'], strict=True)
    vda_weights = base_model.vda_model.state_dict()
    del base_model

    vda_stream = StreamingVDA(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
    vda_stream.load_state_dict(vda_weights, strict=True)
    vda_stream = vda_stream.cuda().eval()

    unet_trt = TRTWrapper(os.path.join(engine_dir, "unet_extractor.engine"))
    gs_trt = TRTWrapper(os.path.join(engine_dir, "gs_regresser.engine"))
    # RVM도 PyTorch torch.hub 모델 대신 TensorRT 엔진으로 실행한다.
    # 입력 img_tensor_raw는 PyNvVideoCodec 경로에서 이미 CUDA Tensor이므로
    # NVDEC/DLPack → RVM-TRT까지 CPU 왕복 없이 이어진다.
    rvm_trt = TRTWrapper(os.path.join(engine_dir, "rvm_resnet50_static.engine"))

    raw_intr = np.load(ref_intr_path)
    ref_extr_path = ref_intr_path.replace('intr.npy', 'extr.npy')
    extr_np = np.load(ref_extr_path)
    if extr_np.shape == (3, 4):
        extr_4x4 = np.eye(4, dtype=np.float32)
        extr_4x4[:3, :] = extr_np
        extr_np = extr_4x4

    intr_504 = raw_intr.copy()
    intr_tensor_504 = torch.from_numpy(intr_504).float().unsqueeze(0).cuda()
    extr_tensor = torch.from_numpy(extr_np).float().unsqueeze(0).cuda()

    render_res = 504
    #target_res = 1008
    target_w = 1920
    target_h = 1080
    fovx = focal2fov(intr_504[0, 0], render_res)
    fovy = focal2fov(intr_504[1, 1], render_res)
    proj_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, K=intr_504, h=render_res, w=render_res).transpose(0, 1).cuda()
    bg_color = getattr(cfg.dataset, 'bg_color', [0.0, 0.0, 0.0])
    base_extr = extr_tensor[0]

    # ============================================================
    # 입력:
    # - MediaMTX RTSP 스트림도 가능하면 PyNvVideoCodec으로 바로 열어
    #   NVDEC/DLPack CUDA Tensor를 받는다.
    # - PyNvVideoCodec이 해당 RTSP URL을 열지 못하면 옵션에 따라
    #   OpenCV latest-frame CPU 경로로 폴백할 수 있다. 단, 폴백은 zero-copy가 아니다.
    # ============================================================
    ffmpeg_reader = None
    pynv_reader = None
    latest_buffer = None
    capture_thread = None
    input_backend_name = "OpenCV stream"

    if is_stream:
        total_frames = 0
        fps = 30.0

        if use_pynvcodec:
            try:
                pynv_reader = PyNvCodecVideoReader(
                    vid_path,
                    gpu_id=gpu_id,
                    buffer_size=pynv_buffer_size,
                )
                fps = pynv_reader.fps
                input_backend_name = "MediaMTX RTSP -> PyNvVideoCodec NVDEC/DLPack"
            except Exception as exc:
                if not allow_opencv_stream_fallback:
                    raise RuntimeError(
                        "MediaMTX RTSP zero-copy 입력 초기화에 실패했습니다.\n"
                        "PyNvVideoCodec ThreadedDecoder가 이 RTSP URL을 지원하지 않거나, "
                        "MediaMTX 스트림 코덱/프로파일을 NVDEC가 열 수 없는 상태입니다.\n"
                        "zero-copy를 포기하고 OpenCV로 받으려면 "
                        "ALLOW_OPENCV_STREAM_FALLBACK=True로 바꾸세요.\n"
                        f"원인: {exc}"
                    ) from exc

                print("[STREAM] PyNvVideoCodec RTSP zero-copy 실패, OpenCV latest-frame으로 폴백합니다.")
                print(f"[STREAM] fallback reason: {exc}")

        if pynv_reader is None:
            latest_buffer = LatestFrameBuffer()
            capture_thread = threading.Thread(
                target=capture_latest_loop,
                args=(vid_path, latest_buffer),
                daemon=True,
            )
            capture_thread.start()

            print("[STREAM] waiting for first frame...")
            while True:
                first_frame, _ = latest_buffer.read()
                if first_frame is not None:
                    break
                if latest_buffer.is_stopped():
                    raise RuntimeError("Capture thread stopped before receiving frames.")
                time.sleep(0.01)

            input_backend_name = "OpenCV/FFmpeg RTSP latest-frame fallback"
    else:
        if use_pynvcodec:
            pynv_reader = PyNvCodecVideoReader(
                vid_path,
                gpu_id=gpu_id,
                buffer_size=pynv_buffer_size,
            )
            total_frames = pynv_reader.total_frames
            fps = pynv_reader.fps
            input_backend_name = "PyNvVideoCodec NVDEC/DLPack"
        else:
            ffmpeg_reader = FFmpegVideoReader(
                vid_path,
                ffmpeg_exe=ffmpeg_exe,
                use_nvdec=use_nvdec,
            )
            total_frames = ffmpeg_reader.total_frames
            fps = ffmpeg_reader.fps
            input_backend_name = "Explicit FFmpeg RGB pipe"

    out_mp4 = os.path.join(out_dir, out_name + "_Explicit_FFmpeg.mp4")
    video_writer = None
    if save_output:
        video_writer = AsyncFFmpegWriter(
            out_mp4,
            width=target_w * 2,
            height=target_h,
            fps=fps,
            ffmpeg_exe=ffmpeg_exe,
            prefer_nvenc=prefer_nvenc,
            queue_size=4,
        )

    # RVM TensorRT static engine recurrent states.
    # rvm_resnet50_static.engine 생성 시 사용한 입력 해상도와 맞아야 한다.
    rec = [
        torch.zeros(1, 16, 256, 144, dtype=torch.float32, device='cuda'),
        torch.zeros(1, 32, 128, 72, dtype=torch.float32, device='cuda'),
        torch.zeros(1, 64, 64, 36, dtype=torch.float32, device='cuda'),
        torch.zeros(1, 128, 32, 18, dtype=torch.float32, device='cuda'),
    ]
    prev_depth, prev_rot, prev_scale, prev_opacity = None, None, None, None
    outlier_threshold = 0.01
    ema_alpha = 0.9

    time_vram_total = 0.0
    time_prep = 0.0
    time_unet = 0.0
    time_vda = 0.0
    time_gs = 0.0
    time_render = 0.0
    time_io = 0.0
    time_input_io = 0.0
    time_output_io = 0.0
    processed_frames = 0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    t_pipeline_start = time.perf_counter()

    # 🆕 스트림 모드용 변수
    last_processed_frame_id = -1
    skipped_frames = 0
    frame_idx = 0

    # 🆕 진행률 표시: 파일이면 total_frames, 스트림이면 max_process_frames 또는 None
    if is_stream:
        pbar_total = max_process_frames if max_process_frames > 0 else None
        pbar = tqdm(total=pbar_total, desc="SRT Stream Inference")
    else:
        pbar = tqdm(total=total_frames, desc="Robust Streaming Inference")

    # ============================================================
    # 🆕 메인 loop: 스트림 vs 파일 양쪽 지원
    # ============================================================
    while True:
        # --- 종료 조건 ---
        if is_stream:
            if latest_buffer is not None and latest_buffer.is_stopped():
                print("[STREAM] capture stopped, exiting main loop")
                break
            if max_process_frames > 0 and processed_frames >= max_process_frames:
                print(f"[STREAM] max_process_frames reached: {max_process_frames}")
                break
        else:
            if total_frames > 0 and frame_idx >= total_frames:
                break

        # --- 프레임 획득 ---
        t_io_start = time.perf_counter()

        frame_rgb = None
        frame_cuda_u8 = None
        if is_stream and pynv_reader is not None:
            # MediaMTX RTSP → PyNvVideoCodec → [3, H, W] CUDA uint8.
            # DLPack 공유이므로 CPU frame/NumPy가 없다.
            frame_cuda_u8 = pynv_reader.read()
            if frame_cuda_u8 is None:
                # 네트워크 스트림에서 일시적으로 빈 배치가 올 수 있으므로 조금 기다린다.
                time.sleep(0.002)
                continue
        elif is_stream:
            frame_bgr, current_frame_id = latest_buffer.read()
            if frame_bgr is None:
                time.sleep(0.005)
                continue
            if current_frame_id == last_processed_frame_id:
                time.sleep(0.001)
                continue
            if last_processed_frame_id >= 0:
                skipped_frames += max(0, current_frame_id - last_processed_frame_id - 1)
            last_processed_frame_id = current_frame_id
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        elif pynv_reader is not None:
            # [3, H, W], CUDA uint8. DLPack 공유이므로 CPU frame/NumPy가 없다.
            frame_cuda_u8 = pynv_reader.read()
            if frame_cuda_u8 is None:
                break
        else:
            # FFmpeg가 RGB24로 직접 전달하므로 BGR->RGB 변환이 필요 없다.
            frame_rgb = ffmpeg_reader.read()
            if frame_rgb is None:
                break

        t_io_end = time.perf_counter()
        input_io_ms = (t_io_end - t_io_start) * 1000
        time_input_io += input_io_ms
        time_io += input_io_ms

        # 입력 Tensor 준비 시간. PyNvVideoCodec 경로는 CPU→GPU 복사 없이 GPU 내부
        # uint8→float32 정규화만 수행한다.
        t_vram_start = time.perf_counter()
        if frame_cuda_u8 is not None:
            img_tensor_raw = frame_cuda_u8.unsqueeze(0).to(dtype=torch.float32)
            img_tensor_raw.mul_(1.0 / 255.0)
        else:
            img_tensor_raw = (
                torch.from_numpy(frame_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device="cuda", dtype=torch.float32)
                .mul_(1.0 / 255.0)
            )
        torch.cuda.synchronize()
        t_vram_end = time.perf_counter()
        time_vram_total += (t_vram_end - t_vram_start)

        t_prep_start = time.perf_counter()
        with torch.no_grad():
            # img_tensor_raw는 PyNvVideoCodec 경로에서는 이미 CUDA Tensor이다.
            # 여기서 바로 RVM TensorRT에 넣으므로 입력부 zero-copy가 RVM까지 유지된다.
            rvm_out = rvm_trt(
                src=img_tensor_raw,
                r1i=rec[0], r2i=rec[1], r3i=rec[2], r4i=rec[3],
            )
            fgr = rvm_out['fgr']
            pha = rvm_out['pha']
            rec = [
                rvm_out['r1o'].clone(),
                rvm_out['r2o'].clone(),
                rvm_out['r3o'].clone(),
                rvm_out['r4o'].clone(),
            ]

            img_clean_tensor = img_tensor_raw * pha
            padded_img = pad_to_square_tensor(img_clean_tensor)
            padded_mask = pad_to_square_tensor(pha)
            img_tensor = F.interpolate(padded_img, size=(render_res, render_res), mode='area')
            mask_tensor = F.interpolate(padded_mask, size=(render_res, render_res), mode='nearest')
            # VDA 출력 안정성을 위해 기존 CPU NumPy 경계는 유지한다.
            img_504_np = (img_tensor[0].permute(1, 2, 0).detach() * 255.0).clamp(0, 255).cpu().numpy().astype(np.uint8)
        torch.cuda.synchronize()
        t_prep_end = time.perf_counter()
        time_prep += (t_prep_end - t_prep_start) * 1000

        start_event.record()
        depth_np = vda_stream.infer_video_depth_one(img_504_np, input_size=504, device='cuda', fp32=True)
        end_event.record()
        torch.cuda.synchronize()
        time_vda += start_event.elapsed_time(end_event)

        depth_pred = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0).cuda()
        img_gps = img_tensor * 2.0 - 1.0

        with torch.no_grad():
            start_event.record()
            unet_out = unet_trt(input_image_gps=img_gps)
            img_feat = (unet_out['feat1'], unet_out['feat2'], unet_out['feat3'])
            end_event.record()
            torch.cuda.synchronize()
            time_unet += start_event.elapsed_time(end_event)

            start_event.record()
            gs_out = gs_trt(img_gps=img_gps, depth=depth_pred, feat1=img_feat[0], feat2=img_feat[1], feat3=img_feat[2])
            rot = gs_out['rot_maps']
            scale = gs_out['scale_maps']
            opacity = gs_out['opacity_maps']

            if prev_depth is None:
                prev_depth, prev_rot, prev_scale, prev_opacity = depth_pred, rot, scale, opacity
            else:
                global_diff = torch.abs(depth_pred - prev_depth).mean().item()
                if global_diff > outlier_threshold:
                    depth_pred = prev_depth
                    rot = prev_rot
                    scale = prev_scale
                    opacity = prev_opacity
                else:
                    depth_pred = ema_alpha * depth_pred + (1.0 - ema_alpha) * prev_depth
                    rot = F.normalize(ema_alpha * rot + (1.0 - ema_alpha) * prev_rot, p=2, dim=1, eps=1e-6)
                    scale = ema_alpha * scale + (1.0 - ema_alpha) * prev_scale
                    opacity = ema_alpha * opacity + (1.0 - ema_alpha) * prev_opacity
                    prev_depth, prev_rot, prev_scale, prev_opacity = depth_pred, rot, scale, opacity

            end_event.record()
            torch.cuda.synchronize()
            time_gs += start_event.elapsed_time(end_event)

        t_render_start = time.perf_counter()
        bs = img_tensor.shape[0]
        data = {'view_0': {'img': img_tensor, 'mask': mask_tensor, 'intr': intr_tensor_504, 'extr': extr_tensor}}
        data['view_0']['depth'] = depth_pred
        data['view_0']['xyz'] = depth2pc(depth_pred, extr_tensor, intr_tensor_504).view(bs, -1, 3)

        valid_mask = (depth_pred > 0.05).view(bs, -1)
        depth_flat = depth_pred.view(valid_mask.shape)
        mask_flat = mask_tensor.view(valid_mask.shape)

        data['view_0']['pts_valid'] = valid_mask & (mask_flat > 0.5)
        data['view_0']['rot_maps'] = rot.view(bs, 4, render_res, render_res)
        data['view_0']['scale_maps'] = scale.view(bs, 3, render_res, render_res)
        data['view_0']['opacity_maps'] = opacity.view(bs, 1, render_res, render_res)

        novel_extr = base_extr.clone()
        R = novel_extr[:3, :3].T
        T = novel_extr[:3, 3]

        world_view_transform = torch.tensor(getWorld2View2(R.cpu().numpy(), T.cpu().numpy(), np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1).float().cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(proj_matrix.unsqueeze(0))).squeeze(0)
        cam_center = world_view_transform.inverse()[3, :3]

        data['novel_view'] = {
            'width': torch.tensor([render_res], dtype=torch.int32).cuda(),
            'height': torch.tensor([render_res], dtype=torch.int32).cuda(),
            'FovX': torch.tensor([fovx], dtype=torch.float32).cuda(),
            'FovY': torch.tensor([fovy], dtype=torch.float32).cuda(),
            'world_view_transform': world_view_transform.unsqueeze(0),
            'full_proj_transform': full_proj_transform.unsqueeze(0),
            'camera_center': cam_center.unsqueeze(0)
        }

        test_ipd = 0.6
        proj_mat = proj_matrix

        data_left = copy.deepcopy({'view_0': data['view_0'], 'novel_view': data['novel_view']})
        data_left['novel_view']['world_view_transform'][0, 3, 0] -= (test_ipd / 2.0)
        data_left['novel_view']['full_proj_transform'] = torch.bmm(data_left['novel_view']['world_view_transform'], proj_mat.unsqueeze(0))
        data_left['novel_view']['camera_center'] = data_left['novel_view']['world_view_transform'].inverse()[0, 3, :3].unsqueeze(0)

        data_right = copy.deepcopy({'view_0': data['view_0'], 'novel_view': data['novel_view']})
        data_right['novel_view']['world_view_transform'][0, 3, 0] += (test_ipd / 2.0)
        data_right['novel_view']['full_proj_transform'] = torch.bmm(data_right['novel_view']['world_view_transform'], proj_mat.unsqueeze(0))
        data_right['novel_view']['camera_center'] = data_right['novel_view']['world_view_transform'].inverse()[0, 3, :3].unsqueeze(0)

        with torch.no_grad():
            render_left, render_right = pts2render(
                {'lmain': data_left['view_0'], 'novel_view': data_left['novel_view']},
                {'lmain': data_right['view_0'], 'novel_view': data_right['novel_view']},
                bg_color=bg_color, is_train=False
            )

        render_tensor_left_504 = render_left['novel_view']['img_pred']
        render_tensor_right_504 = render_right['novel_view']['img_pred']
        render_tensor_left_1008 = F.interpolate(render_tensor_left_504, size=(target_h, target_w), mode='bicubic', align_corners=False)
    
        render_tensor_right_1008 = F.interpolate(render_tensor_right_504, size=(target_h, target_w), mode='bicubic', align_corners=False)

        shift = 300
        render_tensor_left_1008 = render_tensor_left_1008[..., :-shift]
        render_tensor_right_1008 = render_tensor_right_1008[..., shift:]
        render_tensor_left_1008 = F.pad(render_tensor_left_1008, (shift, 0), mode='constant', value=0.0)
        render_tensor_right_1008 = F.pad(render_tensor_right_1008, (0, shift), mode='constant', value=0.0)
        render_tensor_1008 = torch.cat([render_tensor_left_1008, render_tensor_right_1008], dim=3)
        render_chw = render_tensor_1008[0]
        render_hwc = render_chw.permute(1, 2, 0)
        alpha = torch.ones(render_hwc.shape[0], render_hwc.shape[1], 1, device=render_hwc.device)
        render_rgba = torch.cat([render_hwc, alpha], dim=2).contiguous()
        if show_window:
            CudaRuntime1.show_tensor(render_rgba)

        # 파일 저장을 사용할 때만 uint8 출력 프레임을 만든다.
        # 저장하지 않을 때는 GPU→CPU 복사뿐 아니라 불필요한 uint8 변환도 생략한다.
        render_img_1008 = None
        if video_writer is not None:
            render_img_1008 = (
                render_tensor_1008[0]
                .detach()
                .permute(1, 2, 0)
                .mul(255.0)
                .clamp(0, 255)
                .to(torch.uint8)
            )
        torch.cuda.synchronize()

        t_render_end = time.perf_counter()
        time_render += (t_render_end - t_render_start) * 1000

        t_io_start = time.perf_counter()
        if video_writer is not None:
            # 저장이 활성화된 경우에만 GPU 결과를 CPU로 내려 FFmpeg에 전달한다.
            output_frame_rgb = render_img_1008.cpu().numpy()
            video_writer.write(output_frame_rgb)
        t_io_end = time.perf_counter()
        output_io_ms = (t_io_end - t_io_start) * 1000
        time_output_io += output_io_ms
        time_io += output_io_ms

        processed_frames += 1
        frame_idx += 1
        pbar.update(1)

        if is_stream and processed_frames % 30 == 0:
            elapsed_now = time.perf_counter() - t_pipeline_start
            latest_id_text = current_frame_id if latest_buffer is not None else "pynv"
            print(f"[STREAM] processed={processed_frames}, "
                  f"latest_id={latest_id_text}, skipped={skipped_frames}, "
                  f"fps={processed_frames / elapsed_now:.2f}")

    if latest_buffer is not None:
        latest_buffer.stop()
        capture_thread.join(timeout=2.0)
    if pynv_reader is not None:
        pynv_reader.close()
    if ffmpeg_reader is not None:
        ffmpeg_reader.close()
    pbar.close()

    # 비동기 인코더에 남은 프레임까지 모두 기록한 뒤 전체 시간을 끝낸다.
    if video_writer is not None:
        video_writer.close()
    if show_window:
        CudaRuntime1.cleanup()
    t_pipeline_end = time.perf_counter()

    avg_prep = time_prep / processed_frames if processed_frames > 0 else 0
    avg_unet = time_unet / processed_frames if processed_frames > 0 else 0
    avg_vda = time_vda / processed_frames if processed_frames > 0 else 0
    avg_gs = time_gs / processed_frames if processed_frames > 0 else 0
    avg_render = time_render / processed_frames if processed_frames > 0 else 0
    avg_io = time_io / processed_frames if processed_frames > 0 else 0
    avg_input_io = time_input_io / processed_frames if processed_frames > 0 else 0
    avg_output_io = time_output_io / processed_frames if processed_frames > 0 else 0

    avg_network = avg_unet + avg_vda + avg_gs
    avg_cuda_pure = avg_prep + avg_network + avg_render

    print("\n===================================")
    print(" [End-to-End Pipeline Profiling (NVDEC Zero-Copy Input)]")
    print(" - Input Tensor Prepare/Transfer  : {:.2f} s (Total Time)".format(time_vram_total))
    print("-----------------------------------")
    print(" [Per-Frame GPU Inference Time]")
    print(" - Pre-processing (RVM-TRT)  : {:.2f} ms".format(avg_prep))
    print(" - Network Inference     : {:.2f} ms".format(avg_network))
    print("   ├─ U-Net (TRT)        : {:.2f} ms".format(avg_unet))
    print("   ├─ VDA Stream (PT)    : {:.2f} ms".format(avg_vda))
    print("   └─ GS Regresser (TRT) : {:.2f} ms".format(avg_gs))
    print(" - Render & Upsample     : {:.2f} ms".format(avg_render))
    print("   = Pure CUDA Operation : {:.2f} ms / frame".format(avg_cuda_pure))
    print("-----------------------------------")
    print(" - Input Decoder/DLPack   : {:.2f} ms / frame".format(avg_input_io))
    print(" - Input Backend          : {}".format(input_backend_name))
    if video_writer is not None:
        print(" - Output Copy/Enqueue   : {:.2f} ms / frame".format(avg_output_io))
    else:
        print(" - Output Save           : disabled")
    print(" - Total Video I/O       : {:.2f} ms / frame".format(avg_io))
    print(" - Total Pipeline Time   : {:.2f} s".format(t_pipeline_end - t_pipeline_start))
    if video_writer is not None:
        print(" - FFmpeg Encoder        : {}".format(video_writer.encoder))
        print(" - Output File           : {}".format(out_mp4))
    if is_stream:
        print(" - Skipped Input Frames  : {}".format(skipped_frames))
    if t_pipeline_end - t_pipeline_start > 0:
        print(" - Average Target FPS    : {:.2f} FPS".format(processed_frames / (t_pipeline_end - t_pipeline_start)))
    print("===================================\n")


if __name__ == "__main__":
    # MediaMTX에서 제공하는 RTSP 스트림을 기본 입력으로 사용한다.
    # MediaMTX 예: SRT 수신 → rtsp://localhost:8555/test 로 relay
    USE_STREAM = True
    # 결과 MP4 저장 비활성화: 출력 인코딩, GPU→CPU 복사, 큐 대기를 모두 생략한다.
    SAVE_OUTPUT = False
    SHOW_WINDOW = True

    # SAVE_OUTPUT=False일 때는 사용되지 않는다.
    # 나중에 저장을 다시 켤 경우 NVENC를 우선 사용한다.
    PREFER_NVENC = True

    # 파일/MediaMTX RTSP 입력을 PyNvVideoCodec NVDEC + CUDA device memory + DLPack으로 처리한다.
    # VDA 앞의 기존 .cpu().numpy() 경로는 출력 일관성을 위해 그대로 유지한다.
    USE_PYNVVIDEOCODEC = True
    # 실시간 스트림 지연을 줄이려면 2~4 권장. 파일 벤치마크는 8도 괜찮다.
    PYNVC_BUFFER_SIZE = 4
    GPU_ID = 0

    # USE_PYNVVIDEOCODEC=False일 때만 사용하는 기존 외부 FFmpeg fallback 옵션이다.
    USE_NVDEC = False

    # True로 바꾸면 PyNvVideoCodec RTSP가 실패할 때 OpenCV/FFmpeg CPU latest-frame으로 폴백한다.
    # 단, 폴백 경로는 zero-copy가 아니다.
    ALLOW_OPENCV_STREAM_FALLBACK = False

    if USE_STREAM:
        # MediaMTX 경로에 맞게 test/mystream 중 실제 경로로 바꾸면 된다.
        VID_PATH = "rtsp://127.0.0.1:8555/test"
        IS_STREAM = True
        # 테스트용 300프레임. 실제 계속 실행하려면 -1.
        MAX_FRAMES = 300
    else:
        VID_PATH = os.path.join(SCRIPT_DIR, "input_video.mp4")
        IS_STREAM = False
        MAX_FRAMES = -1

    CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "VDA_GPS_0529_Finetune_final.pth")
    ENGINE_DIR = os.path.join(SCRIPT_DIR, "trt_engines")
    REF_INTR_PATH = os.path.join(SCRIPT_DIR, "val", "parm", "0000_000", "0_intr.npy")
    OUT_DIR = os.path.join(SCRIPT_DIR, "inference_results")

    if not IS_STREAM and not os.path.exists(VID_PATH):
        print(f"\n[Error] 비디오 파일을 찾을 수 없습니다: {VID_PATH}")
    else:
        run_ultimate_streaming_inference_filtered(
            VID_PATH,
            CKPT_PATH,
            ENGINE_DIR,
            REF_INTR_PATH,
            OUT_DIR,
            is_stream=IS_STREAM,
            max_process_frames=MAX_FRAMES,
            save_output=SAVE_OUTPUT,
            show_window=SHOW_WINDOW,
            prefer_nvenc=PREFER_NVENC,
            use_nvdec=USE_NVDEC,
            use_pynvcodec=USE_PYNVVIDEOCODEC,
            pynv_buffer_size=PYNVC_BUFFER_SIZE,
            gpu_id=GPU_ID,
            allow_opencv_stream_fallback=ALLOW_OPENCV_STREAM_FALLBACK,
        )
