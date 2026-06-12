# depth_estimation
## Rendering
### Installation
```bash
conda env create --file environment.yml
conda activate gps_gaussian
```

opengl install
```bash
pip install PyOpenGL PyOpenGL_accelerate
pip install glfw
pip install PySide6
```

gaussian rasterization install
```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting/
pip install -e submodules/diff-gaussian-rasterization
cd ..
```

GLFW install
```bash
sudo apt-get install libglfw3-dev
```
pybind/glfw file install

pybind c++ code compile
```
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$(python3 -c "import pybind11; print(pybind11.get_include())")
TORCH_INCLUDE=$CONDA_PREFIX/lib/python3.10/site-packages/torch/include
TORCH_LIB=$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib
CUDA_HOME=/usr/local/cuda

g++ -O2 -shared -fPIC -std=c++17 \
    -I${PYTHON_INCLUDE} \
    -I${PYBIND_INCLUDE} \
    -I${TORCH_INCLUDE} \
    -I${TORCH_INCLUDE}/torch/csrc/api/include \
    -I${CUDA_HOME}/include \
    -Ipybind/glad/include \
    pybind/glad/src/glad.c \
    pybind/CudaRuntime1/CudaRuntime1/viewer.cpp \
    -L${TORCH_LIB} \
    -L${CUDA_HOME}/lib64 \
    -ltorch -ltorch_python -lc10 -lc10_cuda -ltorch_cuda -ltorch_cpu \
    -lcudart \
    -lGL -lglfw \
    -Wl,-rpath,${TORCH_LIB} \
    -o CudaRuntime1.so
```

### Testing
1. 모델 가중치(Checkpoint) 준비
사전 학습된 모델 파일(.pth)을 프로젝트 루트 디렉토리에 위치시킵니다.


2. 양안 렌더링 실행
아래 명령어를 통해 특정 데이터셋에 대한 양안 결과물을 생성할 수 있습니다.
```bash
python test_real_data.py \
--test_data_root 'Dataset Path' \
--ckpt_path 'Model' \
--src_view 0 1 \
--ratio=0.5
```

--src_view: 참고할 소스 뷰 인덱스

--ratio: 시점 보간 비율 (0.5는 중간 지점)


📜 License & Credits
This project is licensed under the MIT License - see the LICENSE file for details.

Original Work: GPS-Gaussian by Shunyuan Zheng.