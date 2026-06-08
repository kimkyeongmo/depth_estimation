프로젝트 설정 및 인퍼런스 가이드
본 프로젝트는 GPS-Gaussian 기반의 실시간 3DGS 스트리밍 파이프라인을 포함하고 있습니다. 원활한 실행을 위해 아래의 환경 설정 단계를 따라주시기 바랍니다.

⚠️ 실행 전 주의사항
모든 인퍼런스 명령어는 GPS-Gaussian/ 폴더 내부에서 실행해야 합니다.

1. 디렉토리 및 모델 경로 설정
프로젝트 실행에 필요한 가중치 파일을 프로젝트 루트의 checkpoints/ 폴더 내에 배치해야 합니다. 아래의 구조를 참고하여 폴더를 생성하고 파일을 이동시켜 주세요.

GPS-Gaussian/
└── checkpoints/
    ├── base/
    │   └── vda_120cm_finetuned_epoch3.pth
    └── VDA_GPS.pth

2. 서브모듈(Submodules) 설치
기존 GPS-Gaussian의 환경 설정과 동일하게 Rasterization 패키지를 설치해야 합니다.

submodules/ 폴더 생성:
mkdir submodules
cd submodules

diff-gaussian-rasterization 다운로드 및 설치:

(기존 GPS-Gaussian 매뉴얼에 따라 패키지를 클론하고 pip install . 명령을 수행하세요.)

3. 카메라 파라미터 데이터 설정
인퍼런스 코드 실행을 위해 필요한 카메라 파라미터 파일들을 val/ 폴더 내에 배치해야 합니다.

경로 구조:
GPS-Gaussian/
└── val/(여기에 제공된 카메라 파라미터 파일들을 배치하세요)


🚀 실행 요약
환경 설정이 완료된 후, GPS-Gaussian/ 폴더에서 아래와 같이 실행 가능합니다:

이미지 인퍼런스: python inference_trt.py

비디오 인퍼런스: python inference_trt_video_filtered.py