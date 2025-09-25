# SNU-AI-Control

딥러닝 기반 모델 예측 제어(Model Predictive Control, MPC) 시스템 구현을 위한 PyTorch 프로젝트입니다.

## 🚀 설치 방법

### 1. 저장소 복제
```bash
git clone https://github.com/yourusername/snu-ai-control.git
cd snu-ai-control
```

### 2. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

## 💻 사용법

### 빠른 시작
기본 설정으로 테스트 스크립트 실행:
```bash
bash run_test.sh
```

### 사용자 정의 학습
명령줄 인자를 사용하여 학습 과정 커스터마이징:
```bash
python main.py \
    --model_name MPCController \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    --gpu_id 0
```

### 설정 파일

#### `configs/default_setting.yaml`
일반적인 학습 설정 포함:
- 학습 하이퍼파라미터 (배치 크기, 학습률, 에폭 수)
- 데이터 로딩 설정
- 로깅 설정

#### `configs/model_setting.yaml`
모델별 구체적인 설정:
- MPC 컨트롤러 파라미터

## 📁 프로젝트 구조

```
snu-ai-control/
│
├── src/
│   │
│   ├── configs/
│   │   ├── default_setting.yaml
│   │   └── model_setting.yaml
│   │
│   ├── data_provider/
│   │   ├── __init__.py
│   │   ├── build_dataset.py
│   │   ├── factory.py
│   │   └── load_dataset.py
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   └── factory.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   ├── mpc_controller.py
│   │   └── tft.py
│   │
│   ├── optimizers/
│   │   ├── __init__.py
│   │   └── factory.py
│   │
│   └── utils/
│       ├── log.py
│       ├── tools.py
│       └── utils.py
│
├── arguments.py
├── exp_builder.py
├── main.py
├── run_test.sh
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```

## 🔧 주요 구성 요소

### 모델
- **MPCController**: 모델 예측 제어와 신경망을 결합한 메인 모델

## 🛠️ 개발

### 새로운 모델 추가하기
1. `src/models/`에 모델 구현
2. `src/models/__init__.py`에 등록
3. `configs/model_setting`에 해당 모델 설정 파일 생성

### 새로운 데이터셋 추가하기
1. `src/data_provider/load_dataset.py`에 데이터 로더 구현
2. 필요에 따라 설정 파일 업데이트

## 📝 라이선스

이 프로젝트는 [LICENSE](LICENSE) 파일에 명시된 조건에 따라 라이선스가 부여됩니다.

---
**참고**: 이 프로젝트는 활발히 개발 중입니다. 정기적으로 업데이트를 확인해주세요.