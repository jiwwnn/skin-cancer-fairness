# Skin Cancer Fairness

피부암 진단 모델의 공정성(fairness)을 개선하기 위한 멀티모달 학습 프로젝트입니다. Fitzpatrick Scale을 기반으로 한 피부 톤 그룹 간 성능 격차를 줄이는 것을 목표로 합니다.

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [데이터셋](#데이터셋)
- [설치](#설치)
- [사용법](#사용법)
- [모델](#모델)
- [공정성 메트릭](#공정성-메트릭)
- [프로젝트 구조](#프로젝트-구조)
- [실험 결과](#실험-결과)

## 개요

이 프로젝트는 피부암 진단에서 Fitzpatrick Scale에 따른 피부 톤 그룹 간의 성능 격차를 해소하기 위한 다양한 방법론을 구현하고 비교합니다. 이미지와 텍스트 설명을 결합한 멀티모달 학습 방식을 사용하여 공정성을 향상시킵니다.

## 주요 기능

- **멀티모달 학습**: 이미지(ResNet)와 텍스트(BERT) 임베딩을 결합한 contrastive learning
- **다양한 공정성 개선 방법론**: 여러 baseline 및 제안 방법 비교
- **공정성 메트릭**: PQD, DPM, EOM 등 다양한 공정성 지표 계산

## 데이터셋

### PAD-UFES-20
- 피부암 이미지 및 병변 설명 텍스트 포함
- Fitzpatrick Scale 정보 포함

### Fitzpatrick17k
- 대규모 피부 질환 이미지 데이터셋
- Fitzpatrick Scale 1-6 분류

데이터셋 전처리는 `preprocessing_pad.py`와 `preprocessing_fitz17.py`를 참고하세요.

## 설치

### 요구사항

```bash
# Python 3.7+
# CUDA 10.2+
# PyTorch 1.6.0+
```

### 의존성 설치

```bash
pip install -r etc/requirements.txt
```

주요 패키지:
- torch==1.6.0
- torchvision==0.7
- transformers==4.1.1
- scikit-learn==0.22.1
- pandas==1.2.0
- pyyaml==5.3.1

## 사용법

### 1. 데이터 준비

```bash
# 데이터셋 다운로드 (필요시)
python download_dataset.py

# 데이터 전처리
python preprocessing_pad.py
python preprocessing_fitz17.py
```

### 2. 설정 파일 수정

각 모델별 config 파일에서 데이터 경로 및 하이퍼파라미터를 설정합니다:

- `ours_config.yaml`: 제안 방법 설정
- `resnet_config.yaml`: ResNet baseline 설정
- `patchalign_config.yaml`: PatchAlign 방법 설정
- `disco_config.yaml`: DISCO 방법 설정
- 기타 모델별 config 파일

### 3. 모델 학습

```bash
# 제안 방법 학습
python ours_run.py

# Baseline 모델 학습
python resnet_run.py

# 다른 방법들
python patchalign_run.py
python disco_run.py
python atrb_run.py
python resm_run.py
python rewt_run.py
```

### 4. Ablation Study

```bash
python ablation_run.py
```

## 모델

### 제안 방법 (Ours)
- **구조**: Dual ResNet encoders + BERT text encoder
- **Loss**: Contrastive loss (NT-Xent) + Classification loss
- **특징**: 병변 설명과 피부 톤 설명을 각각 다른 이미지 인코더와 정렬

### Baseline 모델들
- **ResNet**: 단일 ResNet 기반 이미지 분류
- **PatchAlign**: Patch-level alignment 방법
- **DISCO**: Distribution alignment 방법
- **ATRB**: Attribute-based 방법
- **RESM**: Resampling 방법
- **REWT**: Reweighting 방법

## 공정성 메트릭

프로젝트에서 사용하는 공정성 평가 지표:

- **PQD (Performance Quality Difference)**: 그룹 간 정확도 격차 측정
- **DPM (Demographic Parity Metric)**: 예측 분포의 공정성 측정
- **EOM (Equalized Odds Metric)**: Equalized Odds 준수 정도 측정
- **EOpp0, EOpp1**: Equalized Opportunity 지표
- **EOdd**: Equalized Odds 지표

메트릭 계산은 `fairness_metric.py`에 구현되어 있습니다.

## 프로젝트 구조

```
skin_cancer_fairness/
├── data/                    # 데이터셋 및 메타데이터
│   ├── PAD_UFES_20_images/
│   ├── fitzpatrick17_images/
│   └── *.csv               # 데이터셋 분할 파일
├── models/                 # 모델 구현
│   ├── ours_model.py       # 제안 방법 모델
│   ├── resnet_model.py     # ResNet baseline
│   ├── patchalign_model.py
│   └── ...
├── dataloader/             # 데이터 로더
│   ├── ours_dataset.py
│   ├── baseline_dataset.py
│   └── ...
├── loss/                   # Loss 함수
│   ├── ours_loss.py
│   ├── disco_loss.py
│   └── ...
├── *_config.yaml          # 모델별 설정 파일
├── *_train.py             # 학습 로직
├── *_run.py               # 실행 스크립트
├── fairness_metric.py     # 공정성 메트릭 계산
└── preprocessing_*.py     # 데이터 전처리
```

## 실험 결과

모델 학습 후 다음 정보가 출력됩니다:
- 전체 정확도 (acc_avg)
- 그룹별 정확도 (acc_per_type)
- PQD, DPM, EOM 공정성 메트릭

## 참고사항

- 모든 경로는 절대 경로로 설정되어 있으므로, 환경에 맞게 config 파일을 수정해야 합니다.
- GPU 메모리 부족 시 batch_size를 조정하세요.
- Early stopping이 적용되어 있으며, patience는 10으로 설정되어 있습니다.



