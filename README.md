# ESS 배터리 수명 예측

SKALA 데이터 분석 Mini-Project · 울산캠퍼스 3반 · 강도희

## 프로젝트 목적
초기 100 cycle에서 얻은 충전·방전 신호를 사용해 배터리의 수명(`cycle_life`)을 예측한다.
Batch 1로 회귀 모델을 선택·학습하고 Batch 1 hold-out에서 확인한 뒤, Batch 2에서 최종 성능을 평가한다.
Batch 3은 배치 간 일반화를 추가로 살피기 위한 평가에 사용한다.

## 프로젝트 개요
- **데이터셋:** MIT–Stanford Battery Dataset (Severson et al., *Nature Energy*, 2019)
- **학습 데이터:** Batch 1 (`2017-05-12`)
- **검증 데이터:** Batch 1의 충전 프로토콜 그룹 hold-out
- **테스트 데이터:** Batch 2 (`2018-02-20`)
- **추가 평가:** Batch 3 (`2018-04-12`)
- **태스크:** Regression · 초기 100 cycle 신호로 `cycle_life` 예측
- **평가 지표:** MAPE; 보조 지표로 MAE, RMSE, R² 사용

원자료에는 총 139개 셀이 있다. `cycle_life`가 있는 셀은 129개이며, Batch 2의 8개와 Batch 3의 2개는 라벨이 없다.
누락된 라벨은 대체하지 않았다. 이 셀들은 지도학습 평가에서 제외하고, 라벨을 사용하지 않는 용량 곡선 탐색에만 포함했다.

| 배치 | 전체 셀 | 유효 수명 라벨 | 평균 수명 | 중앙값 | 범위 | `<500` | `>1,000` |
|---|---:|---:|---:|---:|---:|---:|---:|
| Batch 1 | 46 | 46 | 844.7 | 858.5 | 534–1,227 | 0 | 10 |
| Batch 2 | 47 | 39 | 565.7 | 472.0 | 392–1,186 | 28 | 3 |
| Batch 3 | 46 | 44 | 1,059.7 | 1,005.5 | 541–1,935 | 0 | 23 |

## 파일 구조
```text
.
├── assets/figures/       # EDA 시각화
├── data/raw/             # 로컬 원자료; 공개 저장소에서 제외
├── output/pdf/            # DAY 1 설계 PDF
├── outputs/eda/           # 추출 피처, 요약표, 예측에 사용한 EDA 결과
├── outputs/model/         # 모델 지표와 셀별 예측
├── reports/               # README에 반영한 평가 결과
├── src/
│   ├── analysis.py        # 데이터 로드, 피처 생성, EDA
│   ├── model.py           # 모델 탐색, 학습, 평가
│   └── make_day1_pdf.py   # DAY 1 설계 PDF 생성
└── README.md
```

## 환경 설정 및 재현
```bash
git clone https://github.com/Kangdohui/ess-health-cycle-life.git
cd ess-health-cycle-life
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

원자료 세 파일을 공개 Kaggle 데이터셋 또는 [MATR 원본 프로젝트](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204)에서 내려받아 `data/raw/`에 둔다.
- `2017-05-12_batchdata_updated_struct_errorcorrect.mat`
- `2018-02-20_batchdata_updated_struct_errorcorrect.mat`
- `2018-04-12_batchdata_updated_struct_errorcorrect.mat`
```bash
python src/analysis.py
python src/model.py
python src/make_day1_pdf.py
```

원자료 `.mat` 파일은 공개 저장소에 포함하지 않았다. 분석 스크립트는 `outputs/eda/`에 표와 그림을,
모델 스크립트는 `outputs/model/`에 성능 지표와 셀별 예측을 저장한다.

## EDA

### Cycle Life 분포

Batch 2에서만 500 cycle 미만 셀이 관찰되며, 유효 라벨 39개 중 28개(71.8%)가 이 구간에 속한다.
Batch 1에는 500 cycle 미만 셀이 없고 중앙값도 Batch 2보다 386.5 cycle 높다.
Batch 2 최단수명 셀 두 개는 서로 다른 충전 프로토콜이므로, 특정 프로토콜 하나가 단수명을 일으켰다고 단정하지 않았다.

![배치별 Cycle Life 분포](assets/figures/01_cycle_life_by_batch.png)

### 열화 곡선 분석

Cycle 100에서 500까지의 배치별 중앙 Qd 변화는 Batch 1 −2.30%,
Batch 2 −3.84%, Batch 3 −2.00%였다. 15-cycle 중앙값 smoothing 후
탐색한 열화 가속 후보 시점은 Batch 1 479, Batch 2 492, Batch 3 361 cycle이다.
이는 탐색용 knee 후보이며 확정된 물리적 전환점으로 해석하지 않았다.

원자료에서 1.3 Ah를 초과한 Qd 측정값 13건(10개 셀)은 곡선과 knee 요약에서 제외했다. 원자료와 추출 데이터에는 유지했다.

![배치별 Qd 열화 곡선](assets/figures/02_qd_degradation_by_batch.png)

### 초기 ΔQ(V) 곡선

`ΔQ(V) = Qdlin(100 cycle) − Qdlin(10 cycle)`로 차이 곡선을 계산했다.
평균, 표준편차, 절대 면적, 전압 구간별 크기와 피크 전압을 통계 피처로 만들었다.
배치별로 수명과 가장 강하게 연관된 ΔQ 피처의 절대 Spearman 상관계수는 0.73–0.87이었다.
여러 ΔQ 통계량끼리도 상관이 높아 중복 정보를 줄이는 피처 선택과 Ridge 규제를 모델 후보에 포함했다.

![초기 ΔQ(V) 곡선](assets/figures/03_delta_q_voltage_by_batch.png)

### 충전 속도와 수명

C1과 cycle life의 Spearman 상관은 Batch 1 −0.483, Batch 2 +0.055, Batch 3 −0.229였다.
C1과 첫 100 cycle Qd 기울기의 상관도 각각 −0.380, −0.187, −0.068로 배치마다 달랐다.
충전 프로토콜별 수명 차이는 표본 구성을 통제하지 않은 관찰 결과이므로 고속 충전의 인과 효과로 일반화하지 않았다.

![충전 프로토콜과 수명](assets/figures/04_policy_vs_life.png)

### 초기 피처 관계와 중복

첫 100 cycle의 Qd, Qc, IR, 온도, 충전시간 요약치와 ΔQ 통계량을 비교했다. 일부 ΔQ 통계량이 서로 강하게 연관되어 있어 피처 간 중복이 확인됐다. 배치별 상관 구조도 같지 않으므로 단일 피처의 상관만으로 예측 관계가 모든 배치에 유지된다고 보지 않았다.

![초기 피처 상관 및 중복](assets/figures/05_early_feature_relationships.png)

## Modeling

### 피처 엔지니어링 전략

EDA에서 초기 ΔQ 통계량이 수명과 연관되고 서로 중복되는 점을 확인해,
첫 100 cycle의 충전·방전 요약 피처와 ΔQ(V) 통계 피처를 후보로 삼았다.
최종 Ridge 모델은 충전시간 평균과 ΔQ 통계량에서 선택한 8개 피처를 사용했다.
결측값 대체, `SelectKBest` 피처 선택, 표준화는 모두 학습 파이프라인 안에서 fit해 검증·테스트 데이터가 전처리에 섞이지 않도록 했다.

### 모델 선택 및 근거

- **후보 모델:** 중앙값 Dummy 회귀 기준선, Ridge 회귀, 얕은 Random Forest 회귀
- **최종 모델:** Ridge (`alpha=0.1`)
- **선택 근거:** Batch 1 학습 부분의 프로토콜 그룹 교차검증에서 MAPE를 비교했다.
                   Ridge가 CV 평균 MAPE 7.59%로 선택됐으며, hold-out과 두 테스트 배치에서도 Dummy 기준선보다 낮은 MAPE를 기록했다.
- **분할 및 누수 방지:** Batch 1 유효 라벨 46개 중 35개를 학습 부분, 11개를 hold-out으로 나눴다. 같은 충전 프로토콜이 학습과 검증 양쪽에 섞이지 않도록 그룹 분할을 사용했다. 모델 선택 후 Batch 1 전체로 최종 모델을 다시 fit하고 Batch 2에서 한 번 평가했다. Batch 3은 추가 평가로 사용했다.
- **재현성:** 난수 시드 42. 교차검증 fold 수는 학습 부분의 프로토콜 그룹 수에 따라 최대 5개다.

## 성능 결과

회귀 지표는 MAPE로 보고했다. 보조 지표로 MAE, RMSE, R²를 계산했다.

중앙값 Dummy 기준선과 선택 모델을 나란히 비교하면 모든 평가 구간에서 Ridge가 더 낮은 MAPE를 기록했다. 세부 기준선 지표는 `reports/evaluation.csv`에서 확인할 수 있다.

가이드의 성능 보고 형식에 맞춰 핵심 지표를 MAPE로 정리했다.
Gap은 비교 기준에서 평가 구간으로 이동할 때의 MAPE 변화량(평가 구간 MAPE − 비교 기준 MAPE)이다.
따라서 Gap (Train−Valid)은 Valid − Train, Gap (Valid−Test)은 Test − Valid, Gap (Target−Test)은
Batch 2 − 논문 참고 MAPE로 계산했다. 양수는 뒤 구간의 오차가 커졌음을 나타낸다.

| 구분 | MAPE (%) | 비고 |
|---|---:|---|
| Train (Batch 1 CV) | 7.59 ± 1.06 | Batch 1 학습 부분의 그룹 교차검증 평균 ± 표준편차 |
| Valid (Batch 1 Hold-out) | 10.50 | 학습에 포함하지 않은 충전 프로토콜 그룹 |
| Test (Batch 2) | 25.96 | 최종 평가 |
| Gap (Train−Valid) | +2.91 pp | 양수: 학습 내 일반화 저하 / 과적합 가능성 |
| Gap (Valid−Test) | +15.46 pp | 양수: 배치 간 일반화 저하 |
| Gap (Target−Test) | +16.86 pp | Batch 2 MAPE − 논문 참고값 9.1% |

보조 지표로 Batch 2의 MAE는 131.3 cycle, RMSE는 145.5 cycle, R²는 0.56이다.
Batch 3의 추가 평가에서는 MAPE 14.39%, MAE 174.9 cycle, RMSE 278.1 cycle, R² 0.20을 기록했다.
논문은 첫 100 cycle을 이용한 최선 모델의 9.1% test error를 보고했으며, 여기서는 과제 가이드의 참고 기준으로 비교했다.

### Batch 3 추가 평가 (선택)

| 구분 | MAPE (%) | 비고 |
|---|---:|---|
| Test (Batch 3) | 14.39 | 추가 배치 평가; Batch 2와 수명 분포가 다름 |
| Gap (Batch 2−Batch 3) | +11.58 pp | Batch 2 오차가 Batch 3보다 큼 |
| Gap (Target−Test) | +5.29 pp | Batch 3 MAPE − 논문 참고값 9.1% |

Batch 3 MAPE가 Batch 2보다 낮다고 해서 Batch 3에 대한 일반화가 더 우수하다고 단정할 수는 없다.
Batch 3에는 1,000 cycle 초과 셀이 많고 Batch 2에는 500 cycle 미만 셀이 집중되어 있어, 배치별 수명 분포 차이가 성능 차이에 영향을 준다.

## 오류 분석

- Batch 2의 500 cycle 미만 셀 28개에 대한 MAPE는 29.14%였다.
- 학습 Batch 1에는 이 수명 구간이 없었고, 짧은 수명 셀 일부를 실제보다 길게 예측했다.
- Batch 2에서 상대 오차가 큰 사례는 `Batch2_042`(실제 442, 예측 175 cycle; 절대 백분율 오차 60.4%),
   Batch2_016`(396, 620; 56.6%), `Batch2_043`(474, 726; 53.1%)이다.
- Batch 3은 실제 1,000 cycle 초과 셀 23개 중 긴 수명을 낮게 예측하는 사례가 여럿 있다.
- 상대 오차가 큰 사례에는 `Batch3_043`(1,642, 782; 52.4%), `Batch3_039`(1,935, 1,038; 46.4%),
    Batch3_008`(1,836, 1,098; 40.2%)가 포함된다.
- 가능한 원인은 배치별 수명 분포 이동과 학습 표본의 범위 제한이다.
   Batch 1 수명 범위는 534–1,227 cycle인 반면 Batch 3은 최대 1,935 cycle이다.
   피처로 포착되지 않은 배치 조건도 영향을 줄 수 있지만, 현재 자료만으로 원인을 확정할 수 없다.
- 개선 방향은 짧은 수명 셀과 장수명 셀을 추가 확보해 검증하고, 수명 구간·프로토콜별 오차를 계속 보고하며,
   학습 범위에서 벗어난 입력은 별도로 표시하는 것이다. 새 모델이나 조정값은 Batch 2/3 테스트 점수로 선택하지 말고 별도 검증 절차에서 평가해야 한다.

## ESS 도메인 해석

초기 수명 예측은 BESS 운영자가 배터리 교체·점검 우선순위를 검토하거나,
셀 간 수명 편차가 큰 랙을 추가 진단 대상으로 선별하는 보조 신호가 될 수 있다.
이 분석은 실험실 셀의 cycle life를 추정할 뿐, 실제 ESS의 잔여 사용 기간이나 안전 상태를 직접 판정하지 않는다.

실제 BESS 적용 전에는 현장 셀과 운전 조건에 대한 별도 검증, 예측 불확실성 및 범위 밖 감지,
온도·충방전 깊이·운전 이력에 대한 검토, BMS 안전 경보 및 정비 절차와의 연동이 필요하다.
현재 Batch 2/3 결과의 배치 민감도와 수명 구간별 오차를 고려하면, 단독 교체 결정에 사용하기보다
추가 점검을 유도하는 참고 정보로 다루는 것이 적절하다.

## 참고문헌

- Severson, K. A. et al. (2019). [Data-driven prediction of battery cycle life before capacity degradation](https://doi.org/10.1038/s41560-019-0356-8). *Nature Energy*, 4, 383–391. 논문에서 보고한 9.1% test error를 성능 비교 참고값으로 사용했다.
- 데이터 원본: [MATR Battery Data Genome Project](https://data.matr.io/1/projects/5c48dd2bc625d700019f3204).
- 분석에 사용한 데이터 파일 미러: [Kaggle - Data-driven prediction of battery cycle](https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle).
