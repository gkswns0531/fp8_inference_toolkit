# Retrieval Evaluation Dataset Guide

> RARE (Retrieval Accuracy & Relevance Evaluation) 시스템용 데이터셋 설명 및 평가 방법 가이드

---

## 1. 디렉토리 구조

```
dataset/
├── domain/                          # 도메인 데이터셋 (4개)
│   ├── finance_eval_dataset.json    # 금융 (미국 빅테크 연차보고서)
│   ├── patent_eval_dataset.json     # 특허 (미국 특허 문서)
│   ├── legal_eval_dataset.json      # 법률 (미국 연방법)
│   └── hotpot_eval_dataset.json     # 일반 위키 (HotpotQA 기반)
│
├── client/                          # 고객사 데이터셋 (11개)
│   ├── gugak.json                   # 사전 빌드된 JSON (도메인 데이터셋과 동일 스키마)
│   ├── hanhwa_insurance.json
│   ├── isu_system.json
│   ├── jacs.json
│   ├── mirae_asset.json
│   ├── ok_finance.json
│   ├── sejong.json
│   ├── skens.json
│   ├── sumitomo.json
│   ├── trans_cosmos.json
│   ├── yuhan_kimberly.json
│   └── raw/                         # 원본 CSV 보관 (변환 원본)
│       ├── gold_test_data_only.csv   # 전 고객사 평가 QA 데이터
│       ├── vector_db/                # 고객사 원본 코퍼스
│       │   └── {company}_vector_db.csv
│       └── vector_db_clean/          # 정제된 코퍼스
│           └── {company}_vector_db.csv
│
└── DATASET_GUIDE.md                 # 이 문서
```

---

## 2. 도메인 데이터셋 (Domain Datasets)

### 2.1 공통 JSON 스키마

4개 도메인 데이터셋은 동일한 구조를 가진다:

```json
{
  "metadata": {
    "total_queries": 645,
    "total_corpus_chunks": 3281,
    "total_gold_chunks": 9367,
    "avg_gold_chunks_per_query": 14.52,
    "valid_samples": 645,
    "valid_sample_ratio": 1.0,
    "dataset_type": "main",
    "use_original_question": false
  },
  "queries": [
    {
      "query_id": "query_001",
      "question": "What is the trading symbol of the company led by Jensen Huang?",
      "gold_chunk_ids": ["chunk_id_1", "chunk_id_2", ...],
      "gold_chunk_groups": [
        ["chunk_id_1", "chunk_id_2"],    // Group 1 (정보 단위 A)
        ["chunk_id_3"]                    // Group 2 (정보 단위 B)
      ]
    }
  ],
  "corpus": {
    "ANNUAL_REPORT_NVIDIA_2023_page018_chunk002": {
      "content": "문서 텍스트...",
      "metadata": {
        "source_file": "ANNUAL_REPORT_NVIDIA_2023.pdf",
        "page_number": 18,
        "sub_text_index": "2",
        "token_count": 330
      }
    }
  }
}
```

### 2.2 핵심 개념: `gold_chunk_groups` (정보 그룹)

**질문에 답하기 위해 필요한 독립적인 정보 단위들의 집합.**

```
질문: "What is the federal legislature of the country where the Goetheanum is located?"

gold_chunk_groups:
  Group 1: [Goetheanum.txt_chunk001]           → "Goetheanum은 스위스에 있다"
  Group 2: [Federal_Assembly_Switzerland.txt]   → "스위스 연방의회"
```

- 각 그룹은 하나의 **독립적 정보 단위(hop)**를 나타냄
- 그룹 내 청크는 **동일 정보의 대체 가능한 출처** (하나만 검색되면 OK)
- 질문에 완전히 답하려면 **모든 그룹에서 최소 1개씩** 검색되어야 함

### 2.3 도메인별 상세 메타데이터

| 항목 | Finance | Patent | Legal | Hotpot (Wiki) |
|------|---------|--------|-------|---------------|
| **언어** | 영어 | 영어 | 영어 | 영어 |
| **쿼리 수** | 645 | 480 | 605 | 688 |
| **코퍼스 청크 수** | 3,281 | 3,339 | 4,090 | 3,218 |
| **정답 청크 총 수** | 9,367 | 8,708 | 4,470 | 1,647 |
| **쿼리당 평균 그룹 수** | 2.50 | 2.45 | 2.46 | 2.33 |
| **그룹당 평균 청크 수** | 6.69 | 9.84 | 3.21 | 1.03 |
| **멀티그룹 쿼리 비율** | 76.4% | 73.5% | 74.5% | 68.9% |
| **원본 문서 수** | 12 | 50 | 6 | 3,218 |
| **평균 토큰/청크** | 330.4 | 345.7 | 372.2 | 98.5 |
| **토큰 범위** | 1~512 | 1~512 | 2~512 | 12~364 |
| **파일 크기** | 7.0 MB | 7.7 MB | 7.4 MB | 2.4 MB |

### 2.4 도메인별 특성

#### Finance (금융)
- **출처**: 미국 빅테크 기업 연차보고서 (10-K)
- **기업**: Tesla, NVIDIA, Amazon, Apple, Meta, Microsoft (2023~2024)
- **특징**: 재무제표, 경영진 논의, 리스크 요인 등 구조화된 문서. 그룹당 청크 수가 많아(6.69) 동일 정보가 여러 곳에 분산

#### Patent (특허)
- **출처**: 미국 특허 문서 50건
- **분야**: 전자, 의약, 화학, 소프트웨어 등 다양한 기술 분야
- **특징**: 그룹당 청크 수 최대(9.84). 특허 청구항, 상세 설명이 여러 섹션에 걸쳐 반복되는 특성 반영

#### Legal (법률)
- **출처**: 미국 연방법 6종 (저작권법, 특허법, 중재법, 공공계약법, 공공건물법, 일반규정법)
- **특징**: 조문 참조가 많고 법률 용어 중심. 그룹당 청크 수(3.21)가 상대적으로 적어 정답이 명확한 편

#### Hotpot / General Wiki
- **출처**: HotpotQA 기반 위키피디아 문서 3,218건
- **특징**: 문서 1개 = 청크 1개 구조(평균 98.5 토큰). 그룹당 청크가 거의 1개(1.03)로, 정확한 문서를 찾아야만 커버리지가 오름. 가장 희소한(sparse) 정답 분포

---

## 3. 고객사 데이터셋 (Client Datasets)

### 3.1 데이터 구조

고객사 데이터셋은 도메인 데이터셋과 **동일한 JSON 스키마**를 사용한다. 각 고객사별 `{company}.json` 파일은 `metadata`, `queries`, `corpus`를 포함하며, `load_dataset()`으로 바로 로딩 가능하다.

**청크 ID 공식**: `client__{comp_name}__{file_name}__p{page_no:04d}`

#### 원본 CSV (raw/ 하위, 참고용)

원본 CSV는 `dataset/client/raw/`에 보관된다:

- `gold_test_data_only.csv` — 전 고객사 평가 QA 데이터
- `vector_db_clean/{company}_vector_db.csv` — 정제된 코퍼스

JSON 재생성이 필요한 경우: `python evaluation/convert_client_csv_to_json.py`

### 3.2 고객사별 메타데이터

| 고객사 | 쿼리 수 | 코퍼스(clean) | 고유 문서 수 | 도메인 | 언어 |
|--------|---------|--------------|-------------|--------|------|
| **gugak** (국립국악원) | 43 | 1,023 | 1,023 | 국악 사전/자료 | 한국어 |
| **hanhwa_insurance** (한화손해보험) | 183 | 1,615 | 6 | 보험 약관/규정 | 한국어 |
| **isu_system** (이수시스템) | 130 | 219 | 24 | 사내 규정 | 한국어 |
| **jacs** | 32 | 5,599 | 33 | - | 한국어 |
| **mirae_asset** (미래에셋) | 629 | 374 | 57 | 금융/사내규정 | 한국어 |
| **ok_finance** (OK금융) | 22 | 781 | 65 | 금융 | 한국어 |
| **sejong** (세종) | 23 | 5,753 | 130 | 법률/행정 | 한국어 |
| **skens** (SK ENS) | 190 | 626 | 167 | 에너지/산업 | 한국어 |
| **sumitomo** (스미토모) | 95 | 2,427 | 1,018 | 산업 | 일본어 |
| **trans_cosmos** (트랜스코스모스) | 87 | 267 | 5 | IT서비스 | 일본어 |
| **yuhan_kimberly** (유한킴벌리) | 203 | 2,021 | 2,021 | 소비재/사내규정 | 한국어 |
| **합계** | **1,637** | **20,705** | - | - | - |

### 3.3 `vector_db` vs `vector_db_clean`

- `vector_db`: 원본 코퍼스 (총 1.7GB)
- `vector_db_clean`: 정제된 코퍼스 (총 66MB) — **평가에 사용되는 버전**
- 청크 수는 동일하나 content가 정리됨

---

## 4. 평가 방법 (Evaluation Methodology)

### 4.1 평가 메트릭 (4종)

모든 메트릭은 **그룹 기반(group-based)**으로 계산된다.

#### Coverage@K (커버리지)
```
각 정보 그룹에서 최소 1개 청크가 top-K에 포함된 비율

Coverage@K = (커버된 그룹 수) / (전체 그룹 수)
```
- 그룹 내 청크가 **하나라도** top-K에 있으면 해당 그룹은 "커버됨"
- 가장 직관적인 메트릭: "필요한 정보를 얼마나 찾았나"

#### Perfect Match@K (완전 일치)
```
모든 정보 그룹이 커버되었으면 1, 아니면 0

Perfect Match@K = 1.0 if Coverage@K == 1.0 else 0.0
```
- 이진(binary) 메트릭
- 모든 정보를 빠짐없이 찾았는지 측정

#### NDCG@K (순위 품질)
```
각 그룹의 첫 발견 위치에 로그 할인 적용

DCG  = Σ (1 / log₂(rank + 2))   for each group's first hit
IDCG = Σ (1 / log₂(i + 2))      for i = 0..num_groups-1
NDCG = DCG / IDCG
```
- 상위 순위에서 정보를 발견할수록 높은 점수
- 같은 그룹에서 중복 발견은 무시 (첫 발견만 카운트)

#### MRR (평균 역순위)
```
각 그룹에서 첫 번째로 발견된 청크의 역순위 평균

MRR = (1/G) × Σ (1 / first_hit_rank_of_group_i)
```
- top-K 제한 없이 전체 결과에서 계산
- 그룹을 못 찾으면 해당 그룹 점수 = 0

### 4.2 도메인 vs 고객사 평가 차이

| 구분 | 도메인 데이터셋 | 고객사 데이터셋 |
|------|----------------|----------------|
| **평가 단위** | 전체 데이터셋 통합 | 고객사별 개별 평가 |
| **코퍼스 범위** | 전체 청크 풀 | 해당 고객사 청크만 |
| **그룹 분석** | Hop별 분석 (1~4 hop) | 없음 |
| **결과 파일** | 통합 1개 + 모델별 | 고객사별 + 전체 집계 |
| **집계 방식** | 쿼리 단위 평균 | 고객사별 → 전체 평균 |

### 4.3 Hop 분석 (도메인 전용)

도메인 데이터셋은 쿼리의 복잡도에 따라 **hop별 성능 분석**을 수행:

| Hop 수 | 의미 | 예시 |
|--------|------|------|
| **1-hop** | 단일 정보원 필요 | "NVIDIA의 CEO는 누구인가?" |
| **2-hop** | 2개 정보원 조합 필요 | "Goetheanum이 있는 나라의 연방의회는?" |
| **3-hop** | 3개 정보원 조합 필요 | 더 복잡한 추론 질문 |
| **4-hop** | 4개 정보원 조합 필요 | 고난도 멀티스텝 질문 |

hop이 증가할수록 Perfect Match 달성이 어려워지며, 이를 통해 모델의 **멀티스텝 검색 능력**을 측정한다.

### 4.4 평가 파이프라인

```
1. 데이터 로드
   ├─ 도메인: JSON 파일 직접 로드
   └─ 고객사: JSON 파일 직접 로드 (dataset/client/{company}.json)

2. 검색 실행 (모델별)
   ├─ 1차 검색: BM25 / BGE-M3 / Qwen3 등으로 top-N 후보 추출
   └─ [선택] 2차 리랭킹: 리랭커로 top-N → top-K 재정렬

3. 메트릭 계산
   ├─ 쿼리별: Coverage@K, Perfect Match@K, NDCG@K, MRR
   └─ 전체: 쿼리 단위 산술 평균

4. 결과 분석
   ├─ 도메인: 전체 + hop별(1~4) 성능 테이블
   └─ 고객사: 고객사별 + 전체 집계 테이블

5. 결과 저장 → evaluation/results/
```

### 4.5 지원 모델

| 모델 | 유형 | 비고 |
|------|------|------|
| BM25 | 키워드 기반 | 베이스라인 |
| BGE-M3 | Dense Embedding | 다국어 지원 |
| Qwen3-0.6B | Dense Embedding | 경량 모델 |
| Qwen3-4B | Dense Embedding | 중형 모델 |
| Qwen3-8B | Dense Embedding | 대형 모델 |
| Qwen3-VL-2B | Dense Embedding | 비전-언어 모델 |
| Qwen3-VL-8B | Dense Embedding | 비전-언어 대형 |

리랭커로도 위 모델들이 사용되며, `--reranker` 옵션으로 2단계 검색이 가능하다.

---

## 5. 평가 실행 예시

```bash
# 도메인 데이터셋 평가 (finance)
python evaluation/run_evaluation.py \
  --dataset finance \
  --models bge_m3 qwen3_8b \
  --top-k 10

# 리랭킹 포함 평가
python evaluation/run_evaluation.py \
  --dataset patent \
  --models bge_m3 \
  --reranker qwen3_8b \
  --retrieve-top-n 64 \
  --top-k 10

# 고객사 데이터셋 평가 (전체)
python evaluation/run_evaluation.py \
  --dataset client \
  --models bge_m3 \
  --top-k 10

# 특정 고객사만 평가
python evaluation/run_evaluation.py \
  --dataset client_mirae_asset \
  --models bge_m3 \
  --top-k 10
```

---

## 6. 결과 파일 네이밍 규칙

```
evaluation/results/
  # 도메인 데이터셋
  evaluation_results_{model}_top{k}_{dataset}.json
  evaluation_results_{model}_rerank_{reranker}_cand{n}_top{k}_{dataset}.json

  # 고객사 데이터셋
  evaluation_results_{model}_top{k}_client_{company}.json
  evaluation_results_{model}_rerank_{reranker}_cand{n}_top{k}_client_{company}.json
```

예시:
- `evaluation_results_bge_m3_top10_finance_eval_dataset.json`
- `evaluation_results_qwen3_8b_rerank_bge_m3_cand64_top10_client_mirae_asset.json`
