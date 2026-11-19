# npu-multibot-rag

Rebellions NPU 1세트(Qwen3-32B, batch=2)로 RAG 챗봇 2개를 동시에 서비스하는 멀티봇 서버 (Qdrant + FastAPI, 롱폴링 스트리밍)

## 감마봇 (Gammabot) - 멀티봇 RAG 서버

A봇, B봇 2개를 하나의 서버에서 운영하는 멀티봇 시스템.
OOO봇/베타봇(NPU 0~15)과는 완전히 분리된 별도 컨테이너/프로세스로 운영 — 감마봇 장애가 OOO봇에 영향을 주지 않음.
그리고 멀티쓰레드 가능한 상태임(이거 아주 골아픈 문제임 npu는 gpu처럼 자동으로 멀티쓰레드가 안됨, 모델 컴파일 단계에서부터 반영되어야 함)

## 인프라 구성

| 구성요소 | 사양 |
|---------|------|
| LLM | Qwen3-32B (RBLN 컴파일, NPU 16~31 전체, RSD 텐서 병렬, **batch=2 동시 2명 처리**) |
| 임베딩 | BGE-M3 (CPU) |
| 벡터DB | Qdrant (localhost:6333) |
| 스트리밍 | 롱폴링 (`/chat/start` + `/chat/poll`) — SSE 미사용 |
| 포트 | 8000 (내부) → 8002 (호스트) |

> 모델 폴더는 베타봇과 동일한 `qwen3-32b-rbln-batch2`(batch=2 컴파일본)를 읽기 전용으로 공유 마운트.
> 물리 NPU 배정은 컨테이너 실행 시 `RBLN_DEVICES` 환경변수가 정하므로 폴더 공유 자체는 안전함.
> 기동 로그에 `[엔진] batch=2 프로브 성공`이 뜨면 실제 동시 2명 처리가 확정된 것 — 실패 시 batch=1 직렬 모드로 자동 강등.

## 디렉토리 구조

```
gammabot/
├── gammabot_api.py              # 메인 API 서버 (bot_a/bot_b 라우터, 롱폴링)
├── build_vectorstore_qdrant.py  # 문서 적재 스크립트 (변경 없음)
├── chunk_pipeline.py            # 문서 청킹 파이프라인 (베타봇과 공유)
├── Dockerfile                   # gammabot-base:v1 이미지
├── build_gammabot_image.sh      # 이미지 빌드
├── start_gammabot.sh            # 서버 실행
├── data/
│   ├── bot_a/                 # A봇 문서 JSON (chunk_pipeline 출력)
│   └── bot_b/                     # B봇 문서 JSON (chunk_pipeline 출력)
├── static_bot_a/               # A봇 UI
├── static_bot_b/                   # B봇 UI
└── logs/                        # 봇별 채팅 로그
```

## 설치 순서

### 1. Docker 이미지 빌드

```bash
cd ~/workspace/gammabot
sudo bash build_gammabot_image.sh
```

### 2. Qdrant 실행 (이미 띄워져 있으면 생략)

```bash
sudo docker run -d \
  --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v ~/workspace/gammabot/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 3. 문서 청킹 (chunk_pipeline.py)

```bash
python3 chunk_pipeline.py --input ./raw_docs/bot_a/ --output ./data/bot_a/
python3 chunk_pipeline.py --input ./raw_docs/bot_b/ --output ./data/bot_b/
```

### 4. 문서 적재

```bash
cd ~/workspace/gammabot

python3 build_vectorstore_qdrant.py --collection bot_a --data ./data/bot_a/
python3 build_vectorstore_qdrant.py --collection bot_b --data ./data/bot_b/
```

### 5. 서버 실행

```bash
sudo bash start_gammabot.sh
```

## API

| 봇 | UI 경로 | 채팅 시작 | 채팅 폴링 |
|----|---------|----------|----------|
| A봇 | /bot_a/ | POST /bot_a/chat/start | GET /bot_a/chat/poll?job_id=&offset= |
| B봇 | /bot_b/ | POST /bot_b/chat/start | GET /bot_b/chat/poll?job_id=&offset= |
| 헬스체크 | - | GET /health (봇별로도 /bot_a/health, /bot_b/health) |

**흐름**: `/chat/start`에 `{message, history}` POST → `{job_id}` 반환 → `/chat/poll?job_id=..&offset=0`을 반복 호출(응답의 `next_offset`을 다음 요청의 `offset`으로) → `done:true`가 오면 `sources`/`mode` 포함해 종료.

## 주의사항

- OOO봇/베타봇은 NPU 0~15, 별도 컨테이너로 운영 중 → 건드리지 않음
- 감마봇은 NPU 16~31 전체를 Qwen3-32B(batch=2) 단독 사용, 동시 2명까지 처리
- Qdrant는 호스트에서 직접 실행 (Docker 내부 아님)
- Neo4j 없음 (벡터RAG만 사용)
- SSE 대신 롱폴링 사용 — 베타봇에서 확인된 DMZ AP 서버의 완료 시점까지 버퍼링 문제를 원천적으로 회피
- VL 모델(Qwen3-VL-8B)은 사용하지 않음 — NPU 16~31 전체가 텍스트 모델(batch=2)에 배정되어 있어 같은 물리 NPU 위에 VL을 얹을 수 없음


## 벡터스토어 업로드
- 두가지 방식이 있는데 스크립트 직접 실행과 컨테이너에서 스크립트 실행 중 택1한다
- 두 방법이 각기 bge-m3모델의 경로가 다르다는 문제가 있음
- 첫번째 방법은 트랜스포머의 버전과 관련된 이슈가 있는데 현재는 수정 완료

python3 build_vectorstore_qdrant.py --collection bot_a --data ./data/bot_a/

sudo docker run --rm --network host -v ~/workspace/gammabot:/app gammabot-base:v1 bash -c "cd /app && python3 build_vectorstore_qdrant.py --collection bot_a --data ./data/bot_a/"

## 포함하지 않은 것

- 문서 청크 데이터(`data/`), Qdrant 저장소, 채팅 로그
- 로고 이미지(`logo.png`)
- 프롬프트의 기관명·연락처는 `OOO`, `000-0000`, `example.com`으로 바꿔 두었습니다. 실제 운영 시 채워 넣으세요.
