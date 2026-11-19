# ── gammabot-base:v1
# alphabot-base:v8 기반
# 추가: qdrant-client, sentence-transformers, qwen_vl_utils (VL 모델용)

FROM alphabot-base:v8

# ── VL 모델용 추가 패키지

copy wheels/ /tmp/wheels/

RUN pip install --no-index --find-links=/tmp/wheels qdrant-client==1.18.0 && rm -rf /tmp/wheels &&ldconfig

WORKDIR /app
