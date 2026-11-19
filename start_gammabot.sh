#!/bin/bash
# 감마봇 멀티봇 실행 스크립트
# NPU 16~31 사용 (OOO봇: NPU 0~15)
# Qwen3-32B: NPU 16~23 (8개)

sudo docker run -it --rm \
  --network host \
  --privileged \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e RBLN_DEVICES=16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
  -v /home/westadmin/workspace/gammabot:/app \
  -v /home/westadmin/qwen3-32b-rbln-batch2:/home/westadmin/qwen3-32b-rbln-batch2:ro \
  gammabot-base:v1 \
  bash -c "ldconfig && cd /app && python gammabot_api.py"
