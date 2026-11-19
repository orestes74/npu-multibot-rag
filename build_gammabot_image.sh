#!/bin/bash
# gammabot-base:v1 이미지 빌드
# Dockerfile이 있는 디렉토리에서 실행

sudo docker build -t gammabot-base:v1 .
echo "빌드 완료: gammabot-base:v1"
