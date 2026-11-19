"""
build_vectorstore_qdrant.py
============================
JSON 청크 파일을 읽어 Qdrant에 적재합니다.

사용법:
    # bot_a 컬렉션 적재
    python3 build_vectorstore_qdrant.py --collection bot_a --data ./data/bot_a/

    # bot_b 컬렉션 적재
    python3 build_vectorstore_qdrant.py --collection bot_b --data ./data/bot_b/

    # 컬렉션 초기화 후 재적재
    python3 build_vectorstore_qdrant.py --collection bot_a --data ./data/bot_a/ --reset
"""

import os
import json
import sys
import glob
import argparse
import uuid
from typing import List, Dict, Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, OptimizersConfigDiff
)
from tqdm import tqdm

# ── 설정 ──────────────────────────────────────────────────────────────────
BGE_MODEL_NAME="BAAI/bge-m3"
BGE_CACHE_FOLDER = "/home/westadmin/.cache/huggingface/huggingface/hub"
# BGE_MODEL_PATH = "/home/westadmin/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"
QDRANT_HOST    = "localhost"
QDRANT_PORT    = 6333
VECTOR_SIZE    = 1024
BATCH_SIZE     = 32


def load_model():
    print(f"BGE-M3 로딩: {BGE_CACHE_FOLDER}")
    model = SentenceTransformer(BGE_MODEL_NAME, device="cpu", cache_folder=BGE_CACHE_FOLDER)
    model.max_seq_length = 1024
    print("BGE-M3 로드 완료")
    return model


def get_or_create_collection(client: QdrantClient, collection_name: str, reset: bool = False):
    existing = [c.name for c in client.get_collections().collections]
    if reset and collection_name in existing:
        print(f"컬렉션 '{collection_name}' 삭제 중...")
        client.delete_collection(collection_name)
        existing.remove(collection_name)

    if collection_name not in existing:
        print(f"컬렉션 '{collection_name}' 생성 중...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(f"컬렉션 '{collection_name}' 생성 완료")
    else:
        count = client.count(collection_name=collection_name).count
        print(f"기존 컬렉션 '{collection_name}' 사용 (현재 {count}개 포인트)")


def load_json_files(data_path: str) -> List[Dict]:
    """data/ 폴더 또는 단일 JSON 파일에서 청크 로드"""
    chunks = []

    if os.path.isfile(data_path):
        files = [data_path]
    else:
        files = sorted(glob.glob(os.path.join(data_path, "**/*.json"), recursive=True))
        # 노드/관계 파일 제외
        files = [f for f in files if not any(x in os.path.basename(f) for x in ["_nodes", "_relations", "vectorstore"])]

    print(f"파일 {len(files)}개 발견")

    for filepath in files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 형식 1: 청크 리스트
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and ("content" in item or "text" in item):
                        chunks.append({"filepath": filepath, **item})

            # 형식 2: {"chunks": [...]}
            elif isinstance(data, dict) and "chunks" in data:
                doc_info = data.get("document", {})
                for chunk in data["chunks"]:
                    chunks.append({
                        "filepath": filepath,
                        "source_file": doc_info.get("source_file", os.path.basename(filepath)),
                        **chunk
                    })

            print(f"  {os.path.basename(filepath)}: {len(data) if isinstance(data,list) else len(data.get('chunks',[]))}개 청크")

        except Exception as e:
            print(f"  {filepath} 로드 실패: {e}")

    print(f"총 {len(chunks)}개 청크 로드 완료")
    return chunks


def chunk_to_text(chunk: Dict) -> str:
    """임베딩할 텍스트 생성 (topic + content + keywords)"""
    parts = []
    topic = chunk.get("topic", "") or chunk.get("title", "")
    content = chunk.get("content", "") or chunk.get("text", "")
    keywords = chunk.get("keywords", [])
    if isinstance(keywords, list):
        keywords = " ".join(keywords)

    if topic:
        parts.append(topic)
    if content:
        parts.append(content)
    if keywords:
        parts.append(keywords)

    return "\n".join(parts).strip()


def chunk_to_payload(chunk: Dict) -> Dict:
    """Qdrant payload 생성"""
    content = chunk.get("content", "") or chunk.get("text", "")
    keywords = chunk.get("keywords", [])
    if isinstance(keywords, list):
        keywords = ", ".join(keywords)

    return {
        "chunk_id":    chunk.get("chunk_id", chunk.get("id", str(uuid.uuid4())[:8])),
        "text":        content,
        "content":     content,
        "title":       chunk.get("title", chunk.get("topic", "")),
        "topic":       chunk.get("topic", ""),
        "source_file": chunk.get("source_file", os.path.basename(chunk.get("filepath", ""))),
        "doc_id":      chunk.get("doc_id", ""),
        "keywords":    keywords,
        "chunk_type":  chunk.get("chunk_type", ""),
        "char_count":  chunk.get("char_count", len(content)),
    }


def upsert_chunks(client: QdrantClient, model: SentenceTransformer,
                  collection_name: str, chunks: List[Dict]):
    """청크를 임베딩하여 Qdrant에 upsert"""
    total = 0

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="임베딩+적재"):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [chunk_to_text(c) for c in batch]

        # 빈 텍스트 건너뜀
        valid = [(t, c) for t, c in zip(texts, batch) if t.strip()]
        if not valid:
            continue

        texts_v, chunks_v = zip(*valid)
        vectors = model.encode(list(texts_v), normalize_embeddings=True, batch_size=BATCH_SIZE)

        points = []
        for vec, chunk in zip(vectors, chunks_v):
            payload = chunk_to_payload(chunk)
            # chunk_id를 UUID로 변환 (Qdrant는 UUID 또는 정수 ID 필요)
            chunk_id = payload["chunk_id"]
            try:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
            except:
                point_id = str(uuid.uuid4())
            points.append(PointStruct(id=point_id, vector=vec.tolist(), payload=payload))

        client.upsert(collection_name=collection_name, points=points)
        total += len(points)

    print(f"\n적재 완료: {total}개 포인트")
    count = client.count(collection_name=collection_name).count
    print(f"컬렉션 '{collection_name}' 총 포인트 수: {count}")


def main():
    parser = argparse.ArgumentParser(description="Qdrant 벡터스토어 구축")
    parser.add_argument("--collection", required=True, help="컬렉션명 (bot_a, bot_b 등)")
    parser.add_argument("--data", required=True, help="JSON 파일 또는 폴더 경로")
    parser.add_argument("--host", default=QDRANT_HOST, help="Qdrant 호스트")
    parser.add_argument("--port", type=int, default=QDRANT_PORT, help="Qdrant 포트")
    parser.add_argument("--reset", action="store_true", help="기존 컬렉션 삭제 후 재생성")
    args = parser.parse_args()

    # Qdrant 연결
    print(f"Qdrant 연결: {args.host}:{args.port}")
    client = QdrantClient(host=args.host, port=args.port)

    # 컬렉션 준비
    get_or_create_collection(client, args.collection, reset=args.reset)

    # 청크 로드
    chunks = load_json_files(args.data)
    if not chunks:
        print("처리할 청크가 없습니다.")
        sys.exit(1)

    # 임베딩 모델 로드
    model = load_model()

    # 적재
    upsert_chunks(client, model, args.collection, chunks)

    print(f"\n✓ '{args.collection}' 컬렉션 구축 완료!")


if __name__ == "__main__":
    main()
