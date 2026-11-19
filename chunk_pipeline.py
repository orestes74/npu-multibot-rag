# -*- coding: utf-8 -*-
"""
표준 문서 청킹 파이프라인 v2 (OOO봇 벡터스토어용)
====================================================
폴더에 문서(hwpx / txt / pdf / docx)를 넣고 실행하면
텍스트 추출 + 공개 OCR(이미지 페이지)을 거쳐 의미 단위 청크(JSONL)를 생성한다.

기본 동작은 **API 없이 완전 로컬**:
  - 텍스트: 한국 문서 구조(제N조/장/절/번호목록/개조식/표) 인식 청킹
  - 이미지 페이지: PaddleOCR 또는 EasyOCR (설치된 것 자동 감지)
Claude API 청킹은 --api 옵션을 줄 때만 사용한다.

python3 chunk_pipeline.py ./raw_docs/bot_a --out-dir ./data/bot_a 이러한 명령으로 실행

사용법:
    python chunk_pipeline.py ./docs                       # 완전 로컬 (권장 기본)
    python chunk_pipeline.py ./docs --ocr easy            # OCR 엔진 지정
    python chunk_pipeline.py ./docs --api                 # Claude API 청킹 (선택)
    python chunk_pipeline.py ./docs --dry-run             # 처리 계획만 확인

출력:
    <out-dir>/<파일명>.chunks.json   파일별 청크
    <out-dir>/all_chunks.jsonl       통합 청크 (build_vectorstore.py 입력)

청크 스키마 (OOO봇 GraphRAG 메타데이터 호환):
    {"chunk_id": "파일명_0001", "content": "...", "topic": "...",
     "keywords": [...], "chunk_type": "일반|표|목록|규정|그림설명",
     "source_file": "...", "pages": "3-4"}

의존성:
    필수:   pip install pymupdf python-docx
    OCR:    pip install paddleocr  (또는)  pip install easyocr
    API용:  pip install anthropic  (--api 사용 시에만)

폐쇄망 참고: OCR 모델은 첫 실행 시 다운로드되므로, 인터넷 PC에서 한 번 실행해
모델 캐시(EasyOCR: ~/.EasyOCR, Paddle: ~/.paddleocr 또는 ~/.paddlex)를 통째로
옮기면 폐쇄망에서도 동작한다.
"""

import os
import re
import sys
import json
import base64
import zipfile
import argparse
import traceback
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET

SUPPORTED_EXT = {".txt", ".pdf", ".docx", ".hwpx"}

DEFAULT_MODEL      = "claude-sonnet-4-6"
DEFAULT_DPI        = 200        # OCR 인식률을 위해 200 기본 (작은 글자는 300)
SPARSE_PAGE_CHARS  = 50         # 페이지 텍스트가 이 미만이면 이미지성 → OCR
CHUNK_TARGET       = 900        # 청크 목표 크기(자)
CHUNK_MAX          = 1600       # 청크 최대 크기 (표는 예외)
CHUNK_MIN          = 120        # 이보다 작으면 병합
PAGES_PER_CALL     = 3          # --api 비전: 호출당 페이지 수
TEXT_BATCH_CHARS   = 6000       # --api 텍스트: 호출당 글자 수
API_MAX_RETRY      = 2

# ═══════════════════════════════════════════════════════════════════════════
# 1. 텍스트 추출기 (파일 형식별)
# ═══════════════════════════════════════════════════════════════════════════

def extract_txt(path: Path) -> str:
    for enc in ("utf-8", "cp949", "euc-kr", "utf-16"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def extract_hwpx(path: Path) -> str:
    """HWPX = zip 안의 XML. Contents/section*.xml에서 문단 순서대로 텍스트 추출."""
    parts = []
    with zipfile.ZipFile(path) as z:
        sections = sorted(n for n in z.namelist()
                          if re.match(r"Contents/section\d+\.xml$", n))
        if not sections:
            raise ValueError("HWPX 구조가 아닙니다 (Contents/section*.xml 없음)")
        for name in sections:
            root = ET.fromstring(z.read(name))
            for para in root.iter():
                if not para.tag.endswith("}p"):
                    continue
                texts = [t.text for t in para.iter()
                         if t.tag.endswith("}t") and t.text]
                line = "".join(texts).strip()
                if line:
                    parts.append(line)
    return "\n".join(parts)


def extract_docx(path: Path) -> str:
    """python-docx. 문단과 표를 문서 순서 그대로 추출 (표는 파이프 구분)."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = Document(str(path))
    parts = []
    for child in doc.element.body:
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = Paragraph(child, doc).text.strip()
            if text:
                parts.append(text)
        elif tag == "tbl":
            table = Table(child, doc)
            for row in table.rows:
                cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                parts.append("| " + " | ".join(cells) + " |")
            parts.append("")
    return "\n".join(parts)


def analyze_pdf(path: Path):
    """페이지별 텍스트 추출 + 이미지성(텍스트 희박) 여부.
    반환: [(page_no, text, is_sparse), ...]"""
    import fitz
    pages = []
    with fitz.open(str(path)) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            pages.append((i, text, len(text) < SPARSE_PAGE_CHARS))
    return pages


def render_pdf_pages(path: Path, page_numbers, dpi: int):
    """지정 페이지들을 PNG 바이트로 렌더링. 반환: [(page_no, png_bytes), ...]"""
    import fitz
    out = []
    with fitz.open(str(path)) as doc:
        for pno in page_numbers:
            pix = doc[pno - 1].get_pixmap(dpi=dpi)
            out.append((pno, pix.tobytes("png")))
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. 공개 OCR (PaddleOCR / EasyOCR) — 이미지성 PDF 페이지용
# ═══════════════════════════════════════════════════════════════════════════

class OcrEngine:
    """PaddleOCR 우선, 없으면 EasyOCR. 버전별 API 차이를 흡수한다."""

    def __init__(self, prefer: str = "auto"):
        self.name = None
        self._paddle = None
        self._easy = None
        order = {"auto": ["paddle", "easy"],
                 "paddle": ["paddle"],
                 "easy": ["easy"]}[prefer]
        for eng in order:
            if eng == "paddle" and self._init_paddle():
                self.name = "paddleocr"; break
            if eng == "easy" and self._init_easy():
                self.name = "easyocr"; break
        if self.name is None:
            raise RuntimeError(
                "사용 가능한 OCR 엔진이 없습니다.\n"
                "  pip install paddleocr   (한글 인식률 우수)\n"
                "  pip install easyocr     (설치 간단)")

    def _init_paddle(self) -> bool:
        try:
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            from paddleocr import PaddleOCR
        except ImportError:
            return False
        # 버전별 인자 차이 흡수 (3.x: use_textline_orientation / 2.x: use_angle_cls)
        for kwargs in ({"lang": "korean", "use_textline_orientation": True},
                       {"lang": "korean", "use_angle_cls": True},
                       {"lang": "korean"}):
            try:
                self._paddle = PaddleOCR(**kwargs)
                return True
            except (TypeError, ValueError):
                continue
            except Exception as e:
                print(f"  PaddleOCR 초기화 실패: {e}")
                return False
        return False

    def _init_easy(self) -> bool:
        try:
            import easyocr
        except ImportError:
            return False
        try:
            print("  EasyOCR 초기화 (첫 실행 시 모델 다운로드)...")
            self._easy = easyocr.Reader(["ko", "en"], gpu=False)
            return True
        except Exception as e:
            print(f"  EasyOCR 초기화 실패: {e}")
            return False

    def image_to_text(self, png_bytes: bytes) -> str:
        import io
        import numpy as np
        from PIL import Image
        img = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        if self.name == "paddleocr":
            return self._run_paddle(img)
        return self._run_easy(img)

    def _run_paddle(self, img) -> str:
        ocr = self._paddle
        # 3.x: predict() → [{'rec_texts': [...], ...}]
        if hasattr(ocr, "predict"):
            try:
                results = ocr.predict(img)
                lines = []
                for r in results:
                    texts = (r.get("rec_texts") if isinstance(r, dict)
                             else getattr(r, "rec_texts", None))
                    if texts:
                        lines += [t for t in texts if t]
                if lines:
                    return "\n".join(lines)
            except Exception:
                pass
        # 2.x: ocr() → [[[box, (text, conf)], ...]]
        try:
            results = ocr.ocr(img)
        except TypeError:
            results = ocr.ocr(img, cls=True)
        lines = []
        for page in (results or []):
            for item in (page or []):
                try:
                    lines.append(item[1][0])
                except (IndexError, TypeError):
                    continue
        return "\n".join(lines)

    def _run_easy(self, img) -> str:
        results = self._easy.readtext(img, detail=0, paragraph=True)
        return "\n".join(results)


# ═══════════════════════════════════════════════════════════════════════════
# 3. 로컬 구조 인식 청킹 (기본 경로 — API 불필요)
# ═══════════════════════════════════════════════════════════════════════════

# 한국 공문서/규정 헤딩 패턴 (레벨 낮을수록 상위 구조)
HEADING_PATTERNS = [
    (1, re.compile(r"^제\s*\d+\s*[편장]\s")),
    (2, re.compile(r"^제\s*\d+\s*[절관]\s")),
    (2, re.compile(r"^제\s*\d+\s*조\s*(\(|（)")),           # 제1조 (목적)
    (2, re.compile(r"^제\s*\d+\s*조\s")),
    (1, re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\s*[\.、]?\s+\S")),   # Ⅰ. Ⅱ.
    (3, re.compile(r"^\d{1,2}\s*\.\s+\S")),                  # 1. 2. (조항 하위 항목)
    (3, re.compile(r"^\d{1,2}\s*\)\s+\S")),                  # 1) 2)
    (3, re.compile(r"^[가나다라마바사아자차카타파하]\s*[\.\)]\s+\S")),
    (3, re.compile(r"^\(\s*[0-9가-힣]{1,3}\s*\)\s+\S")),     # (1) (가)
    (2, re.compile(r"^[□■◎]\s+\S")),
]
BULLET_RE   = re.compile(r"^[○●◦·•\-\*▶▷–]\s+\S")
TABLE_RE    = re.compile(r"^\|.*\|\s*$")
PAGE_TAG_RE = re.compile(r"^\[페이지\s*(\d+)\]$")

KOR_STOPWORDS = {
    "있다", "없다", "한다", "된다", "하는", "되는", "위한", "대한", "관한",
    "경우", "관련", "이상", "이하", "미만", "초과", "또는", "그리고", "및",
    "등의", "따라", "의한", "하여", "에서", "으로", "까지", "부터", "하며",
    "한다는", "있는", "같다", "말한다", "정한다", "본다", "아니", "모든",
    "다음", "아래", "위와", "해당", "각각", "통해", "대해", "이를", "그러나",
}
JOSA_TAIL = ("에서는", "으로는", "에서", "으로", "이라", "에는", "라는",
             "은", "는", "이", "가", "을", "를", "의", "에", "와", "과",
             "도", "만", "로", "며", "다")


def _strip_josa(tok: str) -> str:
    for j in JOSA_TAIL:
        if len(tok) - len(j) >= 2 and tok.endswith(j):
            return tok[:-len(j)]
    return tok


def extract_keywords(text: str, topn: int = 6):
    toks = re.findall(
        r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9\-]{2,}|\d+(?:\.\d+)?(?:MW|kW|GW|%|억|만|톤|t)",
        text)
    counts = Counter()
    for t in toks:
        t = _strip_josa(t)
        if len(t) < 2 or t in KOR_STOPWORDS:
            continue
        counts[t] += 1
    return [w for w, _ in counts.most_common(topn)]


def _heading_level(line: str):
    for lv, pat in HEADING_PATTERNS:
        if pat.match(line):
            return lv
    return None


def _classify_chunk(lines, topic):
    n_table  = sum(1 for l in lines if TABLE_RE.match(l))
    n_bullet = sum(1 for l in lines if BULLET_RE.match(l) or _heading_level(l) == 3)
    n = max(len(lines), 1)
    if n_table >= 2:
        return "표"
    if re.search(r"제\s*\d+\s*조", topic or "") or \
       (lines and re.match(r"^제\s*\d+\s*조", lines[0])):
        return "규정"
    if n >= 3 and n_bullet / n > 0.5:
        return "목록"
    return "일반"


def _find_sentence_cut(text: str, target: int) -> int:
    """target 근처의 문장/줄 경계 위치를 찾는다."""
    window = text[:target + 200]
    for pat in (r"다\.\s", r"[.!?]\s", r"\n"):
        good = [m.end() for m in re.finditer(pat, window)
                if m.end() >= target * 0.5]
        if good:
            return min(good, key=lambda e: abs(e - target))
    return min(target, len(text))


def structure_chunk(text: str):
    """구조(헤딩/표/목록) 인식 로컬 청킹.

    1) 줄 단위 분류 → 블록 구성 (헤딩이 새 블록 시작, 표는 통짜 유지)
    2) 블록을 목표 크기로 패킹. 상위 헤딩(레벨1~2)은 가급적 새 청크 시작.
    3) 초장문 블록은 문장 경계에서 분할, 소형 청크는 병합.
    """
    lines = [l.rstrip() for l in text.splitlines()]

    # ── 1) 블록 구성 ──
    blocks = []
    cur, cur_head, cur_lv, cur_tbl = [], None, None, False
    page = ""

    def flush_block():
        nonlocal cur, cur_head, cur_lv, cur_tbl
        if cur:
            blocks.append({"lines": cur, "heading": cur_head, "level": cur_lv,
                           "is_table": cur_tbl, "page": page})
        cur, cur_head, cur_lv, cur_tbl = [], None, None, False

    for line in lines:
        s = line.strip()
        if not s:
            continue
        m = PAGE_TAG_RE.match(s)
        if m:
            flush_block()
            page = m.group(1)
            continue
        is_tbl = bool(TABLE_RE.match(s))
        lv = _heading_level(s) if not is_tbl else None
        if lv is not None:
            flush_block()
            cur, cur_head, cur_lv, cur_tbl = [s], s, lv, False
        elif cur and is_tbl != cur_tbl:      # 표 <-> 본문 경계
            head, hlv = cur_head, cur_lv
            flush_block()
            cur, cur_head, cur_lv, cur_tbl = [s], head, hlv, is_tbl
        else:
            if not cur:
                cur_tbl = is_tbl
            cur.append(s)

    flush_block()

    # ── 2) 블록 패킹 ──
    chunks = []
    acc, acc_topic, acc_pages = [], None, set()

    def emit():
        nonlocal acc, acc_topic, acc_pages
        body = "\n".join(acc).strip()
        if not body:
            acc, acc_topic, acc_pages = [], None, set()
            return
        topic = (acc_topic or body.splitlines()[0])[:80]
        pg = sorted(int(p) for p in acc_pages if p)
        pages = ("" if not pg else str(pg[0]) if len(pg) == 1
                 else f"{pg[0]}-{pg[-1]}")
        chunks.append({
            "topic": topic,
            "chunk_type": _classify_chunk(body.splitlines(), topic),
            "keywords": extract_keywords(body),
            "content": body,
            "pages": pages,
        })
        acc, acc_topic, acc_pages = [], None, set()

    for b in blocks:
        btext = "\n".join(b["lines"])
        blen = len(btext)
        cur_len = sum(len(a) + 1 for a in acc)

        # 상위 헤딩(레벨 1~2)은 새 청크 시작 (현재 내용이 어느 정도 있으면)
        if b["level"] in (1, 2) and cur_len >= CHUNK_MIN:
            emit()
            cur_len = 0

        # 표는 통짜 유지 — 크면 단독 청크로
        if b["is_table"] and blen > CHUNK_TARGET:
            emit()
            acc = [btext]
            acc_topic = b["heading"] or "표"
            acc_pages = {b["page"]}
            emit()
            continue

        if cur_len and cur_len + blen > CHUNK_MAX and not b["is_table"]:
            emit()

        if not acc:
            acc_topic = b["heading"]
        acc.append(btext)
        acc_pages.add(b["page"])

        # 초장문 누적 → 문장 경계에서 분할
        while sum(len(a) + 1 for a in acc) > CHUNK_MAX and not b["is_table"]:
            joined = "\n".join(acc)
            cut = _find_sentence_cut(joined, CHUNK_TARGET)
            if cut <= 0 or cut >= len(joined):
                break
            head_part, tail = joined[:cut].strip(), joined[cut:].strip()
            keep_topic, keep_pages = acc_topic, set(acc_pages)
            acc = [head_part]
            emit()
            acc = [tail] if tail else []
            acc_topic, acc_pages = keep_topic, keep_pages

    emit()

    # ── 3) 소형 청크 병합 ──
    merged = []
    for c in chunks:
        if merged and len(c["content"]) < CHUNK_MIN and \
           merged[-1]["chunk_type"] == c["chunk_type"]:
            merged[-1]["content"] += "\n" + c["content"]
            merged[-1]["keywords"] = extract_keywords(merged[-1]["content"])
        else:
            merged.append(c)
    return merged


# ═══════════════════════════════════════════════════════════════════════════
# 4. Claude API 청킹 (--api 옵션일 때만)
# ═══════════════════════════════════════════════════════════════════════════

CHUNK_SCHEMA_GUIDE = """각 청크는 아래 JSON 필드를 갖는다:
- "topic": 청크의 핵심 주제 (한 문장, 검색 키가 되도록 구체적으로)
- "chunk_type": "일반" | "표" | "목록" | "그림설명" | "규정" 중 하나
- "keywords": 검색용 핵심 키워드 3~8개 배열
- "content": 청크 본문

청킹 규칙:
1. 의미 단위로 자른다. 하나의 주제/조항/표/절차가 한 청크.
2. 본문은 원문 표현을 최대한 보존한다 (요약·창작 금지). 표와 그림은
   내용을 빠짐없이 구조화된 문장이나 파이프 표로 옮긴다.
3. 판독이 불확실한 글자·숫자는 절대 지어내지 말고 [판독불가]로 표기한다.
4. 청크당 300~1500자 권장. 표는 잘리지 않게 통째로 유지.
5. 반드시 {"chunks": [...]} 형태의 JSON만 출력한다. 설명·마크다운 금지."""

TEXT_CHUNK_PROMPT = """다음은 사내 문서에서 추출한 텍스트다. 검색(RAG)용으로 청킹하라.

""" + CHUNK_SCHEMA_GUIDE + """

{context_block}<문서 텍스트>
{text}
</문서 텍스트>"""

VISION_CHUNK_PROMPT = """첨부한 이미지는 사내 문서의 페이지들이다. 모든 텍스트, 표, 그림, 차트를
인식하여 검색(RAG)용으로 청킹하라.

""" + CHUNK_SCHEMA_GUIDE + """

추가 규칙:
- 표는 행/열 관계가 보존되도록 파이프(|) 표 또는 "항목: 값" 문장으로 옮긴다.
- 그림·차트는 chunk_type "그림설명"으로 내용과 수치를 서술한다.
- 장식 이미지, 페이지 번호, 머리말/꼬리말은 무시한다."""


def get_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("anthropic 패키지가 없습니다:  pip install anthropic")
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("환경변수 ANTHROPIC_API_KEY가 없습니다. (cmd) set ANTHROPIC_API_KEY=...")
    return anthropic.Anthropic(api_key=key)


def _parse_chunks_json(raw: str):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)  # Qwen3 think 제거
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.DOTALL).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("JSON 객체를 찾을 수 없음")
    data = json.loads(raw[start:end + 1])
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError('"chunks" 배열이 없음')
    out = []
    for c in chunks:
        content = str(c.get("content", "")).strip()
        if not content:
            continue
        out.append({
            "topic":      str(c.get("topic", "")).strip(),
            "chunk_type": str(c.get("chunk_type", "일반")).strip() or "일반",
            "keywords":   [str(k).strip() for k in (c.get("keywords") or []) if str(k).strip()],
            "content":    content,
            "pages":      "",
        })
    return out


def _call_api(client, model, content_blocks, max_tokens=8000):
    last_err = None
    for attempt in range(1, API_MAX_RETRY + 2):
        try:
            resp = client.messages.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": content_blocks}])
            raw = "".join(b.text for b in resp.content if b.type == "text")
            return _parse_chunks_json(raw)
        except Exception as e:
            last_err = e
            print(f"    API 시도 {attempt} 실패: {e}")
    raise RuntimeError(f"API 청킹 실패: {last_err}")


def api_chunk_text(client, model, text):
    paragraphs = [p for p in re.split(r"\n{2,}|\n(?=\S)", text) if p.strip()]
    batches, cur, cur_len = [], [], 0
    for p in paragraphs:
        if cur and cur_len + len(p) > TEXT_BATCH_CHARS:
            batches.append("\n".join(cur)); cur, cur_len = [], 0
        cur.append(p); cur_len += len(p)
    if cur:
        batches.append("\n".join(cur))

    all_chunks, prev_tail = [], ""
    for i, batch in enumerate(batches, start=1):
        print(f"    API 텍스트 배치 {i}/{len(batches)} ({len(batch)}자)")
        ctx = (f"<이전 문맥 (참고만, 청킹하지 말 것)>\n{prev_tail}\n</이전 문맥>\n\n"
               if prev_tail else "")
        prompt = (TEXT_CHUNK_PROMPT.replace("{context_block}", ctx)
                  .replace("{text}", batch))
        all_chunks += _call_api(client, model, [{"type": "text", "text": prompt}])
        prev_tail = batch[-300:]
    return all_chunks


def api_chunk_images(client, model, page_images, dpi):
    results = []
    for i in range(0, len(page_images), PAGES_PER_CALL):
        group = page_images[i:i + PAGES_PER_CALL]
        pnos = [p for p, _ in group]
        pages_str = f"{pnos[0]}" if len(pnos) == 1 else f"{pnos[0]}-{pnos[-1]}"
        print(f"    API 비전 배치 p.{pages_str} ({len(group)}페이지, DPI {dpi})")
        blocks = []
        for pno, png in group:
            blocks.append({"type": "text", "text": f"[페이지 {pno}]"})
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(png).decode()}})
        blocks.append({"type": "text", "text": VISION_CHUNK_PROMPT})
        for c in _call_api(client, model, blocks):
            c["pages"] = pages_str
            results.append(c)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 4-2. 내부망 LLM 청킹 (vLLM 등 OpenAI 호환 API — --llm-base 옵션)
#      예: DGX Spark의 vllm-qwen 컨테이너, 감마봇 Qwen3-VL-8B
# ═══════════════════════════════════════════════════════════════════════════

def _llm_url(base: str) -> str:
    """base가 완전한 completions 엔드포인트면 그대로, 아니면 표준 경로 부착.
    예) http://llm-gateway.example.com/v1/completions  → 그대로 사용
        http://192.168.x.x:8000/v1                             → /chat/completions 부착"""
    b = base.rstrip("/")
    return b if b.endswith("completions") else b + "/chat/completions"


def _extract_content(data) -> str:
    """게이트웨이별 응답 포맷 차이를 흡수해 본문 텍스트를 꺼낸다."""
    try:
        return data["choices"][0]["message"]["content"]     # OpenAI chat 표준
    except (KeyError, IndexError, TypeError):
        pass
    try:
        return data["choices"][0]["text"]                    # completions 구형
    except (KeyError, IndexError, TypeError):
        pass
    for k in ("content", "response", "answer", "output", "result"):
        v = data.get(k) if isinstance(data, dict) else None
        if isinstance(v, str) and v.strip():
            return v
    raise ValueError(f"알 수 없는 응답 포맷 (keys={list(data)[:8] if isinstance(data, dict) else type(data)})")


def _oai_raw_call(base: str, model: str, messages, api_key: str = "",
                  user: str = "chunk_pipeline", xkey: str = "",
                  max_tokens: int = 8000, timeout: int = 600) -> str:
    """내부망 LLM 게이트웨이 호출 → 응답 본문 텍스트 반환."""
    try:
        import requests
    except ImportError:
        sys.exit("requests 패키지가 없습니다:  pip install requests")
    headers = {"Content-Type": "application/json", "X-API-KEY": xkey}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": messages, "user": user,
               "max_tokens": max_tokens, "temperature": 0.2, "stream": False}
    r = requests.post(_llm_url(base), json=payload, headers=headers,
                      timeout=timeout)
    r.raise_for_status()
    return _extract_content(r.json())


def _oai_call(base: str, model: str, messages, api_key: str = "",
              user: str = "chunk_pipeline", xkey: str = "",
              max_tokens: int = 8000):
    """호출 + 청크 JSON 파싱 (재시도 포함)."""
    last_err = None
    for attempt in range(1, API_MAX_RETRY + 2):
        try:
            raw = _oai_raw_call(base, model, messages, api_key=api_key,
                                user=user, xkey=xkey, max_tokens=max_tokens)
            return _parse_chunks_json(raw)
        except Exception as e:
            last_err = e
            print(f"    내부LLM 시도 {attempt} 실패: {e}")
    raise RuntimeError(f"내부LLM 청킹 실패: {last_err}")


def probe_llm(args):
    """--probe: 배치 돌리기 전에 게이트웨이 텍스트/비전 응답 형식을 확인한다."""
    print(f"프로브 대상: {_llm_url(args.llm_base)}  (model={args.llm_model})")
    # 1) 텍스트
    try:
        out = _oai_raw_call(args.llm_base, args.llm_model,
                            [{"role": "user", "content": "안녕하세요. 한 문장으로 답하세요."}],
                            api_key=args.llm_key, user=args.llm_user,
                            xkey=args.llm_xkey, max_tokens=200, timeout=120)
        print(f"[텍스트] 성공: {out[:120]!r}")
    except Exception as e:
        print(f"[텍스트] 실패: {e}")
        return
    # 2) 비전 (OpenAI image_url 규격 수용 여부)
    try:
        from PIL import Image, ImageDraw
        import io
        img = Image.new("RGB", (240, 80), "white")
        ImageDraw.Draw(img).text((10, 30), "TEST 12345", fill="black")
        buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        out = _oai_raw_call(args.llm_base, args.llm_model,
            [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + b64}},
                {"type": "text", "text": "이미지에 적힌 글자를 그대로 답하세요."}]}],
            api_key=args.llm_key, user=args.llm_user, xkey=args.llm_xkey,
            max_tokens=200, timeout=180)
        ok = "12345" in out
        print(f"[비전]   응답: {out[:120]!r}")
        print(f"[비전]   {'성공 — --llm-vision 사용 가능 ✅' if ok else '응답은 왔지만 인식 불확실 — 원문 확인 필요'}")
    except Exception as e:
        print(f"[비전]   실패: {e}")
        print("        → 게이트웨이가 image_url 규격을 안 받는 것일 수 있음. "
              "--llm-vision 대신 OCR 경로를 쓰세요.")


def llm_chunk_text(base: str, model: str, text: str, api_key: str = "",
                   user: str = "chunk_pipeline", xkey: str = ""):
    """내부 LLM 텍스트 청킹 (배치 분할은 Claude 경로와 동일)."""
    paragraphs = [p for p in re.split(r"\n{2,}|\n(?=\S)", text) if p.strip()]
    batches, cur, cur_len = [], [], 0
    for p in paragraphs:
        if cur and cur_len + len(p) > TEXT_BATCH_CHARS:
            batches.append("\n".join(cur)); cur, cur_len = [], 0
        cur.append(p); cur_len += len(p)
    if cur:
        batches.append("\n".join(cur))

    all_chunks, prev_tail = [], ""
    for i, batch in enumerate(batches, start=1):
        print(f"    내부LLM 텍스트 배치 {i}/{len(batches)} ({len(batch)}자)")
        ctx = (f"<이전 문맥 (참고만, 청킹하지 말 것)>\n{prev_tail}\n</이전 문맥>\n\n"
               if prev_tail else "")
        prompt = "/no_think\n" + (TEXT_CHUNK_PROMPT
            .replace("{context_block}", ctx).replace("{text}", batch))
        all_chunks += _oai_call(base, model,
                                [{"role": "user", "content": prompt}],
                                api_key=api_key, user=user, xkey=xkey)
        prev_tail = batch[-300:]
    return all_chunks


def llm_chunk_images(base: str, model: str, page_images, api_key: str = "",
                     user: str = "chunk_pipeline", xkey: str = ""):
    """내부 VLM(Qwen3-VL 등) 비전 청킹. OpenAI image_url(base64 data URI) 규격."""
    results = []
    for i in range(0, len(page_images), PAGES_PER_CALL):
        group = page_images[i:i + PAGES_PER_CALL]
        pnos = [p for p, _ in group]
        pages_str = f"{pnos[0]}" if len(pnos) == 1 else f"{pnos[0]}-{pnos[-1]}"
        print(f"    내부VLM 비전 배치 p.{pages_str} ({len(group)}페이지)")
        content = []
        for pno, png in group:
            content.append({"type": "text", "text": f"[페이지 {pno}]"})
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(png).decode()}})
        content.append({"type": "text",
                        "text": "/no_think\n" + VISION_CHUNK_PROMPT})
        for c in _oai_call(base, model,
                           [{"role": "user", "content": content}],
                           api_key=api_key, user=user, xkey=xkey):
            c["pages"] = pages_str
            results.append(c)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 5. 파일 단위 처리
# ═══════════════════════════════════════════════════════════════════════════

def _chunk_text_backend(text, args, client):
    """텍스트 청킹 백엔드 선택: Claude / 내부LLM / 로컬 구조 인식"""
    if args.api:
        return api_chunk_text(client, args.model, text)
    if args.llm_base:
        return llm_chunk_text(args.llm_base, args.llm_model, text,
                              api_key=args.llm_key, user=args.llm_user,
                              xkey=args.llm_xkey)
    return structure_chunk(text)


def process_file(path: Path, args, client, ocr):
    ext = path.suffix.lower()
    chunks = []

    if ext in (".txt", ".hwpx", ".docx"):
        text = {"txt": extract_txt, "hwpx": extract_hwpx,
                "docx": extract_docx}[ext[1:]](path)
        if not text.strip():
            print("    추출된 텍스트가 없습니다 → 건너뜀")
            return []
        print(f"    텍스트 추출: {len(text)}자")
        chunks = _chunk_text_backend(text, args, client)

    elif ext == ".pdf":
        pages = analyze_pdf(path)
        sparse = [p for p, _, s in pages if s]
        rich   = [(p, t) for p, t, s in pages if not s]
        if args.mode == "text":
            sparse, rich = [], [(p, t) for p, t, _ in pages]
        elif args.mode == "vision":
            sparse, rich = [p for p, _, _ in pages], []
        print(f"    PDF {len(pages)}페이지 (텍스트 {len(rich)} / 이미지성 {len(sparse)})")

        if rich:
            text = "\n\n".join(f"[페이지 {p}]\n{t}" for p, t in rich if t)
            chunks += _chunk_text_backend(text, args, client)

        if sparse:
            if args.api:   # Claude Vision이 직접 인식+청킹
                images = render_pdf_pages(path, sparse, args.dpi)
                chunks += api_chunk_images(client, args.model, images, args.dpi)
            elif args.llm_base and args.llm_vision:   # 내부 VLM (Qwen3-VL 등)
                images = render_pdf_pages(path, sparse, args.dpi)
                chunks += llm_chunk_images(args.llm_base, args.llm_model,
                                           images, api_key=args.llm_key,
                                           user=args.llm_user, xkey=args.llm_xkey)
            elif ocr is not None:   # 공개 OCR → 구조 청킹
                images = render_pdf_pages(path, sparse, args.dpi)
                ocr_parts = []
                for pno, png in images:
                    print(f"    OCR({ocr.name}) p.{pno}")
                    t = ocr.image_to_text(png).strip()
                    if t:
                        ocr_parts.append(f"[페이지 {pno}]\n{t}")
                if ocr_parts:
                    chunks += structure_chunk("\n\n".join(ocr_parts))
            else:
                print(f"    OCR 엔진 없음: 이미지성 페이지 {sparse} 건너뜀")
    else:
        print(f"    미지원 확장자: {ext}")
        return []

    # 메타 부여
    stem = re.sub(r"[^\w가-힣]+", "_", path.stem)
    final = []
    for i, c in enumerate(chunks, start=1):
        final.append({
            "chunk_id":    f"{stem}_{i:04d}",
            "content":     c["content"],
            "topic":       c.get("topic", ""),
            "keywords":    c.get("keywords", []),
            "chunk_type":  c.get("chunk_type", "일반"),
            "source_file": path.name,
            "pages":       c.get("pages", ""),
        })
    return final


# ═══════════════════════════════════════════════════════════════════════════
# 6. 메인 (폴더 순회 + CLI)
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="OOO봇 표준 문서 청킹 파이프라인 v2 (기본: 완전 로컬)")
    ap.add_argument("input_dir", help="문서가 들어있는 폴더")
    ap.add_argument("--out-dir", default="./chunks")
    ap.add_argument("--mode", choices=["auto", "text", "vision"], default="auto",
                    help="PDF 처리 (auto: 페이지별 자동 분기)")
    ap.add_argument("--ocr", choices=["auto", "paddle", "easy", "off"],
                    default="auto", help="이미지 페이지 OCR 엔진")
    ap.add_argument("--api", action="store_true",
                    help="Claude API 청킹 사용 (기본은 로컬 구조 청킹)")
    ap.add_argument("--llm-base", default="",
                    help="내부망 LLM 엔드포인트. 완전한 URL이면 그대로 사용 "
                         "(예: http://llm-gateway.example.com/v1/completions), "
                         "base만 주면 /chat/completions 부착 (예: http://IP:8000/v1)")
    ap.add_argument("--llm-model", default="Normal-Mode",
                    help="모델명 (게이트웨이: Normal-Mode / vLLM: served-model-name)")
    ap.add_argument("--llm-key",
                    default=os.getenv("LLM_API_KEY", ""),
                    help="Bearer 키 (환경변수 LLM_API_KEY)")
    ap.add_argument("--llm-user", default="chunk_pipeline",
                    help="게이트웨이 user 필드")
    ap.add_argument("--llm-xkey", default="", help="X-API-KEY 헤더 값 (필요 시)")
    ap.add_argument("--probe", action="store_true",
                    help="배치 실행 없이 내부LLM 텍스트/비전 응답 형식만 점검")
    ap.add_argument("--llm-vision", action="store_true",
                    help="이미지성 PDF 페이지를 내부 VLM(Qwen3-VL 등)으로 인식")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        sys.exit(f"폴더가 없습니다: {in_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in in_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    if not files:
        sys.exit(f"{in_dir} 안에 지원 문서(hwpx/txt/pdf/docx)가 없습니다.")

    if args.probe:
        if not args.llm_base:
            sys.exit("--probe는 --llm-base와 함께 사용하세요")
        probe_llm(args)
        return

    backend = ("Claude API (" + args.model + ")" if args.api
               else f"내부LLM ({args.llm_model} @ {args.llm_base})" if args.llm_base
               else "로컬 구조 인식")
    print("=== 청킹 파이프라인 v2 ===")
    print(f"입력: {in_dir.resolve()}  ({len(files)}개 파일)")
    print(f"출력: {out_dir.resolve()}")
    print(f"청킹: {backend} | PDF={args.mode} | OCR={args.ocr} | DPI={args.dpi}"
          f"{' | VLM비전' if args.llm_vision else ''}\n")

    if args.dry_run:
        for f in files:
            done = (out_dir / f"{f.stem}.chunks.json").exists()
            print(f"  {'[완료됨]' if done and not args.force else '[대기]  '} {f.name}")
        return

    client = get_client() if args.api else None

    # OCR은 필요할 때만 초기화 (이미지 페이지를 VLM/Claude가 처리하면 불필요)
    ocr = None
    need_ocr = (not args.api and not (args.llm_base and args.llm_vision)
                and args.ocr != "off"
                and any(f.suffix.lower() == ".pdf" for f in files))
    if need_ocr:
        try:
            ocr = OcrEngine(prefer=args.ocr)
            print(f"OCR 엔진: {ocr.name}\n")
        except RuntimeError as e:
            print(f"경고: {e}\n(텍스트 레이어가 있는 페이지만 처리됩니다)\n")

    all_chunks, failed = [], []
    for n, f in enumerate(files, start=1):
        per_file = out_dir / f"{f.stem}.chunks.json"
        print(f"[{n}/{len(files)}] {f.name}")
        if per_file.exists() and not args.force:
            chunks = json.loads(per_file.read_text(encoding="utf-8"))
            print(f"    이미 처리됨 ({len(chunks)}청크) → 스킵 (--force로 재처리)")
            all_chunks += chunks
            continue
        try:
            chunks = process_file(f, args, client, ocr)
            per_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            print(f"    완료: {len(chunks)}청크")
            all_chunks += chunks
        except Exception as e:
            print(f"    실패: {e}")
            traceback.print_exc()
            failed.append(f.name)

    merged = out_dir / "all_chunks.jsonl"
    with open(merged, "w", encoding="utf-8") as fp:
        for c in all_chunks:
            fp.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n=== 완료 ===")
    print(f"총 {len(all_chunks)}청크 → {merged}")
    if failed:
        print(f"실패 파일 {len(failed)}개: {', '.join(failed)}")


if __name__ == "__main__":
    main()
