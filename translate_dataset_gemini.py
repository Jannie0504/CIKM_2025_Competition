import re
import json
import time
from pathlib import Path
from tqdm import tqdm

from google import genai
from google.genai import types

# ─── 0. API 키 설정 ─────────────────────────────────────────────────────
genai_client = genai.Client(api_key="")  # <- 여기에 키 넣기

# ─── (옵션) 배치 크기 설정 ───────────────────────────────────────────────
QC_BATCH_SIZE = 50
QI_BATCH_SIZE = 30

# ─── 1. 언어 코드 → 언어명 매핑 ──────────────────────────────────────────
LANG_NAME = {
    "en": "English",   "fr": "French",    "es": "Spanish",
    "ko": "Korean",    "pt": "Portuguese","ja": "Japanese",
    "de": "German",    "it": "Italian",   "pl": "Polish",
    "ar": "Arabic",    "vn": "Vietnamese","id": "Indonesian",
    "th": "Thai"
}

# ─── 안전 추출 함수 ──────────────────────────────────────────────────────
def safe_text_from_resp(resp):
    t = getattr(resp, "text", None)
    if t:
        return t
    pieces = []
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            if hasattr(part, "text"):
                pieces.append(part.text)
    return "\n".join(pieces)

# ─── 2. 단건 번역 ───────────────────────────────────────────────────────
def gemini_translate_query(query_text: str, src_code: str) -> str:
    src_lang = LANG_NAME.get(src_code, "the source language")
    prompt = f'''Please translate the following {src_lang} text into English as accurately as possible.

"{query_text}"

Return ONLY the translation text (no extra labels).'''

    resp = genai_client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[prompt],
        config=types.GenerateContentConfig(
            max_output_tokens=128,
            temperature=0.0,
            response_mime_type="text/plain"
        )
    )
    raw = safe_text_from_resp(resp)
    return (raw or "").strip().strip('"')

# ─── 3. 배치 번역 (필드 구분 지원) ───────────────────────────────────────
def gemini_translate_batch(entries):
    """
    entries: [{"id":..., "field": "origin_query"|"item_title", "text":...}, ...]
    return: dict[(str(id), field)] = translation
    """
    prompt_text = (
        "Translate each 'text' to English as accurately as possible.\n"
        'Return ONLY a JSON array. Each element must be: '
        '{"id": <id>, "field": <field name>, "translation": <english>}.\n'
        "No comments, no extra keys.\n\n"
        f"{json.dumps(entries, ensure_ascii=False)}"
    )

    resp = genai_client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[prompt_text],
        config=types.GenerateContentConfig(
            max_output_tokens=8192,
            temperature=0.0,
            response_mime_type="application/json"
        )
    )

    raw = safe_text_from_resp(resp)
    # JSON 파싱(백업 포함)
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r'(\[\s*{.*?}\s*\])', raw, flags=re.S)
        if not m:
            raise ValueError("JSON parse failed:\n" + str(raw))
        data = json.loads(m.group(1))

    out = {}
    for d in data:
        pid = str(d.get("id"))
        fld = d.get("field")
        trs = d.get("translation")
        if pid is not None and fld and trs is not None:
            out[(pid, fld)] = trs
    return out

# ─── 4. 재시작 지원: 이미 처리된 id ─────────────────────────────────────
def load_done_ids(out_path: Path):
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(obj["id"])
            except Exception:
                pass
    return done

data_dir  = Path("./data")
file_list = ["test_QI.txt"]   # 필요시 확장: "train_QC.txt", "train_QI.txt", "dev_QC.txt", "dev_QI.txt"

# ─── 5. flush 함수: QC/QI 규칙에 따라 필드 분기 ─────────────────────────
def flush_batch(buffer, fout, retries=2, wait=2.0):
    """
    buffer: [{"item": original_json, "id":..., "lang":..., "text":...}, ...]
      - QC: origin_query만 번역
      - QI: origin_query + item_title 번역
    """
    if not buffer:
        return

    # (id, field, text)로 평탄화
    entries = []
    for x in buffer:
        it = x["item"]
        pid = x["id"]
        lang = x["lang"]

        # 항상 origin_query
        entries.append({
            "id": pid,
            "field": "origin_query",
            "text": it["origin_query"]
        })

        # QI면 item_title도 추가
        if it.get("task") == "QI" and "item_title" in it and it["item_title"] is not None:
            entries.append({
                "id": pid,
                "field": "item_title",
                "text": it["item_title"]
            })

    # 배치 시도
    for attempt in range(retries + 1):
        try:
            result = gemini_translate_batch(entries)  # dict[(id, field)] = translation
            break
        except Exception as e:
            print(f"[Gemini batch error] {e} (try {attempt+1}/{retries})")
            if attempt == retries:
                # 마지막 실패: 단건 폴백
                for x in buffer:
                    it = x["item"]
                    try:
                        it["origin_query"] = gemini_translate_query(it["origin_query"], x["lang"])
                    except Exception:
                        pass
                    if it.get("task") == "QI" and "item_title" in it and it["item_title"] is not None:
                        try:
                            it["item_title"] = gemini_translate_query(it["item_title"], x["lang"])
                        except Exception:
                            pass
                    fout.write(json.dumps(it, ensure_ascii=False) + "\n")
                return
            time.sleep(wait * (2 ** attempt))

    # 성공시 결과 반영
    for x in buffer:
        it  = x["item"]
        pid = str(x["id"])

        it["origin_query"] = result.get((pid, "origin_query"), it["origin_query"])
        if it.get("task") == "QI" and "item_title" in it and it["item_title"] is not None:
            it["item_title"] = result.get((pid, "item_title"), it["item_title"])

        fout.write(json.dumps(it, ensure_ascii=False) + "\n")

# ─── 6. 메인 루프 ────────────────────────────────────────────────────────
for fname in file_list:
    inp  = data_dir / fname
    outp = data_dir / f"{inp.stem}_en{inp.suffix}"

    done_ids = load_done_ids(outp)
    print(f"▶ Resuming {inp.name} → {outp.name} (already {len(done_ids)} lines done)")

    with inp.open("r", encoding="utf-8") as fin, outp.open("a", encoding="utf-8") as fout:
        buffer = []
        for line in tqdm(fin, desc=fname, leave=False):
            item = json.loads(line)
            if item["id"] in done_ids:
                continue  # 이미 처리됨

            code = item.get("language", "en")

            # NOTE: 기존 로직 유지 — 영어가 아니면 그대로 패스(원 코드 유지 요청에 따라)
            if code == "en":
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            # QC/QI 공통: origin_query는 항상 필요. 버퍼에는 원본 item을 그대로 보관
            buffer.append({
                "item": item,
                "id": item["id"],
                "lang": code,
                "text": item["origin_query"]
            })

            if (item["task"] == "QC"):
                if len(buffer) >= QC_BATCH_SIZE:
                    flush_batch(buffer, fout)
                    buffer = []
            elif (item["task"] == "QI"):
                if len(buffer) >= QI_BATCH_SIZE:
                    flush_batch(buffer, fout)
                    buffer = []

        # 남은 것 처리
        flush_batch(buffer, fout)

    print(f"✔ Completed {outp.name}\n")

print("All files translated with Gemini API (QC/QI aware).")
