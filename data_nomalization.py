import argparse, json, re, sys, unicodedata
from pathlib import Path

# -------------------------------
# 1) 공백/기호 정규화 + 소문자화
# -------------------------------
SYM_MAP = {
    "’": "'", "‘": "'", "“": '"', "”": '"',  # quotation
    "–": "-", "—": "-", "−": "-",            # dashes
    "\u00a0": " ", "\u200b": "", "\u200c": "", "\u200d": "", "\u2060": "", "\ufeff": "",
}

WS_RE = re.compile(r"\s+")

def minimal_ascii(s: str) -> str:
    """ 기호 치환 → NFKC 정규화 """
    if not isinstance(s, str):
        return s
    t = s
    for k, v in SYM_MAP.items():
        t = t.replace(k, v)
    # 숫자/문자/기호의 호환성 정규화
    t = unicodedata.normalize("NFKC", t)
    return t

# ------------------------------------
# 2) 숫자·단위 정규화 (공백 제거 규칙)
# ------------------------------------
UNIT_PATTERNS = [
    # inch 류 → "<num>in"
    (re.compile(r'\b(\d+(?:\.\d+)?)\s*(?:inches|inch|in|”|")\b', flags=re.I), lambda m: f"{m.group(1)}in"),
    # 길이 단위 → "<num><unit>"
    (re.compile(r'\b(\d+(?:\.\d+)?)\s*(mm|cm|m)\b', flags=re.I),            lambda m: f"{m.group(1)}{m.group(2).lower()}"),
    # 저장 단위 → "<num><unit>"
    (re.compile(r'\b(\d+(?:\.\d+)?)\s*(gb|mb|kb|tb)\b', flags=re.I),        lambda m: f"{m.group(1)}{m.group(2).lower()}"),
    # 배터리 → "<num>mah"
    (re.compile(r'\b(\d+(?:\.\d+)?)\s*(mah)\b', flags=re.I),                lambda m: f"{m.group(1)}mah"),
    # 기타 단위/지표 → "<num><unit>"
    (re.compile(r'\b(\d+(?:\.\d+)?)\s*(w|v|a|hz|fps|mp)\b', flags=re.I),    lambda m: f"{m.group(1)}{m.group(2).lower()}"),
]

def normalize_units_numbers(text: str) -> str:
    """ 기호/유니코드 정리 → 소문자화 → 단위 결합 → 공백 정리 """
    if not isinstance(text, str) or not text:
        return text
    s = minimal_ascii(text).lower()
    for pat, repl in UNIT_PATTERNS:
        s = pat.sub(repl, s)
    s = WS_RE.sub(" ", s).strip()
    return s

# ------------------------------------
# 3) 모델명 정규화 (표기 통합)
#    - 아래 규칙은 이전 처리에 썼던 것과 동일합니다.
# ------------------------------------
MODEL_RULES = [
    # --- Apple iPhone ---
    (re.compile(r'\biphone\s+(\d{1,2})\s+(pro\s*max)\b', re.I),   r'iphone\1 \2'),
    (re.compile(r'\biphone\s+(\d{1,2})\s+(pro|max|plus|mini)\b', re.I), r'iphone\1 \2'),
    (re.compile(r'\biphone\s+(\d{1,2})\b', re.I),                 r'iphone\1'),
    # --- Apple iPad / Air / mini / Pro ---
    (re.compile(r'\bipad\s+pro\b', re.I),                         r'ipadpro'),
    (re.compile(r'\bipad\s+(mini|air)\b', re.I),                  r'ipad\1'),
    (re.compile(r'\bipad\s+(\d+(?:st|nd|rd|th))\b', re.I),        r'ipad\1'),
    # --- Samsung Galaxy phones ---
    (re.compile(r'\bgalaxy\s+s\s*([0-9]{1,2}[a-z]?)\b', re.I),    r'galaxys\1'),
    (re.compile(r'\bgalaxy\s+a\s*([0-9]{1,2}[a-z]?)\b', re.I),    r'galaxya\1'),
    (re.compile(r'\bgalaxy\s+z\s+(fold|flip)\s*([0-9]+)?\b', re.I), r'galaxyz\1\2'),
    # --- Samsung Tab / Watch ---
    (re.compile(r'\bgalaxy\s*tab\s*([a-z]?[0-9]+)\b', re.I),      r'galaxytab\1'),
    (re.compile(r'\bgalaxy\s*watch\s*([0-9]+)\b', re.I),          r'galaxywatch\1'),
    # --- Google Pixel ---
    (re.compile(r'\bpixel\s*([0-9]+)\s*(pro|xl|a)?\b', re.I),     r'pixel\1\2'),
    # --- OnePlus ---
    (re.compile(r'\bone\s*plus\s*([0-9]+(?:t)?)\b', re.I),        r'oneplus\1'),
    (re.compile(r'\boneplus\s*([0-9]+(?:t)?)\b', re.I),           r'oneplus\1'),
    # --- Xiaomi / Mi / Redmi / POCO ---
    (re.compile(r'\bmi\s*([0-9]+)\s*(lite|pro|ultra)?\b', re.I),  r'mi\1\2'),
    (re.compile(r'\bredmi\s+note\s*([0-9]+[a-z]?)\s*(pro\+?|plus|pro|turbo|prime)?\b', re.I), r'redminote\1\2'),
    (re.compile(r'\bpoco\s*([a-z]?\d{1,3}[a-z+]?)\b', re.I),      r'poco\1'),
    # --- Huawei / Honor / OPPO / Vivo / Realme / Nothing ---
    (re.compile(r'\bmate\s*([0-9]+)\s*(pro|x|rs)?\b', re.I),      r'mate\1\2'),
    (re.compile(r'\bp\s*([0-9]+)\s*(pro|lite)?\b', re.I),         r'p\1\2'),  # Huawei P 시리즈
    (re.compile(r'\bhonor\s*([a-z]?\d{1,3})\s*(pro|plus|x)?\b', re.I), r'honor\1\2'),
    (re.compile(r'\boppo\s*reno\s*([0-9]+)\s*(pro|plus)?\b', re.I), r'opporeno\1\2'),
    (re.compile(r'\bvivo\s*x\s*([0-9]+)\s*(pro|plus)?\b', re.I),  r'vivox\1\2'),
    (re.compile(r'\brealme\s*([0-9]+[a-z]?)\s*(pro|plus)?\b', re.I), r'realme\1\2'),
    (re.compile(r'\bnothing\s*phone\s*\(?([0-9]+)\)?\b', re.I),   r'nothingphone\1'),
    # --- Consoles ---
    (re.compile(r'\bps\s*([345])\b', re.I),                       r'ps\1'),
    (re.compile(r'\bplaystation\s*([345])\b', re.I),              r'ps\1'),
    (re.compile(r'\bxbox\s+one\s+s\b', re.I),                     r'xboxones'),
    (re.compile(r'\bxbox\s+one\s+x\b', re.I),                     r'xboxonex'),
    (re.compile(r'\bxbox\s+series\s+x\b', re.I),                  r'xboxseriesx'),
    (re.compile(r'\bxbox\s+series\s+s\b', re.I),                  r'xboxseriess'),
    # --- Laptops / Lines ---
    (re.compile(r'\bthinkpad\s*t\s*([0-9]+[s]?)\b', re.I),        r'thinkpadt\1'),
    (re.compile(r'\bideapad\s*([0-9]+)\b', re.I),                 r'ideapad\1'),
    (re.compile(r'\bsurface\s*pro\s*([0-9]+)\b', re.I),           r'surfacepro\1'),
    # --- GPUs / CPUs ---
    (re.compile(r'\brtx\s*([34][0-9]{2}ti?)\b', re.I),            r'rtx\1'),
    (re.compile(r'\bradeon\s*rx\s*([0-9]{3,4}[a-z]?)\b', re.I),   r'rx\1'),
    (re.compile(r'\b(core\s*i[3579])[-\s]?([0-9]{4,5}[a-z]{0,2})\b', re.I), r'\1\2'),  # i7-12700H → core i7 12700h → corei7 12700h (후속 공백 정리)
    (re.compile(r'\bryzen\s*([3579])\s*([0-9]{3,5}x?)\b', re.I),  r'ryzen\1 \2'),
    # --- Storage / SSD model lines ---
    (re.compile(r'\b(970|980)\s*(evo|pro)\s*(plus)?\b', re.I),    r'\1\2\3'),  # 970 EVO Plus → 970evoplus
    # --- Cameras / Lenses ---
    (re.compile(r'\bsony\s*a\s*([0-9]+[iv]*)\b', re.I),           r'sonya\1'),
    (re.compile(r'\bcanon\s*(eos)?\s*r\s*([0-9]+[a-z]*)\b', re.I), r'canonr\2'),
    (re.compile(r'\bnikon\s*z\s*([0-9]+[a-z]*)\b', re.I),         r'nikonz\1'),
    (re.compile(r'\b([0-9]{2,3})\s*mm\s*f/?([0-9]\.?[0-9])\b', re.I), r'\1mm f\2'),
    # --- TVs / Routers ---
    (re.compile(r'\bqn([0-9]{2}[a-z])\b', re.I),                  r'qn\1'),    # Samsung QN90B 등
    (re.compile(r'\bax\s*([0-9]{3,5})\b', re.I),                  r'ax\1'),    # AX6000 등
]

def normalize_models(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    s = text
    for pat, repl in MODEL_RULES:
        s = pat.sub(repl, s)
    s = WS_RE.sub(" ", s).strip()
    return s

# ------------------------------------
# 4) 파이프라인 (단계 결합)
# ------------------------------------
def normalize_text_pipeline(s: str) -> str:
    """ 소문자화+비ASCII 정리 → 숫자·단위 결합 → 모델명 정규화 """
    return normalize_models(normalize_units_numbers(s))

# ------------------------------------
# 5) 메인: JSONL 스트리밍 변환
# ------------------------------------
def process_file(in_path: Path, out_path: Path, task: str):
    """
    task: 'QC' -> origin_query만
          'QI' -> origin_query, item_title 둘 다
    """
    n_in = n_out = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            if not line.strip():
                fout.write("\n")
                n_out += 1
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # 문제가 있는 라인은 그대로 통과 (라인 수 보존)
                fout.write(line)
                n_out += 1
                continue

            if task.upper() == "QC":
                if "origin_query" in obj:
                    obj["origin_query"] = normalize_text_pipeline(obj["origin_query"])
            elif task.upper() == "QI":
                if "origin_query" in obj:
                    obj["origin_query"] = normalize_text_pipeline(obj["origin_query"])
                if "item_title" in obj:
                    obj["item_title"]   = normalize_text_pipeline(obj["item_title"])
            else:
                # 지정이 틀리면 경고만 띄우고 원본 저장
                print(f"[warn] unknown task '{task}', line passthrough", file=sys.stderr)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    # 간단한 로그
    print(f"[done] {in_path} -> {out_path} (lines: {n_in} -> {n_out}, same={n_in==n_out})", file=sys.stderr)

# ------------------------------------
# 6) CLI
# ------------------------------------
def main():
    ap = argparse.ArgumentParser(description="QC/QI 데이터 정규화(소문자화+비ASCII, 숫자·단위, 모델명). 데이터 라인 수 보존.")
    ap.add_argument("--task", required=True, choices=["QC", "QI"], help="QC: origin_query만 / QI: origin_query+item_title")
    ap.add_argument("--input", required=True, help="입력 데이터 파일 경로")
    ap.add_argument("--output", required=True, help="출력 데이터 파일 경로")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    process_file(in_path, out_path, args.task)

if __name__ == "__main__":
    main()

# QC: origin_query만 정규화
# python data_nomalization.py --task QC --input data/train_QC_en_clean.txt --output data/train_QC_en_clean_norm.txt

# QI: origin_query + item_title 정규화
# python data_nomalization.py --task QI --input data/train_QI_en_clean.txt --output data/train_QI_en_clean_norm.txt