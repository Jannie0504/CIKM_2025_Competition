"""
This script ONLY applies additional lossless cleanups:
- unicode fractions -> decimals (½ -> 0.5, ¼ -> 0.25, ¾ -> 0.75, …)
- unify range/multiplication symbols (~/–/—/- -> '-', ×/✕/X -> 'x')
- normalize % / °C, °F spacing ( '  %' -> '%', ' °c' -> 'c' )
- remove thousand separators (1,024 -> 1024; 1.024 (locale) -> 1024)
- compress consecutive duplicate tokens ('free free' -> 'free')
- (optional) drop promotional phrases (free shipping, best seller, …)
- ALWAYS preserve line count; unknown/invalid JSON lines are passed through.

Fields:
- QC : origin_query only
- QI : origin_query + item_title
"""

import argparse, json, re, sys
from pathlib import Path

WS_RE = re.compile(r"\s+")
def _strip_ws(s: str) -> str:
    return WS_RE.sub(" ", s).strip() if isinstance(s, str) else s

# --- Extra normalizations (lossless) ---

FRACTIONS = {
    "¼": "0.25", "½": "0.5", "¾": "0.75",
    "⅓": "0.333", "⅔": "0.667",
    "⅕": "0.2",   "⅖": "0.4", "⅗": "0.6", "⅘": "0.8",
}

X_MULTI_RE   = re.compile(r"[✕×X]")             # × ✕ X -> x
RANGE_RE     = re.compile(r"\s*(~|–|—|-)\s*")    # -> '-'
PERCENT_RE   = re.compile(r"\s*%")               # ' %' -> '%'
TEMP_RE      = re.compile(r"\s*°\s*([cfCF])")    # ' °c' -> 'c'
THOUSAND_RE1 = re.compile(r"(?<=\d),(?=\d{3}\b)")# 1,024 -> 1024
THOUSAND_RE2 = re.compile(r"(?<=\d)\.(?=\d{3}\b)")# 1.024 -> 1024 (locale)
DUP_TOKEN_RE = re.compile(r"\b(\w+)(\s+\1\b)+")  # 'aaa aaa' -> 'aaa'

PROMO_WORDS = [
    r"free\s*shipping", r"hot\s*deal", r"best\s*seller", r"new\s*arrival",
    r"limited\s*offer", r"flash\s*sale"
]
PROMO_RE = re.compile(r"|".join(PROMO_WORDS), re.I)

def post_extras(s: str, drop_promos: bool = False) -> str:
    if not isinstance(s, str) or not s:
        return s
    t = s

    # (a) unicode fractions -> decimals
    for frac, val in FRACTIONS.items():
        if frac in t:
            t = t.replace(frac, val)

    # (b) symbols unification
    t = X_MULTI_RE.sub("x", t)             # × ✕ X -> x
    t = RANGE_RE.sub("-", t)               # ~ – — - -> '-'
    t = PERCENT_RE.sub("%", t)             # ' %' -> '%'
    t = TEMP_RE.sub(lambda m: m.group(1).lower(), t)  # ' °C' -> 'c'

    # (c) thousand separators
    t = THOUSAND_RE1.sub("", t)
    t = THOUSAND_RE2.sub("", t)

    # (d) compress consecutive duplicate tokens
    prev = None
    while prev != t:
        prev = t
        t = DUP_TOKEN_RE.sub(lambda m: m.group(1), t)

    # (e) optionally drop promo phrases
    if drop_promos:
        t = PROMO_RE.sub("", t)

    return _strip_ws(t)

# --- JSONL streaming ---

def process_file(in_path: Path, out_path: Path, task: str, drop_promos: bool = False):
    """
    task: 'QC' -> origin_query only
          'QI' -> origin_query + item_title
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
                # keep as-is to preserve line count
                fout.write(line)
                n_out += 1
                continue

            if task.upper() == "QC":
                if "origin_query" in obj:
                    obj["origin_query"] = post_extras(obj["origin_query"], drop_promos=drop_promos)
            elif task.upper() == "QI":
                if "origin_query" in obj:
                    obj["origin_query"] = post_extras(obj["origin_query"], drop_promos=drop_promos)
                if "item_title" in obj:
                    obj["item_title"]   = post_extras(obj["item_title"],   drop_promos=drop_promos)
            else:
                print(f"[warn] unknown task '{task}', line passthrough", file=sys.stderr)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[done] {in_path} -> {out_path} (lines: {n_in} -> {n_out}, same={n_in==n_out})", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Post extras for already preprocessed QC/QI TXT (line-preserving).")
    ap.add_argument("--task", required=True, choices=["QC", "QI"], help="QC: origin_query only / QI: origin_query+item_title")
    ap.add_argument("--input", required=True, help="Input TXT path (already preprocessed)")
    ap.add_argument("--output", required=True, help="Output TXT path")
    ap.add_argument("--drop-promos", action="store_true", help="Drop promo/decoration phrases (optional; default: keep)")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    process_file(in_path, out_path, args.task, drop_promos=args.drop_promos)

if __name__ == "__main__":
    main()

# QC: origin_query만 정규화
# python data_nomalization_up.py --task QC --input data/train_QC_en_clean_norm.txt --output data/train_QC_en_clean_norm_up.txt

# QI: origin_query + item_title 정규화
# python data_nomalization_up.py --task QI --input data/train_QI_en_clean_norm.txt --output data/train_QI_en_clean_norm_up.txt

# QC: origin_query만 정규화 + 판촉 문구까지 제거
# python data_nomalization_up.py --task QC --input data/train_QC_en_clean_norm.txt --output data/train_QC_en_clean_norm_up.txt --drop-promos
# python data_nomalization_up.py --task QC --input data/test_QC_en_clean_norm.txt --output data/test_QC_en_clean_norm_up.txt --drop-promos

# QI: origin_query + item_title 정규화 + 판촉 문구까지 제거
# python data_nomalization_up.py --task QI --input data/train_QI_en_clean_norm.txt --output data/train_QI_en_clean_norm_up.txt --drop-promos
# python data_nomalization_up.py --task QI --input data/test_QI_en_clean_norm.txt --output data/test_QI_en_clean_norm_up.txt --drop-promos