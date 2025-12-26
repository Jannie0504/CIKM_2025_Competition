# sample_per_lang_stratified.py
# 사용법 예시:
#   # 1) 비율 기반: 각 언어에서 10%씩 샘플
#   python sample_data.py \
#       --input data/train_QI_en_clean_norm.txt \
#       --output data/val_QI_en_clean_norm_sample.txt  \
#       --train_out data/train_QI_en_clean_norm_sample.txt \
#       --languages en,fr,es,ko,pt,ja,de,it,pl,ar,th,vn,id \
#       --per_lang_frac 0.10 \
#       --seed 42
#
#   # 2) 개수 기반(이전과 동일): ko/ja=4500, 나머지=4000
#   python sample_data.py \
#       --input data/all.jsonl \
#       --output data/sample_mix.jsonl \
#       --train_out data/rest_except_sample.jsonl \
#       --languages en,fr,es,ko,pt,ja,de,it,pl,ar,th,vn,id \
#       --per_lang_default 4000 \
#       --per_lang_override ko:4500,ja:4500 \
#       --seed 42
#
# 동작:
# - 언어별 타깃 수를 "비율" 또는 "개수"로 결정(비율이 주어지면 비율 우선)
# - 각 언어 내부에서 label(0/1) 분포를 원본에 가깝게 유지하여 층화 샘플링
# - 특정 레이블 부족 시 같은 언어의 다른 레이블로 보충(언어별 총 타깃 수는 맞춤)
# - --train_out 지정 시, 원본 - 샘플 차집합을 별도 저장

import json
import argparse
import random
from collections import defaultdict, Counter
from pathlib import Path
import math

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def save_jsonl(path, items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def summarize(items, title="SUMMARY"):
    n = len(items)
    by_lang = Counter([x.get("language", "unknown") for x in items])
    by_lab  = Counter([x.get("label", None) for x in items])
    print(f"\n[{title}] total={n}")
    print("  label 분포:", dict(sorted(by_lab.items())))
    print("  언어 분포(상위):", dict(sorted(by_lang.items(), key=lambda kv: -kv[1])[:20]))

def _allocate_targets_per_label(counts_per_label, target_total):
    """
    counts_per_label: dict[label] = 원본에서의 레이블 개수
    target_total: 해당 언어에서 뽑을 총 개수
    → 원본 분포 비율(round) 기반으로 label별 타깃 개수 산정 (총합 target_total로 보정)
    """
    total = sum(counts_per_label.values())
    if total == 0:
        return {k: 0 for k in counts_per_label.keys()}

    # 비율 기반 round
    raw = {lab: (counts_per_label[lab] / total) * target_total for lab in counts_per_label}
    alloc = {lab: int(round(v)) for lab, v in raw.items()}

    # 합계 보정
    diff = target_total - sum(alloc.values())
    if diff != 0:
        remainders = sorted(
            ((lab, raw[lab] - int(raw[lab])) for lab in counts_per_label),
            key=lambda x: x[1],
            reverse=(diff > 0)
        )
        i = 0
        while diff != 0 and i < len(remainders):
            lab = remainders[i][0]
            alloc[lab] += 1 if diff > 0 else -1
            diff += -1 if diff > 0 else 1
            i += 1

    # 음수 방지
    for lab in list(alloc.keys()):
        if alloc[lab] < 0:
            alloc[lab] = 0

    # 공급량(cap) 보정
    shortage = 0
    for lab, v in alloc.items():
        cap = counts_per_label[lab]
        if v > cap:
            shortage += (v - cap)
            alloc[lab] = cap

    if shortage > 0:
        # 여유 있는 레이블로 재분배
        for lab in sorted(alloc.keys(), key=lambda k: counts_per_label[k] - alloc[k], reverse=True):
            space = counts_per_label[lab] - alloc[lab]
            if space <= 0:
                continue
            take = min(space, shortage)
            alloc[lab] += take
            shortage -= take
            if shortage == 0:
                break
    return alloc

def parse_languages(arg_languages, detected_langs):
    if arg_languages:
        lang_list = [l.strip() for l in arg_languages.split(",") if l.strip() != ""]
    else:
        lang_list = sorted(detected_langs)
    return lang_list

def parse_overrides(override_str):
    """
    "ko:4500,ja:4500" -> {"ko": 4500, "ja": 4500}
    또는 비율 기반이면 "ko:0.1,ja:0.15" 처럼 float로 파싱해도 됨(호출부에서 선택적으로 사용)
    """
    m = {}
    if not override_str:
        return m
    for tok in override_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            continue
        k, v = tok.split(":", 1)
        k, v = k.strip(), v.strip()
        if not k:
            continue
        try:
            # int 우선, 실패 시 float 시도
            try:
                m[k] = int(v)
            except ValueError:
                m[k] = float(v)
        except Exception:
            pass
    return m

def decide_target_count_for_lang(available_total, lang, *,
                                 mode,                # "frac" or "count"
                                 default,             # float (frac) or int (count)
                                 overrides=None):     # dict per-lang override
    overrides = overrides or {}
    if mode == "frac":
        # per-language override가 float로 들어올 수 있음
        frac = overrides.get(lang, default)
        # 안전클램프
        try:
            frac = float(frac)
        except Exception:
            frac = float(default)
        frac = max(0.0, min(1.0, frac))
        target = int(round(available_total * frac))
        # 0개 방지(원하면 0 허용 가능). 최소 1개로
        target = max(1 if available_total > 0 else 0, target)
        target = min(target, available_total)
        return target
    else:
        # count 모드
        cnt = overrides.get(lang, default)
        try:
            cnt = int(cnt)
        except Exception:
            cnt = int(default)
        cnt = max(0, cnt)
        cnt = min(cnt, available_total)
        return cnt

def stratified_sample_with_targets(items, languages=None, seed=42,
                                   # 비율 모드
                                   per_lang_frac=None, per_lang_frac_override=None,
                                   # 개수 모드
                                   per_lang_default=None, per_lang_override=None):
    rng = random.Random(seed)
    per_lang_frac_override = per_lang_frac_override or {}
    per_lang_override = per_lang_override or {}

    # 그룹핑: 언어 -> 레이블 -> 리스트
    by_lang_label = defaultdict(lambda: defaultdict(list))
    all_langs = set()
    for obj in items:
        lang = obj.get("language", "unknown")
        lab  = obj.get("label", None)
        all_langs.add(lang)
        by_lang_label[lang][lab].append(obj)

    lang_list = parse_languages(languages, all_langs)

    # 모드 결정: 비율이 주어지면 비율 우선
    if per_lang_frac is not None:
        mode = "frac"
        default = float(per_lang_frac)
        overrides = per_lang_frac_override
    else:
        mode = "count"
        default = int(per_lang_default) if per_lang_default is not None else 0
        overrides = per_lang_override

    sampled = []
    for lang in lang_list:
        label_buckets = by_lang_label.get(lang, {})
        for lab in label_buckets:
            rng.shuffle(label_buckets[lab])

        available_total = sum(len(lst) for lst in label_buckets.values())
        target_total = decide_target_count_for_lang(
            available_total, lang, mode=mode, default=default, overrides=overrides
        )

        counts_per_label = {lab: len(lst) for lab, lst in label_buckets.items()}
        target_per_label = _allocate_targets_per_label(counts_per_label, target_total)

        # 각 레이블에서 타깃만큼 꺼내기
        lang_samples = []
        for lab, target in target_per_label.items():
            if target <= 0:
                continue
            lang_samples.extend(label_buckets[lab][:target])

        # 합계가 모자라면 남은 풀에서 보충
        if len(lang_samples) < target_total:
            pool = []
            for lab, lst in label_buckets.items():
                leftover = lst[target_per_label.get(lab, 0):]
                pool.extend(leftover)
            rng.shuffle(pool)
            need = target_total - len(lang_samples)
            lang_samples.extend(pool[:need])

        lang_samples = lang_samples[:target_total]
        sampled.extend(lang_samples)

        print(f"[{lang}] picked {len(lang_samples)} "
              f"(target_total={target_total}, available={counts_per_label})")

    rng.shuffle(sampled)
    return sampled

def save_rest(original_items, sampled_items, out_path):
    """원본 - 샘플 차집합을 out_path에 저장"""
    if not out_path:
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sampled_ids = {str(x.get("id")) for x in sampled_items}
    rest = [x for x in original_items if str(x.get("id")) not in sampled_ids]
    save_jsonl(out_path, rest)
    print(f"나머지(rest) {len(rest)}개 저장 → {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train_out", type=str, default=None,
                    help="샘플(--output) 외에, 원본에서 샘플을 제외한 나머지를 저장할 경로")
    ap.add_argument("--languages", type=str, default=None,
                    help="대상 언어 목록(콤마구분). 미지정 시 전체 언어 자동 탐색")

    # ★ 비율 기반 옵션
    ap.add_argument("--per_lang_frac", type=float, default=None,
                    help="언어별 샘플 비율(예: 0.1 → 10%). 지정 시 비율 모드로 동작하고 개수 옵션은 무시")
    ap.add_argument("--per_lang_frac_override", type=str, default="",
                    help="언어별 비율 오버라이드(예: 'ko:0.15,ja:0.20')")

    # 개수 기반 옵션 (기존)
    ap.add_argument("--per_lang_default", type=int, default=None,
                    help="오버라이드에 지정되지 않은 언어의 기본 샘플 개수(비율 미지정 시에만 사용)")
    ap.add_argument("--per_lang_override", type=str, default="",
                    help="특정 언어별 타깃 개수 오버라이드 (예: 'ko:4500,ja:4500')")

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    items = load_jsonl(args.input)
    summarize(items, "원본")

    # 파싱
    frac_overrides = parse_overrides(args.per_lang_frac_override)
    count_overrides = parse_overrides(args.per_lang_override)

    sampled = stratified_sample_with_targets(
        items,
        languages=args.languages,
        seed=args.seed,
        per_lang_frac=args.per_lang_frac,
        per_lang_frac_override=frac_overrides,
        per_lang_default=args.per_lang_default,
        per_lang_override=count_overrides
    )

    summarize(sampled, "샘플 결과")
    save_jsonl(args.output, sampled)
    print(f"\n저장 완료: {args.output}")

    if args.train_out:
        save_rest(items, sampled, args.train_out)

if __name__ == "__main__":
    main()
