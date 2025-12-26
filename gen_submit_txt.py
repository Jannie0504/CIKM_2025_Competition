import argparse
import json
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Make submit file for a task (QI or QC).")
    parser.add_argument("--task", type=str, choices=["QI", "QC"], required=True,
                        help="Task name: QI or QC")
    args = parser.parse_args()

    task = args.task
    input_path  = Path(f"./predictions/submit_{task}.txt")
    output_path = Path(f"./submit/submit_{task}.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pred = obj.get("prediction", obj.get("predict"))
            if pred is None:
                print(f"[WARN] Missing 'prediction'/'predict': {line}", file=sys.stderr)
                continue
            fout.write(json.dumps({"id": obj["id"], "prediction": pred}, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote: {output_path}")

if __name__ == "__main__":
    main()