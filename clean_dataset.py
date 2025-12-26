import json
import argparse
from clean import SimpleClean

def clean_file(input_path: str, output_path: str, fields: list[str]):
    """
    input_path 에서 JSONL 읽어서,
    fields 에 명시된 키들에 대해 cleaner를 적용한 뒤,
    output_path 에 결과를 JSONL 으로 저장.
    """
    cleaner = SimpleClean()

    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 지정된 각 필드에 대해 전처리
            for key in fields:
                if key in obj and isinstance(obj[key], str):
                    obj[key] = cleaner(obj[key])
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="JSONL 데이터 파일 전처리 (이모지·특수문자 제거 등)")
    parser.add_argument(
        '--task', choices=['QC','QI'], required=True,
        help="전처리할 데이터 타입. QC→['origin_query'], QI→['origin_query','item_title']")
    parser.add_argument(
        '--infile', required=True,
        help="원본 JSONL 파일 경로")
    parser.add_argument(
        '--outfile', required=True,
        help="클린된 결과를 저장할 JSONL 파일 경로")
    args = parser.parse_args()

    # QC와 QI에 따라 처리할 필드 결정
    if args.task == 'QC':
        target_fields = ['origin_query']
    else:  # QI
        target_fields = ['origin_query', 'item_title']

    clean_file(args.infile, args.outfile, target_fields)
    print(f"[Done] {args.infile} → {args.outfile} (fields: {target_fields})")
