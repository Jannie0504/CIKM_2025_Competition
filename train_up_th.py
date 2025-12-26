import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftConfig
)
import torch
from torch.utils.data import DataLoader

from dataset import SentencePairDataset, SentencePairPredictDataset
from parameters import load_parameters

os.environ['WANDB_MODE'] = 'offline'
# os.environ['TRITON_PTXAS_PATH']='/usr/local/cuda/bin/ptxas'

from pathlib import Path

def _ensure_parent_dir(path_or_str):
    if not path_or_str:
        return
    p = Path(path_or_str)
    (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)


# -------------------- 계층 토큰 주입 --------------------
# 카테고리 경로에 계층 토큰 주입: "a,b,c" -> "[L1] a [L2] b [L3] c"
def _mark_levels(cat_path: str) -> str:
    """
    "toys & hobbies,electronic toys,electronic pets"
      -> "[L1] toys & hobbies [L2] electronic toys [L3] electronic pets"
    이미 [L1]가 있으면 재주입하지 않습니다.
    """
    if not isinstance(cat_path, str) or not cat_path:
        return ""
    if "[L1]" in cat_path:
        return cat_path

    parts = [p.strip() for p in cat_path.split(",") if p.strip()]
    return " ".join(f"[L{i+1}] {p}" for i, p in enumerate(parts))

# -------------------- leaf 유틸 & leaf별 threshold 튠 --------------------
def _leaf_name(cat_path: str) -> str:
    # 마지막 토큰(leaf)만 추출
    parts = [p.strip() for p in str(cat_path or "").split(",")]
    return parts[-1] if parts else ""

from collections import defaultdict
def _tune_threshold_per_leaf(trainer, eval_dataset, sentence2_str, default_grid=None):
    """
    eval_dataset에 대해 trainer.predict()로 model_prob를 얻고,
    leaf(category_path 마지막 토큰)별로 F1이 최적인 threshold를 찾는다.
    return: dict {leaf_name: best_thr}
    """
    if default_grid is None:
        # 0.30 ~ 0.70, 0.02 step (가볍고 효과적인 범위)
        default_grid = np.linspace(0.30, 0.70, 21)

    pred_out = trainer.predict(eval_dataset, metric_key_prefix="eval_tune_qc")
    logits = pred_out.predictions                 # (N, 2)
    labels = pred_out.label_ids.astype(int)       # (N,)
    probs1 = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]  # p(y=1)

    # leaf별로 점수/라벨 모으기
    bucket = defaultdict(list)
    for i, ex in enumerate(eval_dataset.data):
        leaf = _leaf_name(ex.get(sentence2_str, ""))
        bucket[leaf].append((probs1[i], labels[i]))

    thr_map = {}
    for leaf, arr in bucket.items():
        S = np.array([s for s, _ in arr]); Y = np.array([y for _, y in arr])
        # leaf에 샘플이 너무 적으면 튠 효과가 불안정 -> 그냥 0.5
        if len(S) < 30:
            thr_map[leaf] = 0.5
            continue
        best = (0.0, 0.5)
        for t in default_grid:
            pred = (S >= t).astype(int)
            f1 = f1_score(Y, pred)
            if f1 > best[0]:
                best = (f1, t)
        thr_map[leaf] = float(best[1])
    return thr_map

# ==================== 원본 설정 ====================
model_args, data_args, training_args = load_parameters()

training_args.load_best_model_at_end = True

_ensure_parent_dir(data_args.eval_outputs)
_ensure_parent_dir(data_args.outputs)

# 마지막 체크포인트 자동 감지
from transformers.trainer_utils import get_last_checkpoint
last_ckpt = None
if os.path.isdir(training_args.output_dir):
    try:
        last_ckpt = get_last_checkpoint(training_args.output_dir)
    except Exception:
        last_ckpt = None
print(f"[Resume] last_ckpt = {last_ckpt}")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Metrics
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, pos_label=1)
    return {"f1": f1}

# Load config/tokenizer/model
config_peft = None
if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path) and 'adapter_config.json' in os.listdir(model_args.model_name_or_path):
    config_peft = PeftConfig.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(config_peft.base_model_name_or_path, num_labels=2)
else:
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=2)

config.hidden_dropout_prob = 0.25
config.attention_probs_dropout_prob = 0.10
if hasattr(config, "classifier_dropout"):
    config.classifier_dropout = 0.20

# Load tokenizer and model
print("use fast tokenizer = {}".format(model_args.use_fast_tokenizer))
tokenizer = AutoTokenizer.from_pretrained(
    config_peft.base_model_name_or_path if isinstance(config_peft, PeftConfig) else model_args.model_name_or_path,
    use_fast=model_args.use_fast_tokenizer
)
if (not hasattr(tokenizer, "pad_token")) or (tokenizer.pad_token is None):
    tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "sep_token", None) or tokenizer.unk_token

config.num_labels = 2
config.problem_type = "single_label_classification"
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config,
)

if model_args.use_lora:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )
    model = get_peft_model(model, peft_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

# Fields
if data_args.task_name == 'QC':
    sentence1_str = 'origin_query'
    sentence2_str = 'category_path'
if data_args.task_name == 'QI':
    sentence1_str = 'origin_query'
    sentence2_str = 'item_title'

# Datasets
train_dataset = SentencePairDataset(data_args.train_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
eval_dataset  = SentencePairDataset(data_args.validation_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
test_dataset  = SentencePairPredictDataset(data_args.test_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
data_collator = DataCollatorWithPadding(tokenizer, padding="longest")

# -------------------- QC일 때 카테고리 경로에 계층 토큰 주입 --------------------
# QC일 때만 category_path에 계층 토큰 주입
if data_args.task_name == "QC":
    for ex in train_dataset.data: ex["category_path"] = _mark_levels(ex.get("category_path", ""))
    for ex in eval_dataset.data:  ex["category_path"] = _mark_levels(ex.get("category_path", ""))
    for ex in test_dataset.data:  ex["category_path"] = _mark_levels(ex.get("category_path", ""))

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# -------------------- leaf별 threshold 맵 저장 변수 --------------------
_qc_thr_map = None  # {leaf_name: thr}

# Train
if training_args.do_train:
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

# Evaluate
if training_args.do_eval:
    print("Evaluating on validation set...")
    metrics = trainer.evaluate(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print(f"Final F1 score on test set: {metrics['eval_f1']:.4f}")

    # === eval 결과 저장 ===
    model.eval()
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator
    )
    with open(data_args.eval_outputs, "w", encoding="utf-8") as fout, torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Eval → file"):
            inputs = {k: v.to(model.device) for k, v in batch.items()
                        if k in ("input_ids", "attention_mask", "token_type_ids")}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            preds = probs.argmax(axis=1).tolist()
            labels = batch["labels"].tolist()

            for i, idx in enumerate(batch["idx"].tolist()):
                item = eval_dataset.data[idx]
                fout.write(json.dumps({
                    "id":           item["id"],
                    "language":     item["language"],
                    sentence1_str:  item[sentence1_str],
                    sentence2_str:  item[sentence2_str],
                    "label":        labels[i],
                    "prediction":   preds[i],
                    "score":        float(probs[i][1]),
                    "model_prob":   float(probs[i][1])  # 추후 튠에 사용
                }, ensure_ascii=False) + "\n")
    print(f"Eval predictions saved to {data_args.eval_outputs}")

# -------------------- QC일 때 leaf별 threshold 자동 튠 --------------------
if data_args.task_name == 'QC':
    _qc_thr_map = _tune_threshold_per_leaf(trainer, eval_dataset, sentence2_str)
    # (선택) 저장하고 싶으면:
    try:
        with open(Path(data_args.outputs).with_suffix(".qc_thr_map.json"), "w", encoding="utf-8") as f:
            json.dump(_qc_thr_map, f, ensure_ascii=False, indent=2)
        print(f"[QC] tuned per-leaf thresholds saved.")
    except Exception:
        pass

# Predict
if training_args.do_predict:
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)

    with open(data_args.outputs, 'w', encoding='utf-8') as f, torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = tokenizer(
                batch[sentence1_str], batch[sentence2_str],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=data_args.max_seq_length
            ).to(model.device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            prediction = probs.argmax(axis=1).tolist()

            for k in range(len(batch['id'])):
                prob1 = float(probs[k][1])
                pred  = prediction[k]

                # -------------------- QC면 leaf별 임계값 적용 --------------------
                if data_args.task_name == 'QC' and _qc_thr_map is not None:
                    leaf = _leaf_name(batch[sentence2_str][k])
                    thr  = _qc_thr_map.get(leaf, 0.5)
                    pred = int(prob1 >= thr)

                output_json = {
                    "id": batch["id"][k].item(),
                    "language": batch["language"][k],
                    sentence1_str: batch[sentence1_str][k],
                    sentence2_str: batch[sentence2_str][k],
                    "prediction": pred,
                    "score":      prob1
                }
                f.write(json.dumps(output_json, ensure_ascii=False) + "\n")

print(f"Predictions saved to {data_args.outputs}")
