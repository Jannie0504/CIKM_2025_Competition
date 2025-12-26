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

import copy

from dataset import SentencePairDataset, SentencePairPredictDataset
from parameters import load_parameters

os.environ['WANDB_MODE'] = 'offline'
# os.environ['TRITON_PTXAS_PATH']='/usr/local/cuda/bin/ptxas'

EMA_DECAY = 0.999  
DISTILL_ALPHA = 0.5  # KL vs CE 혼합 비율
TEMPERATURE   = 2.5 

import re
TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tok(s: str):
    return TOKEN_RE.findall((s or "").lower())

def _jaccard(q: str, t: str):
    Q, T = set(_tok(q)), set(_tok(t))
    inter = len(Q & T); uni = len(Q | T) or 1
    return inter / uni

def _containment(q: str, t: str):
    Q, T = set(_tok(q)), set(_tok(t))
    inter = len(Q & T)
    return inter / (len(Q) or 1)

def _hybrid_qi_score(prob_model_1: float, origin_query: str, item_title: str,
                     w_prob: float, w_jac: float, w_con: float) -> float:
    j = _jaccard(origin_query, item_title)
    c = _containment(origin_query, item_title)
    return w_prob * prob_model_1 + w_jac * j + w_con * c

from pathlib import Path

def _ensure_parent_dir(path_or_str):
    if not path_or_str:
        return
    p = Path(path_or_str)
    # 파일 경로면 부모 폴더 생성, 폴더 경로면 해당 폴더 생성
    (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    
# Configuration
model_args, data_args, training_args = load_parameters()

# === Save best on improvement + EarlyStop(3) 설정 ===
training_args.load_best_model_at_end = True

_ensure_parent_dir(data_args.eval_outputs)
_ensure_parent_dir(data_args.outputs) 

# 마지막 체크포인트 자동 감지 (resume 우선순위 반영) 
from transformers.trainer_utils import get_last_checkpoint

last_ckpt = None
if os.path.isdir(training_args.output_dir):
    try:
        last_ckpt = get_last_checkpoint(training_args.output_dir)
    except Exception:
        last_ckpt = None
print(f"[Resume] last_ckpt = {last_ckpt}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Function to compute metrics
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, pos_label=1)
    return {"f1": f1}


# Load config
config_peft = None
if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path) and 'adapter_config.json' in os.listdir(model_args.model_name_or_path):
    config_peft = PeftConfig.from_pretrained(
        model_args.model_name_or_path
    )
    config = AutoConfig.from_pretrained(
        config_peft.base_model_name_or_path,
        num_labels=2
    )
else:
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2
    )

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

# --- Self‑Distillation: EMA Teacher 생성 ---
teacher_model = copy.deepcopy(model).eval().to(device)
for p in teacher_model.parameters():
    p.requires_grad = False

# Create datasets
if data_args.task_name == 'QC':
    sentence1_str = 'origin_query'
    sentence2_str = 'category_path'
if data_args.task_name == 'QI':
    sentence1_str = 'origin_query'
    sentence2_str = 'item_title'


train_dataset = SentencePairDataset(data_args.train_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
eval_dataset = SentencePairDataset(data_args.validation_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
test_dataset = SentencePairPredictDataset(data_args.test_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
data_collator = DataCollatorWithPadding(tokenizer, padding="longest")

class SelfDistillTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 기본 학생 로짓 + CE 손실
        labels = inputs.pop("labels")
        outputs_student = model(**inputs)
        logits_s = outputs_student.logits
        ce_loss = torch.nn.functional.cross_entropy(logits_s, labels)

        # Teacher 로짓 (no grad)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_t = outputs_teacher.logits

        # KL Distillation loss
        logp_s = torch.log_softmax(logits_s / TEMPERATURE, dim=-1)
        p_t    = torch.softmax(logits_t / TEMPERATURE, dim=-1)
        kl_loss = torch.nn.functional.kl_div(
            logp_s, p_t, reduction="batchmean", log_target=False
        ) * (TEMPERATURE ** 2)

        loss = DISTILL_ALPHA * kl_loss + (1 - DISTILL_ALPHA) * ce_loss

        return (loss, outputs_student) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch):
        # 1) 학생 한 스텝 학습
        loss = super().training_step(model, inputs, num_items_in_batch)

        # 2) EMA Teacher 업데이트
        with torch.no_grad():
            for p_s, p_t in zip(model.parameters(), self.teacher_model.parameters()):
                p_t.data.mul_(EMA_DECAY).add_(p_s.data, alpha=1-EMA_DECAY)
        return loss
    
# Initialize Trainer
trainer = SelfDistillTrainer(
    teacher_model=teacher_model,
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


# Train the model
if training_args.do_train:
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


# Evaluate the model
if training_args.do_eval:
    print("Evaluating on validation set...")
    metrics = trainer.evaluate(eval_dataset)
    
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print(f"Final F1 score on test set: {metrics['eval_f1']:.4f}")

    # === eval 결과 저장 (.txt, JSONL 형식) ===
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

            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()  # (B, 2)
            preds = probs.argmax(axis=1).tolist()
            labels = batch["labels"].tolist()

            for i, idx in enumerate(batch["idx"].tolist()):
                item = eval_dataset.data[idx]
                prob1 = float(probs[i][1])
                final_score = prob1
                final_pred  = preds[i]

                if data_args.task_name == 'QI' and data_args.use_jaccard_qi:
                    oq = item[sentence1_str]  # origin_query
                    it = item[sentence2_str]  # item_title
                    final_score = _hybrid_qi_score(
                        prob1, oq, it,
                        data_args.prob_weight, data_args.jac_weight, data_args.con_weight
                    )
                    final_pred = 1 if final_score >= data_args.hybrid_threshold else 0

                fout.write(json.dumps({
                    "id":           item["id"],
                    "language":     item["language"],
                    sentence1_str:  item[sentence1_str],
                    sentence2_str:  item[sentence2_str],
                    "label":        labels[i],
                    "prediction":   preds[i],
                    "score":        float(probs[i][1])   # 양성 확률
                }, ensure_ascii=False) + "\n")

    print(f"Eval predictions saved to {data_args.eval_outputs}")

# Predict
if training_args.do_predict:
    model.eval()
    dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)

    with open(data_args.outputs, 'w', encoding='utf-8') as f, torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            # Tokenize inputs
            inputs = tokenizer(
                batch[sentence1_str], batch[sentence2_str],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=data_args.max_seq_length
            ).to(model.device)

            # Predict
            outputs = model(**inputs)

            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            # Get prediction (0 or 1)
            prediction = probs.argmax(axis=1).tolist()

            # Create output JSON
            for k in range(len(batch['id'])):
                prob1 = float(probs[k][1])
                final_score = prob1
                final_pred  = prediction[k]

                if data_args.task_name == 'QI' and data_args.use_jaccard_qi:
                    oq = batch[sentence1_str][k]
                    it = batch[sentence2_str][k]
                    final_score = _hybrid_qi_score(
                        prob1, oq, it,
                        data_args.prob_weight, data_args.jac_weight, data_args.con_weight
                    )
                    final_pred = 1 if final_score >= data_args.hybrid_threshold else 0

                output_json = {
                    "id": batch["id"][k].item(),
                    "language": batch["language"][k],
                    sentence1_str: batch[sentence1_str][k],
                    sentence2_str: batch[sentence2_str][k],
                    "prediction": prediction[k],
                    "score":        float(probs[k][1])
                }

                # Write to file
                f.write(json.dumps(output_json, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {data_args.outputs}")
