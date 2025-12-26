from torch.utils.data import Dataset
import torch
import json

# Custom Dataset class
class SentencePairDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, sentence1_str, sentence2_str, soft_labels=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.sentence1_str = sentence1_str
        self.sentence2_str = sentence2_str

        self.soft_labels = soft_labels

        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence1 = item[self.sentence1_str]
        sentence2 = item[self.sentence2_str]
        label = item.get('label', None)

        encoding = self.tokenizer(
            sentence1,
            sentence2,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )

        out = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

        if label is not None:
            out['labels'] = torch.tensor(int(label), dtype=torch.long)
        # Add soft label if available
        if self.soft_labels is not None:
            out['soft_labels'] = torch.tensor(self.soft_labels[idx], dtype=torch.float)

        out['idx'] = idx
        return out

class SentencePairPredictDataset(SentencePairDataset):
    def __getitem__(self, idx):
        return self.data[idx]


class RetrievalDataset(SentencePairDataset):
    def __init__(self, file_path, tokenizer, max_length, s1, s2,
                 retriever, class_name, k=10):
        super().__init__(file_path, tokenizer, max_length, s1, s2)
        self.retriever  = retriever
        self.class_name = class_name
        self.k = k

    def __getitem__(self, idx):
        base = super().__getitem__(idx)
        pair_text = f"{base[self.s1]} [SEP] {base[self.s2]}"
        exclude   = base.get('id', None)
        neighbors = self.retriever.topk(self.class_name, pair_text, self.k, exclude_id=exclude)
        # neighbors 텍스트만 추출
        neg_texts = []
        for n in neighbors:
            if "category_path" in n:
                neg_texts.append(n["category_path"])
            elif "item_title" in n:
                neg_texts.append(n["item_title"])
        base['negatives'] = neg_texts
        return base
