import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from tqdm.notebook import tqdm

from datasets import NerDataset

# CKPT_MODEL = ''
CKPT_MODEL = 'TingChenChang/chinese-ner'
BATCH_SIZE = 32
LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

tokenizer = BertTokenizer.from_pretrained(CKPT_MODEL)
SKIP_TOKEN_IDS = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]

texts = [
    '王小明去台北市立動物園玩',
    '高雄的西子灣是一個散心絕佳的好去處'
]

pred_dataset = NerDataset(tokenizer, texts, for_train=False)

pred_loader = DataLoader(
    dataset=pred_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=pred_dataset.create_mini_batch,
)

model = BertForTokenClassification.from_pretrained(CKPT_MODEL)
model.to(device)

results = []
with torch.no_grad():
    for data in tqdm(pred_loader, desc='predict'):
        input_ids, token_type_ids, attention_mask = [d.to(device) for d in data[:3]]
        infos = data[3]

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        for token_id_list, pred_list, info in zip(input_ids, preds, infos):
            pred_list = [LABELS[i] for i, token_id in zip(pred_list, token_id_list)
                         if token_id not in SKIP_TOKEN_IDS]
            tokens = info['tokens']
            result = list(zip(tokens, pred_list))
            results.append(result)

print('predict result: ')
for result in results:
    print(result)
