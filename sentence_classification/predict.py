import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm.notebook import tqdm

from datasets import SentenceClassificationDataset

# CKPT_MODEL = 'models/'
CKPT_MODEL = 'TingChenChang/sentiment-classification'
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

tokenizer = BertTokenizer.from_pretrained(CKPT_MODEL)

examples = [
    '板金的部分我覺得很脆弱',
    '整體外殼造型我是喜歡的'
]
examples_df = pd.DataFrame(data={'text': examples})

pred_dataset = SentenceClassificationDataset(tokenizer, examples_df, for_train=False)

pred_loader = DataLoader(
    dataset=pred_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=pred_dataset.create_mini_batch,
)

model = BertForSequenceClassification.from_pretrained(CKPT_MODEL)
model.to(device)

pred_labels = []
with torch.no_grad():
    for data in tqdm(pred_loader, desc='predict'):
        input_ids, token_type_ids, attention_mask = [d.to(device) for d in data]

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        pred_labels += outputs.logits.argmax(dim=-1).cpu().tolist()

print('predict result: ', list(zip(examples, pred_labels)))
