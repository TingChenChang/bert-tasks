import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from train import BertForMultiLabelSequenceClassification
from tqdm.notebook import tqdm

from datasets import MultiLabelDataset

# CKPT_MODEL = 'models/'
CKPT_MODEL = 'TingChenChang/toxic-comment-classification'
BATCH_SIZE = 32

tokenizer = BertTokenizer.from_pretrained(CKPT_MODEL)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

examples = [
    'Fuck you! You son of bitch',
    'I will kill you soon'
]
examples_df = pd.DataFrame(data={'comment_text': examples})

pred_dataset = MultiLabelDataset(tokenizer, examples_df, for_train=False)

pred_loader = DataLoader(
    dataset=pred_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=pred_dataset.create_mini_batch,
)

model = BertForMultiLabelSequenceClassification.from_pretrained(CKPT_MODEL)
model.to(device)

pred_labels = []
with torch.no_grad():
    for data in tqdm(pred_loader, desc='predict'):
        input_ids, token_type_ids, attention_mask = [d.to(device) for d in data]

        logits = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        pred_labels += (torch.sigmoid(logits) > 0.5).int().cpu().tolist()

print('predict result: ', list(zip(examples, pred_labels)))
