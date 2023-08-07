# %%
import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForTokenClassification
from tqdm.notebook import tqdm
from seqeval.metrics import f1_score as seq_f1_score

from utils import RunningAverage, tokenize_and_map, LossHistorySaver
from datasets import NerDataset

PRETRAINED_MODEL_NAME = 'bert-base-chinese'
SEED = 1234

TRAINING_SET_RATIO = 0.9
BATCH_SIZE = 32
LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

LEARNING_RATE = 1e-5
EPOCHS = 2
SAVE_CKPT_DIR = f'models/{pd.Timestamp.now():%Y%m%d%H%M}/'
MODEL_PREFIX = 'ner_'

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# %%
# Data
texts = []
tag_lists = []
# with open('data/msra_train_bio.txt') as fr:
with open('data/sample.txt') as fr:
    text = ''
    tag_list = []
    for line in fr.readlines():
        line = line.strip()
        if line == '':
            assert len(text) == len(tag_list)
            texts.append(text)
            tag_lists.append(tag_list)
            text = ''
            tag_list = []
        elif line == '0':
            text += ' '
            tag_list.append('O')
        else:
            char, tag = line.split('\t')
            assert len(char) == 1
            text += char
            tag_list.append(tag)
    texts.append(text)
    tag_lists.append(tag_list)

print('text:', texts[20])
print('tag list:', tag_lists[20])

# %%
# Dataloader
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

SKIP_TOKEN_IDS = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]

dataset = NerDataset(tokenizer, texts, tag_lists)

train_size = int(TRAINING_SET_RATIO * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=dataset.create_mini_batch,
    shuffle=True
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=dataset.create_mini_batch,
)

# %%
# Model
model = BertForTokenClassification.from_pretrained(
    PRETRAINED_MODEL_NAME,
    num_labels=len(LABELS),
    return_dict=True
)
model.to(device)

# %%
# Train
def train_batch(model, data, optimizer, device):
    model.train()
    input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in data]

    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, valid_loader, device):
    model.eval()

    loss_averager = RunningAverage()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in tqdm(valid_loader, desc='evaluate'):
            input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in data]

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss_averager.add(outputs.loss.item())

            preds = outputs.logits.argmax(dim=-1).cpu().tolist()
            for token_id_list, label_list, pred_list in zip(input_ids, labels, preds):
                label_list = [LABELS[i] for i, token_id in zip(label_list, token_id_list)
                              if token_id not in SKIP_TOKEN_IDS]
                pred_list = [LABELS[i] for i, token_id in zip(pred_list, token_id_list)
                             if token_id not in SKIP_TOKEN_IDS]
                all_labels.append(label_list)
                all_preds.append(pred_list)

    f1 = seq_f1_score(all_labels, all_preds)
    return loss_averager.get(), f1

# %%
show_per_iter = 10
valid_per_iter = 100
save_per_iter = valid_per_iter * 2

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss_averager = RunningAverage()
train_loss_saver = LossHistorySaver(os.path.join(SAVE_CKPT_DIR, 'train_loss.csv'))
val_loss_saver = LossHistorySaver(os.path.join(SAVE_CKPT_DIR, 'val_loss.csv'), 'f1')

n_iter = 1
for epoch in range(1, EPOCHS + 1):
    for train_data in train_loader:
        loss = train_batch(model, train_data, optimizer, device)
        train_loss_averager.add(loss)

        if n_iter % show_per_iter == 0:
            print(f'epoch {epoch}, train {n_iter}: loss={train_loss_averager.get():.4f}')
            train_loss_saver.save(epoch, n_iter, train_loss_averager.get())
            train_loss_averager.flush()

        if n_iter % valid_per_iter == 0:
            loss, f1 = evaluate(model, valid_loader, device)
            val_loss_saver.save(epoch, n_iter, loss, f1)
            print(f'epoch {epoch}, valid {n_iter}: loss={loss:.4f}, f1={f1:.2f}')

        if n_iter % save_per_iter == 0:
            path = os.path.join(SAVE_CKPT_DIR, MODEL_PREFIX + f'{n_iter}/')
            print(f'save model at {path}')
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
        
        n_iter += 1
    
    train_loss_saver.plot()
    val_loss_saver.plot()
