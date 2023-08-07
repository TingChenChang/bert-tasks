# %%
import os
import json
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForQuestionAnswering
from tqdm.notebook import tqdm

from utils import RunningAverage, tokenize_and_map, LossHistorySaver
from datasets import QADataset

PRETRAINED_MODEL_NAME = 'bert-base-chinese'
SEED = 1234

TRAINING_SET_RATIO = 0.9
BATCH_SIZE = 32

LEARNING_RATE = 1e-5
EPOCHS = 2
SAVE_CKPT_DIR = f'models/{pd.Timestamp.now():%Y%m%d%H%M}/'
if not os.path.isdir(SAVE_CKPT_DIR):
    os.makedirs(SAVE_CKPT_DIR)
MODEL_PREFIX = 'qa_'

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# %%
# Data
queries = []
contents = []
indexes_in_content = []

# with open('data/DRCD_training.json') as fr:
with open('data/sample.json') as fr:
    rows = json.load(fr)['data']
    for row in rows:
        for paragraph in row['paragraphs']:
            content = paragraph['context']
            for qa in paragraph['qas']:
                query = qa['question']
                for answer in qa['answers']:
                    answer_start = answer['answer_start']
                    answer_text = answer['text']

                    start_index = answer_start
                    end_index = answer_start + len(answer_text) - 1  # end_index落在包含答案的尾巴

                    queries.append(query)
                    contents.append(content)
                    indexes_in_content.append((start_index, end_index))

print('query:', queries[1])
print('content:', contents[1])
start_index, end_index = indexes_in_content[1]
print('answer:', content[start_index:end_index + 1])

# %%
# Dataloader
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

SKIP_TOKEN_IDS = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]

dataset = QADataset(tokenizer, queries, contents, indexes_in_content)

TRAINING_SET_RATIO = 0.9
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
model = BertForQuestionAnswering.from_pretrained(
    PRETRAINED_MODEL_NAME,
    return_dict=True
)
model.to(device)

# %%
# Train
def train_batch(model, data, optimizer, device):
    model.train()
    input_ids, token_type_ids, attention_mask, start_indexes, end_indexes = [d.to(device) for d in data]

    outputs = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        start_positions=start_indexes,
        end_positions=end_indexes
    )
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, valid_loader, device):
    model.eval()

    loss_averager = RunningAverage()
    acc_averager = RunningAverage()

    with torch.no_grad():
        for data in tqdm(valid_loader, desc='evaluate'):
            input_ids, token_type_ids, attention_mask, start_indexes, end_indexes = [d.to(device) for d in data]

            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                start_positions=start_indexes,
                end_positions=end_indexes
            )

            loss_averager.add(outputs.loss.item())

            pred_start_indexes = outputs.start_logits.argmax(dim=-1)
            pred_end_indexes = outputs.end_logits.argmax(dim=-1)
            corrects = torch.logical_and(pred_start_indexes == start_indexes,
                                         pred_end_indexes == end_indexes).cpu().tolist()
            acc_averager.add_all(corrects)

    return loss_averager.get(), acc_averager.get()

# %%
show_per_iter = 10
valid_per_iter = 100
save_per_iter = valid_per_iter * 2

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss_averager = RunningAverage()
train_loss_saver = LossHistorySaver(os.path.join(SAVE_CKPT_DIR, 'train_loss.csv'))
val_loss_saver = LossHistorySaver(os.path.join(SAVE_CKPT_DIR, 'val_loss.csv'), 'acc')

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
            loss, acc = evaluate(model, valid_loader, device)
            val_loss_saver.save(epoch, n_iter, loss, acc)
            print(f'epoch {epoch}, valid {n_iter}: loss={loss:.4f}, acc={acc:.2f}')

        if n_iter % save_per_iter == 0:
            path = os.path.join(SAVE_CKPT_DIR, MODEL_PREFIX + f'{n_iter}/')
            print(f'save model at {path}')
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
        
        n_iter += 1
    
    train_loss_saver.plot()
    val_loss_saver.plot()
