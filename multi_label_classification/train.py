# %%
import os
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel, BertPreTrainedModel
from tqdm.notebook import tqdm
from sklearn import metrics

from utils import RunningAverage, LossHistorySaver
from datasets import MultiLabelDataset

PRETRAINED_MODEL_NAME = 'bert-base-chinese'
SEED = 1234

TRAINING_SET_RATIO = 0.9
BATCH_SIZE = 32
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

LEARNING_RATE = 1e-5
EPOCHS = 2
COLAB_DIR = '/content/drive/MyDrive/Colab Notebooks/BERT/multi_label_cls/'
SAVE_CKPT_DIR = COLAB_DIR + f'models/{pd.Timestamp.now():%Y%m%d%H%M}/'
if not os.path.isdir(SAVE_CKPT_DIR):
    os.makedirs(SAVE_CKPT_DIR)
MODEL_PREFIX = 'toxic_labels_'

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# %%
# Data
df = pd.read_csv('data/toxic_comment_classification.csv')
# df = pd.read_csv('data/sample.csv')
df = df.sample(frac=1).reset_index(drop=True)  # shuffle
print(df.head())

# %%
# Dataloader
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

dataset = MultiLabelDataset(tokenizer, df)

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
class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        
        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = self.dropout(output.pooler_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return logits, loss
        else:
            return logits

model = BertForMultiLabelSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME,
    num_labels=len(LABELS)
)
model.to(device)

# %%
# Train
def train_batch(model, data, optimizer, device):
    model.train()
    input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in data]

    _, loss = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, valid_loader, device):
    model.eval()

    loss_averager = RunningAverage()
    all_preds = {label: [] for label in LABELS}
    all_labels = {label: [] for label in LABELS}

    with torch.no_grad():
        for data in tqdm(valid_loader, desc='evaluate'):
            input_ids, token_type_ids, attention_mask, labels = [d.to(device) for d in data]

            logits, loss = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss_averager.add(loss.item())

            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            labels = labels.cpu().numpy()
            for i, l in enumerate(LABELS):
                all_preds[l] += preds[:, i].tolist()
                all_labels[l] += labels[:, i].tolist()

    f1 = {
        label: metrics.f1_score(all_labels[label], all_preds[label])
        for label in LABELS
    }

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
            val_loss_saver.save(epoch, n_iter, loss)
            print(f'epoch {epoch}, valid {n_iter}: loss={loss:.4f}')
            print({label: f"{fscore:.4f}"for label, fscore in f1.items()})

        if n_iter % save_per_iter == 0:
            path = os.path.join(SAVE_CKPT_DIR, MODEL_PREFIX + f'{n_iter}/')
            print(f'save model at {path}')
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)
        
        n_iter += 1
    
    train_loss_saver.plot()
    val_loss_saver.plot()
