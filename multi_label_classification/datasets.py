import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

class MultiLabelDataset(Dataset):
    def __init__(self, tokenizer, df, max_len=512, for_train=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.for_train = for_train

        self.texts = []
        self.labels = []
        for _, row in df.iterrows():
            self.texts.append(row['comment_text'])
            if for_train:
                self.labels.append([row[col] for col in LABELS])

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.tokenize(text)
        tokens = tokens[:self.max_len - 2]
        processed_tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(processed_tokens))
        token_type_ids = torch.tensor([0] * len(processed_tokens))
        attention_mask = torch.tensor([1] * len(processed_tokens))

        outputs = (input_ids, token_type_ids, attention_mask)

        if self.for_train:
            label = self.labels[idx]
            label = torch.tensor(label)
            outputs += (label, )

        return outputs

    def __len__(self):
        return len(self.texts)

    def create_mini_batch(self, samples):
        outputs = list(zip(*samples))

        # zero pad 到同一序列長度
        input_ids = pad_sequence(outputs[0], batch_first=True)
        token_type_ids = pad_sequence(outputs[1], batch_first=True)
        attention_mask = pad_sequence(outputs[2], batch_first=True)

        batch_output = (input_ids, token_type_ids, attention_mask)
    
        if self.for_train:
            labels = torch.stack(outputs[3])
            batch_output += (labels, )

        return batch_output
