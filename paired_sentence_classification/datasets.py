import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class PairSentenceDataset(Dataset):
    def __init__(self, tokenizer, text_pairs, labels=None, max_len=512, for_train=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.for_train = for_train

        self.text_pairs = text_pairs
        self.labels = labels

    def __getitem__(self, idx):
        text_1, text_2 = self.text_pairs[idx]
        
        text_1 = text_1.lower()
        text_2 = text_2.lower()

        tokens_1 = self.tokenizer.tokenize(text_1)
        tokens_2 = self.tokenizer.tokenize(text_2)

        tokens_1, tokens_2 = self._cut_tokens_pair(tokens_1, tokens_2)

        processed_tokens = ['[CLS]'] + tokens_1 + ['[SEP]'] + tokens_2 + ['[SEP]']

        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(processed_tokens))
        token_type_ids = torch.tensor([0] * (2 + len(tokens_1)) + [1] * (1 + len(tokens_2)))
        attention_mask = torch.tensor([1] * len(processed_tokens))

        outputs = (input_ids, token_type_ids, attention_mask)

        if self.for_train:
            label = self.labels[idx]
            label = torch.tensor(label)

            outputs += (label, )

        return outputs

    def _cut_tokens_pair(self, tokens_1, tokens_2):
        diff = (len(tokens_1) + len(tokens_2)) - (self.max_len - 3)
        if diff > 0:
            half_diff = int(np.ceil(diff / 2))
            tokens_1 = tokens_1[:-half_diff]
            tokens_2 = tokens_2[:-half_diff]
        return tokens_1, tokens_2

    def __len__(self):
        return len(self.text_pairs)

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
