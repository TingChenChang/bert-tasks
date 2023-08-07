import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import tokenize_and_map

LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

class NerDataset(Dataset):
    def __init__(self, tokenizer, texts, tag_lists=None, max_len=512, for_train=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.for_train = for_train

        self.texts = texts
        self.tag_lists = tag_lists

    def __getitem__(self, idx):
        text = self.texts[idx].lower()

        tokens, index_map = tokenize_and_map(self.tokenizer, text)

        cut_index = self.max_len - 2
        if cut_index < len(tokens):
            cut_text_index = index_map.index(cut_index)
            tokens = tokens[:cut_index]
            text = text[:cut_text_index]
            index_map = index_map[:cut_text_index]

        processed_tokens = ['[CLS]'] + tokens + ['[SEP]']

        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(processed_tokens))
        token_type_ids = torch.tensor([0] * len(processed_tokens))
        attention_mask = torch.tensor([1] * len(processed_tokens))

        outputs = (input_ids, token_type_ids, attention_mask)

        if self.for_train:
            labels = []

            tag_list = self.tag_lists[idx]
            for tag, token_index in zip(tag_list, index_map):
                if token_index is None:
                    continue
                if token_index >= len(labels):
                    labels.append(LABELS.index(tag))

            labels = [0] + labels + [0]  # for [CLS] and [SEP]
            labels = torch.tensor(labels)

            assert labels.size(0) == input_ids.size(0)
            outputs += (labels, )

        info = {
            'text': text,
            'tokens': tokens,
            'index_map': index_map
        }
        outputs += (info, )
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
            labels = pad_sequence(outputs[3], batch_first=True)
            batch_output += (labels, )
        else:
            infos = outputs[3]
            batch_output += (infos, )

        return batch_output
