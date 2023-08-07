import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from utils import tokenize_and_map

class QADataset(Dataset):
    def __init__(self, tokenizer, queries, contents, indexes_in_content=None, max_len=512, for_train=True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.for_train = for_train

        self.queries = queries
        self.contents = contents
        self.indexes_in_content = indexes_in_content

    def __getitem__(self, idx):
        query = self.queries[idx].lower()
        content = self.contents[idx].lower()

        query_tokens, query_index_map = tokenize_and_map(self.tokenizer, query)
        content_tokens, content_index_map = tokenize_and_map(self.tokenizer, content)

        cut_index = self.max_len - len(query_tokens) - 3
        if cut_index < len(content_tokens):
            cut_text_index = content_index_map.index(cut_index)
            content_tokens = content_tokens[:cut_index]
            content = content[:cut_text_index]
            content_index_map = content_index_map[:cut_text_index]

        processed_tokens = ['[CLS]'] + query_tokens + ['[SEP]'] + content_tokens + ['[SEP]']

        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(processed_tokens))
        token_type_ids = torch.tensor([0] * (2 + len(query_tokens)) + [1] * (1 + len(content_tokens)))
        attention_mask = torch.tensor([1] * len(processed_tokens))

        outputs = (input_ids, token_type_ids, attention_mask)

        offset = 2 + len(query_tokens)
        if self.for_train:
            start_index_in_content, end_index_in_content = self.indexes_in_content[idx]

            if end_index_in_content >= len(content):
                # end_index is out of max_len => no ans
                start_index_in_content = -1
                end_index_in_content = -1

            start_index = offset + content_index_map[start_index_in_content]
            end_index = offset + content_index_map[end_index_in_content]
            
            start_index, end_index = torch.tensor(start_index), torch.tensor(end_index)
            outputs += (start_index, end_index, )

        content_info = {
            'text': content,
            'tokens': content_tokens,
            'index_map': content_index_map,
            'offset': offset
        }
        outputs += (content_info, )
        return outputs

    def __len__(self):
        return len(self.queries)

    def create_mini_batch(self, samples):
        outputs = list(zip(*samples))

        # zero pad 到同一序列長度
        input_ids = pad_sequence(outputs[0], batch_first=True)
        token_type_ids = pad_sequence(outputs[1], batch_first=True)
        attention_mask = pad_sequence(outputs[2], batch_first=True)

        batch_output = (input_ids, token_type_ids, attention_mask)
    
        if self.for_train:
            start_indexes = torch.stack(outputs[3])
            end_indexes = torch.stack(outputs[4])
            batch_output += (start_indexes, end_indexes, )
        else:
            content_infos = outputs[3]
            batch_output += (content_infos, )

        return batch_output
