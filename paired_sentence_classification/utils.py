import re
import os
import pandas as pd
import matplotlib.pyplot as plt

class RunningAverage:
    def __init__(self):
        self.values = []

    def add(self, val):
        self.values.append(val)

    def add_all(self, vals):
        self.values += vals

    def get(self):
        return sum(self.values) / len(self.values)

    def flush(self):
        self.values = []
        
class LossHistorySaver():
    def __init__(self, path: str, eval_metric: str = None):
        self.path = path
        self.eval_metric = eval_metric
        
        if not os.path.isdir(os.path.dirname(self.path)):
            os.makedirs(os.path.dirname(self.path))
        
        if not os.path.isfile(self.path):
            with open(self.path, 'w') as f:
                f.write(','.join(['epoch', 'iteration', 'loss', f'{eval_metric}']))
                f.write('\n')
    
    def save(self, epoch, iteration, loss, eval_metric=None):
        with open(self.path, 'a') as f:
            f.write(','.join([f'{epoch}', f'{iteration}', f'{loss}', f'{eval_metric}']))
            f.write('\n')
            
    def plot(self):
        data = pd.read_csv(self.path, dtype=float)\
            .drop(columns=['epoch'])\
            .set_index('iteration')
        
        if not self.eval_metric:
            data.drop(columns=[f'{self.eval_metric}'], inplace=True)
            
        data.plot(subplots=True, kind='line')
        plt_path = self.path.split('.')[0] + '.png'
        plt.savefig(plt_path)

def wordize_and_map(text):
    words = []
    index_map_from_text_to_word = []
    while len(text) > 0:
        match_space = re.match(r'^ +', text)
        if match_space:
            space_str = match_space.group(0)
            index_map_from_text_to_word += [None] * len(space_str)
            text = text[len(space_str):]
            continue

        match_en = re.match(r'^[a-zA-Z0-9]+', text)
        if match_en:
            en_word = match_en.group(0)
            index_map_from_text_to_word += [len(words)] * len(en_word)
            words.append(en_word)
            text = text[len(en_word):]
        else:
            index_map_from_text_to_word += [len(words)]
            words.append(text[0])
            text = text[1:]
    return words, index_map_from_text_to_word


def tokenize_and_map(tokenizer, text):
    words, index_map_from_word = wordize_and_map(text)

    tokens = []
    index_map_from_text_to_token = []
    while len(index_map_from_word) > 0:
        if index_map_from_word[0] is None:
            index_map_from_text_to_token.append(None)
            del index_map_from_word[0]
        else:
            word = words.pop(0)
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) == 0 or word_tokens == ['[UNK]']:
                index_map_from_text_to_token += [len(tokens)] * len(word)
                tokens.append('[UNK]')
                del index_map_from_word[:len(word)]
            else:
                for word_token in word_tokens:
                    word_token_len = len(re.sub(r'^##', '', word_token))
                    index_map_from_text_to_token += [len(tokens)] * word_token_len
                    tokens.append(word_token)
                    del index_map_from_word[:word_token_len]

    return tokens, index_map_from_text_to_token
