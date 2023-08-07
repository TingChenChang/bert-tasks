import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering
from tqdm.notebook import tqdm

from datasets import QADataset

# CKPT_MODEL = 'models/'
CKPT_MODEL = 'TingChenChang/chinese-question-answering'
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

tokenizer = BertTokenizer.from_pretrained(CKPT_MODEL)

queries = [
    '陸特和漢斯雷頓開創了哪一地區對梵語的學術研究？',
    '「北京皇家祭壇—天壇」在哪一年的時候，正式被列為世界文化遺產?'
]
contents = [
    '在歐洲，梵語的學術研究，由德國學者陸特和漢斯雷頓開創。後來威廉·瓊斯發現印歐語系，也要歸功於對梵語的研究。此外，梵語研究，也對西方文字學及歷史語言學的發展，貢獻不少。1786年2月2日，亞洲協會在加爾各答舉行。會中，威廉·瓊斯發表了下面這段著名的言論：「梵語儘管非常古老，構造卻精妙絕倫：比希臘語還完美，比拉丁語還豐富，精緻之處同時勝過此兩者，但在動詞詞根和語法形式上，又跟此兩者無比相似，不可能是巧合的結果。這三種語言太相似了，使任何同時稽考三者的語文學家都不得不相信三者同出一源，出自一種可能已經消逝的語言。基於相似的原因，儘管缺少同樣有力的證據，我們可以推想哥德語和凱爾特語，雖然混入了迥然不同的語彙，也與梵語有著相同的起源；而古波斯語可能也是這一語系的子裔。」',
    '北京天壇位於北京市東城區，是明清兩朝帝王祭天、祈穀和祈雨的場所。是現存中國古代規模最大、倫理等級最高的祭祀建築群。1961年，天壇被中華人民共和國國務院公布為第一批全國重點文物保護單位之一。1998年，「北京皇家祭壇—天壇」被列為世界文化遺產。北京天壇最初為明永樂十八年仿南京城形制而建的天地壇，嘉靖九年實行四郊分祀制度後，在北郊覓地另建地壇，原天地壇則專事祭天、祈穀和祈雨，並改名為天壇。清代基本沿襲明制，在乾隆年間曾進行過大規模的改擴建，但年門和皇乾殿是明代建築而無改建除外。1900年八國聯軍進攻北京時，甚至還把司令部設在這裡，並在圜丘壇上架設大炮，攻擊正陽門和紫禁城，聯軍們將幾乎所有的陳設和祭器都席捲而去。1912年中華民國成立後，除了中華民國大總統袁世凱在1913年冬至祭天外，天壇不再進行任何祭祀活動。1918年起闢為公園，正式對民眾開放。目前園內古柏蔥鬱，是北京城南的一座大型園林。'
]

pred_dataset = QADataset(tokenizer, queries, contents, for_train=False)

pred_loader = DataLoader(
    dataset=pred_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=pred_dataset.create_mini_batch,
)

model = BertForQuestionAnswering.from_pretrained(CKPT_MODEL)
model.to(device)

answers = []
with torch.no_grad():
    for data in tqdm(pred_loader, desc='predict'):
        input_ids, token_type_ids, attention_mask = [d.to(device) for d in data[:3]]
        content_infos = data[3]

        outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        for start_logit, end_logit, content_info in zip(outputs.start_logits,
                                                        outputs.end_logits,
                                                        content_infos):
            offset = content_info['offset']
            index_map = content_info['index_map']
            text = content_info['text']
            answer_token_start = start_logit.argmax(dim=-1) - offset
            answer_token_end = end_logit.argmax(dim=-1) - offset
            answer_start = index_map.index(answer_token_start)
            answer_end = index_map.index(answer_token_end) + 1
            if answer_start > answer_end or answer_start <= 0 or answer_end <= 0:
                answer = ''
            else:
                answer = text[answer_start:answer_end]
            answers.append(answer)

print('predict result: ')
for answer in answers:
    print('answer:', answer)
