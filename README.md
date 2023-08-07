# Bert Tasks

## Single Sentence Classification

- **Task**

    Sentiment Classification

- **Dataset**

    `data/chinese_sentiment_classification.csv` from [bigboNed3/chinese_text_cnn](https://github.com/bigboNed3/chinese_text_cnn)

- **Model**

    [transformers.BertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification)

## Paired Sentenece Classification

- **Dataset**

    `data/afqmc_train.json` from [CLUEbenchmark/CLUE](https://github.com/CLUEbenchmark/CLUE)

- **Model**

    [transformers.BertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification)

## Token Classification

- **Task**

    Named Entity Recognition

- **Dataset**

    `data/msra_train_bio.txt` from [OYE93/Chinese-NLP-Corpus](https://github.com/OYE93/Chinese-NLP-Corpus)

- **Model**

    [transformers.BertForTokenClassification](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForTokenClassification)

## Question Answering

- **Task**

    QA

- **Dataset**

    `data/DRCD_training.json` from [DRCKnowledgeTeam/DRCD](https://github.com/DRCKnowledgeTeam/DRCD)

- **Model**

    [transformers.BertForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForQuestionAnswering)


## Multi Labels Classification

- **Task**

    Sentence Multi Labels Classification

- **Dataset**

    `data/toxic_comment_classification.csv` from [kaggle: jigsaw-toxic-comment-classification-challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data?select=train.csv.zip)

- **Model**

    [transformers.BertForPreTraining](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForPreTraining)
