# bert_finetuning
bert 를 korquad 에 맞게 훈련시켜보자.

## Bert
- Dataset: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
  - includes: vocab.txt + bert_config.json

- Process
  - Collect dataset
  - Make vocab
  - Make tfrecord data
  - Pretraining

## Korquad v1


## Reference
- bert paper[https://arxiv.org/abs/1810.04805]
- korquad v1[https://korquad.github.io/category/1.0_KOR.html]
- Bert 를 카톡 데이터를 통해서 학습시켜보자.[https://blog.pingpong.us/dialog-bert-pretrain/]

## Dataset
- Wikipedia dump English: https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
- WikiExtractor: https://github.com/attardi/wikiextractor
