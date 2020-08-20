# Bert Finetuning
bert 를 Fine tuning 해보자

## Bert
- Dataset: https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
  - includes: vocab.txt + bert_config.json + ckpt

- Process
  - Collect dataset
  - Make vocab
  - Make tfrecord data
  - Pretraining
  - Fine tuning

```sh
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

```sh
python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

## SQuAD

### v1.1
Site: 

```sh
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

### v2.0
Site: 

```sh
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True
```

## Korquad
- Dataset:  https://korquad.github.io/

```sh
python run_squad.py \
    --bert_config_file "pretrained 모델 폴더의 bert_config.json" \
    --vocab_file "pretrained 모델 폴더의 vocab.txt" \
    --output_dir "훈련된 모델이 저장될 폴더" (prediction 결과도 이 폴더에 저장된다.) \
    --do_train (훈련을 하겠다는 옵션) \
    --train_file "KorQuAD  데이터셋 폴더의 KorQuAD_v1.0_train.json" \
    --do_predict (predict 하겠다는 옵션) \
    --predict_file "KorQuAD 데이터셋 폴더의 KorQuAD_v1.0_dev.json" \
    --do_lower_case=false (현재 다운받은 Cased 모델은 이 옵션을 적용하지 않는다.) \
    --max_seq_length 적당히 \
    --train_batch_size 적당히 \
    --init_checkpoint "pretrained 모델 폴더" \
```



## Classification Job

### cola: The Corpus of Linguistic Acceptability
```sh
CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
  --task_name=cola \
  --do_train=true \
  --do_eval=true \
  --data_dir=./dataset \
  --vocab_file=./model/vocab.txt \
  --bert_config_file=./model/bert_config.json \
  --init_checkpoint=./model/bert_model.ckpt \
  --max_seq_length=64 \
  --train_batch_size=2 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=./bert_output/ \
  --do_lower_case=False \
  --save_checkpoints_steps 10000
```


## Dataset
- [Wikipedia dump English][https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2]
- [WikiExtractor][https://github.com/attardi/wikiextractor]

## Reference
- [Bert paper][https://arxiv.org/abs/1810.04805]
- [KorQuAD v1][https://korquad.github.io/category/1.0_KOR.html]
- [KorQuAD submition tutorial][https://worksheets.codalab.org/worksheets/0xf9b9efaff9c641a7a48290fe4c94d593]

### Training
- [Bert 를 카톡 데이터를 통해서 학습시켜보자.][https://blog.pingpong.us/dialog-bert-pretrain/]

### Fine Tuning
- [Bert finetuning for classification][https://towardsdatascience.com/beginners-guide-to-bert-for-multi-classification-task-92f5445c2d7c]

- [Bert Multilingual 을 이용해서 KorQuAD 수행해보기][http://mlgalaxy.blogspot.com/2019/01/bert-multilingual-model-korquad-part-1.html]
- [KorQuAD 수행하고 제출하기][http://mlgalaxy.blogspot.com/2019/02/]

