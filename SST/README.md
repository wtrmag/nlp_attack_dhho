# Running instructions on SST-2 dataset

## Data

- Download SST-2 dataset: [sst-2.zip](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/files/?p=%2Fsst-2.zip)
- Download processed SST-2 data for training
  models: [SST2data.zip](https://cloud.tsinghua.edu.cn/d/b6b35b7b7fdb43c1bf8c/files/?p=%2FSST2data.zip)

## Process Data and Train Model

- Process SST-2 Data

```bash
python data_utils.py
```

- Generate Candidate Substitution Words

```bash
python gen_pos.py
python lemma.py
python gen_candidates.py
```

- Train BiLSTM Model

```bash
python train_model.py
```

- Train BERT Model

```bash 
python SST_BERT.py
```

## Craft Adversarial Examples

- Crafting Adversarial Examples for Bi-LSTM

```bash
python AD_dpso_sem.py
```

- Crafting Adversarial Examples for BERT

```bash
python AD_dpso_sem_bert.py
```
