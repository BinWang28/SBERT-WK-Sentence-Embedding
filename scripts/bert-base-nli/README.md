## Model Explained

First, the model is one Pre-trained 12 layer BERT-based models. To increase the model's capability on sentence-level tasks, it is then fine-tuned by Siemease Network with Natural Language Infernece (NLI) data.

Released with paper [Sen-BERT](https://arxiv.org/abs/1908.10084)

## Reproduce our result based on this model:
(in the master directory)

(Best on STS datasets)
```
./scripts/sentence-bert/bert-base-nli.sh
```
(Best on Supervised datasets)
```
./scripts/sentence-bert/bert-base-nli2.sh
```


## Reference Results

### Results on STS dataset

|    Model                 | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg.   |
|--------------------------|-------|-------|-------|-------|-------|-------|--------|--------| 
| Avg. GloVe               | 52.22 | 49.60 | 54.60 | 56.26 | 51.41 | 64.79 | 79.92  |  58.40 |
| InferSent                | 59.33 | 58.85 | 69.57 | 71.26 | 71.46 | 75.74 | 88.35  | 70.65  |
| USE                      | 61.00 | 64.00 | 71.00 | 74.00 | 74.00 | 78.00 | 86.00  | 72.57  |
| BERT - CLS               | 27.58 | 22.52 | 25.63 | 32.11 | 42.69 | 52.14 | 70.05  | 38.96  |
| Avg. BERT                | 46.87 | 52.77 | 57.15 | 63.47 | 64.51 | 65.22 | 80.54  | 61.50  |
| SBERT-WK (bert-base-nli) | 70.22 | 68.13 | 75.46 | 76.94 | 74.51 | 80.00 | 87.38  | 76.09  |

### Result on Supervised Tasks

|    Model                 |   MR  |  CR   |  SUBJ |  MPQA | SST-2 |  TREC |   MRPC | SICK-Entailment | Avg.   |
|--------------------------|-------|-------|-------|-------|-------|-------|--------|-----------------|--------|
| Avg. GloVe               | 77.90 | 79.00 | 91.40 | 87.80 | 81.40 | 83.40 | 73.20  | 79.20           | 81.66  |
| InferSent                | 81.80 | 86.60 | 92.50 | 90.00 | 84.20 | 89.40 | 75.00  | 86.70           | 75.78  |
| USE                      | 80.20 | 86.00 | 93.70 | 87.00 | 86.10 | 93.80 | 72.30  | 83.30           | 85.30  |
| BERT - CLS               | 82.30 | 86.90 | 95.40 | 88.30 | 86.90 | 93.80 | 72.10  | 73.80           | 84.94  |
| Avg. BERT                | 81.70 | 86.80 | 95.30 | 87.80 | 86.70 | 91.60 | 72.50  | 78.20           | 85.08  |
| SBERT-WK (bert-base-nli) | 00.00 | 00.00 | 00.00 | 00.00 | 00.00 | 80.00 | 87.38  | 00              | 00     |

## Transformer Page Reference

The pretrained model is available in the Transformer [Page](https://huggingface.co/binwang/bert-base-nli)