## SBERT-WK: A Sentence Embedding Method By Dissecting BERT-based Word Models
<p align="center">
<img src="figure1.png" alt="Paris" class="center" width="500">
</p>

SBERT-WK provides a way to generate sentence embedding by dissecting deep contextualized models. Because pre-trained language models are quite powerful in a wide range of NLP tasks, but how to generate sentence embedding from deep language models is still challenging. Deep models mostly provide word/token level representation. Previous approaches includes averaging token representations or use CLS tokens provides rather poor performance in either textual similarity tasks, clustering and supervised tasks. Through geometric analysis, our model is capable in finding salient components in representation arocss layers and unified token representations. We evaluate our approach on a wide range of tasks and showed its effectiveness.

Our model is applicable to any deep contextualized models and requires no further training. Details of our method can be found in our publication: [SBERT-WK](https://arxiv.org/abs/2002.06652).


| Section | Description |
|-|-|
| [Installation](#Installation) 									| How to setup the environment  	|
| [Support Architecture](#Support-Architecture) 					| Current support architectures		|
| [Quick Usage Guide](#Quick-Usage-Guide)							| A quick guide 					|
| [Reproduce the result](#Reproduce-the-result)						| Reproduce the result of paper     |
| [Performance](#Performance)										| Performance Comparison		    |
| [Citation](#Citation)												| Reference Link		   		 	|
| [Acknowledge](#Acknowledge)										| Acknowledge		   		 		|



## Installation
We are using Python 3.7 and the model is implemented with Pytorch 1.3.
We also use [transformers v2.2.2](https://github.com/huggingface/transformers)

```
    conda create -n SBERT-WK python=3.7
    conda install numpy
    conda install pytorch=1.3 torchvision cudatoolkit=10.1 -c pytorch
    pip install transformers==2.2.2
    conda install -c anaconda scikit-learn
```

## Support Architecture

### Released Architectures
1. bert-base-uncased:
1. **Released** bert-base-nli:

### Under-preparation
1. bert-base-nli-stsb:
1. bert-large-uncased:
1. bert-large-nli:
1. bert-large-nli-stsb:
1. roberta-base
1. xlnet-base-cased
1. bert-large:
1. bert-large-nli:



## Quick Usage Guide


## Reproduce the result

## Performance

## Citation

If you find our model is useful in your research, please consider cite our paper: [SBERT-WK: A Sentence Embedding Method By Dissecting BERT-based Word Models](https://arxiv.org/abs/2002.06652):

``` 
@article{SBERT-WK,
    title = {{SBERT-WK}: A Sentence Embedding Method By Dissecting BERT-based Word Models},
    author = {Wang, Bin and Kuo, C-C Jay},
    journal={arXiv preprint arXiv:2002.06652},
    year={2020}
}
```

Contact person: Bin Wang, bwang28c@gmail.com

http://mcl.usc.edu/


## Acknowledge

Many thanks for 

1. [Transformer repo](https://github.com/huggingface/transformers)
2. [Sentence-BERT repo](https://github.com/UKPLab/sentence-transformers) in providing pretained models and easy to use architecture.
3. Thanks for SentEval for evlaluation toolkit. [SentEval](https://github.com/facebookresearch/SentEval)







------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Get Started to Duplicate the result from the Paper

**First**, clone our code.

**Second**, download [SentEval](https://github.com/facebookresearch/SentEval) and put it in the folder and rename it 
```
    ./SBERT-WK-Sentence-Embedding-master/SentEval
```
Download data for STS and downstream tasks
```
    ./SBERT-WK-Sentence-Embedding-master/SentEval/data/downstream/get_transfer_data.bash
```

**Third**, run our bash file to reproduce the result.
```
    python SBERT_WK.py \
        --model_type 'binwang/bert-base-nli' \
        --embed_method 'dissecting'  \
        --max_seq_length 128 \
        --batch_size 64 \
        --context_window_size 2 \
        --tasks 'sts' \
```
We have shared 7 models from the https://huggingface.co/models. All the models can be easily accessed by changing the model_type in the above command.
```
    --model_type 'binwang/bert-base-uncased'    # Original BERT Model                        (12 layers)
    --model_type 'binwang/roberta-base'         # Original RoBERTa Model                     (12 layers)
    --model_type 'binwang/xlnet-base-cased'     # Original XLNET Model                       (12 layers)
    --model_type 'binwang/bert-base-nli' #      # BERT Model finetuned on NLI data           (12 layers)
    --model_type 'binwang/bert-base-nli-stsb'   # BERT Model finetuned on NLI and STSB data  (12 layers)
    --model_type 'binwang/bert-large-nli'       # Large BERT finetuned on NLI data           (24 layers)
    --model_type 'binwang/bert-large-nli-stsb'  # Large BERT finetuned on NLI and STSB data  (24 layers)
```
The way to obtain the sentence embedding from the deep contextualized model can be two ways:
```
    --embed_method 'dissecting'
    --embed_method 'ave_last_hidden'
```
Choose tasks to evaluate on:
```
    --tasks 'sts'
    --tasks 'supervised'
    --tasks 'probing'
    --tasks 'all'
```

## Performance on STS tasks (Pearson)

|    Model        | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R |
|-----------------|-------|-------|-------|-------|-------|-------|--------| 
| Avg. GloVe      | 52.22 | 49.60 | 54.60 | 56.26 | 51.41 | 64.79 | 79.92  | 
| InferSent       | 59.33 | 58.85 | 69.57 | 71.26 | 71.46 | 75.74 | 88.35  |
| USE             | 61.00 | 64.00 | 71.00 | 74.00 | 74.00 | 78.00 | 86.00  |
| BERT - CLS      | 27.58 | 22.52 | 25.63 | 32.11 | 42.69 | 52.14 | 70.05  |
| Avg. BERT       | 46.87 | 52.77 | 57.15 | 63.47 | 64.51 | 65.22 | 80.54  |
| Sen-BERT (base-nli)| 64.61 | 67.54 | 73.22 | 74.34 | 70.13 | 74.09 | 84.23  |

|    **SBERT-WK**    | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R |
|--------------------|-------|-------|-------|-------|-------|-------|--------| 
| bert-base-uncased  |
| bert-base-nli      | 70.22 | 68.13 | 75.46 | 76.94 | 74.51 | 80.00 | 87.38  |
| bert-large-nli     | 
|--------------------|-------|-------|-------|-------|-------|-------|--------| 
| bert-base-nli-stsb | 75.53 | 76.34 | 88.62 | 83.06 | 80.96 | 83.02 | 87.79  |
| bert-large-nli-stsb|




## Acknowledge

Many thanks for [Transformer repo](https://github.com/huggingface/transformers) and [Sentence-BERT repo](https://github.com/UKPLab/sentence-transformers) in providing pretained models and easy to use architecture.
Thanks for SentEval for evlaluation toolkit. [SentEval](https://github.com/facebookresearch/SentEval)


## TODO:
Finished Performance Part.

Provide an convenient interface for sentence embedding.
