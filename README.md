# Multiplex Graph Neural Network for Extractive Text Summarization
This is the PyTorch implementation of the paper:

Baoyu Jing, Zeyu You, Tao Yang, Wei Fan and Hanghang Tong [Multiplex Graph Neural Network for Extractive Text Summarization](https://arxiv.org/pdf/2108.12870.pdf), EMNLP'2021 

## Requirements
- torch>=1.7.1
- stanza==1.1.1
- pyrouge==0.1.3
- numpy>=1.19.5
- scipy>=1.5.4
- PyYAML>=6.0 
- sklearn==0.0
- six >= 1.16.0
- tqdm>=4.59.0

Packages can be installed via: ```pip install -r requirements.txt```


## Data Preparation
The preprocessed CNN/DailyMail dataset can be downloaded [here](https://www.dropbox.com/s/c4u6m03m43sgc7h/cnn_dailymail_processed.zip?dl=0).
If you would like to process the raw data by yourself, please follow the instructions below.

### 1. *Dataset*
The raw data for ```CNN/Daily_Mail``` can be downloaded from <https://cs.nyu.edu/~kcho/DMQA/>. 
Data splits can be downloaded from <https://github.com/abisee/cnn-dailymail>, which is already included in this repository.
### 2. *Preprocess*

* *Tokenization*

We follow [Get To The Point: Summarization with Pointer-Generator Networks](https://github.com/abisee/pointer-generator) and use Stanford CoreNLP to tokenize the dataset. 
We use [stanza](https://stanfordnlp.github.io/stanza/index.html) to access CoreNLP. 
Here's the [instruction](https://stanfordnlp.github.io/stanza/corenlp_client.html).


* *Graph Construction*

We build multiplex graphs at both word level and sentence level. 
At word level, we consider the ```syntactic``` and ```semantic``` relations. 
At sentence level, we consider the ```natural connection (same keywords)``` and ```semantic``` relations. 
The semantic graphs are computed on-the-fly within the model, and the other graphs are constructed during preprocessing.

The  ```syntactic``` graph for words within a sentence are constructed based on the dependency graph of the sentence, which is obtained from Stanford CoreNLP. 

The ```natural connection``` graph for sentences are constructed based on their TF-IDF vectors.

* *Oracle Extraction*

We follow [Text Summarization with Pretrained Encoders](https://github.com/nlpyang/PreSumm), and greedily select the sentences within the documents, which have the highest ROUGE scores, as oracles.

### 3. *Others*
We use [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings as the initial word embeddings.
Since we use different numbers of the extracted sentences as summaries for CNN and DailyMail, we need to know whether a document is from CNN or DailyMail during evaluation.
Therefore, you need to run ```cnn_split.py``` to obtain the split files of CNN.

## ROUGE Package
Following previous works, we use ```ROUGE-1.5.5``` to evaluate the model. 
For efficiency, we use ```pyrouge``` to calculate ROUGE scores when extracting oracles.

1. *Download ROUGE-1.5.5 [here](https://github.com/andersjo/pyrouge/tree/master/tools/ROUGE-1.5.5)*

```export ROUGE_EVAL_HOME="/absolute_path_to/ROUGE-1.5.5/data/"```

2. *Install Perl Packages*

```
sudo apt-get install perl
sudo apt-get update
sudo cpan install XML::DOM
```

3. *Remove files to avoid ERROR of the .db files*

```
rm WordNet-2.0.exc.db
./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
```

4. *Install ```pyrouge```*

```
pip install pyrouge
pyrouge_set_rouge_path /absolute_path_to/ROUGE-1.5.5/
```

## Training
1. Specify the configurations in ```model.yml```, ```dataloader.yml```, ```trainer.yml``` and ```vocabulary.yml```.
2. Specify the paths of the configuration files in ```trainer.py```.

```python train.py```

## Evaluation
1. Specify the configurations in ```model.yml```, ```dataloader.yml``` and ```vocabulary.yml```.
2. Specify the paths of the configuration files in ```evaluate.py```.

```python evaluate.py```

## Citation
Please cite the following paper, if you find the repository or the paper useful.

Baoyu Jing, Zeyu You, Tao Yang, Wei Fan and Hanghang Tong [Multiplex Graph Neural Network for Extractive Text Summarization](https://arxiv.org/pdf/2108.12870.pdf), EMNLP'2021 

```
@article{jing2021multiplex,
  title={Multiplex Graph Neural Network for Extractive Text Summarization},
  author={Jing, Baoyu and You, Zeyu and Yang, Tao and Fan, Wei and Tong, Hanghang},
  journal={arXiv preprint arXiv:2108.12870},
  year={2021}
}
```
