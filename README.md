# Using Locally Learnt Word Representations for better Textual Anomaly Detection
This repository provides the implementations of different Anomaly Detection Methods using various textual representations as described in our paper "Using Locally Learnt Word Representations for better Textual Anomaly Detection".

## Citation

You can find our paper at [https://aclanthology.org/2024.insights-1.11/](https://aclanthology.org/2024.insights-1.11/)

If you find our work useful, please also cite the paper:
```
@inproceedings{breidenstein-labeau-2024-using,
    title = "Using Locally Learnt Word Representations for better Textual Anomaly Detection",
    author = "Breidenstein, Alicia and Labeau, Matthieu",
    booktitle = "Proceedings of the Fifth Workshop on Insights from Negative Results in NLP",
    month = jun,
    year = "2024",
    pages = "82--91"
}
```


## Abstract
The literature on general purpose textual Anomaly Detection is quite sparse, as most textual anomaly detection methods are implemented as out of domain detection in the context of pre-established classification tasks. Notably, in a field where pre-trained representations and models are of common use, the impact of the pre-training data on a task that lacks supervision has not been studied. In this paper, we use the simple setting of k-classes out anomaly detection and search for the best pairing of representation and classifier. We show
that well-chosen embeddings allow a simple anomaly detection baseline such as OC-SVM to achieve similar results and even outperform deep state-of-the-art models.

## References
This repository uses (and extends) code provided with these articles :
- [Self-Attentive, Multi-Context One-Class Classification for Unsupervised Anomaly Detection on Text](https://aclanthology.org/P19-1398) (Ruff et al., ACL 2019) : [https://github.com/lukasruff/CVDD-PyTorch](https://github.com/lukasruff/CVDD-PyTorch)
- [DATE: Detecting Anomalies in Text via Self-Supervision of Transformers](https://aclanthology.org/2021.naacl-main.25) (Manolache et al., NAACL 2021) : [https://github.com/bit-ml/date?tab=readme-ov-file#date](https://github.com/bit-ml/date?tab=readme-ov-file#date)


## Installation
This code is written in `Python 3.9` and requires the packages listed in `requirements.txt`.

Clone the repository to your machine and directory of choice:
```
git clone <link of this repository>
```

To run the code, we recommend setting up a virtual environment, e.g. using `virtualenv` or `conda`:

### Creating an environment from an environment.yml file (works usually the best)
You can also create a virtual environment using the environment.yml file provided in this repository (to change the name of the environment, change the name at the top of this file):
```
conda env create -f environment.yml
conda activate TextualAD_env
```

### `virtualenv`
```
# pip install virtualenv
cd <path-to-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### `conda`
```
cd <path-to-directory>
conda create --name myenv
source activate myenv
while read requirement; do conda install -n myenv --yes $requirement; done < requirements.txt
```

After installing the packages, run `python -m spacy download en` and `python -m spacy download fr` to download the [spaCy](https://spacy.io/) `en` and `fr`
library.


## Running experiments

### 1. Download and format the datasets :
The following datasets are used :
- [`20 Newsgroups`](http://qwone.com/~jason/20Newsgroups/)
In the following, the indexation of classes is `[0, 1, 2, 3, 4, 5]` for `['comp', 'misc', 'pol', 'rec', 'rel', 'sci']`.
- [`AG News`](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)
In the following, the indexation of classes is `[0, 1, 2, 3]` for `['business', 'sci', 'sports', 'world']`.
- [`RNCP`](https://www.data.gouv.fr/en/datasets/repertoire-national-des-certifications-professionnelles-et-repertoire-specifique/) : For RNCP, the data was collected from [data.gouv.fr](https://www.data.gouv.fr/en/datasets/repertoire-national-des-certifications-professionnelles-et-repertoire-specifique/), a website from the french government for public data, on 24/05/2023. The dataset was built by joining the titles (initulé) from the training certifications with their ROME code and with the the categories from the corresponding "arborescence thématique" available here : [https://www.data.gouv.fr/fr/datasets/repertoire-operationnel-des-metiers-et-des-emplois-rome/](https://www.data.gouv.fr/fr/datasets/repertoire-operationnel-des-metiers-et-des-emplois-rome/). We provide the data cleaned in data/rncp.csv.
The classes of the dataset contain numbers so their indexation is the number of the class minus one (to start with 0).
```
cd data
python 20ng.py
python ag.py
python rncp.py
```

### 2. Create contaminated training dataset for unsupervised setting
```
# to be executed in ./data too
python contaminate_ds.py
```

### 3. Generate the word vectors (FastText and PPMI) learnt on the dataset and the classes
```
#This operation may take quite long to get all representations computed (especially for AG News representations). Do not hesitate to comment out some code in create_co_oc.py and create_co_oc_cont.py if you do not need all possible representations.
cd word_vectors_saved
python create_co_oc.py
python create_co_oc_cont.py
```

### 4. Run experiments 
The different anomaly detection algorithms can be run by executing the corresponding `main_[algo_name].py` files. These algorithms are :
- CVDD
- DATE
- OCSVM
- LSA
- TONMF

Each of these models can be run with several representations :
- pre-trained representations from FastText, referred to as `FastText_en` of `FastText_fr` (french representations for RNCP dataset). If running with these representations does not work, go to [5](#5-to-download-pre-trained-word-vectors-for-fasttext).
- FastText word representations learnt from the dataset used for anomaly detection or from the class considered as normal in our k-classes-out anomaly detection setting : `FastText_dataset` or `FastText_class`.
- We also experimented with PPMI representations, obtained by computing the PPMI ([Church and Hanks, 1990](https://aclanthology.org/J90-1003)) matrix, which we reduce to the appropriate dimension using the SVD. These representations are referred to as `PPMI_dataset` and `PPMI_class` in the code.
- As DATE is based on ELECTRA ([Clark et al., 2020](https://openreview.net/forum?id=r1xMH1BtvB)), we also experiment with representations obtained through its off-the-shelf version, and through one pre-trained on the dataset or the class. These representations were only implemented with OCSVM and for comparison purposes. To use them, please go to [4.2.3.](#423-saving-sentence-representations-to-be-used-by-ocsvm) first to compute the corresponding representations. In the code, these representations are referred to as `DATE_dataset`,`DATE_class`,`DATE_class_noRMD` (for date representations without rmd task = representations locally learnt with Electra),`DATE_dataset_noRMD`,`Electra_preTrained` (for Electra pre-trained representations).

In the following, we show several examples to run the different algorithms with some example parameters. For better insights on all the possible arguments and options, please look at the corresponding `main_[algo_name].py` files.

### 4.0 General
The following should be executed first before starting to experiment :
```
# activate virtual environment
source myenv/bin/activate  # or 'source activate myenv' for conda

# change to source directory
cd src
```

### 4.1 CVDD
CVDD with FastText_dataset on 20 newsgroup and with normal class 0. The results and main parameters are written into results.txt.
```
python main_cvdd.py --dataset_name newsgroups20 --net_name cvdd_Net --xp_path ../log/test_newsgroups20 --data_path ../data --results_file results.txt --clean_txt --embedding_size 300 --pretrained_model FastText_dataset --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class 0 --cont 0 --seed 0;
```
CVDD with FastText_en on AgNews and with normal class 1 and contamination of 5% of the training set (unsupervised setting).
```
python main_cvdd.py --dataset_name agnews --net_name cvdd_Net --xp_path ../log/test_agnews --data_path ../data --results_file results.txt --clean_txt --embedding_size 300 --pretrained_model FastText_en --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class 1 --cont 5 --seed 0;
```
CVDD with FastText_fr on RNCP and with normal class 0 and contamination of 20% of the training set (unsupervised setting)
```
python main_cvdd.py --dataset_name rncp --net_name cvdd_Net --xp_path ../log/test_rncp --data_path ../data --results_file results.txt --clean_txt --embedding_size 300 --pretrained_model FastText_fr --ad_score context_dist_mean --n_attention_heads 3 --attention_size 150 --lambda_p 1.0 --alpha_scheduler logarithmic --n_epochs 100 --lr 0.01 --lr_milestone 40 --normal_class 0 --cont 20 --seed 0;
```
If you get this error for running experiments with spacy tokenizer in french on RNCP, go to [6](#6-error-with-spacy).

### 4.2 DATE
#### 4.2.1 Setting
First you have to create the masks used by DATE. The parameters for the masks (number of masks, size of the mask or masking percentage can be changed in `python create_masks.py`).
```
cd src/date

# parameters used for 20 newsgroups in original paper from DATE
python create_masks.py --nb_masks 25 --mask_prob 0.25

# parameters used for agnews in original paper from DATE
python create_masks.py --nb_masks 50 --mask_prob 0.5

# if you want to test other number of masks and masking probabilities, you will need to generate other masks with the function above and other parameters
```
DATE needs a testing file containing all the outliers, and other files for saving representations and training a tokenizer on the dataset (for RNCP) which need to be generated as follows :
```
bash generate_files_for_DATE.sh
```

#### 4.2.2 Running experiments
DATE on 20 newsgroups with normal class 0. The results and main parameters are written into results.txt.
```
# you should be in ./src to execute the following lines
cd ..
python main_date.py --dataset_name newsgroups20 --normal_class 0 --results_file results.txt --nb_masks 25 --mask_prob 0.25 --cont 0
```
DATE on agnews with normal class 1 and contamination contamination of 5% of the training set (unsupervised setting).
```
python main_date.py --dataset_name agnews --normal_class 1 --results_file results.txt --nb_masks 50 --mask_prob 0.5 --cont 5
```
DATE on rncp with normal class 0. For rncp, the tokenizer is not pre-trained (as a french tokenizer would be needed) but trained directly on the whole rncp dataset.
```
python main_date.py --dataset_name rncp --normal_class 0 --results_file results.txt --nb_masks 50 --mask_prob 0.5 --cont 0
```
#### 4.2.3 Saving sentence representations to be used by OCSVM
To better understand the impact of local representations on AD, we experimented using the contextual representations from DATE with an OC-SVM model. These representations are learnt locally on each class of the dataset. To get a document-level representation, we used the [CLS] token. We also experimented using Date representations learnt locally without the additional RMD task (= a loaclly learnt Electra model).

To save DATE representations learnt on 20 newsgroups and rncp with normal class 0 :
```
python main_date.py --dataset_name newsgroups20 --normal_class 0 --results_file results.txt --nb_masks 25 --mask_prob 0.25 --cont 0 --extract_reprs date
python main_date.py --dataset_name rncp --normal_class 0 --results_file results.txt --nb_masks 25 --mask_prob 0.25 --cont 0 --extract_reprs date
```
To save ELECTRA representations learnt on 20 newsgroups on class 0 :
```
python main_date.py --dataset_name newsgroups20 --normal_class 0 --results_file results.txt --nb_masks 25 --mask_prob 0.25 --cont 0 --extract_reprs electra
```
To save date or electra dataset representations (which did not provide good results with OCSVM)
```
python main_date.py --dataset_name agnews --normal_class 0 --results_file results.txt --nb_masks 50 --mask_prob 0.5 --cont 0 --extract_reprs date_dataset

python main_date.py --dataset_name agnews --normal_class 0 --results_file results.txt --nb_masks 50 --mask_prob 0.5 --cont 0 --extract_reprs electra_dataset

```

### 4.3 OCSVM
For OCSVM the algorithm taken from sklearn is supposed to be deterministic. It works with DATE representations with OCSVM, but not for other representations (from FastText or PPMI). We suppose this is due to padding in the implementation taken from the code from [https://github.com/lukasruff/CVDD-PyTorch].

OCSVM with rbf kernel with PPMI_dataset representations on 20 newsgroup and with normal class 0. The results and main parameters are written into results.txt.
```
python main_ocsvm.py --dataset_name newsgroups20 --xp_path ../log/test_newsgroups20 --data_path ../data --results_file results.txt --clean_txt --embedding_size 300 --embedding_reduction 'mean' --pretrained_model PPMI_dataset --normal_class 0 --nu 0.5 --kernel linear --use_tfidf_weights False --tokenizer spacy --cont 0 --seed 0;
```

OCSVM with linear kernel with FastText_dataset representations on AG News, with normal class 0 and contamination contamination of 5% of the training set (unsupervised setting).
```
python main_ocsvm.py --dataset_name agnews --xp_path ../log/test_agnews --data_path ../data --results_file results.txt --clean_txt --embedding_size 300 --embedding_reduction 'mean' --pretrained_model FastText_dataset --normal_class 0 --nu 0.5 --kernel linear --use_tfidf_weights False --tokenizer spacy --cont 5 --seed 0;
```

OCSVM with linear kernel with DATE_class (sentence) representations on RNCP, with normal class 0.
```
python main_ocsvm.py --dataset_name rncp --xp_path ../log/test_rncp --data_path ../data --results_file results.txt --clean_txt --embedding_size 300 --embedding_reduction 'mean' --pretrained_model DATE_class --normal_class 0 --nu 0.5 --kernel linear --use_tfidf_weights False --tokenizer spacy --cont 0 --seed 0;
```

To use Date representations, please go to [4.2.3.](#423-saving-sentence-representations-to-be-used-by-ocsvm) first to compute the corresponding representations. In the code, these representations are referred to as `DATE_dataset`,`DATE_class`,`DATE_class_noRMD` (for date representations without rmd task = representations locally learnt with Electra),`DATE_dataset_noRMD`,`Electra_preTrained` (for Electra pre-trained representations). Our experiments did not show good performances of these representations with OCSVM.


### 4.4 LSA
Example of command for LSA with 20 Newsgroups:
```
python main_lsa.py --dataset_name newsgroups20 --xp_path ../log/test_newsgroups20 --data_path ../data --results_file results.txt --clean_txt --normal_class 0 --n_components 300 --seed 0 --cont 0
```

### 4.5 TONMF
The code used for TONMF was adapted into python from [Outlier Detection for Text Data](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974973.55?ref=https://githubhelp.com) (Ramakrishnan Kannan, Hyenkyun Woo, Charu C. Aggarwal, and Haesun Park. 2017) : [https://github.com/ramkikannan/outliernmf](https://github.com/ramkikannan/outliernmf). But the results of this baseline were worse than the ones we obtain with CVDD, DATE and OC-SVM.

Example of command for TONMF with 20 Newsgroups
```
python main_tonmf.py --dataset_name newsgroups20 --xp_path ../log/test_newsgroups20 --data_path ../data --results_file results.txt --clean_txt --normal_class 0 --n_components 300 --seed 0 --n_iter=1 --n_update_wh=1 --cont 0
```

### 5. To download pre-trained word vectors for FastText
If [4](#4-run-experiments) does not work for FastText_en, and you get the error "urllib.error.HTTPError: HTTP Error 404: Not Found", you can download the embeddings manually as follows :
```
cd data/word_vectors_cache
curl -Lo wiki.en.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
curl -Lo wiki.fr.vec https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.fr.vec
```
Otherwise the pre-trained embeddings for FastText and GloVe get downloaded into /word_vectors_cache the first time you use them with step [4](#4-run-experiments).

### 6. Error with Spacy
If you get this error for running experiments with spacy tokenizer in french on RNCP :
```
__init__() got an unexpected keyword argument 'language'
```

Please add
```
if 'language' in kwargs:
            kwargs.pop('language')
```
at line 63 just before this line ```super().__init__(*args, tokenize=partial(_tokenize, tokenizer=self.spacy), **kwargs)```, to get : 
```
if 'language' in kwargs:
            kwargs.pop('language')
        super().__init__(*args, tokenize=partial(_tokenize, tokenizer=self.spacy), **kwargs)
```
In this version of spacy, it is otherwise not possible to use other languages than English, although the package is supposed to support other languages.


## License
License
