# GCN for CECS 551
Implements [GCN]() for [CORA](https://relational.fit.cvut.cz/dataset/CORA) [Citeseer](http://csxstatic.ist.psu.edu/downloads/data) and [Pubmed](https://catalog.data.gov/dataset/pubmed) datasets in an effort to reproduce results in the paper [Semi-Supervised Classification with Graph Neural Networks](https://arxiv.org/abs/1609.02907) by [Kipf](https://tkipf.github.io/) and [Welling](https://staff.fnwi.uva.nl/m.welling/).

Implementation is directly inspired by [Kipf's pytorch implemetnation](https://github.com/tkipf/pygcn). 

## Usage
The following commands assume you have [Anaconda](https://www.anaconda.com/) (or miniconda) installed on your system.

### Create the environment
We have provided an ```environment.yml``` file with all required dependencies defined. To create the environment, run:

```bash
conda env create -f environment.yml # this should be run only once
conda activate gcn551 # this must be done in every new shell
```

### Running
See ```main.py``` for an example of usage. You can run this file on a dataset to test with the default hyperparameters:

```bash
python main.py /path/to/data <cora|citeseer|pubmed>
```