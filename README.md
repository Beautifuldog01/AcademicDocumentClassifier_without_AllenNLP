# AcademicDocumentClassifier_without_AllenNLP

An implementation of Scalable Evaluation and Improvement of Document Set Expansion via Neural Positive-Unlabeled Learning without AllenNLP  
Authors: Alon Jacovi, Gang Niu, Yoav Goldberg, Masashi Sugiyama  
Original Implementation: <https://github.com/alonjacovi/document-set-expansion-pu>  
Paper: <https://arxiv.org/abs/1910.13339>  
Data Available at:<http://nlp.biu.ac.il/~jacovia/pubmed-dse-15.zip>
Feel free to contact me or Alon: Qiuyi.chen2002@student.xjtlu.edu or alonjacovi@gmail.com:)  
## Abstract

This repository contains a non-AllenNLP implementation of the original paper which proposes an approach to document set expansion using Positive-Unlabeled Learning. The problem is framed as an Information Retrieval (IR) task where a small set of cohesive-topic documents serves as a query to retrieve additional documents from a large corpus.  

I have only transcribed the implementation of the nnPU model training and the 'proportional_iterator' trick. As for the generation of the dataset, namely the 'elasticsearch_dse' part, please refer to [Alon's implementation](https://github.com/alonjacovi/document-set-expansion-pu).

## Implementation Details

This implementation diverges from the original repository in key ways:

1. The entire code is based on PyTorch instead of AllenNLP.
2. The text encoder is implemented using Convolutional Neural Networks (CNN) via the nltk and PyTorch libraries.  
Data file follows like this below.  
   <img width="396" alt="image" src="https://github.com/Beautifuldog01/AcademicDocumentClassifier_without_AllenNLP/assets/40363660/45d39ced-85a4-461f-a54e-1bd08a096184">  

By runing ```tree -L 2``` command, you can see the directory structure like this below.  

```
(py310tc2) ➜  AcademicDocumentClassifier_without_AllenNLP git:(main) tree -L 2
.
├── data
│   └── pubmed-dse
├── dataset_pubmed.py
├── LICENSE
├── main_meta_CNN.py
├── models
│   └── nnPUCNN
├── __pycache__
│   ├── dataset_pubmed.cpython-310.pyc
│   └── utils.cpython-310.pyc
├── README.md
├── requirements.txt
├── runs
│   └── nnPU_CNN
└── utils.py
```
## Package requirements

You may use ```pip install -r requirements.txt``` to obatain these packages.  

```
torch==2.0.1  
transformers==4.31.0  
numpy==1.25.2  
nltk==3.8.1  
tqdm==4.66.1  
matplotlib==3.7.2  
tensorboard==2.14.1  
```

## Running the Model

Clone the repository and navigate to the project directory:

```
git clone https://github.com/Beautifuldog01/AcademicDocumentClassifier_without_AllenNLP.git
cd AcademicDocumentClassifier_without_AllenNLP
```

Then cd to the directory and run:

```
python main_meta_CNN.py  --batch_size 64 --num_epochs 100 --lr 0.0001 --prior 0.5
```

```
experiments = [
    "data/pubmed-dse/L50/D000328.D008875.D015658",
    "data/pubmed-dse/L50/D000818.D001921.D051381",
    "data/pubmed-dse/L50/D006435.D007676.D008875",
    "data/pubmed-dse/L20/D000328.D008875.D015658",
    "data/pubmed-dse/L20/D000818.D001921.D051381",
    "data/pubmed-dse/L20/D006435.D007676.D008875",
]
```

To conduct different experiments, you can change the index of the 'experiments' list from 0 to 5.  
Each index corresponds to a different experiment setup defined in the 'experiments' list.  
Simply set ```root_dir``` to the desired experiment by changing the index, for example:  
```root_dir = experiments[0]```  # This will set the root directory to the first experiment.  
```root_dir = experiments[1]```  # This will set the root directory to the second experiment, and so on up to index 5.  


## Cite the Original Work

```
@article{jacovi2019scalable,
  title={Scalable Evaluation and Improvement of Document Set Expansion via Neural Positive-Unlabeled Learning},
  author={Jacovi, Alon and Niu, Gang and Goldberg, Yoav and Sugiyama, Masashi},
  journal={arXiv preprint arXiv:1910.13339},
  year={2019}
}
```
