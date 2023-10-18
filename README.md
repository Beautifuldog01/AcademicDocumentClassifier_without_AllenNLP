# AcademicDocumentClassifier_without_AllenNLP
An implementation of Scalable Evaluation and Improvement of Document Set Expansion via Neural Positive-Unlabeled Learning without AllenNLP  
Authors: Alon Jacovi, Gang Niu, Yoav Goldberg, Masashi Sugiyama  
Original Implementation: https://github.com/alonjacovi/document-set-expansion-pu  
Paper: https://arxiv.org/abs/1910.13339  

## Abstract
This repository contains a non-AllenNLP implementation of the original paper which proposes an approach to document set expansion using Positive-Unlabeled Learning. The problem is framed as an Information Retrieval (IR) task where a small set of cohesive-topic documents serves as a query to retrieve additional documents from a large corpus.

## Implementation Details
This implementation diverges from the original repository in key ways:

1. The entire code is based on PyTorch instead of AllenNLP.
2. The text encoder is implemented using Convolutional Neural Networks (CNN) via the nltk and PyTorch libraries.

## Package requirements
You may use ```pip install -r requirements``` to obatain these packages
torch==2.0.1
transformers==4.31.0
numpy==1.25.2
nltk==3.8.1
tqdm==4.66.1
matplotlib==3.7.2
tensorboard==2.14.1

## Cite the Original Work
```
@article{jacovi2019scalable,
  title={Scalable Evaluation and Improvement of Document Set Expansion via Neural Positive-Unlabeled Learning},
  author={Jacovi, Alon and Niu, Gang and Goldberg, Yoav and Sugiyama, Masashi},
  journal={arXiv preprint arXiv:1910.13339},
  year={2019}
}
```
