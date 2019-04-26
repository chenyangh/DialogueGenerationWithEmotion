# Dialogue Generation with Expressed Emotions
<img src="img/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repo contains the implementation of the two papers:

[*Automatic dialogue generation with expressed emotions*](http://www.aclweb.org/anthology/N18-2008)


[*Generating Responses Expressing Emotion in an Open-domain Dialogue System*](https://conversations2018.files.wordpress.com/2018/10/conversations_2018_paper_9_preprint2.pdf)

The second paper is basically an extension of the first, it shows four more approaches to express specified emotions. 

The following figure shows an overview of all the 7 models. 

![models](img/models.jpg)

## Instructions
The code is originally written in PyTorch0.3 and Python3.6

This project is heavily relying on emotion classifier. In this code ,we use a very simple Bi-LSTM model. The performance would very but not too much depending what kinda of text classifier you are using.

CBET dataset can be accessed through this [link](https://github.com/chenyangh/CBET-dataset). It is balanced in single labeled emotions and preprocessed. 

To replicate the results in the paper, you need to follow the following instructions:

1.  Firstly, train an emotion classifier using CBET dataset.  



2. Download jiwei's dataset as in his [github page](), I made a code that converts his dataset from token IDs to actual tokens.


```
python jiwei_dataset.py
```

3. 


## Citation
If you find our work is helpful, please consider citing one of the following papers.

[*Automatic dialogue generation with expressed emotions*](http://www.aclweb.org/anthology/N18-2008)

```
@inproceedings{huang2018automatic,
  title={Automatic dialogue generation with expressed emotions},
  author={Huang, Chenyang and Za\"{i}ane, Osmar and Trabelsi, Amine and Dziri, Nouha},
  booktitle={Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  volume={2},
  pages={49--54},
  year={2018}
}
``` 


[*Generating Responses Expressing Emotion in an Open-domain Dialogue System*](https://conversations2018.files.wordpress.com/2018/10/conversations_2018_paper_9_preprint2.pdf)

```
@incollection{Huang2019,
  doi = {10.1007/978-3-030-17705-8_9},
  url = {https://doi.org/10.1007/978-3-030-17705-8_9},
  year = {2019},
  publisher = {Springer International Publishing},
  pages = {100--112},
  author = {Chenyang Huang and Osmar R. Za\"{i}ane},
  title = {Generating Responses Expressing Emotion in an Open-Domain Dialogue System},
  booktitle = {Internet Science}
}
```

