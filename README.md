## INFUNE
This repository contains the source code for the paper *A Novel Framework with Information Fusion and Neighborhood Enhancement for User Identity Linkage* published in ECAI 2020. The paper can be downloaded from [Arxiv](https://arxiv.org/pdf/2003.07122) or the official site [ECAI](https://ebooks.iospress.nl/volumearticle/55084).

## Requirements
- Ubuntu 16.04
- python3.6
- numpy>=1.17.2
- scipy>=1.1.0
- torch>=1.1.0
- torch-geometric>=1.3.0

## Data description
The data can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1gbJUaE59t4c8uLbJ7H03bljm2G0asoOM?usp=sharing). 

### Data for the information fusion component
All data requires for the information fusion component lie in `data/`.
- `adj_s.pkl, adj_t.pkl`: adjacency matrices of the source network and the target network, respectively.
- `sims_p.pkl, sims_c.pkl`: the ground truth similarity matrices of profile and content, respectively.
- `train_test_{}.pkl`: randomly split training and testing anchor user pairs at ratios range from 0.1 to 0.9.

### Extra data for neighborhood enhancement component
The neighborhood enhancement component requires extra inputs such as the pre-trained node embeddings, candidate users and etc. To ease the evaluation, the ratio of the training set is set to be 0.8, and the pre-trained node embeddings and other data are provided in `results/`.
- `emb_0.8.pkl`: the node embeddings.
- `candidate_0.8.pkl`: candidate user matrix.
- `nei_0.8.pkl`: `{(u, v): [[], [], [], []]}`, potential matched and unmatched neighbors of u and v.
- `adj_list.pkl`: the adjacency list of nodes in the source network and the target network.
## How to run
All arguments can be modified in [config.py](./config.py). Specify the information by modifying the argument --options. The default value is 'structure profile content'.
- Information fusion component
```bash
cd runs
python IF.py
```
- Neighborhood enhancement component
```bash
cd runs
python NE.py
```

## Citation
If you find this repository useful in your research, please consider citing the following paper:
```
@inproceedings{chen2020infune,
	title={A Novel Framework with Information Fusion and Neighborhood Enhancement for User Identity Linkage},
	author={Chen, Siyuan and Wang, Jiahai and Du, Xin and Hu, Yanqing},
	booktitle={24th European Conference on Artificial Intelligence (ECAI)},
	pages={1754--1761},
	year={2020}
}
```