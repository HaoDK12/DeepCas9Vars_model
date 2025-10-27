# DeepCas9Vars_model

A unitied multi-task deep-learning model for sgRNA efficiency prediction of 17 SpCas9 variants simultaneously.

## Introduction

The DeepCas9Vars model is designed to optimize gRNA design across diverse Cas9 variants for CRISPR-Cas9-mediated research and therapeutic application. The implemented model predicts the editing activities of gRNAs at specified target sites, allowing users to evaluate the likelihood of successful DNA cleavage (DSBs) induced by different Cas9 enzymes. By scanning the input sequence for potential target sites and evaluating them across a panel of Cas9 variants, DeepCas9Vars provides accurate indel frequency predictions, which estimate the relative editing activity for each gRNA–Cas9 combination in advance.

> DeepCas9Vars shows state-of-the-art accuracy of sgRNA efficiency estimation across various SpCas9 variants and is freely available at [dreamdb.biomed.au.dk/DeepCas9Vars](https://dreamdb.biomed.au.dk/DeepCas9Vars/home).

## Installation
```bash
git clone https://github.com/HaoDK12/DeepCas9Vars_model.git
cd DeepCas9Vars_model
```
Set environment as needed by ``requirements.txt``. You can import and use the model in scripts or run it via CLI.

## Run DeepOne via command line tool
Run the DeepCas9Vars-model command line interface (CLI) with the command in the terminal:
```
python DeepCas9Vars-model.py --input_seq TTATCTTCGCTATCACCTCCGCCGGGGTCACCCATTAT --selected_cas9s SpCas9 HiFi-Cas9 --out_path results.tsv
```
| Argument       | Type  | Required | Description                                                                                          |
| -------------- | ----- | -------- | ---------------------------------------------------------------------------------------------------- |
| `--input_seq`  | `str` | Yes    | Genomic DNA sequence (31–2000 bp) without spaces, line breaks, or numbers.                           |
| `--selected_cas9s`  | `str` | Yes    | Name of SpCas9 variant used for prediction. Supported: `HiFi-Cas9`, `HypaCas9`, `LZ3-Cas9`, `Sniper-Cas9`, `SpCas9`,`SpCas9-HF1`, `SpCas9-NG`, `SpCas9-NRCH`, `SpCas9-NRRH`, `SpCas9-NRTH`, `SpG`, `SpRY`, `SuperFi-Cas9`, `VRQR`, `eSpCas9`, `evoCas9`, `xCas9`. |
| `--out_path`   | `str` | No    | Output file path for saving the prediction results (e.g., `results.tsv`).                            |
| `--help`, `-h` | flag  | No     | Show help message and exit.                                                                          |

Note: Supported variants are ``["HiFi-Cas9", "HypaCas9", "LZ3-Cas9", "Sniper-Cas9", "SpCas9","SpCas9-HF1", "SpCas9-NG", "SpCas9-NRCH", "SpCas9-NRRH", "SpCas9-NRTH","SpG", "SpRY", "SuperFi-Cas9", "VRQR", "eSpCas9", "evoCas9", "xCas9"]``, which can be multiple choice.

## Contact
We greatly appreciate your feedback. If bug reports or suggestions, Please contact us (au735018@uni.au.dk).

## Cite
If you are using DeepCas9Vars in your publication, please cite:  
Hao Y, ..., Lin L, Yonglun L#, A unified multi-task deep learning model predicts sgRNA efficiency for 17 Cas9 variants simultaneously. 2025 (Manuscript under revision)  

DeepCas9Vars is trained on data reported in the publication above and in papers from Kim et.al, please cite that as well.  
Kim N, Choi S, Kim S, Song M, Seo JH, Min S, Park J, Cho SR, Kim HH. Deep learning models to predict the editing efficiencies and outcomes of diverse base editors. Nat Biotechnol. 2024 Mar;42(3):484-497.
