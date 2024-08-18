# AdaFlow
## Introduction
This repository is for the paper "AdaFlow: Learning and Utilizing Workflows for Enhanced Service Recommendation in Dynamic Environments." The AdaFlow model proposed in the paper aims to uncover potential service pattern within a mashup by learning the workflows between internal APIs, and it is used for service recommendation. 
It shows superior performance across various metrics, marking a significant advancement over previous methods.

## Implementation
It is recommended to understand the model code in conjunction with the paper.

## Dataset
The raw dataset used for experiments in the paper comes from Programmable Web and is stored in the Dataset_generation/Correted-ProgrammableWeb-dataset-main directory. 
We provide four scripts to generate the datasets used in the paper, with the generated files stored in the DS1 and DS2 folders under Dataset_generation, representing the two datasets used in the paper.
For each DS, the three core files are: api_dev_embeds.npz, invocation.json, and mashup_dev_embeds.npz:
- api_dev_embeds.npz: embeddings of APIs
- mashup_dev_embeds.npz: embeddings of mashups
- invocation.json: the invocation relationships between mashups and APIs

## How to replicate
### Directory Structure
- configs:
  - experiment: configuration files for model experiments
  - model: configuration files for model parameters
  - datamodule: data source configurations
- src:
  - datamodules:
    - PWDatamodule.py: used to load training, validation, and test sets
    - ProgrammableWebDataset.py: used to build DGLDataset datasets and compute propensity scores for training
    - models: contains the AdaFlow model and comparison models
- run.py: start-up script
### Steps to Replicate
1. Generate the corresponding dataset files using the scripts in the Dataset_generation folder and place them in your specified directory.
2. When using a particular DS, first configure the parameters for the API and mashup in ProgrammableWebDataSet, and the parameters in the configuration file under configs/model for the corresponding model.
3. Before starting training, configure the experiment configuration file under configs/experiment for the corresponding model.
4. Start the training by running the command 'python run.py experiment=AdaFlow.yaml' or set 'experiment=AdaFlow.yaml' in the IDE’s Configuration and launch run.py with a single click.