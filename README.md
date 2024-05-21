# UnCLe SAM -- Unleashing Continual Learning for SAM (MIDL 2024)

This repository represents the official PyTorch code base for our MIDL 2024 published paper **UnCLeSAM: Unleashing SAM’s Potential for Continual Prostate MRI Segmentation**. For more details, please refer to [our paper](https://openreview.net/forum?id=jRtUQ2VnNi).


## Table Of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [How to get started?](#how-to-get-started)
4. [Data and pre-trained models](#data-and-pre-trained-models)
5. [Citations](#citations)
6. [License](#license)

## Introduction

This MIDL 2024 submission currently includes the following methods for Continual Learning:
* Sequential Training
* Riemannian Walk
* Elastic Weight Consolidation

## Installation

The simplest way to install all dependencies is by using [Anaconda](https://conda.io/projects/conda/en/latest/index.html):

1. Create a Python 3.9 environment as `conda create -n <your_conda_env> python=3.9` and activate it as `conda activate  <your_conda_env>`.
2. Install CUDA and PyTorch through conda with the command specified by [PyTorch](https://pytorch.org/). The command for Linux was at the time `conda install pytorch torchvision cudatoolkit=11.3 -c pytorch`. Our code was last tested with version 1.13. Pytorch and TorchVision versions can be specified during the installation as `conda install pytorch==<X.X.X> torchvision==<X.X.X> cudatoolkit=<X.X> -c pytorch`. Note that the cudatoolkit version should be of the same major version as the CUDA version installed on the machine, e.g. when using CUDA 11.x one should install a cudatoolkit 11.x version, but not a cudatoolkit 10.x version.
3. Navigate to the project root (where `setup.py` lives).
4. Execute `pip install -r requirements.txt` to install all required packages.


## How to get started?
- The easiest way to start is using our `train_abstract_*.py` python files. For every baseline and Continual Learning method, we provide specific `train_abstract_*.py` python files, located in the [scripts folder](https://github.com/MECLabTUDA/UnCLeSAM/tree/main/scripts/torch).
- The [eval folder](https://github.com/MECLabTUDA/UnCLeSAM/tree/main/eval) contains several jupyter notebooks that were used to calculate performance metrics and plots used in our submission.


## Data and pre-trained models
- **Data**: In our paper, we used four publicly available prostate datasets from:
  - [Multi-site Dataset for Prostate MRI Segmentation](https://liuquande.github.io/SAML/)
  - [Prostate Dataset from the Medical Decathlon Challenge](https://drive.google.com/file/d/1Ff7c21UksxyT4JfETjaarmuKEjdqe1-a/view?usp=share_link)
- **Models**: Our pre-trained models from our submission can be provided by contacting the [main author](mailto:amin.ranem@gris.informatik.tu-darmstadt.de) upon request.

For more information about UnCLe SAM, please read the following paper:
```
Ranem, A., Aflal, M. A. M., Fuchs, M., & Mukhopadhyay, A. (2024, February).
UnCLe SAM: Unleashing SAM’s Potential for Continual Prostate MRI Segmentation. In Medical Imaging with Deep Learning.
```

## Citations
If you are using UnCLe SAM or our code base for your article, please cite the following paper:
```
@inproceedings{ranem2024uncle,
  title={UnCLe SAM: Unleashing SAM’s Potential for Continual Prostate MRI Segmentation},
  author={Ranem, Amin and Aflal, Mohamed Afham Mohamed and Fuchs, Moritz and Mukhopadhyay, Anirban},
  booktitle={Medical Imaging with Deep Learning},
  year={2024}
}
```

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
