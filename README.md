# Machine Learning Engineer Nanodegree
## Capstone Proposal
Michel Moreau
July 2nd, 2019

## Proposal  
[**CellSignal** - _Disentangling biological signal from experimental noise in cellular images_](https://www.rxrx.ai/)

## Introduction
My final project is to participale in the NeurIPS competition on Kaggle called **CellSignal** - _Disentangling biological signal from experimental noise in cellular images_. More information about this competition is available here:  
- Competition's website https://www.rxrx.ai  
- Kaggle competition's link: https://www.kaggle.com/c/recursion-cellular-image-classification/overview

Full disclosure: Some text in this proposal will be taken word for word from the competition's websites.

## Domain Background  
Recursion Pharmaceuticals, creators of the industry’s largest dataset of biological images, generated entirely in-house, believes AI has the potential to dramatically improve and expedite the drug discovery process. More specifically, machine learning could help understand how drugs interact with human cells.

This competition is designed to disentangle experimental noise from real biological signals. The goal is to classify images of cells under one of 1,108 different genetic perturbations, and thus eliminate the noise introduced by technical execution and environmental variation between \[drug\] experiments.


### Datasets and Inputs  
The data is available on the Kaggle's competition site https://www.kaggle.com/c/recursion-cellular-image-classification/data .   
  
One of the main challenges for applying AI to biological microscopy data is that even the most careful replicates of a process will not look identical. This dataset challenges you to develop a model for identifying replicates that is robust to experimental noise.

The same siRNAs (effectively genetic perturbations) have been applied repeatedly to multiple cell lines, for a total of 51 experimental batches. Each batch has four plates, each of which has 308 filled wells. For each well, microscope images were taken at two sites and across six imaging channels. Not every batch will necessarily have every well filled or every siRNA present.

For more information about the dataset, see the [competition's website](https://rxrx.ai) or the Kaggle link above.  


### Solution Statement  
The solution for this problem will likely be resolved with the type of model architecture used in computer vision and image classification. This is a multiclass classification problem, but algorithms and model architectures we've seen in the dogs classification project https://github.com/MichelML/udacity-dog-project/, such as VGG-16 and ResNet-50, will be considered for this project.

### Benchmark Model
Related to the solution statement, ResNet-50 will likely be our benchmark model, having achieved very good results in multiclass image classification problems in the past [source](https://arxiv.org/abs/1903.10035), having reached 98.87% accuracy when classifying histopathology images. ResNet-50 will be measured with multiclass accuracy, which is the same measure the competition's evaluation uses.

### Evaluation Metrics
Submissions will be evaluated on Multiclass Accuracy, which is simply the average number of observations with the correct label.

#### Submission File
For each id_code in the test set, we will predict the correct siRNA. As per the competition's indications, The file should contain a header and have the following format:

```  
id_code,sirna
HEPG2-08_1_B03,911
HEPG2-08_1_B04,911
etc.   
```

### Project Design

-----------  

