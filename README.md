
#Introduction
Corresponding code to the paper "Adversarial Sample Detection for Deep Neural Network through Model Mutation Testing" by Jingyi Wang, Guoliang Dong, Jun Sun, Xinyu Wang, Peixin Zhang,accepted by ICSE 2019.

Implementations of this artifacts in pytorch 0.41 with python 2.7. 
#Code Structure

This artifact includes four function modules, each of which is independent.

- Adversarial Sample Generation (attacks)
- Mutation Model Generation (lcr_auc)
- Label Change Rate(lcr) and AUC over adversarial samples (model_mutation)
- Adversarial Sample Detection (detect)

*scripts* contains some bash shell files related to the four function modules. The **craftAdvSamples.sh** is used to check if the adversarial samples are valid. The folder **default-test** contains two shell fiels to analyze the lcr and auc over MNIST and CIFAR10 dataset respectively, and the results of auc are listed as follows:

#### AUC Report for Cifar10 Data 
attack|NAI|GF|NS|WS
---|---|---|---|---
fgsm|**0.9000**|0.8814|0.7687|0.8476
jsma|0.9737|**0.9757**|0.9534|0.9737
cw|**0.9264**|0.9230|0.8499|0.8964
bb|**0.8707**|0.8370|0.705|0.8076
df|0.9686|**0.9769**|0.9502|0.9731

#### AUC Report for MNIST Data
attack|NAI|GF|NS|WS
---|---|---|---|---
fgsm|**0.9647**|0.9639|0.9354|0.9515
jsma|0.9941|0.9957|**0.9960**|0.9955	
cw|**0.9576**|0.9501|0.9089|0.9225
bb|**0.9690**|0.9674|0.9439|0.9590
df|**0.9862**|0.9837|0.9812|0.9836

---

The folder **build-in-resource** contains some essential resources,including: 

- dataset: 
	- the complete mnist and cifar10 dataset
	- the adversarial samples with the attack manners described in our paper
- mutated_model: the mutated models used in the paper
- nr-lcr: the label change rate of legitimate sampels. 
- pretrained-model: lenet for mnsit,and googlenet for cifar10.

# Useage
NOTE: we enable users to specific the required parameters just following the instructions printed by the shell file.
 
## Running Attack
```
cd ./scripts
./craftAdvSamples.sh
```
## Running Model Mutation
```
cd ./scripts
./modelMuated.sh
```
## Running Adversarial Samples Detection
```
cd ./scripts
./detect.sh
```
## Running LCR and AUC Analysis
```
cd ./scripts
./lcr_acu_analysis.sh
```

# Reference
- [carlini/nn_robust_attacks](https://github.com/carlini/nn_robust_attacks)
- [DeepFool](https://github.com/paulasquin/DeepFool)













 