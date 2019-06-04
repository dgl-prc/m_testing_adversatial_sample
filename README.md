
# Introduction

Code to the paper "Adversarial Sample Detection for Deep Neural Network through Model Mutation Testing" by Jingyi Wang, Guoliang Dong, Jun Sun, Xinyu Wang, Peixin Zhang in ICSE 2019.

The implementation of this artifact is based on pytorch 0.41 with python 2.7. 
#Code Structure

This artifact includes four independent modules.

- Adversarial Sample Generation (attacks)
- Mutation Model Generation (lcr_auc)
- Label Change Rate(lcr) and AUC over adversarial samples (model_mutation)
- Adversarial Sample Detection (detect)

*scripts* contains some bash shell files related to the four function modules. The **craftAdvSamples.sh** is used to check if the adversarial samples are valid. The folder **default-test** contains two shell files to analyze the LCR and AUROC over MNIST and CIFAR10 dataset respectively, and the results of auc are listed as follows:

#### AUROC Report for Cifar10 Data 
attack|NAI|GF|NS|WS
---|---|---|---|---
fgsm|**0.9000**|0.8814|0.7687|0.8476
jsma|0.9737|**0.9757**|0.9534|0.9737
cw|**0.9264**|0.9230|0.8499|0.8964
bb|**0.8707**|0.8370|0.705|0.8076
df|0.9686|**0.9769**|0.9502|0.9731

#### AUROC Report for MNIST Data
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
	- the complete MNIST and CIFAR10 dataset
	- the adversarial samples with the attack methods described in our paper
- mutated_model: the mutated models used in the paper
- nr-lcr: the label change rate of normal sampels. 
- pretrained-model: Lenet for mnsit,and GooGleNet for cifar10.

# Useage
NOTE: we enable users to specific the required parameters just following the instructions printed by the shell file.
 
We recommend using the docker image. See INSTALL file in this directory to get about its useage.
Now, we assume that one has successfully load the image.

###1. Enter the Work Directory
To quickly start the experiment, we provide some script for each module in folder "scripts". Just enter the folder and start the experiments.

```
cd scripts/
```
###2. Adversarial Samples Generation

run the script:

```
./craftAdvSamples.sh
```
You will see the following info:

```
To quickly yield adversarial samples, we provide a default setting for each attack manner.Do you want to perform
an attack with the default settings?y/n
```
You can use the default settings with "y", or specific parameters by yourself with "n". We recommend one to choose "n" firstly to see which parameters are required and the default settings. We demonstrate the useage of default mode here.

Back to the useage example, input "y" to continue.

```
# you need choose which dataset you want to attack. Here mnist is selected.
dataType ( [0] mnist; [1] cifar10): 0
# Then,choose which attac manner you want to use. Here fgsm is selected.
attackType:fgsm
```
If everything goes well, you will see the following info after typing "Enter":

```
=======>Please Check Parameters<=======
modelName: lenet
modelPath: ../build-in-resource/pretrained-model/lenet.pkl
dataType: 0
sourceDataPath: ../build-in-resource/dataset/mnist/raw
attackType: fgsm
attackParameters: 0.35,true
savePath: ../artifacts_eval/adv_samples/mnist/fgsm
device: -1
<======>Parameters=======>
```
If you do not need to alter the settings, then press any key to continue and you will see a log info immediately:

```
Crafting Adversarial Samples....
```
Note, it maybe take some time to generate the adversarail samples for some specific attacks.

If successful, the following info will be printed on the screen finally.

```
successful samples 2054
Done!
icse19-eval-attack-fgsm: rename 125, remove 40,success 1889
Adversarial samples are saved in ../artifacts_eval/adv_samples/mnist/fgsm/2019-01-13_03:48:45
DONE!
```

Additionally, you can check if the adversarial samples is valid.

```
./advSampelsVerify.sh dataType filePath device advGround

```
The "device" is the No. of GPU. "-1" indicates that only using cpu.

The "advGround" indicates whether or not the ground truth should be the adversarial label.

For example: 

```
./advSampelsVerify.sh 0 ../artifacts_eval/adv_samples/mnist/fgsm/2019-01-13_03:48:45 -1 1
```

The expected ouput in the last line is:

```
Total:2014,Success:2014
```

### 3. Mutation Model Generation
Run the script:

```
./modelMuated.sh
```

You will see the following info:

```
To quickly verify the mutation process, we provide a group of default parameters，do you want to quickly start the
program?y/n
```

You can use the default settings with "y", or specific parameters by yourself with "n". By default, we choose "GF" as the mutation operator. When the defualt settings is selected, the following info will be printed.

```
=======>Parameters<=======
modelName: lenet
modelPath: ../build-in-resource/pretrained-model/lenet.pkl
accRation: 0.9
dataType: 0
numMModels: 10
mutatedRation: 0.001
opType: GF
savePath: ../artifacts_eval/modelMuation/
device: -1
<======>Parameters=======>
```
If you do not need to alter the settings, press any key to continue and the logs as follows will be output:

```
2019-01-13 13:28:47,651 - INFO - data type:mnist
2019-01-13 13:28:47,655 - INFO - >>>>>>>>>>>>Start-new-experiment>>>>>>>>>>>>>>>>
2019-01-13 13:28:48,275 - INFO - orginal model acc=0.9829
2019-01-13 13:28:48,275 - INFO - acc_threshold:88.0%
2019-01-13 13:28:48,275 - INFO - seed_md_name:lenet,op_type:GF,ration:0.001,acc_tolerant:0.9,num_mutated:10
2019-01-13 13:28:48,305 - INFO - 61/61706 weights to be fuzzed
2019-01-13 13:28:48,885 - INFO - Mutated model: accurate 0.9827
2019-01-13 13:28:48,886 - INFO - Progress:1/10
2019-01-13 13:28:48,892 - INFO - 61/61706 weights to be fuzzed
2019-01-13 13:28:49,493 - INFO - Mutated model: accurate 0.9832
...
```
When finished, the last line in the screen is expected be:
```
The mutated models are stored in ../artifacts_eval/modelMuation/2019-01-13_13:28:47/gf0.001/lenet
```

###4. Label change rate and auc statistics
To get the auc over adversarial samples, we should obtain the lcr of normal samples in advance. In this script, the default setting is designed for calculating adversarial samples. For simplicity, we have placed the lcr of normal samples for all the four operators and the relative path of results are as follows:

```
../build-in-resource/nr-lcr/mnsit/lenet/gf/5e-2p/nrLCR.npy
../build-in-resource/nr-lcr/mnsit/lenet/nai/5e-2p/nrLCR.npy
../build-in-resource/nr-lcr/mnsit/lenet/ns/5e-2p/nrLCR.npy
../build-in-resource/nr-lcr/mnsit/lenet/ws/5e-2p/nrLCR.npy

../build-in-resource/nr-lcr/cifar10/googlenet/gf/5e-3p
../build-in-resource/nr-lcr/cifar10/googlenet/nai/5e-3p
../build-in-resource/nr-lcr/cifar10/googlenet/ns/5e-3p
../build-in-resource/nr-lcr/cifar10/googlenet/ws/5e-3p
```

NOTE: One can always generates his own lcr results with this scripts by setting the parameters.

To be consistent with the running example, we select the following lcr result. 

```
../build-in-resource/nr-lcr/mnsit/lenet/gf/5e-2p/nrLCR.npy 
```

Now, let's begin.

Run the script:

```
./lcr_acu_analysis.sh 
```
You will see the following info:

```
To quickly label change rate and auc statistics , we provide a group of default parameters，do you want to quickly start the
program?y/n
```
Then, choose "y" to use the default setting. For demonstration, we only use 10 mutated model to perform lcr and auc statistics. To reproduce the results in the paper, one can use scripts placed in the directory "default-test".

Following the below instruction 

```
Please provide the lcr result of normal samples for the auc computinglease test.
Do you have the lcr results of normal samples?(y/n)y
Path of normal's lcr list:../build-in-resource/nr-lcr/mnsit/lenet/gf/5e-2p/nrLCR.npy
```
The complete default settings are as follows:

```
=======>Please Check Parameters<=======
dataType: mnist
device: -1
testType: adv
useTrainData: False
batchModelSize: 2
maxModelsUsed: 10
mutatedModelsPath: ../build-in-resource/mutated_models/mnist/lenet/gf/5e-2p/
testSamplesPath: ../build-in-resource/dataset/mnist/adversarial/jsma/
seedModelName: lenet
test_result_folder: ../lcr_auc-testing-results/mnist/lenet/gf/5e-2p/jsma/
The test will be divided into 5 batches
The logs will be saved in: ../lcr_auc-testing-results/mnist/lenet/gf/5e-2p/jsma/-2019-01-14-08
is_adv: True
nrLcrPath: ../build-in-resource/nr-lcr/mnsit/lenet/gf/5e-2p/nrLCR.npy
<======>Parameters=======>
```
If everything goes well, the expected input is:

```
batch:1
model_start_no:1
batch:2
model_start_no:3
batch:3
model_start_no:5
batch:4
model_start_no:7
batch:5
model_start_no:9
Testing Done!
 >>>>>>>>>>>seed data:mnist,mutated_models:../build-in-resource/mutated_models/mnist/lenet/gf/5e-2p/<<<<<<<<<<
>>>>>>>>>>>mnist<<<<<<<<<<<<<<
Total Samples Used:1000,auc:0.9863,avg_lcr:0.5033,std:0.1711,confidence(95%):0.0106,confidence(98%):0.0126,confidence(99%):0.0140
```
###5. Adversarial Sample Detection

Run the script:

```
./detect.sh
```
You will see the following info:

```
To quickly perform adversarial detection, we provide a group of default parameters，do you want to quickly start the
program?y/n
```
Then, choose "y" to use the default setting. The following info is expected:

```
=======>Please Check Parameters<=======
threhold: 0.0441
extendScale: 1.0
relaxScale: 0.1
mutatedModelsPath: ../build-in-resource/mutated_models/mnist/lenet/nai/5e-2p/
alpha: 0.05
beta: 0.05
testSamplesPath: ../build-in-resource/dataset/mnist/adversarial/jsma/
dataType: 0
testType: adv
seedModelPath: ../build-in-resource/pretrained-model/lenet.pkl
mutatedModelsPath: ../build-in-resource/mutated_models/mnist/lenet/nai/5e-2p/
device: -1
<======>Parameters=======>
``` 
Then, press any key to continue. If everything goes well, there is a dynamic progress showing how many samples has been processed,like:

```
Processed:37.60 %
```
When finished,the average accuracy and average number of mutated models used are expected ouput:

```
Processed:100.00 %adverage accuracy:0.998, avgerage mutated used:35.814
```


# Reference
- [carlini/nn_robust_attacks](https://github.com/carlini/nn_robust_attacks)
- [DeepFool](https://github.com/paulasquin/DeepFool)













 
