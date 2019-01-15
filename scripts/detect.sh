#!/usr/bin/env bash

#############
# Function
#############
get_char()
{
    SAVEDSTTY=`stty -g`
    stty -echo
    stty cbreak
    dd if=/dev/tty bs=1 count=1 2> /dev/null
    stty -raw
    stty echo
    stty $SAVEDSTTY
}

exe_file=../detect/adv_detect.py

echo -e "NOTE: Our experiments are only based on two datasets: mnist and cifar10,\n
         but it is a piece of cake to extend to other datasets only providing a \n
         proper pytorch-style data loader tailored to himself datasets."

echo "To quickly verify the mutation process, we provide a group of default parameters，do you want to quickly start the
program?y/n"

read choice
####################
# read parameters
####################

'''

threhold threhold
extendScale extendScale
relaxScale relaxScale
mutatedModelsPath mutatedModelsPath
alpha alpha
beta beta
testSamplesPath testSamplesPath
dataType dataType
testType testType
seedModelPath seedModelPath

'''

if test "$choice" = "y"
then
    # mnsit 0.05 ns,fgsm
    threshold=0.0124
    extendScale=1.0
    relaxScale=0.1
    mutatedModelsPath="../build-in-resource/mutated_models/mnist/lenet/ns/5e-2p/"
    alpha=0.05
    beta=0.05
    testSamplesPath="../build-in-resource/dataset/mnist/adversarial/fgsm/"
    dataType=0
    testType="adv"
    seedModelPath="../build-in-resource/pretrained-model/lenet.pkl"
    device=-1
else
    python $exe_file --help
    echo "Tha above is the description of each paprameter. Please input them one by one."
    echo

    read -p "threshold:" threshold
    read -p "extendScale:" extendScale
    read -p "relaxScale:" relaxScale
    read -p "mutatedModelsPath:" mutatedModelsPath
    read -p "alpha:" alpha
    read -p "beta:" beta
    read -p "testSamplesPath:" testSamplesPath
    read -p "dataType:" dataType
    read -p "testType:" testType
    read -p "seedModelPath:" seedModelPath
    read -p "mutatedModelsPath:" mutatedModelsPath
    read -p "device:" device

fi

echo "=======>Please Check Parameters<======="
    echo "threhold:" $threshold
    echo "extendScale:" $extendScale
    echo "relaxScale:" $relaxScale
    echo "mutatedModelsPath:" $mutatedModelsPath
    echo "alpha:" $alpha
    echo "beta:" $beta
    echo "testSamplesPath:" $testSamplesPath
    echo "dataType:" $dataType
    echo "testType:" $testType
    echo "seedModelPath:" $seedModelPath
    echo "mutatedModelsPath:" $mutatedModelsPath
    echo "device:" $device
echo "<======>Parameters=======>"

echo "Press any key to start mutation process"
echo " CTRL+C break command bash..." # 组合键 CTRL+C 终止命令!
char=`get_char`


python -u $exe_file --extendScale $extendScale \
                    --relaxScale $relaxScale \
                    --mutatedModelsPath $mutatedModelsPath \
                    --alpha $alpha \
                    --beta $beta \
                    --testSamplesPath $testSamplesPath \
                    --dataType $dataType \
                    --testType $testType \
                    --seedModelPath $seedModelPath \
                    --threshold  $threshold \
                    --device $device
