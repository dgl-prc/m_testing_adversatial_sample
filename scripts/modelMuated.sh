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

exe_file=../model_mutation/generate_mutated_models.py
echo -e "NOTE: Our experiments are only based on two datasets: mnist and cifar10,\n
         but it is a piece of cake to extend to other datasets only providing a \n
         proper pytorch-style data loader tailored to himself datasets."

echo "To quickly verify the mutation process, we provide a group of default parameters，do you want to quickly start the
program?y/n"
read choice

if test "$choice" = "y"
then
    modelName="lenet"
    modelPath="../build-in-resource/pretrained-model/lenet.pkl"
    accRation=0.9
    dataType=0
    numMModels=10
    mutatedRation=0.001
    opType="GF"
    savePath="../artifacts_eval/modelMuation/"
    device=-1

else

    python $exe_file --help

    echo "Tha above is the description of each paprameter. Please input them one by one."
    echo
    read -p "modelName:" modelName
    read -p "modelPath:" modelPath
    read -p "accRation:" accRation
    read -p "dataType:" dataType
    read -p "numMModels:" numMModels
    read -p "mutatedRation:" mutatedRation
    read -p "opType:" opType
    read -p "savePath:" savePath
    read -p "device:" device
fi


##########
# show default parameters
###########
echo"=======>Parameters<======="
echo "modelName:" $modelName
echo "modelPath:" $modelPath
echo "accRation:" $accRation
echo "dataType:" $dataType
echo "numMModels:" $numMModels
echo "mutatedRation:" $mutatedRation
echo "opType:" $opType
echo "savePath:" $savePath
echo "device:"  $device

echo "Press any key to start mutation process"
echo " CTRL+C break command bash..." # 组合键 CTRL+C 终止命令!
char=`get_char`

python $exe_file --modelName ${modelName} \
                 --modelPath ${modelPath} \
                 --accRation ${accRation} \
                 --dataType ${dataType} \
                 --numMModels  ${numMModels}   \
                 --mutatedRation  ${mutatedRation} \
                 --opType  ${opType} \
                 --savePath  ${savePath} \
                 --device  ${device}


