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

exe_file=../lcr/lcr_threshold.py
analyze_file=../lcr/log_analysis.py

echo -e "NOTE: Our experiments are only based on two datasets: mnist and cifar10,\n
         but it is a piece of cake to extend to other datasets only providing a \n
         proper pytorch-style data loader tailored to himself datasets."

echo "To quickly verify the mutation process, we provide a group of default parameters，do you want to quickly start the
program?y/n"
read choice

####################
# read parameters
####################

if test "$choice" = "y"
then
    dataType=1
    device=1
    testType="adv"
    useTrainData="True"
    batchModelSize=10
    maxModelsUsed=10
    mutatedModelsPath="../build-in-resource/mutated_models/cifar10/googlenet/ns/3e-3p/"
    testSamplesPath="../build-in-resource/dataset/cifar10/adversarial/jsma/"
    seedModelName="lenet"
    test_result_folder="../lcr-testing-results/cifar10/googlenet/ns/3e-2p/jsma/"
    seedModelPath="../build-in-resource/pretrained-model/lenet.pkl"
else
    python $exe_file --help
    echo "Tha above is the description of each paprameter. Please input them one by one."
    echo
    read -p "dataType:" dataType
    read -p "device:"  device
    read -p "testType:" testType
    read -p "useTrainData:" useTrainData
    read -p "batchModelSize:" batchModelSize
    read -p "mutatedModelsPath:" mutatedModelsPath
    read -p "testSamplesPath:" testSamplesPath
    read -p "seedModelName:" seedModelName
    read -p "test_result_folder:" test_result_folder
    read -p "maxModelsUsed:" maxModelsUsed
fi
date=`date +%Y-%m-%d-%H`
logpath=${test_result_folder}${date}
if [[ ! -d "$logpath" ]];then
    mkdir -p $logpath
fi
totalbatches=$(( $(( $maxModelsUsed / $batchModelSize )) + $(( $maxModelsUsed % $batchModelSize )) ))

is_adv="True"
if test "$testType" = "normal"
then
    is_adv="False"
fi

echo "=======>Please Check Parameters<======="
    echo "dataType:" $dataType
    echo "device:" $device
    echo "testType:" $testType
    echo "useTrainData:" $useTrainData
    echo "batchModelSize:" $batchModelSize
    echo "maxModelsUsed:" $maxModelsUsed
    echo "mutatedModelsPath:" $mutatedModelsPath
    echo "testSamplesPath:" $testSamplesPath
    echo "seedModelName:" $seedModelName
    echo "test_result_folder:" $test_result_folder
    echo "The test will be divided into "$totalbatches" batches"
    echo "The logs will be saved in:" $logpath
    echo "is_adv:" $is_adv
echo "<======>Parameters=======>"

echo "Press any key to start mutation process"
echo " CTRL+C break command bash..." # 组合键 CTRL+C 终止命令!
char=`get_char`

for((no_batch=1;no_batch<=${totalbatches};no_batch++))
do
        echo batch:${no_batch}
        model_start_no=$(( $(( $(( no_batch-1 ))*${batchModelSize} ))+1 ))
        echo model_start_no:${model_start_no}
        python  -u $exe_file --dataType ${dataType} \
                         --device  ${device} \
                         --testType ${testType} \
                         --useTrainData ${useTrainData} \
                         --startNo ${model_start_no} \
                         --batchModelSize ${batchModelSize} \
                         --mutatedModelsPath ${mutatedModelsPath} \
                         --testSamplesPath ${testSamplesPath} \
                         --seedModelName ${seedModelName} \
                         --seedModelPath ${seedModelPath} \
                         > $logpath/${no_batch}.log
done

echo "Testing Done!"
##############
# analyze the LCR
##############
python -u $analyze_file $logpath $maxModelsUsed $is_adv




