#!/usr/bin/env bash

cd ../
exe_file=../lcr_auc/mutated_testing.py
analyze_file=../lcr_auc/lcr_auc_analysis.py
function normal_lcr(){

opType=$1
device=2
dataType=1
testType="normal"
is_adv="True"
useTrainData="False"
batchModelSize=50
maxModelsUsed=500
mutatedModelsPath="../build-in-resource/mutated_models/cifar10/googlenet/${opType}/5e-3p/"
testSamplesPath="../build-in-resource/dataset/cifar10/raw/"
seedModelName="googlenet"
lcrSavePath="../build-in-resource/nr-lcr/googlenet/${opType}/5e-3p/nrLCR.npy"
seedModelPath="../build-in-resource/pretrained-model/googlenet.pkl"
test_result_folder="../lcr_auc-testing-results/cifar10/googlenet/${opType}/5e-3p/normal/"

date=`date +%Y-%m-%d-%H`
logpath=${test_result_folder}-${date}
totalbatches=$(( $(( $maxModelsUsed / $batchModelSize )) + $(( $maxModelsUsed % $batchModelSize )) ))


echo "=======>Please Check Parameters<======="
    if test $dataType = 0
    then
        echo "dataType:" "mnist"
    else
        echo "dataType:" "cifar10"
    fi
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
    if test "$is_adv" = "True"
    then
        echo "nrLcrPath:" $nrLcrPath
    else
        echo "lcrSavePath:" $lcrSavePath
    fi
echo "<======>Parameters=======>"

echo "Press any key to start mutation process"
echo " CTRL+C break command bash..." # 组合键 CTRL+C 终止命令!

if [[ ! -d "$logpath" ]];then
    mkdir -p $logpath
fi
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
#########################
# analyze the LCR and AUC
#########################
if test "$is_adv" = "True"
then
    python -u $analyze_file --logPath $logpath \
                        --maxModelsUsed $maxModelsUsed \
                        --isAdv $is_adv \
                        --nrLcrPath $nrLcrPath
else
    python -u $analyze_file --logPath $logpath \
                    --maxModelsUsed $maxModelsUsed \
                    --isAdv $is_adv \
                    --lcrSavePath $lcrSavePath
fi
}



function log_analyze(){

    analyze_file=../lcr_auc/lcr_auc_analysis.py
    opType=$1
    logpath=$2
    lcrSavePath="../build-in-resource/nr-lcr/googlenet/${opType}/5e-3p/nrLCR.npy"
    python -u $analyze_file --logPath $logpath \
                    --maxModelsUsed 500 \
                    --isAdv "False" \
                    --lcrSavePath $lcrSavePath
}

for op in "ns" "gf" "ws" "nai"
do
    echo "==============>$op<================"
    if test "$op" = "nai"
    then
        logpath="../lcr_auc-testing-results/cifar10/googlenet/nai/5e-3p/normal/-2019-01-07-02"
    elif test "$op" = "ns"
    then
        logpath="../lcr_auc-testing-results/cifar10/googlenet/ns/5e-3p/normal/-2019-01-06-22"
    elif test "$op" = "gf"
    then
        logpath="../lcr_auc-testing-results/cifar10/googlenet/gf/5e-3p/normal/-2019-01-06-23"
    else test "$op" = "ws"
        logpath="../lcr_auc-testing-results/cifar10/googlenet/ws/5e-3p/normal/-2019-01-07-01"
    fi
    log_analyze $op $logpath
done








