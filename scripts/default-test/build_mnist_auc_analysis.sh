#!/usr/bin/env bash

function aucAnalysis(){

<<<<<<< HEAD
    exe_file=../../lcr_auc/mutated_testing.py
    analyze_file=../../lcr_auc/lcr_auc_analysis.py
=======
    exe_file=../lcr_auc/mutated_testing.py
    analyze_file=../lcr_auc/lcr_auc_analysis.py
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3

    ####################
    # read parameters
    ####################
    opType=$1
    attatck=$2
    is_adv="True"
    dataType=0
    useTrainData="False"
    batchModelSize=500
    maxModelsUsed=500
    seedModelName="lenet"
<<<<<<< HEAD
    mutatedModelsPath="../../build-in-resource/mutated_models/mnist/lenet/${opType}/5e-2p/"
    nrLcrPath="../../build-in-resource/nr-lcr/${opType}/5e-2p/nrLCR.npy"
=======
    mutatedModelsPath="../build-in-resource/mutated_models/mnist/lenet/${opType}/5e-2p/"
    nrLcrPath="../build-in-resource/nr-lcr/${opType}/5e-2p/nrLCR.npy"
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3
    seedModelPath=None
    testType="adv"  # normal,adv,wl

    device=1
<<<<<<< HEAD
    testSamplesPath="../../build-in-resource/dataset/mnist/adversarial/${attatck}"
    test_result_folder="../../artifacts_eval/lcr_auc-testing-results/mnist/lenet/${opType}/5e-2p/${attatck}"
=======
    testSamplesPath="../build-in-resource/dataset/mnist/adversarial/${attatck}"
    test_result_folder="../artifacts_eval/lcr_auc-testing-results/mnist/lenet/${opType}/5e-2p/${attatck}"
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3
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

<<<<<<< HEAD
=======
cd ../
>>>>>>> 4052834f1df04c1ea8d67bb049eb1bdb2d82a4e3
for op in "ns" "gf" "ws" "nai"
do
    for attack in "fgsm" "deepfool" "bb" "jsma" "cw"
    do
        echo "==============>$op,$attack<================"
        aucAnalysis $op $attack
    done
done





