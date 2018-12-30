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

exe_file=../attacks/craft_adversarial_img.py
echo -e "NOTE: Our experiments are only based on two datasets: mnist and cifar10,\n
         but it is a piece of cake to extend to other datasets only providing a \n
         proper pytorch-style data loader tailored to himself datasets. Each attack manner has different parameters. All
         the parameters are organized in a list.The order of the parameters can be found in the REDME in this folder."

echo "To quickly yield adversarial samples, we provide a default setting for each attack manner.Do you want to perform
an attack with the default settings?y/n"
read choice

if test "$choice" = "y"
then
    read -p "dataType ( [0] mnist; [1] cifar10):" dataType
    if test "$dataType" = "0"
    then
        modelName="lenet"
        modelPath="../build-in-resource/pretrained-model/lenet.pkl"
        sourceDataPath="../build-in-resource/dataset/mnist/raw"
    elif test "$dataType" = "1"
    then
         modelName="googlenet"
         modelPath="../build-in-resource/pretrained-model/googlenet.pkl"
         sourceDataPath="../build-in-resource/dataset/cifar10/raw"
    else
        echo "Invalid data type:" $dataType
        exit
    fi

    read -p "attackType:" attackType
    if test "$attackType" = "fgsm"
    then
       if test "$dataType" = "0" # mnist
       then
            attackParameters=(0.35,true)
       else # cifar10
            attackParameters=(0.03,true)
       fi
    elif test "$attackType" = "jsma"
    then
       attackParameters=(0.12)
    elif test "$attackType" = "bb"
    then
       if test "$dataType" = "0" # mnist
       then
        #eps,max_iter,submodel_epoch,seed_data_size,step_size
            attackParameters=(0.35,6,10,200,0.1)
       else # cifar10
            attackParameters=(0.35,4,10,200,0.015)
       fi
    elif test "$attackType" = "deepfool"
    then
        attackParameters=(0.02,50)
    elif test "$attackType" = "cw"
    then
       if test "$dataType" = "0" # mnist
       then
            attackParameters=(0.6,10000)
       else # cifar10
            attackParameters=(0.6,1000)
       fi
    else
        echo "Invalid attack type:" $attackType
        exit
    fi

    if test "$dataType" = "0"
    then
        savePath="../artifacts_eval/adv_samples/mnist/"$attackType
    else
        savePath="../artifacts_eval/adv_samples/cifar10/"$attackType
    fi
    device=1
else
    python $exe_file --help
    echo "Tha above is the description of each paprameter. Please input them one by one."
    echo
    read -p "modelName:" modelName
    read -p "modelPath:" modelPath
    read -p "dataType:" dataType
    read -p "sourceDataPath:" sourceDataPath
    read -p "attackType:" attackType
    read -p "attackParameters:" attackParameters
    read -p "savePath:" savePath
    read -p "device:" device
fi

##########################
# show default parameters
##########################
echo "=======>Please Check Parameters<======="
echo "modelName:" $modelName
echo "modelPath:" $modelPath
echo "dataType:" $dataType
echo "sourceDataPath:" $sourceDataPath
echo "attackType:" $attackType
echo "attackParameters:" $attackParameters
echo "savePath:" $savePath
echo "device:"  $device
echo "<======>Parameters=======>"


echo "Press any key to start mutation process"
echo " CTRL+C break command bash..." # 组合键 CTRL+C 终止命令!
char=`get_char`

python  -u $exe_file --modelName ${modelName} \
                 --modelPath ${modelPath} \
                 --dataType ${dataType} \
                 --sourceDataPath ${sourceDataPath} \
                 --attackType  ${attackType}   \
                 --attackParameters  ${attackParameters} \
                 --savePath  ${savePath} \
                 --device  ${device}


