#!/usr/bin/env bash

exe_file=../attacks/attack_util.py

dataType=$1
filePath=$2
device=$3
advGround=$4


python $exe_file  --dataType dataType \
                  --filePath filePath \
                  --device device \
                  --advGround advGround


