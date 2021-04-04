#!/bin/bash
dir_nikon="./NIKON_D700/DNG/"
dir_canon="./Canon_EOS_5D/DNG/"
if [ ! -d "$dir_nikon" ];then
mkdir $dir_nikon
fi
if [ ! -d "$dir_canon" ];then
mkdir $dir_canon
fi
wget -P./NIKON_D700/DNG -i NIKON_D700.txt
wget -P./Canon_EOS_5D/DNG -i Canon_EOS_5D.txt
# python data_preprocess.py
# python data_preprocess.py --camera="Canon_EOS_5D"

