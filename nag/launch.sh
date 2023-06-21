#!/bin/bash
## run project
## launch
# Hongda 10/19/2017


projname=$1
currentFolder=$PWD
motherFolder="output/new_fidelity/"
cd $motherFolder

echo ">> checking folder..."$motherFolder$projname
if [ -d "$projname" ];
   then
	# Take action if $DIR exists. #
	read -p "!! project exist, continue?" yn
	case $yn in
	[Yy]* ) ;;
	[Nn]* ) exit;;
	* ) echo "please answer yes or no"; exit;;
	esac
   else
	echo ">> creating foler: "$projname
	mkdir "$projname"
fi

cd $currentFolder
#cp prm_dict.json $motherFolder$projname
cp param.json $motherFolder$projname
cp config.json $motherFolder$projname
cp launch.sh $motherFolder$projname
echo ">> launching main.py"
python main.py >$motherFolder$projname"/out" 2>$motherFolder$projname"/err" &

echo ">> job submitted!"
