#!/bin/bash
## run project
## launch
# Hongda 10/19/2017


# read project folder from config.json
projfolder=$( cat config.json | python -c "import sys, json; print(json.load(sys.stdin)['dir'])")
currentFolder=$PWD
echo ">> project folder: "$projfolder
if [ -d "$projfolder" ];
   then
	# Take action if $DIR exists. #
	read -p "!! project exist, continue?" yn
	case $yn in
	[Yy]* ) ;;
	[Nn]* ) exit;;
	* ) echo "please answer yes or no"; exit;;
	esac
   else
	echo ">> creating foler: "$projfolder
	mkdir "$projfolder"
fi

cd $currentFolder
cp param.json $projfolder
cp config.json $projfolder
cp launch.sh $projfolder
echo ">> launching main.py"
python main.py >$projfolder"/out" 2>$projfolder"/err" &

echo ">> job submitted!"
