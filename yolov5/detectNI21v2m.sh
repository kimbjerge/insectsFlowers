#!/bin/bash
dataYear="$1"
dateDate="$2"
filepath="/mnt/Dfs/Tech_TTH-KBE/NI_2/$dataYear/$dateDate/"
csv=".csv"
inputSystems="systems$dateDate.txt"
ls $filepath > $inputSystems
while IFS= read -r system
do
	echo $filepath$system
	input="wsct$dateDate.txt"
	ls "$filepath$system" > $input
	while IFS= read -r line
	do
		echo "$filepath$system/$line"
		csvFile="$dataYear-$dateDate-$system-$line$csv"
		echo "$csvFile"
		python detectCSVNI2v2m.py --weights insectsNI21m-bestF1-1280m6.pt --img 1280 --conf 0.25 --save-txt --nosave --source "$filepath$system/$line/" --name "res$2" 
		mv "res$2.csv" "NI2m/$csvFile"
	done < $input
        rm -r "runs/detect/res$2*"
done < $inputSystems
