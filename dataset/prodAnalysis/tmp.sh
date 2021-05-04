#! /bin/bash

for var in *.json
do
	numlines=$(wc -l $var | cut -f 1 -d ' ')
	tmp=$(echo "0.75 * $numlines" | bc)
	train=$(printf "%.*f\n" 0 $tmp)
	test=$(echo "$numlines - $train" | bc)

	echo $train
	echo $test

	# take the file, shuffle it
	tmpFile=egg.txt
	shuf $var > $tmpFile
	# take the train/test bit and chuck it into new file
	head -$train $tmpFile > "train_$var"
	tail -$test $tmpFile > "test_$var"
	rm $tmpFile
done
