#!/bin/bash
# Start verification testing

# TESTING PARAMETERS
workers=(1 2 4 8)
multi=1
w_tree=64
p_tree=16

echo -e "*********************************************************\n\t\tVERIFICATION BATCH TEST\n*********************************************************"

# Detect compile flag if present
if [ "$1" != "" ]; then
  if [ "$1" == "-c" ]; then
    echo -e "\n*********************\n* COMPILATION PHASE *\n*********************\n"
    nvcc -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner
  else
    echo -e "Unknown input: $1 \nUSAGE: ./batch_test.sh (-c)"
  fi
else
  echo -e "\n******************************\n* SKIPPING COMPILATION PHASE *\n******************************\n"
fi




echo -e "\n*****************\n* TESTING PHASE *\n*****************\n"
for i in "${workers[@]}"
do
  ./cuda_miner -diff -2 -w $i -t 4 &> ./test_results/worker${i}.txt
  python python/minerPostprocess.py -w $i -wTree $w_tree -pTree $p_tree --silent

  if [ "$multi" == "1" ]; then
    ./cuda_miner -diff -2 -w $i -t 4 --multi &> ./test_results/worker${i}_multi.txt
    python python/minerPostprocess.py -w $i -wTree $w_tree -pTree $p_tree --multi --silent
  fi
done
