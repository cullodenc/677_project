#!/bin/bash
# Start verification testing

# Device Limitation options
# NUMBER OF AVAILABLE MULTIPROCESSORS ON THE SYSTEM   Default: 10, system dependent
num_multiprocessors=10

#CUDA CAPABILITY VERSION FOR CODE GENERATION
cc=60

# ENABLE PYTHON POSTPROCESSING (Requires xlsxwriter and Python 2.7)
use_py=0




# TESTING PARAMETERS
workers=(1 2 4 8 16)
multi=1

w_tree=4
p_tree=32
# Difficulty
diff=0
dscale=0
dbits=0  # 224 is good, 240 better
dlimit=32
# Number of rounds to run
# 250 for extended test
# 30 minutes: 1 worker, 1 hour: 2 workers
rounds=400

time_min=120
time_sec=$((60*$time_min))




echo -e "*********************************************************\n\t\tVERIFICATION BATCH TEST\n*********************************************************"


# VARIABLE ASSIGNMENTS
SM="-DSM=$num_multiprocessors"


# Detect compile flag if present
if [ "$1" != "" ]; then
  if [ "$1" == "-c" ]; then
    echo -e "\n*********************\n* COMPILATION PHASE *\n*********************\n"
    nvcc $SM -gencode arch=compute_${cc},code=sm_${cc} cuda_miner.cu -o cuda_miner
  else
    echo -e "Unknown input: $1 \nUSAGE: ./batch_test.sh (-c)"
  fi
else
  echo -e "\n******************************\n* SKIPPING COMPILATION PHASE *\n******************************\n"
fi




echo -e "\n*****************\n* TESTING PHASE *\n*****************\n"
for i in "${workers[@]}"
do
  #worker_rounds=$(($i*$rounds))
  worker_rounds=$(($rounds))
  #./cuda_miner -diff $diff -w $i -t $worker_rounds -wTree $w_tree -pTree $p_tree -dscale $dscale -dlimit $dlimit &> ./test_results/worker${i}.txt
  #./cuda_miner -diff $diff -w $i -t $worker_rounds -wTree $w_tree -pTree $p_tree -dscale $dscale -dlimit $dlimit | tee ./test_results/worker${i}.txt
  #python python/minerPostprocess.py -w $i -wTree $w_tree -pTree $p_tree --silent
  #CHANGED NEW PYTHON SCRIPT ITERATES OVER ALL WORKERS IN THE INPUT ARRAY
  #python python/minerPostprocess.py -w ${workers[@]} -wTree $w_tree -pTree $p_tree --silent

  if [ "$multi" == "1" ]; then
    echo "Multilevel Enabled"
    #./cuda_miner -diff $diff -w $i -t $worker_rounds --multi -wTree $w_tree -pTree $p_tree -dscale $dscale -dlimit $dlimit &> ./test_results/worker${i}_multi.txt

    # CHANGED COMMENT THIS SECTION OUT TEMPORARILY TO AVOID OVERWRITING MINING DATA
    ./cuda_miner -diff $diff -w $i -t $worker_rounds --multi -wTree $w_tree -pTree $p_tree -dscale $dscale -dlimit $dlimit -dbits $dbits -timeout $time_sec &> ./test_results/worker${i}_multi.txt
    #python python/minerPostprocess.py -w $i -wTree $w_tree -pTree $p_tree --multi --silent
  else
    ./cuda_miner -diff $diff -w $i -t $worker_rounds -wTree $w_tree -pTree $p_tree -dscale $dscale -dlimit $dlimit -dbits $dbits -timeout $time_sec &> ./test_results/worker${i}.txt

  fi
done

if [ "$use_py" == "1" ]; then
echo "STARTING PYTHON SCRIPTS"
python python/minerPostprocess.py -w ${workers[@]} -wTree $w_tree -pTree $p_tree --silent
python python/minerPostprocess.py -w ${workers[@]} -wTree $w_tree -pTree $p_tree --multi --silent

else
echo "PYTHON SCRIPTS DISABLED"
fi
