
# FULL BATCH TEST
NUM_THREADS=(256 512 1024)
workers=(1 2 4 8 16)

difficulty=-2


echo "*************************** STARTING BENCHMARK BATCH TEST***************************"

bin_dir="bin/benchmark/"
if [[ ! -e $bin_dir ]]; then
  mkdir -p $bin_dir
fi

# Detect compile flag if present
if [ "$1" != "" ]; then
  if [ "$1" == "-c" ]; then
    echo -e "\n*********************\n* COMPILATION PHASE *\n*********************\n"
    for j in "${NUM_THREADS[@]}"
    do
      echo -e "\n\nCOMPILING ARCHITECTURE FOR $j THREADS PER BLOCK \n"
      nvcc -DCUSTOM_THREADS=$j -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o ${bin_dir}cuda_miner$j
    done
  else
    echo -e "Unknown input: $1 \nUSAGE: ./thread_test.sh (-c)"
  fi
else
  echo -e "\n******************************\n* SKIPPING COMPILATION PHASE *\n******************************\n"
  # COMPILE ANYWAY IF THE FILES DONT EXIST YET
  for j in "${NUM_THREADS[@]}"
  do
    if [[ ! -e ${bin_dir}cuda_miner$j ]]; then
      echo -e "\n\nEXECUTABLE FOR $j THREADS PER BLOCK NOT FOUND, COMPILING ARCHITECTURE \n"
      nvcc -DCUSTOM_THREADS=$j -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o ${bin_dir}cuda_miner$j
    fi
  done
fi


for j in "${NUM_THREADS[@]}"
do
  echo -e "\n\nBENCHMARKS FOR $j THREADS PER BLOCK \n"
  for i in "${workers[@]}"
  do
    # Create output directories if neccessary
    dir_single="outputs/benchtest/results_${i}_chains/"
    dir_multi="outputs/benchtest/results_${i}_pchains/"

    if [[ ! -e $dir_single ]]; then
      mkdir -p $dir_single
    fi
    if [[ ! -e $dir_multi ]]; then
      mkdir -p $dir_multi
    fi

    echo "STARTED $i WORKER BENCHMARK - SINGLE LEVEL"
    ./${bin_dir}cuda_miner$j -w $i -diff $difficulty --dryrun --benchmark > ${dir_single}bench${j}.log

    echo "STARTED $i WORKER BENCHMARK - MULTILEVEL"
    ./${bin_dir}cuda_miner$j -w $i -diff $difficulty --dryrun --benchmark  --multi > ${dir_multi}bench${j}.log

  done
done

echo "STARTING PYTHON SCRIPT"
python ./python/read_benchmarks.py ./outputs/benchtest/
