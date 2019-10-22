#!/bin/bash
# Start verification testing
# USAGE:
# verify.sh [Num Workers] [Multilevel?]

# Default testing values
NUM_WORKERS=1
MULTILEVEL=0

################ PARSE INPUT ARGUMENTS ################

# Detect command line arguments if present
if [ "$1" != "" ]; then
  if ! [[ "$1" =~ ^[1-9]+$ ]]; then
    echo "Invalid number of workers selected"
  else
    NUM_WORKERS=$1
  fi
fi

if [ "$2" != "" ]; then
  if ! [[ "$2" =~ ^[0-1]+$ ]]; then
    echo "Invalid chain level selected"
  else
    MULTILEVEL=$2
  fi
fi



################ RUN VERIFICATION TEST ################
if [ "$MULTILEVEL" == 0 ]; then
  echo "Starting verification test with $NUM_WORKERS workers"
  #./cuda_miner --debug -w $NUM_WORKERS
  #python verify.py $NUM_WORKERS $MULTILEVEL
  #python python/verify.py $NUM_WORKERS $MULTILEVEL
  python python/minerPostprocess.py -w $NUM_WORKERS
else
  echo "Starting multilevel verification test with $NUM_WORKERS workers"
  python python/minerPostprocess.py -w $NUM_WORKERS --multi
  #./cuda_miner --debug -w $NUM_WORKERS --multi
fi
