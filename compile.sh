#!/bin/bash
# General compilation script for the application
################################################
####### Compilation Option Settings ###########
###############################################

# NUMBER OF AVAILABLE MULTIPROCESSORS ON THE SYSTEM   Default: 10, system dependent
num_multiprocessors=10

#CUDA CAPABILITY VERSION FOR CODE GENERATION
cc=60


# NUMBER OF THREADS PER BLOCK   Default: 1024, should be a power of 2
block_size=1024

# NUMBER OF MULTIPROCESSORS USED BY THE PARENT MINER (Default: 2) Must be non-zero, and greater than 0
parent_procs=2

# HARDWARE DEBUGGING LEVEL (0-3)  Default: 1, disable with 0
debug_level=1

#ENABLE PROFILING FEATURES (0-1) Default: 0 (disabled) DO NOT USE IF PROFILING LIBRARY IS UNAVAILABLE
profile=0

#ENABLE MEMORY CHECK (0-1) Default: 0 (disabled) MUST BE TURNED OFF FOR MINING TO FUNCTION PROPERLY
mem_check=0




###############################################
##### Compilation Argument Generation #########
###############################################

# VARIABLE ASSIGNMENTS
SM="-DSM=$num_multiprocessors"

THREADS="-DCUSTOM_THREADS=$block_size"

PARENT="-DPARENT_PROC=$parent_procs"


if [ "$debug_level" == "0" ]; then
  # Empty debug parameter
  DEBUG=""
else
  # Set the desired debug level argument
  DEBUG="-DDEV_DEBUG=$debug_level"
fi


if [ "$profile" == "0" ]; then
  # Empty debug parameter
  PROFILE=""
else
  # Set the desired debug level argument
  PROFILE="-DUSE_NVTX=1 -lnvToolsExt"
fi


if [ "$mem_check" == "0" ]; then
  # Empty debug parameter
  MEM_CHECK=""
else
  # Set the desired debug level argument
  MEM_CHECK="-DMEM_CHECK -lineinfo -Xptxas -v"
fi



###############################################
######## Compilation Function Call ############
###############################################
#original
#nvcc -DDEV_DEBUG=1 -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner
nvcc $SM $DEBUG $THREADS $PARENT $MEM_CHECK $PROFILE -gencode arch=compute_${cc},code=sm_${cc} cuda_miner.cu -o cuda_miner
