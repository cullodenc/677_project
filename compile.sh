nvcc -D DEV_DEBUG=1 -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner
