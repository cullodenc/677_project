#DEVICE DEBUG FUNCTIONS
#nvcc -D DEV_DEBUG=2 -Xptxas -v cuda_miner.cu -o cuda_miner
nvcc -D DEV_DEBUG=2 -Xptxas -v -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner
