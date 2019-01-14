nvcc -maxrregcount 32 -Xptxas -v cuda_miner.cu -o cuda_miner
#nvcc -maxrregcount 32 -Xptxas -v -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner
