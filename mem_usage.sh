#nvcc -maxrregcount 32 -Xptxas -v cuda_miner.cu -o cuda_miner
#nvcc -Xptxas -v cuda_miner.cu -o cuda_miner
#nvcc -D DEV_DEBUG=1 -Xptxas -v cuda_miner.cu -o cuda_miner
#nvcc -maxrregcount 32 -Xptxas -v -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner
#nvcc -Xptxas -v -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner

# GENERATE LINE NUMBER INFORMATION FOR DEVICE CODE
nvcc -lineinfo -Xptxas -v -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner
# DEBUG DEVICE OPTIONS ENABLED
#nvcc -G -Xptxas -v -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner
