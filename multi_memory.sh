nvcc -Xptxas -v -gencode arch=compute_30,code=sm_30 cuda_miner.cu -o cuda_miner
nvcc -Xptxas -v -gencode arch=compute_35,code=sm_35 cuda_miner.cu -o cuda_miner
nvcc -Xptxas -v -gencode arch=compute_50,code=sm_50 cuda_miner.cu -o cuda_miner
nvcc -Xptxas -v -gencode arch=compute_52,code=sm_52 cuda_miner.cu -o cuda_miner
nvcc -Xptxas -v -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner
nvcc -Xptxas -v -gencode arch=compute_61,code=sm_61 cuda_miner.cu -o cuda_miner

#nvcc -Xptxas -v -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 -O2 -o cuda_miner -c cuda_miner.cu
