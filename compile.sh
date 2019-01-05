#nvcc -rdc=true sha256.cu cuda_sha.cu cuda_miner.cu -o cuda_miner
#nvcc -rdc=true sha256.cu cuda_miner.cu -o cuda_miner
#nvcc -rdc=true sha256_cpy.cu cuda_miner.cu -o cuda_miner
nvcc cuda_miner.cu -o cuda_miner
