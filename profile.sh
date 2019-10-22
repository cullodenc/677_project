#nvcc -D USE_NVTX=1 cuda_miner.cu -o cuda_miner -lnvToolsExt
nvcc -DUSE_NVTX=1 -lineinfo -gencode arch=compute_60,code=sm_60 cuda_miner.cu -o cuda_miner -lnvToolsExt
