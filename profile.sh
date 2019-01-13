nvcc -maxrregcount 32 -D USE_NVTX=1 cuda_miner.cu -o cuda_miner -lnvToolsExt
