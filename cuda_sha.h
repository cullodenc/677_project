#ifndef CUDASHA_H
#define CUDASHA_H

/*************************** HEADER FILES ***************************/
#include <stdlib.h>
#include <memory.h>
#include "sha256.h"
#include <cuda.h>

#include <stdio.h>

/**************************** DATA TYPES ****************************/
typedef unsigned char BYTE;             // 8-bit byte

/*********************** FUNCTION DECLARATIONS **********************/
extern "C" __global__ void genTestHashes(BYTE * hash_df, BYTE * seed, int num_blocks);

//extern "C" __global__ void mineBlock(BYTE * block_d, BYTE * hash_d, BYTE * hash_f, BYTE * nonce_f, BYTE * target, int num_blocks);

//extern "C" __global__ void minerKernel(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, BYTE * hash_i, BYTE * hash_f, SHA256_CTX * ctx, BYTE * target, BYTE * time_d, int * flag_d, int compare, int num_blocks);
extern "C" __global__ void minerKernel(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, BYTE * target, BYTE * time_d, int * flag_d, int compare, int num_blocks);

extern "C" __global__ void benchKernel(BYTE * block_d);

extern "C" __global__ void getMerkleRoot(BYTE * pHash_d, BYTE * pRoot_d, int buffer_blocks);
//extern "C" __global__ void getMerkleRoot(BYTE * pHash_d, BYTE * pRoot_d, int buffer_blocks);

extern "C" __global__ void getMerkleRootCpy(BYTE * pHash_d, BYTE * ret);

//extern "C" __global__ void cudaTest(void);

extern "C" __global__ void threadTest(BYTE * largeVal, BYTE * ret_val);


#endif   // SHA256_H
