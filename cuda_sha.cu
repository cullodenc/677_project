// ECE 677
// Term Project
// Due: December 6, 2018
// Programmer: Connor Culloden

// Compile with nvcc -rdc=true sha256.cu cuda_sha.cu -o cuda_sha
// rdc enables device linking

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <math.h>
#include "sha256.h"
#include <string.h>

#include <cuda.h>

typedef unsigned char BYTE;
//#define num_threads  256
//#define num_blocks 1
#define hash_size 32 * sizeof(BYTE)
#define parent_block_size 16

//extern  __shared__ BYTE local_mem[];

__host__ void getBlockHeader(BYTE * block, BYTE * version, BYTE * prevBlock, BYTE * merkleRoot, BYTE * time_b, BYTE * target, BYTE * nonce);

__global__ void testSha(BYTE * block_d, BYTE * hash_df);

extern "C" __global__ void genTestHashes(BYTE * hash_df, BYTE * seed, int num_blocks);

//extern "C" __global__ void mineBlock(BYTE * block_d, BYTE * hash_d, BYTE * hash_f, BYTE * nonce_f, BYTE * target, int num_blocks);
//extern "C" __global__ void mineBlock(BYTE * block_d, BYTE * hash_d);

//extern "C" __global__ void minerKernel(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, BYTE * hash_i, BYTE * hash_f, SHA256_CTX * ctx, BYTE * target, BYTE * time_d, int * flag_d, int compare, int num_blocks);
extern "C" __global__ void minerKernel(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, BYTE * target, BYTE * time_d, int * flag_d, int compare, int num_blocks);

extern "C" __global__ void benchKernel(BYTE * block_d);

extern "C" __global__ void getMerkleRoot(BYTE * pHash_d, BYTE * pRoot_d,  int buffer_blocks);
//extern "C" __global__ void getMerkleRoot(BYTE * pHash_d, BYTE * pRoot_d, int buffer_blocks);

extern "C" __global__ void getMerkleRootCpy(BYTE * pHash_d, BYTE * ret);

extern "C" __global__ void threadTest(BYTE * largeVal, BYTE * ret_val){
	//SHA256_CTX ctx;
//	printf("MERKLE ROOT COPY TEST PRINT: \n");
		ret_val[threadIdx.x] =  largeVal[threadIdx.x];

//	printf("MERKLE ROOT COPY TEST FINISHED\n");
}
/*
extern "C" __global__ void cudaTest(void){
	//SHA256_CTX ctx;
//	printf("MERKLE ROOT COPY TEST PRINT: \n");
		printf("THREAD %i WORKING\n", threadIdx.x);

//	printf("MERKLE ROOT COPY TEST FINISHED\n");
}
*/

extern "C" __global__ void getMerkleRootCpy(BYTE * pHash_d, BYTE * ret){
	//SHA256_CTX ctx;
//	printf("MERKLE ROOT COPY TEST PRINT: \n");
		ret[threadIdx.x] =  pHash_d[threadIdx.x];

//	printf("MERKLE ROOT COPY TEST FINISHED\n");
}
/*
extern "C" __global__ void getMerkleRoot(BYTE * pHash_d, BYTE * pRoot_d, int buffer_blocks){
	if(threadIdx.x < 32){
		pRoot_d[threadIdx.x] =  pHash_d[threadIdx.x];
	}

}
*/
/*
int main(int argc, char *argv[]){

  // Create event variables for timing
  cudaEvent_t t0, t1, t2;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  cudaEventCreate(&t2);

  float timingResults[3];

  // Allocate the array, with _h for the host and _d for the device based pointers
  BYTE *block_h, *block_d;
//  BYTE *hash_hi, *hash_di;
  BYTE *hash_hf, *hash_df;

  cudaEventRecord(t0);

  size_t size_hash = num_blocks * num_threads *(32 * sizeof(BYTE));
//  hash_hi = (BYTE *) malloc(size_hash);
  hash_hf = (BYTE *) malloc(size_hash);
//  cudaMalloc((void **) &hash_di, size_hash);
  cudaMalloc((void **) &hash_df, size_hash);

  size_t size_block = 80 * sizeof(BYTE);
  block_h = (BYTE *) malloc(size_block);
  cudaMalloc((void **) &block_d, size_block);

  cudaEventRecord(t1);

  BYTE * seed_h;
  BYTE * seed_d;
  size_t size_seed = (30 * sizeof(BYTE));
  seed_h = (BYTE *)malloc(size_seed);
  cudaMalloc((void **) &seed_d, size_seed);

  srand(time(0));
  for(int i = 0; i < 30; i++){
      // Create and store 30 random
      seed_h[i] = (rand() % 256) & 0xFF;
  }
  cudaMemcpy(seed_d, seed_h, size_seed, cudaMemcpyHostToDevice);



  BYTE version[] = {0x01,0x00,0x00,0x00};
  BYTE prevBlock[] = {0x81,0xcd,0x02,0xab,0x7e,0x56,0x9e,0x8b,0xcd,0x93,0x17,0xe2,0xfe,0x99,0xf2,0xde,0x44,0xd4,0x9a,0xb2,0xb8,0x85,0x1b,0xa4,0xa3,0x08,0x00,0x00,0x00,0x00,0x00,0x00};
  BYTE merkleRoot[] = {0xe3,0x20,0xb6,0xc2,0xff,0xfc,0x8d,0x75,0x04,0x23,0xdb,0x8b,0x1e,0xb9,0x42,0xae,0x71,0x0e,0x95,0x1e,0xd7,0x97,0xf7,0xaf,0xfc,0x88,0x92,0xb0,0xf1,0xfc,0x12,0x2b};
  BYTE time_b[] = {0xc7,0xf5,0xd7,0x4d};
  BYTE target[]={0xf2,0xb9,0x44,0x1a};
  BYTE nonce[] = {0x42,0xa1,0x46,0x95};
  printf("Getting Block Header\n");

  // Test printing bytes
  printf("Version String: %02x %02x %02x %02x\n", version[0], version[1], version[2], version[3]);

  getBlockHeader(block_h, version, prevBlock, merkleRoot, time_b, target, nonce);

  cudaMemcpy(block_d, block_h, size_block, cudaMemcpyHostToDevice);

  printf("Starting Kernel\n");
//  testSha <<<num_blocks, num_thread>>>(block_d, hash_di, hash_df);

  genTestHashes<<<num_blocks, num_threads>>>(hash_df, seed_d);
  printf("Finished Kernel\n");

  cudaDeviceSynchronize();

  cudaEventRecord(t2);
  cudaEventSynchronize(t2);

//  cudaMemcpy(hash_hi, hash_di, size_hash, cudaMemcpyDeviceToHost);
  cudaMemcpy(hash_hf, hash_df, size_hash, cudaMemcpyDeviceToHost);

  cudaEventElapsedTime(&timingResults[0], t0, t1);
  cudaEventElapsedTime(&timingResults[1], t1, t2);
  cudaEventElapsedTime(&timingResults[2], t0, t2);
/*
  printf("Hash Results for Thread:\n Intermediate: \t\t");
  for(int i = 0; i < SHA256_BLOCK_SIZE; i++){
    printf("%02x", hash_hi[i]);
  }
  */
/*
  printf("\n\n Final: \t\t\n");
  for(int i = 0; i < num_blocks; i++){
    for(int j = 0; j < num_threads; j++){
      printf("Thread [%i,%i]:", i, j);
      for(int k = 0; k < SHA256_BLOCK_SIZE; k++){
        printf("%02x", hash_hf[i*num_blocks + j*num_threads + k]);
      }
      printf("\n");
    }
  }
  printf("\n\n");

  printResults(hash_hf, num_blocks, num_threads);

  printf("Timing Results:\n Start: %f \n Hash: %f \n", timingResults[0], timingResults[1]);
  printf("finished timing results\n");

  // Free seeds
  free(seed_h);
  cudaFree(seed_d);

  // Free Blocks
  free(block_h);
  printf("Freed host block_h\n");

//  free(hash_hi);
  free(hash_hf);
  printf("Freed hash_h\n");

  cudaFree(block_d);
  printf("Freed block_d\n");

//  cudaFree(hash_di);
  cudaFree(hash_df);
  printf("Freed hash_d\n");

  return 0;
}
*/
/*
__global__ void testSha(BYTE * block_d,  BYTE * hash_df){
	SHA256_CTX ctx;
  // LOCAL Intermediate hash
  BYTE hash_di[] = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};

	sha256_init(&ctx);
	sha256_update(&ctx, block_d, 64);
  sha256_update(&ctx, &(block_d[64]), 16);
	sha256_final(&ctx, hash_di);

  sha256_init(&ctx);
  sha256_update(&ctx, hash_di, 1);
  sha256_final(&ctx, &hash_df[blockIdx.x*num_blocks +threadIdx.x]);


}
*/
/*
__device__ void printHash(BYTE * hash){
	char temp[3];
	BYTE total[65];
	for(int i = 0; i < 32; i+=1){

		sprintf(temp, "%03x", hash[i]);
		total[i*2] = temp[1];
		total[i*2+1] = temp[2];
	}
	total[64] = '\0';
	printf("%s\n", total);
	return;
}
*/
extern "C" __global__ void benchKernel(BYTE * block_d){

	  unsigned int nonce = 0x00000000;
	  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
		int inc_size = blockDim.x * gridDim.x;

		SHA256_CTX thread_ctx;
	  nonce += idx;

	  BYTE threadBlock[80];
		#pragma unroll 80
	  for(int i = 0; i < 80; i++){
	    threadBlock[i] = block_d[i];
	  }


		BYTE hash_t_i[32];
	  BYTE hash_t_f[32];
		#pragma unroll 32
	  for(int i = 0; i < 32; i++){
	    hash_t_i[i] = 0x00;
	    hash_t_f[i] = 0x00;
	  }

		while(nonce < 0x0FFFFFFF){
	    threadBlock[76] = (BYTE)(nonce >> 24) & 0xFF;
	    threadBlock[77] = (BYTE)(nonce >> 16) & 0xFF;
	    threadBlock[78] = (BYTE)(nonce >> 8) & 0xFF;
	    threadBlock[79] = (BYTE)(nonce & 0xFF);

	    sha256_init(&thread_ctx);
	    sha256_update(&thread_ctx, threadBlock, 64);
	//    sha256_update(&thread_ctx, block_d, 64);
	    sha256_update(&thread_ctx, &(threadBlock[64]), 16);
			sha256_final(&thread_ctx, hash_t_i);

	    sha256_init(&thread_ctx);
	  	sha256_update(&thread_ctx, hash_t_i, 32);
	  	sha256_final(&thread_ctx, hash_t_f);

			if(idx == 0){
				printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x \n", hash_t_f[0], hash_t_f[1], hash_t_f[2], hash_t_f[3], hash_t_f[4], hash_t_f[5], hash_t_f[6], hash_t_f[7], hash_t_f[8], hash_t_f[9],\
				hash_t_f[10], hash_t_f[11], hash_t_f[12], hash_t_f[13], hash_t_f[14], hash_t_f[15], hash_t_f[16], hash_t_f[17], hash_t_f[18], hash_t_f[19],\
				hash_t_f[20], hash_t_f[21], hash_t_f[22], hash_t_f[23], hash_t_f[24], hash_t_f[25], hash_t_f[26], hash_t_f[27], hash_t_f[28], hash_t_f[29], hash_t_f[30], hash_t_f[31]);

			}

			nonce += inc_size;
	  }
}


extern "C" __global__ void genTestHashes(BYTE * hash_df, BYTE * seed, int num_blocks){
	SHA256_CTX ctx;
  /*

	sha256_init(&ctx);
	sha256_update(&ctx, block_d, 64);
  sha256_update(&ctx, &(block_d[64]), 16);
	sha256_final(&ctx, hash_di);
*/
  BYTE block = (BYTE)(blockIdx.x & 0xFF);
//  BYTE thread[2];
//  thread[0] = (BYTE)(threadIdx.x >> 8) & 0xFF;
  BYTE thread = (BYTE)(threadIdx.x & 0xFF);
	int offset = 32*threadIdx.x + blockIdx.x * blockDim.x;

//  printf("Thread %i seed: %02x %02x %02x\n", threadIdx.x, block, thread);

  BYTE seed_hash[32];
  #pragma unroll 30
  for(int i = 0; i < 30; i++){
    seed_hash[i] = seed[i];
  }

  seed_hash[30] = block;
  seed_hash[31] = thread;

  sha256_init(&ctx);
  sha256_update(&ctx, seed_hash, 32);
//  sha256_final(&ctx, &hash_df[hash_size*(blockIdx.x*num_blocks +threadIdx.x)]);
	sha256_final(&ctx, &hash_df[offset]);
}
/*
extern "C" __global__ void mineBlock(BYTE * block_d, BYTE * hash_d, BYTE * hash_f, BYTE * nonce_f, BYTE * target, int num_blocks){
//extern "C" __global__ void mineBlock(BYTE * block_d, BYTE * hash_d){
	SHA256_CTX ctx;
  BYTE hash_di[32]; //Intermediate hash
  BYTE hash_df[32]; //Intermediate hash

  unsigned int nonce = 0;
  int iteration = 0;
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int inc_size = blockIdx.x * blockDim.x * num_blocks;
  unsigned int nonce_base = 0;
  nonce = nonce_base + idx;
//  while(nonce < 0xFFFFFFF){
    iteration++;
    if(idx == 0){
      printf("Iteration %i \n", iteration);
    }
    block_d[76] = (BYTE)(nonce >> 24) & 0xFF;
    block_d[77] = (BYTE)(nonce >> 16) & 0xFF;
    block_d[78] = (BYTE)(nonce >> 8) & 0xFF;
    block_d[79] = (BYTE)(nonce & 0xFF);


    sha256_init(&ctx);
    sha256_update(&ctx, &block_d, 64);
    sha256_update(&ctx, &(block_d[64]), 16);
    sha256_final(&ctx, &hash_di);

    sha256_init(&ctx);
  	sha256_update(&ctx, &hash_di, 32);
  	sha256_final(&ctx, &hash_df);



//  }


}
*/

//extern "C" __global__ void minerKernel(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, BYTE * hash_i, BYTE * hash_f, SHA256_CTX * ctx, BYTE * target, BYTE * time_d, int * flag_d, int compare, int num_blocks){
extern "C" __global__ void minerKernel(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, BYTE * target, BYTE * time_d, int * flag_d, int compare, int num_blocks){
//	printf("MINER FUNCTION!!\n");

  int success = 0;

//  BYTE hash_i[32]; //Intermediate hash
//  BYTE hash_f[32];
//  hash_i = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
//  hash_f = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};

  unsigned int nonce = 0x00000000;
  int iteration = 0;
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	//	SHA256_CTX * thread_ctx = (SHA256_CTX*)&(ctx[idx]);
	SHA256_CTX thread_ctx;
	int inc_size = blockDim.x * num_blocks;
	int max_iteration = 0xffffffff / inc_size;
/*
	if(threadIdx.x == 0){
			inc_size = blockDim.x * num_blocks;
			max_iteration = 0xffffffff / inc_size;
	}
*/
//	int inc_size = blockDim.x * num_blocks;
  nonce += idx;
//	int max_iteration = 0xffffffff / inc_size;

  // ONLY NEEDS TO BE 1 FOR FIRST 64 BYTES
/*
	__shared__ BYTE baseBlock[64];
	if(threadIdx.x == 0){
		#pragma unroll 64
		for(int i = 0; i < 64; i++){
			baseBlock[i] = block_d[i];
		}
	}

	// EACH THREAD HAS ITS OWN VARIABLE FOR TOP 16 BYTES
	BYTE threadBlock[16];
	#pragma unroll 16
	for(int i = 0; i < 16; i++){
		threadBlock[i] = block_d[i+64];
	}
*/


  BYTE threadBlock[80];
	#pragma unroll 80
  for(int i = 0; i < 80; i++){
    threadBlock[i] = block_d[i];
  }


	BYTE hash_t_i[32];
  BYTE hash_t_f[32];
	#pragma unroll 32
  for(int i = 0; i < 32; i++){
    hash_t_i[i] = 0x00;
    hash_t_f[i] = 0x00;
  }


//  BYTE * hash_t_i = (BYTE*)&(hash_i[idx*32]);
//  BYTE * hash_t_f = (BYTE*)&(hash_f[idx*32]);
	/*
	#pragma unroll 32
  for(int i = 0; i < 32; i++){
    hash_t_i[i] = 0x00;
    hash_t_f[i] = 0x00;
  }
	*/

//  int count = 0;
//  printf("Nonce: %08x \t Flag: %i\n", nonce, flag_d[0]);
//  while(nonce < 0xFFFFFFFF && flag_d[0] == 0){
	while(flag_d[0] == 0){
		if(iteration < max_iteration){
			iteration++;
		}else{ // UPDATE TIME
			iteration = 0;
			threadBlock[68] = time_d[0];
			threadBlock[69] = time_d[1];
			threadBlock[70] = time_d[2];
			threadBlock[71] = time_d[3];
			if(idx == 0){
				printf("NEW TIME %02x%02x%02x%02x\n\n", time_d[0], time_d[1], time_d[2], time_d[3]);
			}
		}
//		if(idx == 0){
//			printf("%08x\n", nonce);
//		}

  //  if(idx == 0){
  //    printf("Iteration %i: Trying nonce %08x \n", iteration, nonce);
//    }
    threadBlock[76] = (BYTE)(nonce >> 24) & 0xFF;
    threadBlock[77] = (BYTE)(nonce >> 16) & 0xFF;
    threadBlock[78] = (BYTE)(nonce >> 8) & 0xFF;
    threadBlock[79] = (BYTE)(nonce & 0xFF);
  //  hash_in = {"abc"};


    sha256_init(&thread_ctx);
    sha256_update(&thread_ctx, threadBlock, 64);
//    sha256_update(&thread_ctx, block_d, 64);
    sha256_update(&thread_ctx, &(threadBlock[64]), 16);
		sha256_final(&thread_ctx, hash_t_i);

    sha256_init(&thread_ctx);
  	sha256_update(&thread_ctx, hash_t_i, 32);
  	success = sha256_final_target(&thread_ctx, hash_t_f, target, compare);

	//	success = 1;
		/*
		#pragma unroll 32
		for(int i = 0; i < 32; i++){
			hash_t_f[i] = i;
		}

		#pragma unroll 16
		for(int i = 0; i < 16; i++){
			threadBlock[i] = i;
		}
*/

    if(success == 0){
			nonce += inc_size;
		}else{
      flag_d[0] = 1;
      nonce_f[0] = threadBlock[76];
      nonce_f[1] = threadBlock[77];
      nonce_f[2] = threadBlock[78];
      nonce_f[3] = threadBlock[79];
//      printf("Success! Thread %i found winning block! Nonce: %02x%02x%02x%02x\n", idx, nonce_f[0], nonce_f[1], nonce_f[2], nonce_f[3]);
			block_d[76] = threadBlock[76];
			block_d[77] = threadBlock[77];
			block_d[78] = threadBlock[78];
			block_d[79] = threadBlock[79];
/*
			nonce_f[0] = threadBlock[12];
			nonce_f[1] = threadBlock[13];
			nonce_f[2] = threadBlock[14];
			nonce_f[3] = threadBlock[15];
			printf("Success! Thread %i found winning block! Nonce: %02x%02x%02x%02x\n", idx, nonce_f[0], nonce_f[1], nonce_f[2], nonce_f[3]);
			block_d[76] = threadBlock[12];
			block_d[77] = threadBlock[13];
			block_d[78] = threadBlock[14];
			block_d[79] = threadBlock[15];
*/
		//	 printf("Success! Thread %i found winning block! Nonce: %02x%02x%02x%02x\n", idx, nonce_f[0], nonce_f[1], nonce_f[2], nonce_f[3]);
/*
			#pragma unroll 64
			for(int i = 0; i < 64; i++){
				block_d[i] = baseBlock[i];
			}
			#pragma unroll 16
      for(int i = 0; i < 16; i++){
        block_d[i+64] = threadBlock[i];
      }
*/
			#pragma unroll 80
			for(int i = 0; i < 80; i++){
				block_d[i] = threadBlock[i];
			}
			#pragma unroll 32
      for(int i = 0; i < 32; i++){
        hash_d[i] = hash_t_f[i];
      }
      break;
    }
//    count++;
  }



}

/*
// UPDATED MINER KERNEL FOR USE WITH 3D ADDRESSING
extern "C" __global__ void minerKernel(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, BYTE * target, BYTE * time_d, int * flag_d, int compare, int num_blocks){
	SHA256_CTX ctx;
  int success = 0;

//  BYTE hash_i[32]; //Intermediate hash
//  BYTE hash_f[32];
//  hash_i = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
//  hash_f = {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};

  unsigned int nonce = 0x00000000;
  int iteration = 0;
	// X-DIM = 16
	// Y-DIM = 16
	// Z-DIM = 4
	// BLOCK INC = 0x400
	//								0x00 -> 0x0F    0x10 -> 0xF0         	     0x000 -> 0x300
	unsigned int idx = threadIdx.x + blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z + blockDim.x*blockDim.y*blockDim.z*blockIdx.x;
//  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
//  int inc_size = blockDim.x * num_blocks;
	int inc_size = blockDim.x*blockDim.y*blockDim.z*gridDim.x;
//  printf("Increment Size: %i \t BlockDim: %i \n", inc_size, blockDim.x);
  nonce += idx;
	int max_iteration = 0xffffffff / inc_size;

  BYTE threadBlock[80];
	#pragma unroll 80
  for(int i = 0; i < 80; i++){
    threadBlock[i] = block_d[i];
  }

  BYTE hash_i[32];
  BYTE hash_f[32];
	#pragma unroll 32
  for(int i = 0; i < 32; i++){
    hash_i[i] = 0x00;
    hash_f[i] = 0x00;
  }

//  int count = 0;
//  printf("Nonce: %08x \t Flag: %i\n", nonce, flag_d[0]);
//  while(nonce < 0xFFFFFFFF && flag_d[0] == 0){
	while(flag_d[0] == 0){
		if(iteration < max_iteration){
			iteration++;
		}else{ // UPDATE TIME
			iteration = 0;
			threadBlock[68] = time_d[0];
			threadBlock[69] = time_d[1];
			threadBlock[70] = time_d[2];
			threadBlock[71] = time_d[3];
			if(idx == 0){
				printf("NEW TIME %02x%02x%02x%02x\n\n", time_d[0], time_d[1], time_d[2], time_d[3]);
			}
		}

  //  if(idx == 0){
  //    printf("Iteration %i: Trying nonce %08x \n", iteration, nonce);
//    }
    threadBlock[76] = (BYTE)(nonce >> 24) & 0xFF;
    threadBlock[77] = (BYTE)(nonce >> 16) & 0xFF;
    threadBlock[78] = (BYTE)(nonce >> 8) & 0xFF;
    threadBlock[79] = (BYTE)(nonce & 0xFF);
  //  hash_in = {"abc"};

//    printf("threadBlock: ");
//    for(int i = 0; i < 80; i++){
//      printf("%02x", threadBlock[i]);
//    }
//    printf("\n");


    sha256_init(&ctx);
    sha256_update(&ctx, threadBlock, 64);
//    sha256_update(&ctx, block_d, 64);
    sha256_update(&ctx, &(threadBlock[64]), 16);
    sha256_final(&ctx, hash_i);

//    printf("Intermediate Hash: ");
//    for(int i = 0; i < 80; i++){
//      printf("%02x", hash_i[i]);
//    }
//    printf("\n");


    sha256_init(&ctx);
  	sha256_update(&ctx, hash_i, 32);
  	success = sha256_final_target(&ctx, hash_f, target, compare);

//    printf("Final Hash: ");
//    for(int i = 0; i < 80; i++){
//      printf("%02x", hash_f[i]);
//    }
//    printf("\n");


    if(success == 0){
			nonce += inc_size;
		}else{
      flag_d[0] = 1;
      nonce_f[0] = threadBlock[76];
      nonce_f[1] = threadBlock[77];
      nonce_f[2] = threadBlock[78];
      nonce_f[3] = threadBlock[79];
//      printf("Success! Thread %i found winning block! Nonce: %02x%02x%02x%02x\n", idx, nonce_f[0], nonce_f[1], nonce_f[2], nonce_f[3]);
			block_d[76] = threadBlock[76];
			block_d[77] = threadBlock[77];
			block_d[78] = threadBlock[78];
			block_d[79] = threadBlock[79];

		//	 printf("Success! Thread %i found winning block! Nonce: %02x%02x%02x%02x\n", idx, nonce_f[0], nonce_f[1], nonce_f[2], nonce_f[3]);
			#pragma unroll 80
      for(int i = 0; i < 80; i++){
        block_d[i] = threadBlock[i];
      }
			#pragma unroll 32
      for(int i = 0; i < 32; i++){
        hash_d[i] = hash_f[i];
      }
      break;
    }
//    count++;
  }



}
*/

extern "C" __global__ void getMerkleRoot(BYTE * pHash_d, BYTE * pRoot_d, int buffer_blocks){
	SHA256_CTX ctx;
	// Shared memory for sharing hash results
//	BYTE * shared_data = (BYTE*)local_mem;
	__shared__ BYTE local_mem_in[parent_block_size][64];
	__shared__ BYTE local_mem_out[parent_block_size][32];

//	BYTE * hash_ref = (BYTE*)&pHash_d[32*threadIdx.x];
	int tree_size = pow(2.0, ceil(log2((double)buffer_blocks)));
/*
	if(threadIdx.x == 0){
		printf("TREE SIZE: %i\n", tree_size);
	}
*/



	// SET UP HASH TREE THREADS
	if(threadIdx.x < buffer_blocks){
		//SET UP UNIQUE THREADS
//		if(threadIdx.x < buffer_blocks){
				for(int i = 0; i < 32; i++){
			//			local_mem_in[threadIdx.x][i] = hash_ref[i];
					local_mem_in[threadIdx.x][i] = pHash_d[threadIdx.x*32+i];
				}
//		}else{ // SET UP PADDING BRANCHES
	//		for(int i = 0; i < 32; i++){
		//			local_mem_in[threadIdx.x][i] = hash_ref[i];
//				local_mem_in[threadIdx.x][i] = pHash_d[(buffer_blocks-1)*32+i];
//			}
//		}


		// Calculate first hash, store in shared memory
		sha256_init(&ctx);
		sha256_update(&ctx, local_mem_in[threadIdx.x], 32);
		sha256_final(&ctx, local_mem_out[threadIdx.x]);

		#pragma unroll 32
		for(int i = 0; i < 32; i++){
			local_mem_in[threadIdx.x][i] = local_mem_out[threadIdx.x][i];
		}

		sha256_init(&ctx);
		sha256_update(&ctx, local_mem_in[threadIdx.x], 32);
		sha256_final(&ctx, local_mem_out[threadIdx.x]);

		// Sequential hash reduction
		// First iteration 0 = 0|1	2=2|3 	4=4|5		6=6|7
		// Second iteration 0 = (0|1)|(2|3) 	4=(4|5)|(6|7)
		// Third iteration 0 = ((0|1)|(2|3))|((4|5)|(6|7)), etc...

		// For 32:
		// 2  -> 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
		// 4  -> 0, 		4, 		8, 		 12, 		 16, 		 20,		 24, 		 28,
		// 8  -> 0, 		  		8, 		    		 16, 		    		 24,
		// 16 -> 0, 												 16,
		// 32 -> 0

//		printf("MERKLE TREE LEVEL 1 THREAD %i HASH: %02x%02x%02x%02x\n", threadIdx.x, local_mem_out[threadIdx.x][0], local_mem_out[threadIdx.x][1], local_mem_out[threadIdx.x][2], local_mem_out[threadIdx.x][3]);

		// Progressively loop to combine hashes
//		printf("\n\nSTART LOOP THREAD %i\n\n", threadIdx.x);
//		for(int i = 2; i <= buffer_blocks; i*=2){
		for(int i = 2; i <= tree_size; i*=2){
			if(threadIdx.x % i == 0){
				int mid = i/2;
				if(threadIdx.x + mid < buffer_blocks){
//					printf("MERKLE TREE LEVEL %i THREAD %i \n", i, threadIdx.x);

					#pragma unroll 32
					for(int j = 0; j < 32; j++){
						local_mem_in[threadIdx.x][j] = local_mem_out[threadIdx.x][j];
						local_mem_in[threadIdx.x][32+j]= local_mem_out[threadIdx.x+mid][j];
					}
				}else{ // HASH TOGETHER DUPLICATES FOR UNMATCHED BRANCHES
//					printf("MERKLE TREE EDGE LEVEL %i THREAD %i \n", i, threadIdx.x);
					#pragma unroll 32
					for(int j = 0; j < 32; j++){
						local_mem_in[threadIdx.x][j] = local_mem_out[threadIdx.x][j];
						local_mem_in[threadIdx.x][32+j]= local_mem_out[threadIdx.x][j];
					}
				}


				sha256_init(&ctx);
				sha256_update(&ctx, local_mem_in[threadIdx.x], 64);
				sha256_final(&ctx, local_mem_out[threadIdx.x]);

				#pragma unroll 32
				for(int j = 0; j < 32; j++){
					local_mem_in[threadIdx.x][j] = local_mem_out[threadIdx.x][j];
				}

				sha256_init(&ctx);
				sha256_update(&ctx, local_mem_in[threadIdx.x], 32);
				sha256_final(&ctx, local_mem_out[threadIdx.x]);

//				printf("MERKLE TREE LEVEL %i THREAD %i \n\
HASH 1: %02x%02x%02x%02x\nHASH 2: %02x%02x%02x%02x\nHASH OUT: %02x%02x%02x%02x\n\n", i, threadIdx.x, local_mem_in[threadIdx.x][0], local_mem_in[threadIdx.x][1], local_mem_in[threadIdx.x][2], local_mem_in[threadIdx.x][3], local_mem_in[threadIdx.x][32], local_mem_in[threadIdx.x][33], local_mem_in[threadIdx.x][34], local_mem_in[threadIdx.x][35], local_mem_out[threadIdx.x][0], local_mem_out[threadIdx.x][1], local_mem_out[threadIdx.x][2], local_mem_out[threadIdx.x][3]);

			}
		}
		// All values coalesce into thread 0 shared memory space, and then get read back
		if(threadIdx.x == 0){
			for(int i = 0; i < 32; i++){
		//			local_mem_in[threadIdx.x][i] = hash_ref[i];
				pRoot_d[i] = local_mem_out[0][i];
	//			local_mem_in[threadIdx.x][i] = pHash_d[(buffer_blocks-1)*32+i];
			}
		}

	}
/*
	if(threadIdx.x < 32){
		pRoot_d[threadIdx.x] =  local_mem_in[0][threadIdx.x];
	}
*/
}
