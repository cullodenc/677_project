// ECE 677
// Term Project
// Programmer: Connor Culloden

/* PROJECT DESCRIPTION
 *********************************************************************************
 * The following program was developed to test the performance potential of
 * blockchain based applications which utilize several child blockchains which are
 * unified under a single parent blockchain, forming a tree-like structure.
 * Such a framework could potentially enable much higher transaction verification
 * rates across the framework, as much of the work would be performed on child chains
 * which use a traditional Proof of Work consensus protocol to maintain security.
 * This can enable the parent chain to oversee the contributing child chains
 * using a less intensive protocol such as Proof of Stake, while mitigating the
 * possibility for the 'nothing at stake' problem to arise. Further enhancements
 * may allow the framework to operate with far lower memory requirements as users
 * and miners for each child chain would only need the subset of transaction data
 * to verify a transactions authenticity. The parent-child architecture also allows
 * for pruning of spent chains to reduce the total framework memory overhead.
 *
 * This particular project will primarily focus on the mining comparison across
 * various architecture cores, and many of the critical features for a fully
 * functional blockchain application are not present, and are planned to be
 * implemented in the future if promising results are obtained.
 *
 * The algorithm utilized here follows a very similar framework to Bitcoin,
 * sharing the same basis for Block Headers and double SHA256 hashing. The input
 * 'transaction' data consists of a set of randomly generated hash values which
 * are created as needed, and are representative of the merkle roots for blocks
 * of transactions. Due to this, the program relies soley on variations in the
 * nonce and time fields when searching for a solution to each block.
 *
 * Parent chain architectures include a Merkle Tree hashing algorithm to collect
 * child transactions into a merkle root, though these parent blocks are of a user
 * specified fixed size to keep things simple.
 *
 * This implementation was designed to run multiple mining algorithms on a
 * single CUDA enabled GPU (compute compatibility 6.1) to best fit the testing
 * environment, though this could be extended to multiple GPUs of a different
 * generation with a bit of modification. Running this application across many
 * GPU clusters may require a bit more effort, as an intermediate framework would
 * most likely be neccessary to enable intercluster communication.
 *
 *********************************************************************************
 * PROGRAM PARAMETERS
 *********************************************************************************
 * Many of the program parameters are modifiable using various command line
 * arguments, which enable the testing and comparison of various architectures,
 * and allow for other uses such as code profiling and benchmarking. Mining options
 * are also available to scale this application to meet hardware constraints,
 * such as initial difficulty targets and exit conditions, which can
 * drastically reduce the work required to test an architecture.
 *
 * The difficulty scaling utilized here has also been modified a fair amount
 * compared to traditional blockchain architectures, as it is designed to sweep
 * over a range of difficulty targets, instead of changing to maintain a consistent
 * mining rate across a network. The difficulty is incremented bytewise, creating
 * 255 (0xFF) difficulty levels for each target exponent. This combined with
 * the ability to lower the diffiulty adjustment period allows a large range of
 * diffiulties to be tested in a matter of hours instead of weeks.
 *
 *********************************************************************************
 * PROGRAM USAGE
 *********************************************************************************
 * This program can be compiled by running the included bash script 'compile.sh'
 * This operation can also be performed on non-linux based systems using the
 * following command: FIXME: THIS IS PROBABLY GOING TO CHANGE IN THE FUTURE
 *                   nvcc -rdc=true sha256.cu cuda_sha.cu host.cu -o cuda_sha
 *
 * Once compiled, the program can be run by executing the created executable,
 * followed by a list of run options which determine the architecture and many
 * other optional features.
 * To find out more, try using the '--help' option to see an updated list of
 * accepted parameters.
 *
 * The main mining operation produces numerous output files in a unique directory (FIXME)
 * located in either the default 'outputs' folder, or a user specified folder (FIXME)
 * For each worker chain, the folder will contain an outputs_#.txt file,
 * which displays the basic information for each block mined, along with some
 * timing statistics for each difficulty level. An error file is also provided to
 * isolate error messages created by events such as when the end of an input file
 * is reached or when the parent chain buffer fills up before the previous block
 * has finished, creating a lag in the system.
 *
 * Multilevel architectures also include a file to detail the hashes that went into
 * each parent block and the total time taken to fill the parent buffer (pHashOutputs),
 * and a file that consolidates the parent blocks, along with the timing statistics
 * for each parent difficulty level.
 */

/* TECHNICAL REFERENCE
 *********************************************************************************
 * Each block header follows the same structure used for the Bitcoin blockchain
 * The total block size is 80 Bytes, with the following breakdown
 *_____________________________________________________________________________
 *______NAME______|___SIZE___|___________________DESCRIPTION___________________|
 * Version        | 4  Bytes | Software Version                                |
 * hashPrevBlock  | 32 Bytes | Hash of the previous block in the chain         |
 * hashMerkleRoot | 32 Bytes | Merkle Root of the current block                |
 * Time           | 4  Bytes | Current Timestamp (sec) since last Epoch        |
 * Bits           | 4  Bytes | Compact form of the target difficulty           |
 * Nonce          | 4  Bytes | Variable value to try and find a solution       |
 *------------------------------------------------------------------------------
 *
 * The algorithm implemented uses a constant software version, and a zero value
 * initial previous block hash. The rest of the chain builds off of this.
 * The mining algorithm also varies a bit from the standard bitcoin algorithm by
 * updating the time after all nonces have been tried, and resetting the nonce
 * to zero. This eliminates some of the additional complexity that would result
 * from constantly modifying the time, or implementing the extraNonce value.
 *
 * More details on the block hashing algorithm can be found here:
 * https://en.bitcoin.it/wiki/Block_hashing_algorithm
 *
 */


/******************************************************************************
 ****************************** TREE OF CONTENTS ******************************
 ******************************************************************************
 cuda_miner
 │
 ├───PREPROCESSOR DIRECTIVES
 │   ├───Library Inclusions
 │   ├───Type Definitions
 │   ├───Macro Definitions
 │   └───Constant Definitions
 │
 ├───DECLARATIONS
 │   ├───Global Variable Declarations
 │   └───Function Declarations
 │
 └───FUNCTION DEFINITIONS
     ├───Main Function
     ├───Host Core Process
     ├───HOST_FUNCTIONS
     │   ├───TESTING
     │   │   ├───hostDeviceQuery
     │   │   ├───hostFunctionalTest
		 │   │   ├───testMiningHash
		 │   │   ├───miningBenchmarkTest
     FIXME
     │   ├───MEMORY
     │   │   ├───ALLOCATION
     │   │   │   ├───allocWorkerMemory
     │   │   │   ├───allocParentMemory
     │   │   │   ├───allocMiningMemory
     │   │   │   └───allocFileStrings
     │   │   │
     │   │   ├───FREEING
     │   │   │   ├───freeWorkerMemory
     │   │   │   ├───freeParentMemory
     │   │   │   ├───freeMiningMemory
     │   │   │   └───freeFileStrings
     │   │   │
     │   │   ├───CUDA
     │   │   │   ├───createCudaVars
     │   │   │   └───destroyCudaVars
     │   │   │
     │   │   └───TIMING
     │   │       ├───initTime
     │   │       └───freeTime
     │   │
     │   ├───MINING
     │   │   ├───INITIALIZATION
     │   │   │   ├───initializeBlockHeader
     │   │   │   ├───initializeWorkerBlock
     │   │   │   └───initializeParentBlock
     │   │   │
     │   │   ├───UPDATE
     │   │   │   ├───updateBlock
     │   │   │   ├───updateParentRoot
     │   │   │   ├───updateParentHash
     │   │   │   ├───updateDifficulty
     │   │   │   └───updateTime
     │   │   │
     │   │   ├───GETTERS
     │   │   │   ├───getTime
     │   │   │   └───getDifficulty
     │   │   │
     │   │   └───CALCULATIONS
     │   │       ├───calculateDifficulty
     │   │       ├───calculateTarget
		 │   │       └───calculateMiningTarget
     │   │
     │   ├───KERNELS
     │   │   ├───launchGenHash
     │   │   ├───launchMerkle
     │   │   ├───launchMiner
     │   │   └───returnMiner
     │   │
     │   ├───UTILITIES
     │   │   ├───HEX_CONVERSION
     │   │   │   ├───encodeHex
     │   │   │   ├───decodeHex
     │   │   │   ├───printHex
     │   │   │   └───printHexFile
     │   │   │
     │   │   └───LOGGING
     │   │       ├───printLog
     │   │       ├───printDebug
     │   │       ├───printError
     │   │       ├───logStart
     │   │       └───printProgress
     │   │
     │   └───I/0
     │       ├───INPUT
     │       │   ├───initializeHashes
     │       │   ├───initializeInputFile
     │       │   ├───printInputFile
     │       │   └───readNextHash
     │       │
     │       └───OUTPUT
     │           ├───initializeOutputs
     │           ├───initializeParentOutputs
     │           ├───printDifficulty
     │           ├───printErrorTime
     │           └───printOutputFile
     │
     ├───GLOBAL_FUNCTIONS
     │   ├───benchmarkKernel
     │   ├───hashTestKernel
     │   ├───genHashKernel
     │   ├───minerKernel
     │   └───merkleKernel
     │
     └───DEVICE_FUNCTIONS
		 		 ├───get_smid
		 		 ├───get_warpid
		 	 	 ├───get_laneid
         ├───printHash
				 ├───printBlock
				 ├───sha256_mining_transform
				 ├───sha256_mining_transform_short
				 ├───scheduleExpansion
				 ├───scheduleExpansion_short
				 └───sha256_blockHash
*/



/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/**************************  ___________________________________________________________________________________________________________________  **************************/
/**************************  |    _____    _____    ______   _____    _____     ____     _____   ______    _____    _____    ____    _____     |  **************************/
/**************************  |   |  __ \  |  __ \  |  ____| |  __ \  |  __ \   / __ \   / ____| |  ____|  / ____|  / ____|  / __ \  |  __ \    |  **************************/
/**************************  |   | |__) | | |__) | | |__    | |__) | | |__) | | |  | | | |      | |__    | (___   | (___   | |  | | | |__) |   |  **************************/
/**************************  |   |  ___/  |  _  /  |  __|   |  ___/  |  _  /  | |  | | | |      |  __|    \___ \   \___ \  | |  | | |  _  /    |  **************************/
/**************************  |   | |      | | \ \  | |____  | |      | | \ \  | |__| | | |____  | |____   ____) |  ____) | | |__| | | | \ \    |  **************************/
/**************************  |   |_|      |_|  \_\ |______| |_|      |_|  \_\  \____/   \_____| |______| |_____/  |_____/   \____/  |_|  \_\   |  **************************/
/**************************  |_________________________________________________________________________________________________________________|  **************************/
/**************************                                                                                                                       **************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>     // NEEDED FOR DIRECTORY CREATION
#include <math.h>         // NEEDED FOR MORE COMPLEX MATH
#include <string.h>       // NEEDED FOR STRING OPERATIONS
#include <ctype.h>        // NEEDED FOR char OPERATION tolower
#include <time.h>         // NEEDED FOR TIMESTAMPING

// libraries for sha256
#include <stddef.h>
#include <memory.h>

// NOTE USED FOR ALTERNATIVE TIMING TO FIX TIMING UPDATE BUG
// USED TO QUERY NUMBER OF CPU THREADS SUPPORTED (linux only)
//#include <unistd.h>

// CODE TO QUERY NUMBER OF THREADS AVAILABLE
//int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
//printf("Detected %i threads supported by this system\n", numCPU);


#include <cuda.h>
// INCLUDE PROFILER LIBRARIES IF USE_NVTX IS ENABLED IN NVCC COMPILE
#ifdef USE_NVTX
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <nvToolsExtCuda.h>
#include <nvToolsExtCudaRt.h>
#endif

/***************************************************************************************************************************************************************************/
/*****************************************************************************TYPE DEFINITIONS******************************************************************************/
/***************************************************************************************************************************************************************************/

typedef unsigned char BYTE;             // 8-bit byte
typedef unsigned int  WORD;             // 32-bit word

typedef struct{
	// ID OF THE CURRENT WORKER
	int id;
	/*----------------------------MAIN VARIABLES-----------------------------*/
	WORD * block_h;				// Host storage for current block
	WORD * block_d;				// Device storage for current block

	WORD * buffer_h;				// Host buffer for merkle hashing
	WORD * buffer_d;				// Device buffer for merkle hashing

	WORD * hash_h;				// Host storage for the result hash
	WORD * hash_d;				// Device storage for the result hash

	// Variables for storing the intermediate hash of the constant block header
	WORD * basestate_h;		// Host storage for the base state, copied to constant memory for mining
	WORD * basestate_d;		// Device storage for the base state, can be used to either compute the basestate on device side, or pass in the basestate to the miner

	BYTE *hash_byte;						// Device byte storage for result hash

	int buff_size;					// MAXIMUM BUFFER SIZE
	int buff_blocks;

	/*----------------------------CUDA VARIABLES-----------------------------*/
	// STREAMS
	cudaStream_t stream;
	// TODO ADD H2D AND D2H STREAMS HERE
	// EVENTS
	cudaEvent_t t_start, t_stop;
	cudaEvent_t t_diff_start, t_diff_stop;
	// TIMING VARS
	float t_result;
	float t_diff;
	/*---------------------------IO FILE VARIABLES---------------------------*/
	FILE * inFile;
	char outFile[50];
	int readErr;

	/*----------------------------MINING VARIABLES---------------------------*/
	// FLAGS
	int alive;				// INDICATE IF MINER IS STILL ACTIVE
	int * flag;				// SIGNAL WHEN A SOLUTION IS FOUND ON THE DEVICE

	// MINING VARIABLES
	WORD * target;
	int target_len;
	double difficulty;
	int blocks;
	int diff_level;

} WORKLOAD;

/***************************************************************************************************************************************************************************/
/****************************************************************************MACRO DEFINITIONS******************************************************************************/
/***************************************************************************************************************************************************************************/
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define GET_T1(x, y, z, c, k, m) (CH(x,y,z) + EP1(x) + c + k + m)
#define GET_T2(x,y,z) (MAJ(x,y,z) + EP0(x))

#define SHFTCMP(x, y, n) (((x >> n) & 0x000000ff) <= ((y >> n) & 0x000000ff))
#define COMPARE(x, y) (SHFTCMP(x,y,24) & SHFTCMP(x,y,16) & SHFTCMP(x,y,8) & SHFTCMP(x,y,0))

/***************************************************************************************************************************************************************************/
/**************************************************************************CONSTANT DEFINITIONS*****************************************************************************/
/***************************************************************************************************************************************************************************/
#define HASH_SIZE_BYTE sizeof(BYTE)*32				// SIZE OF HASH IN BYTES
#define BLOCK_SIZE sizeof(WORD)*20 			// SIZE OF EACH BLOCK IN WORDS
#define HASH_SIZE sizeof(WORD)*8				// SIZE OF BLOCK BASE IN WORDS

#define MAX_WORKERS 16 // 16 WORKERS MAX BASED ON MAX BLOCK SIZE
#define BLOCK_CONST_SIZE (MAX_WORKERS+1)*8 		// SAVE STATE OF FIRST BLOCK HASH
#define TARGET_CONST_SIZE (MAX_WORKERS+1)*8

WORD k_host[64] = { // SHA256 constants
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// FIXME Not currently used. Device side SHA256 constants as a single array
__constant__ WORD k[64] = { // SHA256 constants
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// SPLIT SHA CONSTANTS
__constant__ WORD k_s[4][16] = { // SHA256 constants
	{0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	 0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174},
	{0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	 0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967},
	{0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	 0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070},
	{0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	 0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2}
};

// INITIAL STATE CONSTANT
__constant__ WORD i_state[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
	0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// PRECOMPUTED SCHEDULE PADDING VALUES FOR 80 BYTE BLOCK HASH
__constant__ WORD msgSchedule_80B[16] = {
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x80000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000280
};

// SPLIT PRECOMPUTED MESSAGE SCHEDULE VALUES FOR 64 BYTE BLOCK HASH
__constant__ WORD msgSchedule_64B_s[4][16] = {
	{0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
	 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000200},
	{0x80000000, 0x01400000, 0x00205000, 0x00005088, 0x22000800, 0x22550014, 0x05089742, 0xa0000020,
	 0x5a880000, 0x005c9400, 0x0016d49d, 0xfa801f00, 0xd33225d0, 0x11675959, 0xf6e6bfda, 0xb30c1549},
	{0x08b2b050, 0x9d7c4c27, 0x0ce2a393, 0x88e6e1ea, 0xa52b4335, 0x67a16f49, 0xd732016f, 0x4eeb2e91,
	 0x5dbf55e5, 0x8eee2335, 0xe2bc5ec2, 0xa83f4394, 0x45ad78f7, 0x36f3d0cd, 0xd99c05e8, 0xb0511dc7},
	{0x69bc7ac4, 0xbd11375b, 0xe3ba71e5, 0x3b209ff2, 0x18feee17, 0xe25ad9e7, 0x13375046, 0x0515089d,
	 0x4f0d0f04, 0x2627484e, 0x310128d2, 0xc668b434, 0x420841cc, 0x62d311b8, 0xe59ba771, 0x85a7a484}
};

// PRECOMPUTED SCHEDULE PADDING VALUES FOR 32 BYTE BLOCK HASH
__constant__ WORD msgSchedule_32B[16] = {
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x80000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000100
};

// COPY PRECOMPUTED SCHEDULE PADDING VALUES FOR 32 BYTE BLOCK HASH
__constant__ WORD msgSchedule_32B_cpy[16] = {
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x80000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000100
};

/*----------------------------------------------------------------------------CONSTANT SYMBOLS-----------------------------------------------------------------------------*/
// MINING CONSTANTS
__constant__ WORD block_const[BLOCK_CONST_SIZE];
__constant__ WORD target_const[TARGET_CONST_SIZE];
__constant__ WORD time_const;

/***************************************************************************************************************************************************************************/
/**************************************************************************PROFILING DEFINITIONS****************************************************************************/
/***************************************************************************************************************************************************************************/

int PROFILER = 0;         // PROFILER SWITCH, DISABLED BY DEFAULT
int TEST_COUNT = 0;
//#define USE_NVTX 1

// INCLUDE PROFILER FUNCTIONS IF USE_NVTX IS ENABLED IN NVCC COMPILE
#ifdef USE_NVTX
	// PROFILER COLOR DEFINITIONS

	const uint32_t colors[4][12] ={
																//		0 					1						2						3						4						5						6						7						8						9						10					11
/*GRAYSCALE					(SPECIAL)*/	{ 0xff000000, 0xff101010, 0xff202020, 0xff303030, 0xff404040, 0xff505050, 0xff606060, 0xff707070, 0xff808080, 0xff909090, 0xffa0a0a0, 0xffb0b0b0 },
/*BRIGHT RAINBOW 		(LEVEL 0)*/	{ 0xffff0000, 0xffff8000, 0xffffe000, 0xffd0ff00, 0xff00ff40, 0xff00ffff, 0xff00b0ff, 0xff0060ff, 0xff0020ff, 0xff8000ff, 0xffff00ff, 0xffff0080 },
/*DULL RAINBOW 			(LEVEL 0)*/	{ 0xff800000, 0xff804000, 0xff808000, 0xff408000, 0xff008040, 0xff0080a0, 0xff004080, 0xff000080, 0xff400080, 0xff800080, 0xff800040, 0xff800040  },
																{ 0xffff4080, 0xffff8040, 0xff40ff80, 0xff80ff40, 0xff4080ff, 0xff8040ff, 0xffff4080, 0xffff8040, 0xff40ff80, 0xff80ff40, 0xff4080ff, 0xff8040ff }
};
 // TODO SET SPECIAL CASE FOR MINING, DIFFICULTY IS GRAY SCALE, BLOCKS PROCEED FROM A LIGHT SHADE, UP TO DARK

	const int num_colors = sizeof(colors[0])/sizeof(uint32_t);	// COLORS PER PALETTE
	const int num_palettes = sizeof(colors)/(sizeof(uint32_t)*num_colors); 																// TOTAL NUMBER OF COLOR PALETTES

	#define NUM_PALETTES num_palettes
	#define NUM_COLORS num_colors

	// TEST TO SEE IF PROFILING MACRO WAS PASSED IN
	#define PRINT_MACRO printf("MACRO PASSED SUCCESSFULLY!!\n\n")
	#define START_PROFILE cudaProfilerStart()
	#define STOP_PROFILE cudaProfilerStop()

	#define NAME_STREAM(stream, name) { 	\
		if(PROFILER == 1){ 									\
			nvtxNameCuStreamA(stream, name); 	\
		}																		\
	}

	// DEFAULT RANGE MANAGEMENT FUNCTIONS
	#define PUSH_RANGE(name,cid) { 													\
		if(PROFILER == 1){ 																		\
			int color_id = cid; 																\
			color_id = color_id%num_colors;											\
			nvtxEventAttributes_t eventAttrib = {0};				 		\
			eventAttrib.version = NVTX_VERSION; 								\
			eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; 	\
			eventAttrib.colorType = NVTX_COLOR_ARGB; 						\
			eventAttrib.color = colors[0][color_id]; 						\
			eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; 	\
			eventAttrib.message.ascii = name; 									\
			nvtxRangePushEx(&eventAttrib); 											\
		}}

	#define POP_RANGE if(PROFILER == 1){nvtxRangePop();}

	// DOMAIN MANAGEMENT FUNCTIONS
	#define DOMAIN_HANDLE nvtxDomainHandle_t

	#define DOMAIN_CREATE(handle, name){ 										\
			if(PROFILER == 1){ 																	\
				handle = nvtxDomainCreateA(name); 								\
	}}

	#define DOMAIN_DESTROY(handle){ 												\
			if(PROFILER == 1){ 																	\
				nvtxDomainDestroy(handle); 												\
	}}

	// ID specifies color related pattern, send -2 for time, -1 for parent
	#define PUSH_DOMAIN(handle, name, id, level, cid) { 						\
		if(PROFILER == 1){ 																						\
			int worker_id = id; 																				\
			int color_id = cid; 																				\
			int palette_id = level; 																		\
			worker_id = worker_id%num_colors; 													\
			color_id = color_id%num_colors;															\
			palette_id = palette_id%num_palettes; 											\
			uint32_t color = colors[palette_id][color_id]; 							\
			if(id > -1){																								\
				if(level == 2){   																				\
				/*	color = color ^ ~colors[3][worker_id];		*/					\
				}																													\
			}																														\
			/*ADD IF STATEMENT HERE FOR ID*/ 														\
			nvtxEventAttributes_t eventAttrib = {0}; 										\
			eventAttrib.version = NVTX_VERSION;	 												\
			eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; 					\
			eventAttrib.colorType = NVTX_COLOR_ARGB; 										\
			eventAttrib.color = color; 																	\
			eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;	\
			eventAttrib.payload.llValue = level; 												\
			eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; 					\
			eventAttrib.message.ascii = name; 													\
			nvtxDomainRangePushEx(handle, &eventAttrib); 								\
		}}

	#define POP_DOMAIN(handle) if(PROFILER == 1){nvtxDomainRangePop(handle);}

#else // EMPTY FUNCTIONS WHEN NVTX IS DISABLED OR UNAVAILABLE
	#define PRINT_MACRO printf("MACRO WAS NOT PASSED!!\n\n")
	#define START_PROFILE
	#define STOP_PROFILE
	#define NAME_STREAM(stream, name)
	#define PUSH_RANGE(name,cid)
	#define POP_RANGE
	#define DOMAIN_HANDLE int
	#define DOMAIN_CREATE(handle, name)
	#define DOMAIN_DESTROY(handle)
	#define PUSH_DOMAIN(handle, name, id, level, cid)
	#define POP_DOMAIN(handle)
#endif

// ENABLE DEVICE SIDE DEBUGGING
// DEVICE_PRINT IS FOR LOGGING USING A SINGLE THREAD
// DEVICE PRINT ANY WILL PRINT FOR ALL THREADS (BEST FOR BRANCHES)
// DEVICE_DEBUG WILL EXECUTE ANY ENCLOSED CODE
//#ifdef DEV_DEBUG
#if DEV_DEBUG == 1
	// basic debug, enable time log
	#define DEVICE_TIME(msg, arg){												\
		if(threadIdx.x+blockIdx.x*blockDim.x == 0){					\
			printf(msg, arg);																	\
		}																										\
	}
	#define DEVICE_PRINT_SOLN(msg, args...){}
	#define DEVICE_PRINT(msg, args...){}
	#define DEVICE_PRINT_ANY(msg, args...){}
	#define DEVICE_DEBUG(args...){}
#elif DEV_DEBUG == 2
	#define DEVICE_TIME(msg, arg){												\
		if(threadIdx.x+blockIdx.x*blockDim.x == 0){					\
			printf(msg, arg);																	\
		}																										\
	}
	#define DEVICE_PRINT_SOLN(msg, args...){							\
		printf(msg, args);																	\
	}
	#define DEVICE_PRINT(msg, args...){}
	#define DEVICE_PRINT_ANY(msg, args...){}
	#define DEVICE_DEBUG(args...){}
#elif DEV_DEBUG == 3
	#define DEVICE_TIME(msg, arg){												\
		if(threadIdx.x+blockIdx.x*blockDim.x == 0){					\
			printf(msg, arg);																	\
		}																										\
	}
	#define DEVICE_PRINT_SOLN(msg, args...){							\
		printf(msg, args);																	\
	}
	#define DEVICE_PRINT(msg, args...){										\
		if(threadIdx.x+blockIdx.x*blockDim.x == 0){					\
			printf(msg, args);																\
		}																										\
	}
	#define DEVICE_PRINT_ANY(msg, args...){printf(msg, args);}
	#define DEVICE_DEBUG(args...){args}
#else
	#define DEVICE_TIME(msg, arg){}
	#define DEVICE_PRINT_SOLN(msg, args...){}
	#define DEVICE_PRINT(msg, args...){}
	#define DEVICE_PRINT_ANY(msg, args...){}
	#define DEVICE_DEBUG(args...){}
#endif

/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************************  ______________________________________________________________________________________________________________________  *************************/
/************************  |    _____    ______    _____   _                   _____               _______   _____    ____    _   _    _____    |  *************************/
/************************  |   |  __ \  |  ____|  / ____| | |          /\     |  __ \      /\     |__   __| |_   _|  / __ \  | \ | |  / ____|   |  *************************/
/************************  |   | |  | | | |__    | |      | |         /  \    | |__) |    /  \       | |      | |   | |  | | |  \| | | (___     |  *************************/
/************************  |   | |  | | |  __|   | |      | |        / /\ \   |  _  /    / /\ \      | |      | |   | |  | | | . ` |  \___ \    |  *************************/
/************************  |   | |__| | | |____  | |____  | |____   / ____ \  | | \ \   / ____ \     | |     _| |_  | |__| | | |\  |  ____) |   |  *************************/
/************************  |   |_____/  |______|  \_____| |______| /_/    \_\ |_|  \_\ /_/    \_\    |_|    |_____|  \____/  |_| \_| |_____/    |  *************************/
/************************  |____________________________________________________________________________________________________________________|  *************************/
/************************                                                                                                                          *************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/



/***************************************************************************************************************************************************************************/
/***************************  _________________________________________________________________________________________________________________  ***************************/
/***************************  |     ___   _       ___    ___     _     _       __   __    _     ___   ___     _     ___   _      ___   ___    |  ***************************/
/***************************  |    / __| | |     / _ \  | _ )   /_\   | |      \ \ / /   /_\   | _ \ |_ _|   /_\   | _ ) | |    | __| / __|   |  ***************************/
/***************************  |   | (_ | | |__  | (_) | | _ \  / _ \  | |__     \ V /   / _ \  |   /  | |   / _ \  | _ \ | |__  | _|  \__ \   |  ***************************/
/***************************  |    \___| |____|  \___/  |___/ /_/ \_\ |____|     \_/   /_/ \_\ |_|_\ |___| /_/ \_\ |___/ |____| |___| |___/   |  ***************************/
/***************************  |_______________________________________________________________________________________________________________|  ***************************/
/***************************                                                                                                                     ***************************/
/***************************************************************************************************************************************************************************/

/***************************************************************************************************************************************************************************/
/***********************************************************************DEFAULT DEVICE CONSTRAINTS**************************************************************************/
/***************************************************************************************************************************************************************************/
// TODO Add in compiler options for different design parameters
// TODO Define global variables using these values

// HARDWARE CONTRAINTS
#define HOST_MULTIPROCESSORS 8					// AVAILABLE CORES ON THE CPU (COULD AFFECT TIMING WITH MANY WORKERS)
#define DEVICE_MULTIPROCESSORS 10 			// TOTAL NUMBER OF STREAMING MULTIPROCESSORS ON THE GPU
//#define DEVICE_MINIMUM_VERSION 3					// MINIMUM COMPUTE COMPATIBILITY REQUIRED

// DEVICE THREAD CONSTRAINTS
#define MAX_THREADS_PER_BLOCK 1024 			// MAXIMUM THREADS PER BLOCK
#define MAX_THREADS_PER_SM 2048					// MAXIMUM THREADS PER MULTIPROCESSOR

//DEVICE MEMORY CONSTRAINTS
#define SHARED_MEM_PER_BLOCK 49152 			// (BYTES) LIMITS MERKLE THREAD LIMIT
#define REG_PER_BLOCK 65536
#define REG_PER_SM 65536

/***************************************************************************************************************************************************************************/
/***********************************************************************PROGRAM DESIGN CONSTRAINTS**************************************************************************/
/***************************************************************************************************************************************************************************/
// MINING KERNEL USAGE
#define MINING_REG_PER_THREAD 32
#define MINING_SHARED_MEM	16384				// 16B per thread

// MERKLE KERNEL USAGE
#define MERKLE_REG_PER_THREAD 48
#define MERKLE_SHARED_MEM	96				// 96B per thread
#define MAX_MERKLE_THREADS SHARED_MEM_PER_BLOCK/MERKLE_SHARED_MEM 					// 512 threads shared memory limit


// USER DEFINED NUMBER OF THREADS
#ifdef CUSTOM_THREADS
	#define NUM_THREADS CUSTOM_THREADS
#else
	#define NUM_THREADS 1024
#endif

// DEVICE LIMITATIONS
#define SM_THREAD_LIMIT_REGS REG_PER_SM/MINING_REG_PER_THREAD   // 2048
#define MINING_BLOCKS_PER_SM SM_THREAD_LIMIT_REGS/NUM_THREADS		// 2 @1024 THREADS

// CALCULATED MAX BLOCKS FOR MINING OPERATIONS
#define AVAILABLE_BLOCKS MINING_BLOCKS_PER_SM*DEVICE_MULTIPROCESSORS  // 20 @1024 THREADS, 40 @ 512 THREADS,..., 320 @ 64 THREADS


// QUESTION Is there a more efficient way of determining the number of blocks to be allocated for the parent chain?
// For example: Set it to be calculated based on # workers and available multiprocessors

// Workers get 80% of resources when using multilevel mining, varies depending on the number of multiprocessors available on the device
// 16 @1024 threads, 32 @512 threads, 64 @256, 128 @128, 256 @64
#define MAX_BLOCKS MINING_BLOCKS_PER_SM*(DEVICE_MULTIPROCESSORS-2)

// USER DEFINED PARAMETER DEFAULTS
#define MERKLE_THREADS 512			// 512 MAXIMUM DUE TO SHARED MEMORY LIMIT (WAS 64 FOR TESTING)
int WORKER_BUFFER_SIZE = 32;
int PARENT_BLOCK_SIZE = 16;
int DIFFICULTY_LIMIT = 32;

// FIXME SEPARATE VARIABLES BY TYPE
/***************************************************************************************************************************************************************************/
/****************************************************************************GLOBAL VARIABLES*******************************************************************************/
/***************************************************************************************************************************************************************************/

//#define TARGET_DIFFICULTY 256
//#define TARGET_DIFFICULTY 1024
int TARGET_DIFFICULTY = 1;

#define TARGET_BLOCKS DIFFICULTY_LIMIT*TARGET_DIFFICULTY

// INPUTS GENERATED = LOOPS * NUM_THREADS * NUM_BLOCKS
#define INPUT_LOOPS 25

// Exponentially reduce computation time, 0 is normal, negative values down to -3 drastically reduce difficulty, highest difficulty is 26
int DIFF_REDUCE = -1;

// INITIALIZE DEFAULT GLOBAL VARIABLES FOR COMMAND LINE OPTIONS
// INFORMATIVE COMMAND OPTIONS
int DEBUG = 0;            // DEBUG DISABLED BY DEFAULT
int MINING_PROGRESS = 0;  // MINING PROGRESS INDICATOR DISABLED BY DEFAULT (ONLY ENABLE IF NOT SAVING CONSOLE OUTPUT TO A FILE, OTHERWISE THE STATUS WILL OVERTAKE THE WRITTEN OUTPUT)

// ARCHITECTURE COMMAND OPTIONS
int MULTILEVEL = 0;       // MULTILEVEL ARCHITECTURE DISABLED BY DEFAULT
int NUM_WORKERS = 1;      // NUMBER OF WORKERS 1 BY DEFAULT

// MINING COMMAND OPTIONS
// FIXME: ADD NUM_THREADS, MAX_BLOCKS, OPTIMIZE_BLOCKS, etc. here

// NOTE: reduces the number of blocks allocated to workers if the parent also requires space on the GPU
#define WORKER_BLOCKS ((MULTILEVEL == 1) ? MAX_BLOCKS: AVAILABLE_BLOCKS)/NUM_WORKERS
//#define WORKER_BLOCKS MAX_BLOCKS/NUM_WORKERS
#define PARENT_BLOCKS AVAILABLE_BLOCKS-MAX_BLOCKS


// NUMBER OF LOOPS IN THE BENCHMARK
#define BENCHMARK_LOOPS 10

int DIFF_SCALING = 1;
int DIFFICULTY_BITS = 0;

// Timeout variables
int TIMEOUT = 0;		// Set to 1 to enable timeout
int TIME_LIMIT = 0;	// Set to number of seconds till timeout

#define START_POW (0X1D - DIFF_REDUCE)
#define START_BITS (0x00FFFF - (DIFFICULTY_BITS << 8))
#define START_DIFF ((START_POW << 24) | START_BITS)

/***************************************************************************************************************************************************************************/
/****************************************  _______________________________________________________________________________________  ****************************************/
/****************************************  |    _  _   ___   ___  _____   ___  _   _  _  _   ___  _____  ___  ___   _  _  ___    |  ****************************************/
/****************************************  |   | || | / _ \ / __||_   _| | __|| | | || \| | / __||_   _||_ _|/ _ \ | \| |/ __|   |  ****************************************/
/****************************************  |   | __ || (_) |\__ \  | |   | _| | |_| || .` || (__   | |   | || (_) || .` |\__ \   |  ****************************************/
/****************************************  |   |_||_| \___/ |___/  |_|   |_|   \___/ |_|\_| \___|  |_|  |___|\___/ |_|\_||___/   |  ****************************************/
/****************************************  |_____________________________________________________________________________________|  ****************************************/
/****************************************                                                                                           ****************************************/
/***************************************************************************************************************************************************************************/
__host__ void hostCoreProcess(int num_chains, int multilevel);

/***************************************************************************************************************************************************************************/
/*****************************************************************************TESTING FUNCTIONS*****************************************************************************/
/***************************************************************************************************************************************************************************/
/*-----------------------------------------------------------------------------QUERY FUNCTIONS-----------------------------------------------------------------------------*/
__host__ void hostDeviceQuery(void);

/*-----------------------------------------------------------------------------TEST FUNCTIONS------------------------------------------------------------------------------*/
__host__ void hostFunctionalTest(void);
__host__ void testMiningHash(WORKLOAD * t_load, BYTE * test_str, BYTE * correct_str, WORD diff_pow, char ** logStr);
__host__ void testDoubleHash(WORKLOAD * t_load, BYTE * test_str, BYTE * correct_str, int test_size, char ** logStr);
__host__ void testMerkleHash(WORKLOAD * t_load, BYTE * test_str, BYTE * correct_str, int test_size, char ** logStr);
__host__ void miningBenchmarkTest(int num_workers);
__host__ void miningBenchmarkTest_full(int num_workers);
__host__ void colorTest(int num_colors, int num_palettes);

// TODO ADD TESTING CORES HERE
/*-----------------------------------------------------------------------------------||------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/************************************************************************MEMORY MANAGEMENT FUNCTIONS************************************************************************/
/***************************************************************************************************************************************************************************/
/*---------------------------------------------------------------------------WORKLOAD MANAGEMENT---------------------------------------------------------------------------*/
__host__ void allocWorkload(int id, WORKLOAD * load, int buffer_size);
__host__ void freeWorkload(WORKLOAD * load);

/*-------------------------------------------------------------------------CUDA STREAM MANAGEMENT--------------------------------------------------------------------------*/
__host__ void createCudaVars(cudaEvent_t * timing1, cudaEvent_t * timing2, cudaStream_t * stream);
__host__ void destroyCudaVars(cudaEvent_t * timing1, cudaEvent_t * timing2, cudaStream_t * stream);

/*-------------------------------------------------------------------------CUDA TIMING MANAGEMENT--------------------------------------------------------------------------*/
__host__ void initTime(cudaStream_t * tStream, WORD ** time_h);
__host__ void freeTime(cudaStream_t * tStream, WORD ** time_h);

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/***********************************************************************MINING MANAGEMENT FUNCTIONS*************************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------MINING INITIALIZATION---------------------------------------------------------------------------*/
__host__ void initializeBlockHeader(WORD * block, WORD version, WORD * prevBlock, WORD * merkleRoot, WORD time_b, WORD target, WORD nonce);
__host__ void initializeWorkerBlock(WORKLOAD * load);
__host__ void initializeParentBlock(WORD * pBlock_h);

/*-----------------------------------------------------------------------------MINING UPDATES------------------------------------------------------------------------------*/
__host__  int updateBlock(FILE * inFile, WORD * block_h, WORD * hash_h, WORD * buffer_h);
__host__ int updateBlock_load(WORKLOAD * load);
__host__ void updateParentHash(WORD * block_h, WORD * hash_h);
__host__ void updateDifficulty(WORD * block_h, int diff_level);
__host__ void updateTime(cudaStream_t * tStream, WORD * time_h, DOMAIN_HANDLE prof_handle);

/*-----------------------------------------------------------------------------MINING GETTERS------------------------------------------------------------------------------*/
__host__ WORD getTime(void);
__host__ void getDifficulty(WORKLOAD * load);

/*---------------------------------------------------------------------------MINING CALCULATIONS---------------------------------------------------------------------------*/
__host__ double calculateDifficulty(BYTE * bits);
__host__ int calculateTarget(BYTE * bits, BYTE * target);
__host__ int calculateMiningTarget(BYTE * bits, BYTE * target_bytes, WORD * target);
__host__ void calculateSchedule(WORD m[]); // CALCULATE MINING SCHEDULE PRIOR TO STARTING THE MINER
__host__ void calculateFirstState(WORD state[], WORD base[]); // CALCULATE FIRST HALF OF FIRST HASH

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/************************************************************************KERNEL MANAGEMENT FUNCTIONS************************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------INPUT GENERATION KERNEL-------------------------------------------------------------------------*/
__host__ void launchGenHash(WORD ** hash_hf, WORD ** hash_df, WORD ** seed_h, WORD ** seed_d, size_t size_hash);

/*----------------------------------------------------------------------------MERKLE TREE KERNEL---------------------------------------------------------------------------*/
__host__ void launchMerkle(WORKLOAD * load);

/*------------------------------------------------------------------------------MINING KERNEL------------------------------------------------------------------------------*/
__host__ void launchMiner(WORKLOAD * load);
__host__ void returnMiner(WORKLOAD * load);

__host__ void launchWorkflow(WORKLOAD * load);
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/*****************************************************************************UTILITY FUNCTIONS*****************************************************************************/
/***************************************************************************************************************************************************************************/
/*------------------------------------------------------------------------HEX CONVERSION FUNCTIONS-------------------------------------------------------------------------*/
__host__ void encodeHex(BYTE * str, BYTE * hex, int len);
__host__ void encodeWord(BYTE * str, WORD * hex, int len);
__host__ void decodeHex(BYTE * hex, BYTE * str, int len);
__host__ void decodeWord(WORD * hex, BYTE * str, int len);
__host__ void printHex(BYTE * hex, int len);
__host__ void printHexFile(FILE * outfile, BYTE * hex, int len);
__host__ void printWords(WORD * hash, int len);
__host__ void printMerkle(WORKLOAD * load);
__host__ void host_convertHash_Word2Byte(WORD * in, BYTE* out);
__host__ void host_convertHash_Byte2Word(BYTE * in, WORD* out, int len);

/*------------------------------------------------------------------------STATUS LOGGING FUNCTIONS-------------------------------------------------------------------------*/
__host__ void printLog(const char* msg);
__host__ void printDebug(const char * msg);
__host__ void printError(const char * msg);
__host__ void logStart(int workerID, int block, WORD * start_hash);
__host__ int printProgress(int mining_state, int multilevel,int num_workers,int pchain_blocks, int *chain_blocks);

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/*************************************************************************I/O MANAGEMENT FUNCTIONS**************************************************************************/
/***************************************************************************************************************************************************************************/
/*--------------------------------------------------------------------------INPUT FILE FUNCTIONS---------------------------------------------------------------------------*/
__host__ int initializeHash(WORKLOAD * load);      // INIT A SINGLE HASH FILE
__host__ void initializeInputFile(FILE * inFile, char * filename);
__host__ void printInputFile(WORD * hash_f, char * filename, int blocks, int threads);
__host__ int readNextHash(FILE * inFile, WORD * hash_w);

/*--------------------------------------------------------------------------OUTPUT FILE FUNCTIONS--------------------------------------------------------------------------*/
__host__ int initializeOutfile(char * outFile, char * out_dir_name, int worker_id);
__host__ int initializeBenchmarkOutfile(char * outFile, char * out_dir_name, int worker_id);
__host__ int initializeParentOutputs(char * bfilename, char * hfilename);
__host__ void printDifficulty(char* diff_file, int worker_num, double difficulty, float time, int num_blocks);
__host__ void printErrorTime(char* err_file, char *err_msg, float err_time);
__host__ void printOutputFile(char * outFileName, WORD * block_h, WORD * hash_f, int block, float calc_time, double difficulty, int id, int log_flag);

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/**********************************  __________________________________________________________________________________________________  ***********************************/
/**********************************  |     ___  _     ___   ___    _    _      ___  _   _  _  _   ___  _____  ___  ___   _  _  ___    |  ***********************************/
/**********************************  |    / __|| |   / _ \ | _ )  /_\  | |    | __|| | | || \| | / __||_   _||_ _|/ _ \ | \| |/ __|   |  ***********************************/
/**********************************  |   | (_ || |__| (_) || _ \ / _ \ | |__  | _| | |_| || .` || (__   | |   | || (_) || .` |\__ \   |  ***********************************/
/**********************************  |    \___||____|\___/ |___//_/ \_\|____| |_|   \___/ |_|\_| \___|  |_|  |___|\___/ |_|\_||___/   |  ***********************************/
/**********************************  |________________________________________________________________________________________________|  ***********************************/
/**********************************                                                                                                      ***********************************/
/***************************************************************************************************************************************************************************/

/*------------------------------------------------------------------------------TEST KERNELS-------------------------------------------------------------------------------*/
template <int blocks, int id>
__global__ void miningBenchmarkKernel(WORD * block_d, WORD * result_d, BYTE * hash_d, int * flag_d, int * total_iterations);
template <int sel>
__global__ void hashTestDoubleKernel(WORD * test_block, WORD * result_block);
__global__ void hashTestMiningKernel(WORD * test_block, WORD * result_block, int * success);

/*------------------------------------------------------------------------------MINING KERNELS-----------------------------------------------------------------------------*/
template <int blocks, int id>
__global__ void minerKernel(WORD * block_d, WORD * result_d, BYTE * hash_d, int * flag_d);
__global__ void genHashKernel(WORD * hash_df, WORD * seed, int num_blocks);
__global__ void merkleKernel(WORD * pHash_d, WORD * block_d, int buffer_blocks,  int tree_size);
__global__ void merkleKernel_workflow(WORD * pHash_d, WORD * block_d, WORD * basestate_d, int buffer_blocks,  int tree_size);

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/************************************  _______________________________________________________________________________________________  ************************************/
/************************************  |    ___   ___ __   __ ___  ___  ___   ___  _   _  _  _   ___  _____  ___  ___   _  _  ___    |  ************************************/
/************************************  |   |   \ | __|\ \ / /|_ _|/ __|| __| | __|| | | || \| | / __||_   _||_ _|/ _ \ | \| |/ __|   |  ************************************/
/************************************  |   | |) || _|  \ V /  | || (__ | _|  | _| | |_| || .` || (__   | |   | || (_) || .` |\__ \   |  ************************************/
/************************************  |   |___/ |___|  \_/  |___|\___||___| |_|   \___/ |_|\_| \___|  |_|  |___|\___/ |_|\_||___/   |  ************************************/
/************************************  |_____________________________________________________________________________________________|  ************************************/
/************************************                                                                                                   ************************************/
/***************************************************************************************************************************************************************************/

/*--------------------------------------------------------------------------DEVICE DEBUG FUNCTIONS-------------------------------------------------------------------------*/
static __device__ __inline__ uint32_t get_smid();
static __device__ __inline__ uint32_t get_warpid();
static __device__ __inline__ uint32_t get_laneid();

/*-------------------------------------------------------------------------DEVICE UTILITY FUNCTIONS------------------------------------------------------------------------*/
__device__ void printHash(BYTE * hash);
__device__ void printBlock(BYTE * hash);
__device__ void printState(WORD * hash);
__device__ void printBlockW(WORD * hash);
__device__ __inline__ void convertHash_Word2Byte(WORD * in, BYTE* out);

/*-----------------------------------------------------------------------MESSAGE SCHEDULE FUNCTION------------------------------------------------------------------------*/
__device__ __inline__ void scheduleExpansion_short( WORD m[]);

/*-----------------------------------------------------------------------PARTIAL TRANSFORM FUNCTIONS------------------------------------------------------------------------*/
__device__ __inline__ void sha256_hashQuarter(WORD state[8], WORD m[], int offset);
__device__ __inline__ void sha256_hashSingle(WORD * base, WORD * state, WORD * m);

/*-------------------------------------------------------------------------FULL TRANSFORM FUNCTIONS-------------------------------------------------------------------------*/
__device__ __inline__ int sha256_blockHash(WORD * uniquedata, WORD * base, WORD * state, WORD * target);
__device__ __inline__ void sha256_merkleHash_64B(WORD * hash_data, WORD * state);
__device__ __inline__ void sha256_merkleHash_32B(WORD * hash_data, WORD * state);
__device__ __inline__ void sha256_merkleHash_base(WORD * hash_data, WORD * state);

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
// TODO DOCUMENT THESE FUNCTIONS
/* NOTE Basic callback function templates
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *load);
void CUDART_CB myHostNodeCallback(void *load);
*/

/***************************************************************************************************************************************************************************/
/************************************************************************END FUNCTION DECLARATIONS**************************************************************************/
/***************************************************************************************************************************************************************************/

// TEMPLATED FUNCTION CALLS
// NEW BENCHMARK LAUNCHER WHICH USES BROADER ID BASED TEMPLATING
// USED TO SIMULATE A FULL WORKLOAD DURING BENCHMARKING TO PREVENT INFLATED PERFORMANCE FOR LOW WORKLOADS PER SM
// FOR ADDITIONAL KERNELS, BEST USED WITH HIGH DIFFICULTY TO ENSURE CONTINUOUS OPERATION THROUGHOUT THE BENCHMARK, REQUIRES MANUAL EXIT AT THE END (BY CHANGING THE WORKER FLAG)
#define LAUNCH_BENCHMARK_TEST(w_blocks, id, stream, block, result, hash, flag, iterations){ 																														\
	if(MULTILEVEL == 0){																																																																	\
		switch (w_blocks) {																																																																	\
			case 1: START_BENCHMARK(AVAILABLE_BLOCKS, id, stream, block, result, hash, flag, iterations);	break;																							\
			case 2:	START_BENCHMARK((AVAILABLE_BLOCKS/2), id, stream, block, result, hash, flag, iterations);	break;																					\
			case 4:	START_BENCHMARK((AVAILABLE_BLOCKS/4), id, stream, block, result, hash, flag, iterations);	break;																					\
			case 8:	START_BENCHMARK((AVAILABLE_BLOCKS/8), id, stream, block, result, hash, flag, iterations);	break;																					\
			case 16:START_BENCHMARK((AVAILABLE_BLOCKS/16), id, stream, block, result, hash, flag, iterations);	break;																				\
			default:																																																																					\
				printf("ERROR LAUNCHING MINER: MINING WITH %i BLOCKS IS CURRENTLY NOT SUPPORTED\n SUPPORTED VALUES ARE [1, 2, 4, 8, 16]\n", w_blocks);					\
				break;																																																																					\
		} 																																																																									\
	} else {																																																																							\
		switch (w_blocks) {																																																																	\
			case 1:	START_BENCHMARK(MAX_BLOCKS, id, stream, block, result, hash, flag, iterations);	break;																										\
			case 2:	START_BENCHMARK((MAX_BLOCKS/2), id, stream, block, result, hash, flag, iterations);	break;																								\
			case 4:	START_BENCHMARK((MAX_BLOCKS/4), id, stream, block, result, hash, flag, iterations);	break;																								\
			case 8:	START_BENCHMARK((MAX_BLOCKS/8), id, stream, block, result, hash, flag, iterations);	break;																								\
			case 16:START_BENCHMARK((MAX_BLOCKS/16), id, stream, block, result, hash, flag, iterations);	break;																							\
			default:																																																																					\
				printf("ERROR LAUNCHING MINER: MINING WITH %i BLOCKS IS CURRENTLY NOT SUPPORTED\n SUPPORTED VALUES ARE [1, 2, 4, 8, 16]\n", w_blocks);					\
				break;																																																																					\
		} 																																																																									\
	}																																																																											\
}


#define START_BENCHMARK(w_blocks, id, stream, block, result, hash, flag, iterations){																																		\
	switch (id) {																																																																					\
		case 0:	 miningBenchmarkKernel<w_blocks, 0><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;										\
		case 1:	 miningBenchmarkKernel<w_blocks, 1><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;										\
		case 2:	 miningBenchmarkKernel<w_blocks, 2><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;										\
		case 3:	 miningBenchmarkKernel<w_blocks, 3><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;										\
		case 4:	 miningBenchmarkKernel<w_blocks, 4><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;										\
		case 5:	 miningBenchmarkKernel<w_blocks, 5><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;										\
		case 6:	 miningBenchmarkKernel<w_blocks, 6><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;										\
		case 7:	 miningBenchmarkKernel<w_blocks, 7><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;										\
		case 8:	 miningBenchmarkKernel<w_blocks, 8><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;										\
		case 9:	 miningBenchmarkKernel<w_blocks, 9><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;										\
		case 10: miningBenchmarkKernel<w_blocks, 10><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;									\
		case 11: miningBenchmarkKernel<w_blocks, 11><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;									\
		case 12: miningBenchmarkKernel<w_blocks, 12><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;									\
		case 13: miningBenchmarkKernel<w_blocks, 13><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;									\
		case 14: miningBenchmarkKernel<w_blocks, 14><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;									\
		case 15: miningBenchmarkKernel<w_blocks, 15><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;									\
		case 16: miningBenchmarkKernel<w_blocks, 16><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag, iterations);	break;									\
	} 																																																																										\
}

// TEMPLATE FOR MINER KERNEL
//FIXME CHANGE BLOCKS TO CALCULATE FROM NUM THREADS AND AVAILABLE RESOURCES
// IE. Current value of 20480 = threads/SM * available SMs
// WORKER BLOCKS = (((Total SMs)*(threads/SM))/NUM_WORKERS)/NUM_THREADS
// CURRENTLY TAKES THE NUMBER OF WORKERS AS THE INPUT,
#define LAUNCH_MINER(w_blocks, id, stream, block, result, hash, flag){ 																																							\
	if(id <= 16 && id >= 0){ /* ONLY ACCEPT BLOCKS WITH A VALID WORKER ID*/																																						\
		if(MULTILEVEL == 0){																																																														\
			switch (w_blocks) {																																																														\
				case 0:  START_MINER(PARENT_BLOCKS, id, stream, block, result, hash, flag);   break;																												\
				case 1:  START_MINER(AVAILABLE_BLOCKS, id, stream, block, result, hash, flag);   break;																											\
				case 2:  START_MINER(AVAILABLE_BLOCKS/2, id, stream, block, result, hash, flag);   break;																										\
				case 4:  START_MINER(AVAILABLE_BLOCKS/4, id, stream, block, result, hash, flag);   break;																										\
				case 8:  START_MINER(AVAILABLE_BLOCKS/8, id, stream, block, result, hash, flag);   break;																										\
				case 16: START_MINER(AVAILABLE_BLOCKS/16, id, stream, block, result, hash, flag);  break;																										\
				default:																																																																		\
					printf("ERROR LAUNCHING MINER: MINING WITH %i BLOCKS IS CURRENTLY NOT SUPPORTED\n SUPPORTED VALUES ARE [1, 2, 4, 8, 16]\n", w_blocks);		\
					break;																																																																		\
			}																																																																							\
		} else{ 																																																																				\
			switch (w_blocks) {																																																														\
				case 0:  START_MINER(PARENT_BLOCKS, id, stream, block, result, hash, flag);   break;																												\
				case 1:  START_MINER(MAX_BLOCKS, id, stream, block, result, hash, flag);   break;																											 			\
				case 2:  START_MINER(MAX_BLOCKS/2, id, stream, block, result, hash, flag);   break;																													\
				case 4:  START_MINER(MAX_BLOCKS/4, id, stream, block, result, hash, flag);   break;																													\
				case 8:  START_MINER(MAX_BLOCKS/8, id, stream, block, result, hash, flag);   break;																													\
				case 16: START_MINER(MAX_BLOCKS/16, id, stream, block, result, hash, flag);  break;																													\
				default:																																																																		\
					printf("ERROR LAUNCHING MINER: MINING WITH %i BLOCKS IS CURRENTLY NOT SUPPORTED\n SUPPORTED VALUES ARE [1, 2, 4, 8, 16]\n", w_blocks);		\
					break;																																																																		\
			}																																																																							\
		}																																																																								\
	} else{																																																																						\
		printf("WORKER ID OF %i IS INVALID. THE WORKER ID MUST BE A POSITIVE INTEGER LESS THAN OR EQUAL TO 16 \n", id);																	\
	}																																																																									\
}

// TEMPLATE INSTANTIATIONS WITH TEMPLATED ID TO ELIMINATE REGISTER GAIN FROM CONSTANT MEMORY ACCESSES
// MEM CHECK VERSION ONLY WORKS WITH 1 WORKER
#ifdef MEM_CHECK  // TEMPLATE FOR FAST COMPILATION, REDUCES EXCESS DETAILS FROM MEMORY USAGE RESULTS, WILL ONLY WORK FOR SINGLE WORKER DESIGNS
	#define START_MINER(w_blocks, id, stream, block, result, hash, flag){																							\
		switch (id) {																																																		\
			case 0: minerKernel<w_blocks, 0><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag); break; 		\
			case 1: minerKernel<w_blocks, 1><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag); break; 		\
		}																																																								\
	}
#else		// FULL TEMPLATE FOR CONSTANT MEMORY ID, TAKES LONGER TO COMPILE
	#define START_MINER(w_blocks, id, stream, block, result, hash, flag){																							\
		switch (id) {																																																		\
			case 0:	 minerKernel<w_blocks, 0><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag);  break;		\
			case 1:	 minerKernel<w_blocks, 1><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag);  break;		\
			case 2:	 minerKernel<w_blocks, 2><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag);  break;		\
			case 3:	 minerKernel<w_blocks, 3><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag);  break;		\
			case 4:	 minerKernel<w_blocks, 4><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag);  break;		\
			case 5:	 minerKernel<w_blocks, 5><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag);  break;		\
			case 6:	 minerKernel<w_blocks, 6><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag);  break;		\
			case 7:	 minerKernel<w_blocks, 7><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag);  break;		\
			case 8:	 minerKernel<w_blocks, 8><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag);  break;		\
			case 9:	 minerKernel<w_blocks, 9><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag);  break;		\
			case 10: minerKernel<w_blocks, 10><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag); break;		\
			case 11: minerKernel<w_blocks, 11><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag); break;		\
			case 12: minerKernel<w_blocks, 12><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag); break;		\
			case 13: minerKernel<w_blocks, 13><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag); break;		\
			case 14: minerKernel<w_blocks, 14><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag); break;		\
			case 15: minerKernel<w_blocks, 15><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag); break;		\
			case 16: minerKernel<w_blocks, 16><<<w_blocks, NUM_THREADS, 0, stream>>>(block, result, hash, flag); break;		\
		} 																																																							\
	}
#endif

#define HASH_DOUBLE_KERNEL(sel, stream, test_block, result_block){																\
	switch (sel) {																																									\
		case 32: hashTestDoubleKernel<32><<<1, 1, 0, stream>>>(test_block, result_block); 		break;	\
		case 64: hashTestDoubleKernel<64><<<1, 1, 0, stream>>>(test_block, result_block); 		break;	\
		default: printf("ERROR: INCORRECT PARAMETER SIZE %i FOR DOUBLE HASH TEST! \n", sel); 	break;	\
	}																																																\
}

// HOST INITIALIZATION, BEGIN WITH PARSING COMMAND LINE ARGUMENTS
int main(int argc, char *argv[]){
  // IMPROVED COMMAND LINE ARGUMENT PARSING
	PRINT_MACRO;
  if(argc == 1){ // DEFAULT MODE SELECTED, PRINT OUTPUT SPECIFYING OPTIONS WITH DEFAULTS
    printf("WARNING: NO OPTIONS SELECTED, RUNNING DEFAULT IMPLEMENTATION\n\
BASIC INPUT OPTIONS: \n\n\
\t --help  \t HELP FLAG: DISPLAY ALL INPUT OPTIONS (NO DESIGN RUN)\n\
\t --debug \t ENABLE MORE DETAILED CONSOLE OUTPUTS (DEFAULT: DISABLED)\n\
\t --multi \t MULTILEVEL ARCHITECTURE (DEFAULT: DISABLED)\n\
\t  -w #   \t NUMBER OF WORKER CHAINS (DEFAULT: 1)\n\n\
FOR A LIST OF ALL AVAILABLE OPTIONS, TRY '%s --help'\n\n\n", argv[0]);
  }

  // INITIALIZE PROGRAM FLAGS
  int err_flag = 0;
  int help_flag = 0;

  // PERFORM DRY RUN (NO MINING)
  int dry_run = 0;

  // FLAGS FOR ADDITIONAL TESTING AND INFORMATION
  int query_flag = 0;
  int test_flag = 0;
  int bench_flag = 0;

  // TODO ADD OPTION FOR SELECTING THE OPTIMAL THREAD AND BLOCK COUNT BASED ON DEVICE QUERIES


  char arg_in[50];
  for(int i = 1; i < argc; i++){
      // COPY INPUT ARG TO ALL LOWERCASE STRING
      strcpy(arg_in, argv[i]);
      char * p = arg_in;
      for( ; *p; ++p) *p = tolower(*p);
      //printf("\nARGUMENT %i: %s\n", i, arg_in);
      // CHECK FOR INFORMATION OPTIONS AND FUNCTION SWITCHES FIRST
      if(strcmp(arg_in, "--help") == 0){  // HELP OPTION
        help_flag = 1;
        break;
      }
      else if(strcmp(arg_in, "--debug") == 0){  // DEBUG OPTION
        DEBUG = 1;
        printDebug("DEBUG SETTINGS ENABLED\n");
      }
      else if(strcmp(arg_in, "--dryrun") == 0){  // DRY RUN OPTION
        dry_run = 1;
        printf("DRY RUN ENABLED, MINING WILL NOT BE INITIATED\n");
      }
      else if(strcmp(arg_in, "--profile") == 0){  // PROFILER OPTION
        PROFILER = 1;
        printf("PROFILER FUNCTIONS ENABLED\n");
				// TODO ADD NVTX TO LABEL STREAMS AND ADD EVENTS (SEE NVIDIA PROFILER GUIDE FOR MORE DETAILS)
				// TODO PRIOR TO EXECUTION, SET DIFF_REDUCE TO 2 OR MORE TO TRY REDUCE PROFILING OVERHEAD
				//DIFF_REDUCE = 2; // TOO EASY,
      }
      else if(strcmp(arg_in, "--indicator") == 0){  // MINING INDICATOR OPTION
        MINING_PROGRESS = 1;
        printf("WARNING: MINING PROGRESS INDICATOR ENABLED! THIS MAY CAUSE UNDESIRABLE BEHAVIOR IF WRITING CONSOLE OUTPUT TO A FILE!\n");
      }
      // CHECK FOR TESTING INTERFACE OPTIONS
      else if(strcmp(arg_in, "--query") == 0){ // DEVICE QUERY OPTION
        query_flag = 1;
      }
      else if(strcmp(arg_in, "--test") == 0){  // FUNCTIONAL VERIFICATION TEST OPTION
        // FIXME: ADD FUNCTIONAL VERIFICATION TEST
        test_flag = 1;
      }
      else if(strcmp(arg_in, "--benchmark") == 0){  // BENCHMARKING OPTION
        bench_flag = 1;
      }
      // CHECK FOR DESIGN PARAMETERS
      else if(strcmp(arg_in, "--multi") == 0){
        printf("MULTITHREADED DESIGN ENABLED!\n");
        MULTILEVEL = 1;
      }
      else if(strcmp(arg_in, "-w") == 0){
          if(i+1 < argc){
            if(atoi(argv[i+1]) > 0){
              NUM_WORKERS = atoi(argv[i+1]);
              printf("NUMBER OF WORKERS SET TO %i\n", NUM_WORKERS);
              i++;
            } else{
              printf("%s   fatal:  OPTION '-w' EXPECTS A POSITIVE NON-ZERO INTEGER ARGUMENT, RECEIVED '%s' INSTEAD\n\n", argv[0], argv[i+1]);
              err_flag = 1;
              break;
            }
          } else{
            printf("%s   fatal:  ARGUMENT EXPECTED AFTER '-w'\n\n", argv[0]);
            err_flag = 1;
            break;
          }
      }
			else if(strcmp(arg_in, "-t") == 0){
					if(i+1 < argc){
						if(atoi(argv[i+1]) > 0){
							TARGET_DIFFICULTY = atoi(argv[i+1]);
							printf("TARGET DIFFICULTY SET TO %i, MINING GOAL OF %i TOTAL BLOCKS\n", TARGET_DIFFICULTY, TARGET_BLOCKS);
							i++;
						} else{
							printf("%s   fatal:  OPTION '-t' EXPECTS A POSITIVE NON-ZERO INTEGER ARGUMENT, RECEIVED '%s' INSTEAD\n\n", argv[0], argv[i+1]);
							err_flag = 1;
							break;
						}
					} else{
						printf("%s   fatal:  ARGUMENT EXPECTED AFTER '-t'\n\n", argv[0]);
						err_flag = 1;
						break;
					}
			}
			else if(strcmp(arg_in, "-timeout") == 0){
					if(i+1 < argc){
						if(atoi(argv[i+1]) > 0){
							TIME_LIMIT = atoi(argv[i+1]);
							TIMEOUT = 1;
							printf("TIMEOUT ENABLED, SET TO %i SECONDS\n", TIME_LIMIT);
							i++;
						} else{
							printf("%s   fatal:  OPTION '-timeout' EXPECTS A POSITIVE NON-ZERO INTEGER ARGUMENT, RECEIVED '%s' INSTEAD\n\n", argv[0], argv[i+1]);
							err_flag = 1;
							break;
						}
					} else{
						printf("%s   fatal:  ARGUMENT EXPECTED AFTER '-timeout'\n\n", argv[0]);
						err_flag = 1;
						break;
					}
			}
			else if(strcmp(arg_in, "-diff") == 0){
					if(i+1 < argc){
						if(atoi(argv[i+1]) >= -3 && atoi(argv[i+1]) <= 26){
							DIFF_REDUCE = atoi(argv[i+1]);
							printf("STARTING DIFFICULTY MODIFIER SET TO %i\n", DIFF_REDUCE);
							i++;
						} else{
							printf("%s   fatal:  OPTION '-diff' EXPECTS AN INTEGER ARGUMENT BETWEEN -3 AND 26, RECEIVED '%s' INSTEAD\n\n", argv[0], argv[i+1]);
							err_flag = 1;
							break;
						}
					} else{
						printf("%s   fatal:  ARGUMENT EXPECTED AFTER '-diff'\n\n", argv[0]);
						err_flag = 1;
						break;
					}
			}
			else if(strcmp(arg_in, "-dscale") == 0){
					if(i+1 < argc){
						if(atoi(argv[i+1]) >= 0){
							DIFF_SCALING = atoi(argv[i+1]);
							printf("DIFFICULTY SCALING SET TO %i\n", DIFF_SCALING);
							i++;
						} else{
							printf("%s   fatal:  OPTION '-dscale' EXPECTS AN INTEGER ARGUMENT GREATER THAN ZERO, RECEIVED '%s' INSTEAD\n\n", argv[0], argv[i+1]);
							err_flag = 1;
							break;
						}
					} else{
						printf("%s   fatal:  ARGUMENT EXPECTED AFTER '-dscale'\n\n", argv[0]);
						err_flag = 1;
						break;
					}
			}
			else if(strcmp(arg_in, "-dlimit") == 0){
					if(i+1 < argc){
						if(atoi(argv[i+1]) >= 0){
							DIFFICULTY_LIMIT = atoi(argv[i+1]);
							printf("DIFFICULTY LIMIT SET TO %i\n", DIFFICULTY_LIMIT);
							i++;
						} else{
							printf("%s   fatal:  OPTION '-dlimit' EXPECTS AN INTEGER ARGUMENT GREATER THAN ZERO, RECEIVED '%s' INSTEAD\n\n", argv[0], argv[i+1]);
							err_flag = 1;
							break;
						}
					} else{
						printf("%s   fatal:  ARGUMENT EXPECTED AFTER '-dlimit'\n\n", argv[0]);
						err_flag = 1;
						break;
					}
			}
			else if(strcmp(arg_in, "-dbits") == 0){
					if(i+1 < argc){
						if(atoi(argv[i+1]) >= 0 && atoi(argv[i+1]) < 255){
							DIFFICULTY_BITS = atoi(argv[i+1])+1;
							printf("DIFFICULTY BITS SET TO %i\n", DIFFICULTY_BITS);
							i++;
						} else{
							printf("%s   fatal:  OPTION '-dbits' EXPECTS AN INTEGER BETWEEN ZERO AND 254, RECEIVED '%s' INSTEAD\n\n", argv[0], argv[i+1]);
							err_flag = 1;
							break;
						}
					} else{
						printf("%s   fatal:  ARGUMENT EXPECTED AFTER '-dbits'\n\n", argv[0]);
						err_flag = 1;
						break;
					}
			}
			else if(strcmp(arg_in, "-ptree") == 0){
					if(i+1 < argc){
						if(atoi(argv[i+1]) >= 1 && atoi(argv[i+1]) <= 512){
							PARENT_BLOCK_SIZE = atoi(argv[i+1]);
							printf("PARENT MERKLE TREE SIZE SET TO %i\n", PARENT_BLOCK_SIZE);
							i++;
						} else{
							printf("%s   fatal:  OPTION '-ptree' EXPECTS AN INTEGER ARGUMENT BETWEEN O AND 512, RECEIVED '%s' INSTEAD\n\n", argv[0], argv[i+1]);
							err_flag = 1;
							break;
						}
					} else{
						printf("%s   fatal:  ARGUMENT EXPECTED AFTER '-ptree'\n\n", argv[0]);
						err_flag = 1;
						break;
					}
			}
			else if(strcmp(arg_in, "-wtree") == 0){
					if(i+1 < argc){
						if(atoi(argv[i+1]) >= 1 && atoi(argv[i+1]) <= 512){
							WORKER_BUFFER_SIZE = atoi(argv[i+1]);
							printf("WORKER MERKLE TREE SIZE SET TO %i\n", WORKER_BUFFER_SIZE);
							i++;
						} else{
							printf("%s   fatal:  OPTION '-wtree' EXPECTS AN INTEGER ARGUMENT BETWEEN O AND 512, RECEIVED '%s' INSTEAD\n\n", argv[0], argv[i+1]);
							err_flag = 1;
							break;
						}
					} else{
						printf("%s   fatal:  ARGUMENT EXPECTED AFTER '-wtree'\n\n", argv[0]);
						err_flag = 1;
						break;
					}
			}
			else{
				printf("%s   fatal:  UNKNOWN ARGUMENT '%s'\n\n", argv[0], argv[i]);
				err_flag = 1;
				break;
			}
      //FIXME: ADD ADDITIONAL OPTIONS HERE FOR OTHER DESIGN PARAMETERS
  }

  // TODO ADD VARIABLE VERIFICATION HERE, RAISE ERROR FLAG IF A PROBLEM IS ENCOUNTERED
  // TODO SET BLOCKS PER WORKER BASED ON NUMBER OF WORKERS SELECTED AND BLOCKS AVAILABLE
        // NOTE TECHNICALLY, MAX BLOCKS IS 2^32, THOUGH THESE OBVIOUSLY WOULDNT BE CONCURRENT

  // TODO ADD OPTION TO GET IDEAL UTILIZATION BASED ON USAGE STATISTICS



  // ERROR IN COMMAND LINE OPTIONS
  if(err_flag == 1){
    printf("ONE OR MORE ERRORS DETECTED IN COMMAND LINE OPTIONS, UNABLE TO CONTINUE OPERATION\nTRY '%s --help' TO SEE A LIST OF AVAILABLE OPTIONS\n", argv[0]);
  }
  // HELP OPTIONS
  else if(help_flag == 1){
    printf("\nAVAILABLE OPTIONS FOR '%s' (LETTER-CASE DOES NOT MATTER):\n\n\
 PROGRAM QUERY AND TESTING INTERFACES (INFORMATION OPTIONS)\n\n\
\t --help  \t\t HELP FLAG: DISPLAY ALL INPUT OPTIONS (NO DESIGN RUN)\n\
\t --query \t\t DEVICE QUERY FLAG: RUN QUERY TO SHOW BASIC DEVICE HARDWARE SPECIFICATIONS \n\
\t --test  \t\t TEST FLAG: RUN TEST CORE TO VERIFY KERNEL OUTPUTS ARE CORRECT\n\
\t --benchmark  \t\t BENCHMARK FLAG: RUN SIMPLE MINING CORE TO DETERMINE DESIGN PERFORMANCE\n\n\
 PROGRAM FUNCTION SWITCHES (ENABLE OR DISABLE CERTAIN FEATURES)\n\n\
\t --debug \t\t ENABLE MORE DETAILED CONSOLE OUTPUTS (DEFAULT: DISABLED)\n\
\t --dryrun \t\t DISABLES THE MAIN MINING FUNCTION FOR THIS RUN (DEFAULT: DISABLED)\n\
\t --profile \t\t ENABLE CAPTURE FUNCTIONS FOR USE WITH NVIDIA VISUAL PROFILER (DEFAULT: DISABLED)\n\
\t --indicator \t\t ENABLE PROGRESS INDICATOR (DEFAULT: DISABLED)\n\t\t\t\t\t [!!WARNING!!-DO NOT USE INDICATOR IF WRITING CONSOLE OUTPUT TO A FILE]\n\n\
 DESIGN SPECIFIERS\n\n\
\t --multi \t\t MULTILEVEL ARCHITECTURE (DEFAULT: DISABLED)\n\
\t  -w #   \t\t NUMBER OF WORKER CHAINS AS A POSITIVE NON-ZERO INTEGER (DEFAULT: 1)\n\
\t  -t #   \t\t THE TARGET DIFFICULTY AS A POSITIVE NON-ZERO INTEGER (DEFAULT: 1)\n\
\t  -timeout #   \t\t THE PROGRAM TIMEOUT IN SECONDS AS A POSITIVE NON-ZERO INTEGER (DEFAULT: DISABLED)\n\
\t  -diff #   \t\t STARTING DIFFICULTY MODIFIER AS AN INTEGER, HIGHER VALUES ARE MORE DIFFICULT [-3 MINIMUM, 0 NORMAL, 26 MAXIMUM] (DEFAULT: -1)\n\
\t  -dscale # \t\t DIFFICULTY SCALING MODIFIER AS AN INTEGER, HIGHER VALUES INCREASE THE DIFFICULTY SCALING RATE, MINIMUM OF ZERO FOR CONSTANT DIFFICULTY (DEFAULT: 1)\n\
\t  -dbits # \t\t STARTING DIFFICULTY BITS AS AN INTEGER, HIGHER VALUES INCREASE THE STARTING DIFFICULTY [0 MINIMUM, 254 MAXIMUM] (DEFAULT: 0)\n\
\t  -dlimit # \t\t NUMBER OF BLOCKS PER DIFFICULTY LEVEL, MUST BE AN INTEGER GREATER THAN ZERO (DEFAULT: 32)\n\
\t  -wTree # \t\t WORKER MERKLE TREE BUFFER SIZE, MINIMUM OF 1 FOR NO MERKLE HASHING, MAXIMUM OF 512 IS THE SYSTEM LIMITATION (DEFAULT: 64)\n\
\t  -pTree # \t\t PARENT MERKLE TREE BUFFER SIZE, MINIMUM OF 1 FOR NO MERKLE HASHING, MAXIMUM OF 512 IS THE SYSTEM LIMITATION (DEFAULT: 16)\n", argv[0]);
  }
  // RUN THE SELECTED IMPLEMENTATION(S)
  else{
    // RUN DEVICE QUERY TO SEE AVAILABLE RESOURCES
    if(query_flag == 1){
      hostDeviceQuery();
    }
    // RUN FUNCTIONAL TEST FOR THE HASHING FUNCTIONS
    if(test_flag == 1){
      printf("FUNCTIONAL TESTING SELECTED!!!!!\n\n");
      hostFunctionalTest();
			//colorTest(NUM_COLORS, NUM_PALETTES);
    }
    // RUN BENCHMARK TEST FOR DEVICE PERFORMANCE
    if(bench_flag == 1){
      printf("BENCHMARK TESTING SELECTED!!!!!\n");
/* CHANGED FOR ALTERNATE BENCHMARK TESTING
			miningBenchmarkTest(NUM_WORKERS);
//*/
			miningBenchmarkTest_full(NUM_WORKERS);
    }
    // START MINING IF DRY RUN IS NOT SELECTED
    if(dry_run == 0){
      // TODO CHECK FOR PROFILER ENABLED, INCLUDE LOGGING OF ENABLED SETTINGS
      hostCoreProcess(NUM_WORKERS, MULTILEVEL);
			  //
    } else{
      printLog("MINING DISABLED FOR DRY RUN TESTING. NOW EXITING...\n\n");
    }
  }
	cudaDeviceReset();

  return 0;
}

/****************************************************************************************************************************************************************************/
/****************************************************************************************************************************************************************************/
// CORE MINING PROCESS
// INCLUDES CODE TO INITIALIZE A SPECIFIED NUMBER OF WORKERS AND A PARENT CHAIN IF NECCESSARY
// USING THE MULTILEVEL COMMAND ON EXECUTION ENABLES THE PARENT CHAIN FUNCTIONALITY
// ADDITIONAL OUTPUT CAN BE VIEWED BY USING THE DEBUG OPTION ON EXECUTION
/****************************************************************************************************************************************************************************/
/****************************************************************************************************************************************************************************/
__host__ void hostCoreProcess(int num_workers, int multilevel){
    printf("STARTING%s CORE PROCESS WITH %i WORKERS\n",(multilevel==1 ? " MULTILEVEL": ""), num_workers);

		START_PROFILE;
		char stream_name[50];
		// INITIALIZE PROFILING DOMAINS
		#ifdef USE_NVTX
			DOMAIN_HANDLE t_handle;
			DOMAIN_HANDLE p_handle;
			DOMAIN_HANDLE w_handle[NUM_WORKERS];
		#else
			int t_handle = 0;
		#endif

/*----------------------------GLOBAL TIMING VARIABLES-----------------------------*/
sprintf(stream_name, "TIME STREAM");
DOMAIN_CREATE(t_handle, stream_name);
PUSH_DOMAIN(t_handle, stream_name, -2, 0, 0);		// BLACK LABEL

float total_time[6];
cudaStream_t g_timeStream;
cudaEvent_t g_timeStart, g_timeFinish;
createCudaVars(&g_timeStart, &g_timeFinish, &g_timeStream);

// ADD NAME TO TIME STREAM
NAME_STREAM(g_timeStream, stream_name);

cudaEvent_t g_time[4];
for(int i = 0; i < 4; i++){
  cudaEventCreate(&g_time[i]);
}
PUSH_DOMAIN(t_handle, "ALLOC", -2, 2, 0);
cudaEventRecord(g_timeStart, g_timeStream);

char out_location[30];
if(multilevel == 1){
  sprintf(out_location, "outputs/results_%i_pchains", num_workers);
}else{
  sprintf(out_location, "outputs/results_%i_chains", num_workers);
}

char time_filename[100];
sprintf(time_filename,"%s/timing.out", out_location);


float err_time;
cudaStream_t errStream;
cudaEvent_t errStart, errFinish;
createCudaVars(&errStart, &errFinish, &errStream);

char error_filename[100];
sprintf(error_filename,"%s/error.out", out_location);
FILE * errFile;
if(errFile = fopen(error_filename, "w")){
  fprintf(errFile, "ERROR LOG FILE\n\n");
  fclose(errFile);
}
/**********************************************************************************************************************************/
/********************************************************WORKER ALLOCATION*********************************************************/
/**********************************************************************************************************************************/
		for(int i = 0; i < num_workers; i++){ // START WORKER DOMAINS AND ALLOCATION PROFILING
			sprintf(stream_name, "WORKER %i", i);
			DOMAIN_CREATE(w_handle[i], stream_name);
			PUSH_DOMAIN(w_handle[i], stream_name, i, 1, i);
			PUSH_DOMAIN(w_handle[i], "ALLOC", i, 2, 0);
		}
/**************************VARIABLE DECLARATIONS**************************/
/*--------------------------WORKLOAD VARIABLE----------------------------*/
		WORKLOAD * w_load;			// MAIN POINTER TO WORKER VARIABLES
		WORKLOAD * w_ptr;				// HELPER POINTER FOR SIMPLIFYING WORKER REFERENCES
/*----------------------------MINING VARIABLES---------------------------*/
    int chain_blocks[num_workers];
    int errEOF[num_workers];

		// ALLOCATE WORKLOAD VARIABLES
		w_load = (WORKLOAD*)malloc(sizeof(WORKLOAD)*num_workers);

		for(int i = 0; i < num_workers; i++){
			// ALLOCATE WORKLOAD INNER VARIABLES
			allocWorkload(i+1, &w_load[i], WORKER_BUFFER_SIZE);
			POP_DOMAIN(w_handle[i]); // END WORKER ALLOCATION RANGE
		}

/*------------------------------------------------------------------------*/
/**************************************************************************/

/**********************************************************************************************************************************/
/********************************************************PARENT ALLOCATION*********************************************************/
/**********************************************************************************************************************************/
		if(multilevel == 1){
			// Profiling functions
			sprintf(stream_name, "PARENT");
			DOMAIN_CREATE(p_handle, stream_name);
			PUSH_DOMAIN(p_handle, stream_name, -1, 0, 8);
			PUSH_DOMAIN(p_handle, "ALLOC", -1, 2, 0);
		}
/**************************VARIABLE DECLARATIONS**************************/
/*-------------------------MAIN PARENT VARIABLES-------------------------*/
		WORKLOAD * p_load;							// PARENT WORKING VARIABLES

/*-------------------------PARENT CUDA VARIABLES--------------------------*/
    // GET TIME NEEDED TO CREATE EACH PARENT BLOCK
    float pbuff_timing = 0;
    double pbuff_diffSum = 0;
    cudaEvent_t buff_p1, buff_p2;
/*------------------------PARENT IO FILE VARIABLES-------------------------*/
    char bfilename[50];
    char hfilename[50];
/*------------------------PARENT MINING VARIABLES--------------------------*/
    int worker_record[PARENT_BLOCK_SIZE];

    int parentFlag=0;
    int pchain_blocks=0;
/*-----------------------------------------------------------------------*/
/****************************PARENT ALLOCATION****************************/
    if(multilevel == 1){
				p_load = (WORKLOAD*)malloc(sizeof(WORKLOAD));
				allocWorkload(0, p_load, PARENT_BLOCK_SIZE);
				POP_DOMAIN(p_handle);  // POP ALLOC RANGE
    }

/*------------------------------------------------------------------------*/
/**************************************************************************/
POP_DOMAIN(t_handle); // END ALLOC RANGE

/**********************************************************************************************************************************/
/**********************************************************INITIALIZATION**********************************************************/
/**********************************************************************************************************************************/
PUSH_DOMAIN(t_handle, "FILES", -2, 2, 1); // START FILE INITIALIZATION RANGE

/*-------------------------BLOCK INITIALIZATION--------------------------*/
// WORKER INITIALIZE WITH WORKLOAD
for(int i = 0; i < num_workers; i++){
	initializeHash(&w_load[i]);
	initializeWorkerBlock(&w_load[i]);
	initializeOutfile((&w_load[i])->outFile, out_location, (&w_load[i])->id);
}

POP_DOMAIN(t_handle); // FINISH FILE INIT

/*------------------------------------------------------------------------*/
/**************************************************************************/
PUSH_DOMAIN(t_handle, "INIT", -2, 2, 2); // START VARIABLES INIT

/*-------------------------FLAG INITIALIZATION----------------------------*/
    WORD * time_h;
    cudaStream_t tStream;
    initTime(&tStream, &time_h);
		sprintf(stream_name, "TIME UPDATE");
		NAME_STREAM(tStream, stream_name);

		// Variables for time based stop conditions
		WORD * start_time;
		start_time = (WORD *)malloc(sizeof(WORD));

		WORD * elapsed_time;
		elapsed_time = (WORD *)malloc(sizeof(WORD));

		*start_time = *time_h;
		*elapsed_time = 0;

    int FLAG_TARGET = 0;
    int PROC_REMAINING = num_workers+multilevel;

    int mining_state;

/*------------------------------------------------------------------------*/
/**************************************************************************/

/**********************************************************************************************************************************/
/******************************************************WORKER INITIALIZATION*******************************************************/
/**********************************************************************************************************************************/

/*------------------------THREAD INITIALIZATION---------------------------*/
    for(int i = 0; i < num_workers; i++){
				PUSH_DOMAIN(w_handle[i], "INIT", i, 2, 2);
				sprintf(stream_name, "WORKER_%i", i);
				NAME_STREAM((&w_load[i])->stream, stream_name);
        chain_blocks[i] = 0; errEOF[i] = 0;
				// GETS AND SETS WORKER DIFFICULTY
				getDifficulty(&w_load[i]);
				POP_DOMAIN(w_handle[i]); // POP WORKER INIT RANGE
    }
/*------------------------------------------------------------------------*/
/**************************************************************************/

/**********************************************************************************************************************************/
/******************************************************PARENT INITIALIZATION*******************************************************/
/**********************************************************************************************************************************/
if(multilevel == 1){
			PUSH_DOMAIN(p_handle, "INIT", -1, 2, 2);
	/*-------------------------BLOCK INITIALIZATION--------------------------*/
			sprintf(bfilename, "outputs/results_%i_pchains/pBlockOutputs.txt",num_workers);
			sprintf(hfilename, "outputs/results_%i_pchains/pHashOutputs.txt",num_workers);
			initializeParentOutputs(bfilename, hfilename);
	/*------------------------CHAIN INITIALIZATION---------------------------*/
			sprintf(stream_name, "PARENT");
			NAME_STREAM(p_load->stream, stream_name);
			cudaEventCreate(&buff_p1);
			cudaEventCreate(&buff_p2);
			initializeParentBlock(p_load->block_h);
			getDifficulty(p_load);
			POP_DOMAIN(p_handle);  // POP ALLOC RANGE
}

/*------------------------------------------------------------------------*/
/**************************************************************************/

/**********************************************************************************************************************************/
/********************************************************MINING LOOP BEGIN*********************************************************/
/**********************************************************************************************************************************/
//POP_DOMAIN(t_handle); // END PARENT INIT
POP_DOMAIN(t_handle); // END TIMING INIT
cudaEventRecord(g_time[0], g_timeStream);
PUSH_DOMAIN(t_handle, "START", -2, 2, 3); // START STREAM INIT
/*--------------------------------------------------------------------------------------------------------------------------------*/
/**************************************************INITIALIZE ASYNCHRONOUS STREAMS*************************************************/
		if(multilevel == 1){
			PUSH_DOMAIN(p_handle, "MINING", -1, 2, 3);
			PUSH_DOMAIN(p_handle, "DIFF", -1, 0, 5); //FIXME
		}
		for(int i = 0; i < num_workers; i++){
			PUSH_DOMAIN(w_handle[i], "MINING", i, 2, 3);
			PUSH_DOMAIN(w_handle[i], "DIFF", i, 0, 5);
			PUSH_DOMAIN(w_handle[i], "START", i, 2, 3);  // START WORKER MINING
		}
    for(int i = 0; i < num_workers; i++){
				//logStart((&w_load[i])->id, 1, (&w_load[i])->buffer_h);
        //cudaEventRecord((&w_load[i])->t_start, (&w_load[i])->stream); // HANDLED IN launchWorkflow
        cudaEventRecord((&w_load[i])->t_diff_start, (&w_load[i])->stream);
				// TODO MODIFY TO ENABLE MERKLE HASHING ON A SECOND STREAM (REQUIRES PARENT MULTISTREAM FOR COMPUTE QUEUE)
				launchWorkflow(&w_load[i]);
				/* FIXME OLD Miner
				launchMiner(&w_load[i]);
				*/
				POP_DOMAIN(w_handle[i]); // POP START
				PUSH_DOMAIN(w_handle[i], "B", i, 2, 5);  // START BLOCKS
				// TODO START DIFFICULTY RANGE & BLOCK COUNT HERE
    }
    // START PARENT TIMERS
    if(multilevel == 1){
      cudaEventRecord(buff_p1, p_load->stream);
      cudaEventRecord(p_load->t_diff_start, p_load->stream);
    }
		POP_DOMAIN(t_handle); // END STREAM INITIALIZATION
    cudaEventRecord(g_time[1], g_timeStream);
		PUSH_DOMAIN(t_handle, "MINING", -2, 2, 5); // START MINING LOOP
    /*--------------------------------------------------------------------------------------------------------------------------------*/
    /********************************************BEGIN MINING UNTIL TARGET BLOCKS ARE FOUND********************************************/
    int block_total = 0;

		// MINING LOOP UNTIL THE TARGET NUMBER OF BLOCKS ARE MINED OR THE TIME LIMIT IS REACHED
		while( (block_total < TARGET_BLOCKS && ((TIMEOUT == 1)?((*elapsed_time) < TIME_LIMIT):1)) || PROC_REMAINING != 0){
      updateTime(&tStream, time_h, t_handle);
			*elapsed_time = (*time_h - *start_time);
      if(MINING_PROGRESS == 1){
        mining_state = printProgress(mining_state, multilevel, num_workers, pchain_blocks, chain_blocks);
      }
      // SET FLAG_TARGET TO 1
			// BEGIN SHUTDOWN PROCESS IF AN END CONDITION IS MET
			if((block_total >= TARGET_BLOCKS || (TIMEOUT == 1 && ((*elapsed_time) >= TIME_LIMIT))) && FLAG_TARGET == 0){
          FLAG_TARGET = 1;

					// END MINING SECTION, MOVE ON TO FINAL HASH
					for(int i = 0; i < num_workers; i++){
						POP_DOMAIN(w_handle[i]); // POP BLOCKS, REPLACE WITH FINAL
						PUSH_DOMAIN(w_handle[i], "FINAL", i, 2, 6);  // START FINAL MINING
					}
					POP_DOMAIN(t_handle); // END MINING LOOP
          cudaEventRecord(g_time[2], g_timeStream);
					PUSH_DOMAIN(t_handle, "FINAL", -2, 2, 6); // START FINAL LOOP

					if(TIMEOUT == 1 && ((*elapsed_time) >= TIME_LIMIT)){
						printLog("\n\n**************************************************\nTIME LIMIT REACHED, FINISHING REMAINING PROCESSES*\n**************************************************\n\n");
					}
					else{
          	printLog("\n\n**********************************************\nTARGET REACHED, FINISHING REMAINING PROCESSES*\n**********************************************\n\n");
					}
      }
      /*--------------------------------------------------------------------------------------------------------------------------------*/
      /*******************************************LOOP OVER MINERS TO CHECK STREAM COMPLETION********************************************/
      for(int i = 0; i < num_workers; i++){
				w_ptr = &w_load[i];

        if(multilevel == 1){  // CHECK PARENT MINER COMPLETION STATUS IF MULTILEVEL
					if(p_load->alive == 1){ // MAKE SURE PARENT STREAM IS ALIVE BEFORE CHECKING IT
	          if(cudaStreamQuery(p_load->stream) == 0 && parentFlag == 1){   // PARENT CHAIN RESULTS ARE READY, PROCESS OUTPUTS AND PRINT
	            // processParent
	            p_load->blocks++;
							cudaEventRecord(p_load->t_stop, p_load->stream);
							returnMiner(p_load);
	            cudaEventSynchronize(p_load->t_stop);
	            cudaEventElapsedTime(&p_load->t_result, p_load->t_start, p_load->t_stop);
							printOutputFile(bfilename, p_load->block_h, p_load->hash_h, p_load->blocks, p_load->t_result, p_load->difficulty, -1, 1);
							updateParentHash(p_load->block_h, p_load->hash_h);
	            parentFlag = 0;
							POP_DOMAIN(p_handle); // POP THE PREVIOUS BLOCK
	          }
	          // PARENT CHAIN IS STILL PROCESSING LAST BLOCK, WAIT FOR COMPLETION
	          else if(parentFlag == 1 && p_load->buff_blocks == PARENT_BLOCK_SIZE){
	                cudaError_t pErr = cudaStreamQuery(p_load->stream);
	                char alert_buf_full[1000];
	                char alert_start[150] = "\n***********************************************************************\nALERT: PARENT BUFFER IS FULL AND PREVIOUS BLOCK IS NOT YET FINISHED!!!*\n";
	                char alert_end[150] = "BLOCKING UNTIL MINING RESOURCES ARE AVAILABLE...                      *\n***********************************************************************\n";
	                sprintf(alert_buf_full, "%sPARENT STREAM STATUS: [CODE: %i]:(%s: %s)*\n%s", alert_start, pErr, cudaGetErrorName(pErr), cudaGetErrorString(pErr), alert_end);
	                printDebug(alert_buf_full);
	                cudaEventRecord(errStart, errStream);
	                cudaEventRecord(buff_p2, errStream);		// FIXME THIS WILL CAUSE TIMING BLOCK ON THE PARENT STREAM (MOVE TO DEFAULT IF STREAMS ARE NONBLOCKING)
									cudaEventSynchronize(buff_p2);
									cudaEventElapsedTime(&pbuff_timing, buff_p1, buff_p2);
	                // WAIT FOR PARENT TO FINISH, THEN RETRIEVE RESULTS
	                while(cudaStreamQuery(p_load->stream) != 0){
	                  updateTime(&tStream, time_h, t_handle);
	                  if(MINING_PROGRESS == 1){
	                    mining_state = printProgress(mining_state, multilevel, num_workers, pchain_blocks, chain_blocks);
	                  }
	                  // MONITOR WORKER TIMING WHILE WAITING
	                  for(int j = 0; j < num_workers; j++){
											if((&w_load[j])->alive == 1){ // ONLY CHECK LIVING WORKERS
												// CHECK IF STREAM IS READY
												if(cudaStreamQuery((&w_load[j])->stream) == cudaSuccess){
													// UPDATE TIMING RESULT IF NECCESSARY
													if((&w_load[j])->t_result == 0){
														cudaEventRecord((&w_load[j])->t_stop, (&w_load[j])->stream);
		                        cudaEventSynchronize((&w_load[j])->t_stop);
		                        cudaEventElapsedTime(&(&w_load[j])->t_result, (&w_load[j])->t_start, (&w_load[j])->t_stop);
													}
													if((&w_load[j])->t_diff == 0 && ((&w_load[j])->blocks >= (&w_load[j])->diff_level * DIFFICULTY_LIMIT || FLAG_TARGET == 1)){
														cudaEventRecord((&w_load[j])->t_diff_stop, (&w_load[j])->stream);
		                        cudaEventSynchronize((&w_load[j])->t_diff_stop);
		                        cudaEventElapsedTime(&(&w_load[j])->t_diff, (&w_load[j])->t_diff_start, (&w_load[j])->t_diff_stop);
													}
												}
											}
	                  }
	                }
	                cudaEventRecord(errFinish, errStream);
	                cudaStreamSynchronize(errStream);
	                cudaEventElapsedTime(&err_time, errStart, errFinish);
	                printErrorTime(error_filename, (char*)"PARENT BUFFER IS FULL AND PREVIOUS BLOCK IS NOT YET FINISHED!!!", err_time);

									p_load->blocks++;
									cudaEventRecord(p_load->t_stop, p_load->stream);
									returnMiner(p_load);
			            cudaEventSynchronize(p_load->t_stop);
			            cudaEventElapsedTime(&p_load->t_result, p_load->t_start, p_load->t_stop);
									printOutputFile(bfilename, p_load->block_h, p_load->hash_h, p_load->blocks, p_load->t_result, p_load->difficulty, -1, 1);
									updateParentHash(p_load->block_h, p_load->hash_h);
	                parentFlag = 0;
									POP_DOMAIN(p_handle); // POP THE PREVIOUS BLOCK
	          }
	          // PARENT BUFFER IS READY, EXIT FOR LOOP TO BEGIN PARENT EXECUTION
	          if(p_load->buff_blocks == PARENT_BLOCK_SIZE){
	              printDebug("NEW PARENT BLOCK IS READY!\n");
	              break;
	          }
					}
        } // END PARENT CHAIN MONITOR
        // PROCESS WORKER RESULTS AND START NEXT BLOCK IF THE TARGET HAS NOT BEEN MET
				if(w_ptr->alive == 1){ // ONLY PROCEED IF THE STREAM ISN'T DEAD
					if(cudaStreamQuery(w_ptr->stream) == cudaSuccess && errEOF[i] != 1){
						// RECORD WORKER TIME IF NOT DONE ALREADY
						if(w_ptr->t_result == 0){
							cudaEventRecord(w_ptr->t_stop, w_ptr->stream);
							cudaEventSynchronize(w_ptr->t_stop);
							cudaEventElapsedTime(&w_ptr->t_result, w_ptr->t_start, w_ptr->t_stop);
						}
						// UPDATE WORKER COUNTERS
						w_ptr->blocks++;
						chain_blocks[i]++;
						block_total++;
						// GET RESULTS AND TIME FOR PRINTING
						returnMiner(w_ptr);
						printOutputFile(w_ptr->outFile, w_ptr->block_h, w_ptr->hash_h, w_ptr->blocks, w_ptr->t_result, w_ptr->difficulty, i, 1);
						// PRINT TO PARENT HASH FILE AND ADD RESULTS TO PARENT BUFFER IF MULTILEVEL
						POP_DOMAIN(w_handle[i]); // POP CURRENT BLOCK

						if(multilevel == 1){
							printOutputFile(hfilename, w_ptr->block_h, w_ptr->hash_h, w_ptr->blocks, w_ptr->t_result, w_ptr->difficulty, i, 0);
							// COPY HASH TO THE PARENT BUFFER
							for(int j = 0; j < 8; j++){
								p_load->buffer_h[p_load->buff_blocks*8 + j] = w_ptr->hash_h[j];
							}
							worker_record[p_load->buff_blocks] = w_ptr->id;
							pbuff_diffSum+=w_ptr->difficulty;
							p_load->buff_blocks++;
						}
						// INCREMENT DIFFICULTY IF THE LIMIT HAS BEEN REACHED (PRINT IF TARGET HAS BEEN REACHED)
						if(w_ptr->blocks >= w_ptr->diff_level * DIFFICULTY_LIMIT || FLAG_TARGET == 1){
							// PRINT DIFFICULTY BLOCK STATISTICS
							if(w_ptr->t_diff == 0){ // DIFF TIMER NOT YET RECORDED, RECORD EVENT NOW
								cudaEventRecord(w_ptr->t_diff_stop, w_ptr->stream);
								cudaEventSynchronize(w_ptr->t_diff_stop);
								cudaEventElapsedTime(&w_ptr->t_diff, w_ptr->t_diff_start, w_ptr->t_diff_stop);
							}
							printDifficulty(w_ptr->outFile, w_ptr->id, w_ptr->difficulty, w_ptr->t_diff, (w_ptr->blocks-(w_ptr->diff_level-1)*DIFFICULTY_LIMIT));

							// INCREMENT IF TARGET HASN'T BEEN REACHED
							if(FLAG_TARGET == 0){
								POP_DOMAIN(w_handle[i]); // POP CURRENT DIFF
								updateDifficulty(w_ptr->block_h, w_ptr->diff_level);
								getDifficulty(w_ptr);
								cudaEventRecord(w_ptr->t_diff_start, w_ptr->stream);
								w_ptr->diff_level++;
								w_ptr->t_diff = 0;
								PUSH_DOMAIN(w_handle[i], "DIFF", i, 2, 5);  // START NEW DIFF
							}
						}

						// MINE NEXT BLOCK ON THIS WORKER IF TARGET HASN'T BEEN REACHED
						if(FLAG_TARGET == 0){
							PUSH_DOMAIN(w_handle[i], "B", i, 2, 5);  // START NEXT BLOCK
							// CHANGED Added update for workload
//							errEOF[i] = updateBlock(w_ptr->inFile, w_ptr->block_h, w_ptr->hash_h, w_ptr->buffer_h);
							errEOF[i] = updateBlock_load(w_ptr);
							if(errEOF[i] == 1){
								char eof_str[20];
								sprintf(eof_str, "WORKER %i INPUT EOF!", i+1);
								printErrorTime(error_filename, eof_str, 0.0);
							}
							//logStart(w_ptr->id, (w_ptr->blocks)+1, w_ptr->buffer_h);
							// RESET TIMING RESULT TO ZERO FOR NEXT BLOCK
							w_ptr->t_result = 0;
							launchWorkflow(w_ptr);
							/*
							cudaEventRecord(w_ptr->t_start, w_ptr->stream);
							launchMiner(w_ptr);
							*/
						} else{ // EXECUTION COMPLETED, MARK WORKER AS NO LONGER ACTIVE
							w_ptr->alive = 0;
							// END WORKER FINAL, START CLEANUP
							POP_DOMAIN(w_handle[i]); // POP DIFF
							POP_DOMAIN(w_handle[i]); // POP MINING
							PUSH_DOMAIN(w_handle[i], "CLEAN", i, 2, 9);  // END WORKER MINING
							PROC_REMAINING--;
						}
					}
				}
			} // FOR LOOP END
      /*--------------------------------------------------------------------------------------------------------------------------------*/
      /**********************************************START PARENT MINING WHEN BUFFER IS FULL*********************************************/
      // PROC_REMAINING == 1 INDICATES THAT THIS IS THE FINAL ITERATION, MUST BE AT LEAST 1 BLOCK IN BUFFER FROM PRIOR WORKER BLOCKS
      if((multilevel == 1 && parentFlag == 0) && (p_load->buff_blocks == PARENT_BLOCK_SIZE || PROC_REMAINING == 1)){
    //    if(pbuffer_blocks > 0){
          // COPY IN THE CURRENT BUFFER CONTENTS
          char merkle_debug[50+PARENT_BLOCK_SIZE*100];
          char hash_entry[80];
          BYTE temp_hash[65];

					// TODO ADD WORKLOAD VARS TO HANDLE MERKLE HASHING (CAN BE USED FOR HASH INPUTS TOO)
					if(DEBUG == 1){
						sprintf(merkle_debug, "PARENT BLOCK %i CONTENTS:  \n", pchain_blocks+1);
	          for(int i = 0; i < p_load->buff_blocks; i++){
							decodeWord(&(p_load->buffer_h[i*8]), temp_hash, 8);
	            sprintf(hash_entry, "WORKER %i\t%s\n", worker_record[i], (char*)temp_hash);
	            strcat(merkle_debug, hash_entry);
	          }
	          // PRINT PARENT BLOCK CONTENTS
	          printDebug(merkle_debug);
					}

          // PARENT DIFFICULTY SCALING
          if(p_load->blocks >= p_load->diff_level * DIFFICULTY_LIMIT){ // Increment difficulty
						POP_DOMAIN(p_handle); // POP THE PREVIOUS DIFFICULTY
            cudaEventRecord(p_load->t_diff_stop, p_load->stream);
            cudaEventSynchronize(p_load->t_diff_stop);
            cudaEventElapsedTime(&p_load->t_diff, p_load->t_diff_start, p_load->t_diff_stop);
            printDifficulty(bfilename, -1, p_load->difficulty, p_load->t_diff, (p_load->blocks-(p_load->diff_level-1)*DIFFICULTY_LIMIT));
						updateDifficulty(p_load->block_h, p_load->diff_level);
						getDifficulty(p_load);
            cudaEventRecord(p_load->t_diff_start, p_load->stream);
            p_load->diff_level++;
						PUSH_DOMAIN(p_handle, "DIFF", -1, 0, 5); // PUSH NEW DOMAIN
          }
					PUSH_DOMAIN(p_handle, "B", -1, 2, 5); // START NEXT BLOCK

					// PRINT OUT BUFFER STATS
					if(pbuff_timing == 0){ // NEW BUFFER TIMER NOT YET RECORDED, RECORD EVENT NOW
						cudaEventRecord(buff_p2, p_load->stream);
						cudaEventSynchronize(buff_p2);
						cudaEventElapsedTime(&pbuff_timing, buff_p1, buff_p2);
					}
					pbuff_diffSum /= p_load->buff_blocks;
					printDifficulty(hfilename, 0, pbuff_diffSum, pbuff_timing, p_load->buff_blocks);
					pbuff_diffSum = 0;
					pbuff_timing = 0;
					cudaEventRecord(buff_p1, p_load->stream);

//					cudaEventRecord(p_load->t_start, p_load->stream);
					// CHANGED Using workflow for parent
					launchWorkflow(p_load);

					/*
					launchMerkle(p_load); // UPDATE BLOCK AT THE END OF MERKLE HASHING
					logStart(p_load->id, p_load->blocks+1, &p_load->block_h[9]); // TODO Callback after merkle
					launchMiner(p_load);
					*/

//          cudaEventRecord(p_load->t_stop, p_load->stream);
          p_load->buff_blocks = 0;
          parentFlag = 1;

          // FINAL ITERATION, WAIT FOR PARENT STREAM TO FINISH
          if(PROC_REMAINING == 1){
            while(cudaStreamQuery(p_load->stream) != 0){
              updateTime(&tStream, time_h, t_handle);
              if(MINING_PROGRESS == 1){
                mining_state = printProgress(mining_state, multilevel, num_workers, p_load->blocks, chain_blocks);
              }
            }
            p_load->blocks++;
						cudaEventRecord(p_load->t_stop, p_load->stream);
						returnMiner(p_load);
            cudaEventSynchronize(p_load->t_stop);
            cudaEventElapsedTime(&p_load->t_result, p_load->t_start, p_load->t_stop);
						printOutputFile(bfilename, p_load->block_h, p_load->hash_h, p_load->blocks, p_load->t_result, p_load->difficulty, -1, 1);
						updateParentHash(p_load->block_h, p_load->hash_h);
            parentFlag = 0;
						POP_DOMAIN(p_handle); // POP THE PREVIOUS BLOCK

            cudaEventRecord(p_load->t_diff_stop, p_load->stream);
            cudaEventSynchronize(p_load->t_diff_stop);
            cudaEventElapsedTime(&p_load->t_diff, p_load->t_diff_start, p_load->t_diff_stop);
            printDifficulty(bfilename, -1, p_load->difficulty, p_load->t_diff, (p_load->blocks-(p_load->diff_level-1)*DIFFICULTY_LIMIT));

						// FINISH PARENT, MOVE ON TO CLEANUP
						POP_DOMAIN(p_handle); //POP DIFF
						POP_DOMAIN(p_handle); //POP MINING
						PUSH_DOMAIN(p_handle, "CLEAN", -1, 2, 9);

						p_load->alive = 0;
            cudaEventDestroy(buff_p1);
            cudaEventDestroy(buff_p2);
            PROC_REMAINING--;
          }
      }
    } // WHILE LOOP END
		POP_DOMAIN(t_handle); // END FINAL LOOP
    cudaEventRecord(g_time[3], g_timeStream);
		PUSH_DOMAIN(t_handle, "CLEAN", -2, 2, 9); // START MEMORY FREEING
		cudaDeviceSynchronize();
    printLog("FINISHED PROCESSING, FREEING MEMORY");
    /**********************************************************************************************************************************/
    /***************************************************FREE HOST AND DEVICE MEMORY****************************************************/
    /**********************************************************************************************************************************/

    /*--------------------------------------------------------------------------------------------------------------------------------*/
    /*********************************************************CLOSE INPUT FILES********************************************************/
    destroyCudaVars(&errStart, &errFinish, &errStream);
    for(int i = 0; i < num_workers; i++){
      fclose((&w_load[i])->inFile);
    }
    /*--------------------------------------------------------------------------------------------------------------------------------*/
    /*******************************************************FREE MINING VARIABLES******************************************************/
    printDebug((const char*)"FREEING MINING MEMORY");
    freeTime(&tStream, &time_h);
		free(start_time);
		free(elapsed_time);
    /*--------------------------------------------------------------------------------------------------------------------------------*/
    /*************************************************FREE PARENT AND WORKER VARIABLES*************************************************/
    printDebug((const char*)"FREEING WORKER MEMORY");
		for(int i = 0; i < num_workers; i++){
			freeWorkload(&w_load[i]);
		}
		free(w_load);

		// DESTROY WORKER PROFILING DOMAINS
		for(int i = 0; i < num_workers; i++){
			POP_DOMAIN(w_handle[i]);POP_DOMAIN(w_handle[i]);
			DOMAIN_DESTROY(w_handle[i]);
		}
		if(multilevel == 1){
      printDebug((const char*)"FREEING PARENT MEMORY");
			freeWorkload(p_load);
			free(p_load);
			// DESTROY PARENT PROFILING DOMAINS
			POP_DOMAIN(p_handle);POP_DOMAIN(p_handle);
			DOMAIN_DESTROY(p_handle);
		}

    /**********************************************************************************************************************************/
    /******************************************************PRINT TIMING ANALYSIS*******************************************************/
    /**********************************************************************************************************************************/
    // GET TIMING INTERVALS
    cudaEventRecord(g_timeFinish, g_timeStream);
    cudaStreamSynchronize(g_timeStream);
    cudaEventElapsedTime(&total_time[0], g_timeStart, g_time[0]);
    cudaEventElapsedTime(&total_time[1], g_time[0], g_time[1]);
    cudaEventElapsedTime(&total_time[2], g_time[1], g_time[2]);
    cudaEventElapsedTime(&total_time[3], g_time[2], g_time[3]);
    cudaEventElapsedTime(&total_time[4], g_time[3], g_timeFinish);
    cudaEventElapsedTime(&total_time[5], g_timeStart, g_timeFinish);

    // CREATE TIMING ANALYSIS STRING
    char time_str[1000];
sprintf(time_str, "\n/****************************TIMING ANALYSIS FOR %i WORKER CHAINS%s****************************/\n\
TIMING-1: VARIABLE_INITIALIZATION: %f\n\
TIMING-2: STREAM_INITIALIZATION: %f\n\
TIMING-3: MAIN_LOOP: %f\n\
TIMING-4: FINAL_ITERATION: %f\n\
TIMING-5: MEMORY_CLEANUP: %f\n\
/**********************************************************************************************/\n\
TOTAL_EXECUTION_TIME: %f\n\
/**********************************************************************************************/\n\
", num_workers, (multilevel == 1 ? " WITH PARENT CHAIN": ""), total_time[0],total_time[1],total_time[2],total_time[3],total_time[4],total_time[5]);

    FILE * time_outFile;
    if(time_outFile = fopen(time_filename, "w")){
      fprintf(time_outFile, "\n%s\n", time_str);
      fclose(time_outFile);
    }else{
      printError("TIMING ANALYSIS WRITING FAILED!!");
      printErrorTime(error_filename, (char*)"TIMING ANALYSIS WRITING FAILED!!", 0.0);
    }
    printLog(time_str);
    printDebug("TIMING ANALYSIS COMPLETE, FREEING TIMING VARIABLES");
    destroyCudaVars(&g_timeStart, &g_timeFinish, &g_timeStream);
    for(int i = 0; i < 4; i++){
      cudaEventDestroy(g_time[i]);
    }

		// DESTROY TIMING PROFILING DOMAINS
		POP_DOMAIN(t_handle); // END MEMORY FREE LOOP
		POP_DOMAIN(t_handle); // END TIMING RANGE
		DOMAIN_DESTROY(t_handle); // FREE TIMING DOMAIN
		STOP_PROFILE; // END PROFILING

    printLog("APPLICATION FINISHED. NOW EXITING...");
    cudaDeviceSynchronize();
    return;
}




/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
/*********************  ____________________________________________________________________________________________________________________________________________    *********************/
/*********************  |                                                                                                                                           |   *********************/
/*********************  |    /$$   /$$  /$$$$$$   /$$$$$$  /$$$$$$$$       /$$$$$$$$ /$$   /$$ /$$   /$$  /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$  /$$$$$$   |   *********************/
/*********************  |   | $$  | $$ /$$__  $$ /$$__  $$|__  $$__/      | $$_____/| $$  | $$| $$$ | $$ /$$__  $$|__  $$__/|_  $$_/ /$$__  $$| $$$ | $$ /$$__  $$  |   *********************/
/*********************  |   | $$  | $$| $$  \ $$| $$  \__/   | $$         | $$      | $$  | $$| $$$$| $$| $$  \__/   | $$     | $$  | $$  \ $$| $$$$| $$| $$  \__/  |   *********************/
/*********************  |   | $$$$$$$$| $$  | $$|  $$$$$$    | $$         | $$$$$   | $$  | $$| $$ $$ $$| $$         | $$     | $$  | $$  | $$| $$ $$ $$|  $$$$$$   |   *********************/
/*********************  |   | $$__  $$| $$  | $$ \____  $$   | $$         | $$__/   | $$  | $$| $$  $$$$| $$         | $$     | $$  | $$  | $$| $$  $$$$ \____  $$  |   *********************/
/*********************  |   | $$  | $$| $$  | $$ /$$  \ $$   | $$         | $$      | $$  | $$| $$\  $$$| $$    $$   | $$     | $$  | $$  | $$| $$\  $$$ /$$  \ $$  |   *********************/
/*********************  |   | $$  | $$|  $$$$$$/|  $$$$$$/   | $$         | $$      |  $$$$$$/| $$ \  $$|  $$$$$$/   | $$    /$$$$$$|  $$$$$$/| $$ \  $$|  $$$$$$/  |   *********************/
/*********************  |   |__/  |__/ \______/  \______/    |__/         |__/       \______/ |__/  \__/ \______/    |__/   |______/ \______/ |__/  \__/ \______/   |   *********************/
/*********************  |___________________________________________________________________________________________________________________________________________|   *********************/
/*********************                                                                                                                                                  *********************/
/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/





/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/********  _______________________________________________________________________________________________________________________________________________________  ********/
/********  |    _______   ______    _____   _______   _____   _   _    _____     ______   _    _   _   _    _____   _______   _____    ____    _   _    _____    |  ********/
/********  |   |__   __| |  ____|  / ____| |__   __| |_   _| | \ | |  / ____|   |  ____| | |  | | | \ | |  / ____| |__   __| |_   _|  / __ \  | \ | |  / ____|   |  ********/
/********  |      | |    | |__    | (___      | |      | |   |  \| | | |  __    | |__    | |  | | |  \| | | |         | |      | |   | |  | | |  \| | | (___     |  ********/
/********  |      | |    |  __|    \___ \     | |      | |   | . ` | | | |_ |   |  __|   | |  | | | . ` | | |         | |      | |   | |  | | | . ` |  \___ \    |  ********/
/********  |      | |    | |____   ____) |    | |     _| |_  | |\  | | |__| |   | |      | |__| | | |\  | | |____     | |     _| |_  | |__| | | |\  |  ____) |   |  ********/
/********  |      |_|    |______| |_____/     |_|    |_____| |_| \_|  \_____|   |_|       \____/  |_| \_|  \_____|    |_|    |_____|  \____/  |_| \_| |_____/    |  ********/
/********  |_____________________________________________________________________________________________________________________________________________________|  ********/
/********                                                                                                                                                           ********/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/******************************************************************************QUERY FUNCTIONS******************************************************************************/
// USE DEVICE PROPERTIES AND ATTRIBUTES TO DISPLAY HARDWARE INFORMATION
__host__ void hostDeviceQuery(void){
  printf("STARTING DEVICE QUERY\n\n");
  int device;
  int value;
  cudaGetDevice(&device);
  printf("GOT DEVICE: %i\n", device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("Device Number: %d\n", device);
  printf("  Device name: %s\n", prop.name);
  printf("MEMORY INFORMATION\n\n");
  printf("  Memory Clock Rate (KHz): %d\n",prop.memoryClockRate);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %f\n\n",2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  printf("  Total Global Memory: %lu\n",prop.totalGlobalMem);
  printf("  Total Constant Memory: %lu\n",prop.totalConstMem);
  printf("  Shared Memory Per Block: %lu (BYTES)\n",prop.sharedMemPerBlock);
  printf("  Registers Per Block: %i\n",prop.regsPerBlock);
  printf("BLOCK STATS \n\n");
  printf("  Warp Size: %i\n",prop.warpSize);
  printf("  Max Threads Per Block: %i\n",prop.maxThreadsPerBlock);
  printf("  Max Threads (x-dim): %i\n",prop.maxThreadsDim[0]);
  printf("  Max Threads (y-dim): %i\n",prop.maxThreadsDim[1]);
  printf("  Max Threads (z-dim): %i\n\n",prop.maxThreadsDim[2]);
  printf("  Max Grid (x-dim): %i\n",prop.maxGridSize[0]);
  printf("  Max Grid (y-dim): %i\n",prop.maxGridSize[1]);
  printf("  Max Grid (z-dim): %i\n",prop.maxGridSize[2]);
  printf("MACRO STATS \n\n");
  printf("  Multiprocessor Count: %i\n",prop.multiProcessorCount);
  printf("  Concurrent Kernels: %i\n",prop.concurrentKernels);
  printf("  Compute Capability: %i   %i\n", prop.major, prop.minor);
  printf("ATTRIBUTE QUERIES \n\n");
  cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerMultiProcessor ,device);
  printf("  Max threads per multi processor: %i\n", value);
  cudaDeviceGetAttribute(&value, cudaDevAttrAsyncEngineCount  ,device);
  printf("  Number of asynchronous engines: %i\n", value);
  cudaDeviceGetAttribute(&value, cudaDevAttrStreamPrioritiesSupported  ,device);
  printf("  Device supports stream priorities: %i\n", value);
  cudaDeviceGetAttribute(&value, cudaDevAttrGlobalL1CacheSupported   ,device);
  printf("  Device supports caching globals in L1: %i\n", value);

  cudaDeviceGetAttribute(&value, cudaDevAttrLocalL1CacheSupported  ,device);
  printf("  Device supports caching locals in L1: %i\n", value);

  cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerMultiprocessor  ,device);
  printf("  Maximum shared memory available per multiprocessor in bytes: %i\n", value);

  cudaDeviceGetAttribute(&value, cudaDevAttrMaxRegistersPerMultiprocessor  ,device);
  printf("  Maximum number of 32-bit registers available per multiprocessor: %i\n", value);

  cudaDeviceGetAttribute(&value, (cudaDeviceAttr)86  ,device);
  printf("  Link between the device and the host supports native atomic operations: %i\n", value);

  cudaDeviceGetAttribute(&value, (cudaDeviceAttr)87  ,device);
  printf("  Ratio of single precision performance to double precision performance(FP/sec): %i\n", value);

  cudaDeviceGetAttribute(&value, (cudaDeviceAttr)90  ,device);
  printf("  Device supports Compute Preemption: %i\n", value);

  cudaDeviceGetAttribute(&value, (cudaDeviceAttr)95  ,device);
  printf("  Device supports launching cooperative kernels via cudaLaunchCooperativeKernel: %i\n", value);

  cudaDeviceGetAttribute(&value, (cudaDeviceAttr)101   ,device);
  printf("  Host can directly access managed memory on the device without migration: %i\n", value);

  cudaDeviceGetAttribute(&value, (cudaDeviceAttr)99 ,device);
  printf("  Device supports host memory registration via cudaHostRegister: %i\n", value);

	cudaDeviceGetAttribute(&value, (cudaDeviceAttr)15 ,device);
	printf("\n  GPU OVERLAP: Device can possibly copy memory and execute a kernel concurrently: %i\n", value);

	cudaDeviceGetAttribute(&value, (cudaDeviceAttr)17 ,device);
	printf("\n  KernelExecTimeout: Specifies whether there is a run time limit on kernels: %i\n", value);

  return;
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*******************************************************************************TEST FUNCTIONS******************************************************************************/
// NEW HOST FUNCTIONAL TEST USING WORDS INSTEAD OF BYTES
__host__ void hostFunctionalTest(void){
  printf("STARTING FUNCTIONAL TEST\n");
	// INITIALIZE BENCHMARK VARIABLES
	WORKLOAD * t_load;
	t_load = (WORKLOAD*)malloc(sizeof(WORKLOAD));
	allocWorkload(0, t_load, 16);

	// ADD NAME TO STREAM
	NAME_STREAM(t_load->stream, "TEST STREAM");

	// STORE DIFF_REDUCE TO BE SET LATER
	int temp_reduce = DIFF_REDUCE;
	DIFF_REDUCE = 0;

	BYTE test_str[161];
	BYTE correct_str[65];

	int logSize = 500;
  char logResult[8000];
	char * logStr;
	char logMsg[logSize];

	BYTE merkle_str[1025];

	// Prepare logging variables
  logStr = (char*)malloc(sizeof(char) * logSize);
	strcpy(logResult, "\n****************************HASHING FUNCTIONAL TESTS****************************\n");

	// INITIALIZE TEST PROFILING DOMAIN
	#ifdef USE_NVTX
		DOMAIN_HANDLE handle;
	#endif

	DOMAIN_CREATE(handle, "FUNCTIONAL TESTS");
	// 80 BYTE MESSAGE (DOUBLE HASH)
	PUSH_DOMAIN(handle, "80B MINING TEST", -2, 0, 4);
		// NEW DOUBLE HASH FUNCTION
		PUSH_DOMAIN(handle, "ACCEL HASH", -2, 0, 8);
			strcpy((char*)test_str, "0100000000000000000000000000000000000000000000000000000000000000000000001979507de7857dc4940a38410ed228955f88a763c9cccce3821f0a5e65609f565c2ffb291d00ffff01004912");
			strcpy((char*)correct_str, "265a66f42191c9f6b26a1b9d4609d76a0b5fdacf9b82b6de8a3b3e904f000000");
			testMiningHash(t_load, test_str, correct_str, 0x1e00ffff, &logStr);
			sprintf(logMsg, "NEW DOUBLE HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
	POP_DOMAIN(handle);

	// VARIOUS DIFFICULTIES TEST
	PUSH_DOMAIN(handle, "DIFFICULTY TEST", -2, 2, 1);
		// 2 ZEROS (DIFFICULTY: 0x2000ffff)
		PUSH_DOMAIN(handle, "D=0x2000ffff", -2, 1, 0);
			strcpy((char*)test_str, "01000000a509fafcf42a5f42dacdf8f4fb89ff525c0ee3acb0d68ad364f2794f2d8cd1007d750847aac01636528588e2bccccb01a91b0b19524de666fdfaa4cfad669fcd5c39b1141d00ffff00005cc0");
			strcpy((char*)correct_str, "d1bca1de492c24b232ee591a1cdf16ecd8c51400d4da49a97f9536f27b286e00");
			testMiningHash(t_load, test_str, correct_str, 0x2000ffff, &logStr);
			sprintf(logMsg, "DIFFICULTY TEST 1 [0x2000ffff]: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		// 4 ZEROS (DIFFICULTY: 0x1f00ffff)
		PUSH_DOMAIN(handle, "D=0x1f00ffff", -2, 1, 1);
			strcpy((char*)test_str, "010000008e2e5fd95b75846393b579f7368ebbee8ca593ed574dd877b4255e1385cd0000286e0824b41e054a6afea14b0b4588017895ace8f9cc4837279074e238462cd75c340d171d00ffff0002043d");
			strcpy((char*)correct_str, "fbbb3f2adadd66d9d86cdacc735f99edece886faed7a0fbc17594da445820000");
			testMiningHash(t_load, test_str, correct_str, 0x1f00ffff, &logStr);
			sprintf(logMsg, "DIFFICULTY TEST 2 [0x1f00ffff]: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		// 6 ZEROS (DIFFICULTY: 0x1e00ffff)
		PUSH_DOMAIN(handle, "D=0x1e00ffff", -2, 1, 2);
			strcpy((char*)test_str, "010000000298ff1c6d24d9f04ed441ce3f3a4b695d7fdb8cc13bc7f7417a68a44b000000d49d1c71552793e1d9182ab63ca5fe8d23f2711ecb26f7b0f9ad931c5980aadb5c340d521c00ffff020caca2");
			strcpy((char*)correct_str, "46b26c30b35175ecb88ddbe08f2d56070f616b2d6f302ef334286fc575000000");
			testMiningHash(t_load, test_str, correct_str, 0x1e00ffff, &logStr);
			sprintf(logMsg, "DIFFICULTY TEST 3 [0x1e00ffff]: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		// 8 ZEROS (DIFFICULTY: 0x1d00ffff)
		PUSH_DOMAIN(handle, "D=0x1d00ffff", -2, 1, 3);
			strcpy((char*)test_str, "01000000ac44a5ddb3c7a252ab2ea9278ab4a27a5fd88999ff192d5f6e86f66b000000009984a9337cf3852ef758d5f8baf090700c89133ba9c19e27f39b465942d8e7465c3440bd1b00ffffdba51c5e");
			strcpy((char*)correct_str, "30498d768dba64bd6b1455ae358fefa3217096449f05800b61e2e93b00000000");
			testMiningHash(t_load, test_str, correct_str, 0x1d00ffff, &logStr);
			sprintf(logMsg, "DIFFICULTY TEST 4 [0x1d00ffff]: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		// 16 ZEROS (DIFFICULTY: 0x1900ffff)
		PUSH_DOMAIN(handle, "D=0x1900ffff", -2, 1, 4);
			strcpy((char*)test_str, "0100000081cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a308000000000000e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0f1fc122bc7f5d74df2b9441a42a14695");
			strcpy((char*)correct_str, "1dbd981fe6985776b644b173a4d0385ddc1aa2a829688d1e0000000000000000");
			testMiningHash(t_load, test_str, correct_str, 0x1900ffff, &logStr);
			sprintf(logMsg, "DIFFICULTY TEST 5 [0x1900ffff]: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
	POP_DOMAIN(handle);

	// VARIOUS DIFFICULTIES TEST
	PUSH_DOMAIN(handle, "DOUBLE HASH TEST", -2, 2, 2);
		// DOUBLE HASH 32B | 32B TEST
		PUSH_DOMAIN(handle, "HASH 32B|32B", -2, 1, 5);
			strcpy((char*)test_str, "1979507de7857dc4940a38410ed228955f88a763c9cccce3821f0a5e65609f56");
			strcpy((char*)correct_str, "b3ee97623477d3efda34eb42750e362422cc571547be546e1b1763ade855fdb0");
			testDoubleHash(t_load, test_str, correct_str, 32, &logStr);
			sprintf(logMsg, "32B DOUBLE HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		PUSH_DOMAIN(handle, "HASH 64B|32B", -2, 1, 6);
		strcpy((char*)test_str, "0100000000000000000000000000000000000000000000000000000000000000000000001979507de7857dc4940a38410ed228955f88a763c9cccce3821f0a5e");
		strcpy((char*)correct_str, "03761a41afdfc48a021ff6852de90f9b5972cf8a4d0338e43cb8eb4f6044786b");
			testDoubleHash(t_load, test_str, correct_str, 64, &logStr);
			sprintf(logMsg, "64B DOUBLE HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
	POP_DOMAIN(handle);

	PUSH_DOMAIN(handle, "MERKLE TEST", -2, 2, 3);
		// MERKLE HASH TESTS
		PUSH_DOMAIN(handle, "MERKLE 1", -2, 1, 0);
			strcpy((char*)merkle_str, "6be0ad2cd9b2014644504878974800baf96d52f0767d5ba68264139f95df4869");
			strcpy((char*)correct_str, "ba26064e7dad783f2e3a49071e674accc2efcaf45254b42149abf861dfce033f");
			testMerkleHash(t_load, merkle_str, correct_str, 1, &logStr);
			sprintf(logMsg, "MERKLE 1 HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		PUSH_DOMAIN(handle, "MERKLE 2-1", -2, 1, 1);
			strcpy((char*)merkle_str, "6be0ad2cd9b2014644504878974800baf96d52f0767d5ba68264139f95df4869");
			strcat((char*)merkle_str, "7a97ceb4c13ae5ecd87317d3bce4305af9de043800b9e0dde83fb0967c52b162");
			strcpy((char*)correct_str, "f5eb35cd8091643a174f0e7eda768f6f51a5d3e61691eb1b302653c7149cff2c");
			testMerkleHash(t_load, merkle_str, correct_str, 2, &logStr);
			sprintf(logMsg, "MERKLE 2-1 HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		PUSH_DOMAIN(handle, "MERKLE 2-2", -2, 1, 2);
			strcpy((char*)merkle_str, "4a999e696ac674fdbf7a94876d9e230aa31ba4282d21e564d064e5950afb225e");
			strcat((char*)merkle_str, "a16da6f6849fe9d9e6a02667d9bcce28b411b64bfad7869d136112f9dfabeeb8");
			strcpy((char*)correct_str, "561dbd4591dfbd2352da56036881b18bf8e1dc7771397b807bba500449ee8243");
			testMerkleHash(t_load, merkle_str, correct_str, 2, &logStr);
			sprintf(logMsg, "MERKLE 2-2 HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		PUSH_DOMAIN(handle, "MERKLE 4-1", -2, 1, 3);
			strcpy((char*)merkle_str, "6be0ad2cd9b2014644504878974800baf96d52f0767d5ba68264139f95df4869");
			strcat((char*)merkle_str, "7a97ceb4c13ae5ecd87317d3bce4305af9de043800b9e0dde83fb0967c52b162");
			strcat((char*)merkle_str, "4a999e696ac674fdbf7a94876d9e230aa31ba4282d21e564d064e5950afb225e");
			strcat((char*)merkle_str, "a16da6f6849fe9d9e6a02667d9bcce28b411b64bfad7869d136112f9dfabeeb8");
			strcpy((char*)correct_str, "9469e5f693434dab893fbd7adc376a1df75011bde71aa1b30e5fd37db038f7f4");
			testMerkleHash(t_load, merkle_str, correct_str, 4, &logStr);
			sprintf(logMsg, "MERKLE 4-1 HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		PUSH_DOMAIN(handle, "MERKLE 4-2", -2, 1, 4);
			strcpy((char*)merkle_str, "fa5412058b60f2c5877a5ab55ce3d4d40623439f2234edfc9bfa829ebf1646ec");
			strcat((char*)merkle_str, "2384040c97479c51cead374a9b093ae2571dff5921856b31c956270609388fbb");
			strcat((char*)merkle_str, "8a301aceff3f16a6c441237492c2b358c7e2346cb299be4c6b88fc0c4f949bec");
			strcat((char*)merkle_str, "4ee8b360b8a9a9b2c2f0ab3f02ca3da20fd1b2fd96a4c74b991a4b98c544feed");
			strcpy((char*)correct_str, "9b3b36b2099e2715c5eab4b54c4def46119726bffb0451936ec49a6a56f5d55c");
			testMerkleHash(t_load, merkle_str, correct_str, 4, &logStr);
			sprintf(logMsg, "MERKLE 4-2 HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		PUSH_DOMAIN(handle, "MERKLE 8-1", -2, 1, 5);
			strcpy((char*)merkle_str, "6be0ad2cd9b2014644504878974800baf96d52f0767d5ba68264139f95df4869");
			strcat((char*)merkle_str, "7a97ceb4c13ae5ecd87317d3bce4305af9de043800b9e0dde83fb0967c52b162");
			strcat((char*)merkle_str, "4a999e696ac674fdbf7a94876d9e230aa31ba4282d21e564d064e5950afb225e");
			strcat((char*)merkle_str, "a16da6f6849fe9d9e6a02667d9bcce28b411b64bfad7869d136112f9dfabeeb8");
			strcat((char*)merkle_str, "fa5412058b60f2c5877a5ab55ce3d4d40623439f2234edfc9bfa829ebf1646ec");
			strcat((char*)merkle_str, "2384040c97479c51cead374a9b093ae2571dff5921856b31c956270609388fbb");
			strcat((char*)merkle_str, "8a301aceff3f16a6c441237492c2b358c7e2346cb299be4c6b88fc0c4f949bec");
			strcat((char*)merkle_str, "4ee8b360b8a9a9b2c2f0ab3f02ca3da20fd1b2fd96a4c74b991a4b98c544feed");
			strcpy((char*)correct_str, "e3ef39f376e7e60d21f19d55571c93096ba841c7edfbbbd60d304521dfa6f679");
			testMerkleHash(t_load, merkle_str, correct_str, 8, &logStr);
			sprintf(logMsg, "MERKLE 8-1 HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		PUSH_DOMAIN(handle, "MERKLE 8-2", -2, 1, 6);
			strcpy((char*)merkle_str, "c060aff8cd43ac565db9cc16d2c955f2950666392f37e650f933087ef0a3521f");
			strcat((char*)merkle_str, "0a0fcd4ac910e2a4d999dc1749b0fb151227f9814032cd7ff87c086c35a0c29d");
			strcat((char*)merkle_str, "6d63b050cb7259a40b95aa4735ae0405a967449b0e1189af1f4a798cf81a8733");
			strcat((char*)merkle_str, "11dc07d576f64a25a5a5dc3f0af7b07138070c1bb3461c9261795d31ca5f78d5");
			strcat((char*)merkle_str, "709a961120f2824e5e737284ecd9bc597c88abbd756d3c356d90ca248158049d");
			strcat((char*)merkle_str, "be55800cc10c078eecb039f0e4157ddef779c32baabfc113e0794437a22f16f2");
			strcat((char*)merkle_str, "72ea245bf08809e7645e9fcf8b02cf3497e2715bbb9214d1896aaa6069fd611e");
			strcat((char*)merkle_str, "f4456bc878b17beee82089ce413ec2362d51d3e01ba9071a420bd391a5421045");
			strcpy((char*)correct_str, "a3dd4163da9d676e1c59bc46fbd9f2489fe8d638ce6c04349a14ff31f2245c41");
			testMerkleHash(t_load, merkle_str, correct_str, 8, &logStr);
			sprintf(logMsg, "MERKLE 8-2 HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		PUSH_DOMAIN(handle, "MERKLE 16", -2, 1, 7);
			strcpy((char*)merkle_str, "6be0ad2cd9b2014644504878974800baf96d52f0767d5ba68264139f95df4869");
			strcat((char*)merkle_str, "7a97ceb4c13ae5ecd87317d3bce4305af9de043800b9e0dde83fb0967c52b162");
			strcat((char*)merkle_str, "4a999e696ac674fdbf7a94876d9e230aa31ba4282d21e564d064e5950afb225e");
			strcat((char*)merkle_str, "a16da6f6849fe9d9e6a02667d9bcce28b411b64bfad7869d136112f9dfabeeb8");
			strcat((char*)merkle_str, "fa5412058b60f2c5877a5ab55ce3d4d40623439f2234edfc9bfa829ebf1646ec");
			strcat((char*)merkle_str, "2384040c97479c51cead374a9b093ae2571dff5921856b31c956270609388fbb");
			strcat((char*)merkle_str, "8a301aceff3f16a6c441237492c2b358c7e2346cb299be4c6b88fc0c4f949bec");
			strcat((char*)merkle_str, "4ee8b360b8a9a9b2c2f0ab3f02ca3da20fd1b2fd96a4c74b991a4b98c544feed");
			strcat((char*)merkle_str, "c060aff8cd43ac565db9cc16d2c955f2950666392f37e650f933087ef0a3521f");
			strcat((char*)merkle_str, "0a0fcd4ac910e2a4d999dc1749b0fb151227f9814032cd7ff87c086c35a0c29d");
			strcat((char*)merkle_str, "6d63b050cb7259a40b95aa4735ae0405a967449b0e1189af1f4a798cf81a8733");
			strcat((char*)merkle_str, "11dc07d576f64a25a5a5dc3f0af7b07138070c1bb3461c9261795d31ca5f78d5");
			strcat((char*)merkle_str, "709a961120f2824e5e737284ecd9bc597c88abbd756d3c356d90ca248158049d");
			strcat((char*)merkle_str, "be55800cc10c078eecb039f0e4157ddef779c32baabfc113e0794437a22f16f2");
			strcat((char*)merkle_str, "72ea245bf08809e7645e9fcf8b02cf3497e2715bbb9214d1896aaa6069fd611e");
			strcat((char*)merkle_str, "f4456bc878b17beee82089ce413ec2362d51d3e01ba9071a420bd391a5421045");
			strcpy((char*)correct_str, "55ac8c4a3074053c9ceb102416cb6e8e78dfc84df3369150203744d638b90d1b");
			testMerkleHash(t_load, merkle_str, correct_str, 16, &logStr);
			sprintf(logMsg, "MERKLE 16 HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);

	POP_DOMAIN(handle);

	// DESTROY FUNCTIONAL TEST DOMAIN
	DOMAIN_DESTROY(handle);

	strcat(logResult, "********************************************************************************\n\n");
  printLog(logResult);

	// RETURN DIFF_REDUCE TO ITS ORIGINAL VALUE
	DIFF_REDUCE = temp_reduce;
  return;
}

__host__ void testMiningHash(WORKLOAD * t_load, BYTE * test_str, BYTE * correct_str, WORD diff_pow, char ** logStr){
	BYTE result_str[65];
	BYTE correct_hex[32];
	int hash_match;

	int * success_h;
	int * success_d;

	success_h = (int*)malloc(sizeof(int));
	cudaMalloc((void **) &success_d, sizeof(int));

	t_load->block_h[18] = diff_pow;
	getDifficulty(t_load);

	encodeWord(test_str, t_load->block_h, 160);
	cudaMemcpyAsync(t_load->block_d, t_load->block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, t_load->stream);

	calculateFirstState(t_load->basestate_h, t_load->block_h);
	cudaMemcpyToSymbolAsync(block_const, t_load->basestate_h, HASH_SIZE, 0, cudaMemcpyHostToDevice, t_load->stream);

	hashTestMiningKernel<<<1, 1, 0, t_load->stream>>>(t_load->block_d, t_load->hash_d, success_d);
	cudaMemcpyAsync(t_load->hash_h, t_load->hash_d, HASH_SIZE, cudaMemcpyDeviceToHost, t_load->stream);
	cudaMemcpyAsync(success_h, success_d, sizeof(int), cudaMemcpyDeviceToHost, t_load->stream);
	cudaDeviceSynchronize();

	// Compare results
	decodeWord(t_load->hash_h, result_str, 8);
	encodeHex(correct_str, correct_hex, 64);
	hash_match = strcmp((char*)result_str, (char*)correct_str);
	if(hash_match == 0){
		sprintf(*logStr, "SUCCESS, TARGET MET VALUE: %i", *success_h);
	}else{
		sprintf(*logStr, "FAILED, TARGET MET VALUE: %i\n \t\tEXPECTED: %s\n \t\tRECEIVED: %s", *success_h, correct_str, result_str);
	}

	free(success_h);
	cudaFree(success_d);
	return;
}

__host__ void testDoubleHash(WORKLOAD * t_load, BYTE * test_str, BYTE * correct_str, int test_size, char ** logStr){
	BYTE result_str[65];
	BYTE correct_hex[32];
	int hash_match;

	encodeWord(test_str, t_load->block_h, 160);
	cudaMemcpyAsync(t_load->block_d, t_load->block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, t_load->stream);

	HASH_DOUBLE_KERNEL(test_size, t_load->stream, t_load->block_d, t_load->hash_d);
	cudaMemcpyAsync(t_load->hash_h, t_load->hash_d, HASH_SIZE, cudaMemcpyDeviceToHost, t_load->stream);
	cudaDeviceSynchronize();

	// Compare results
	decodeWord(t_load->hash_h, result_str, 8);
	encodeHex(correct_str, correct_hex, 64);
	hash_match = strcmp((char*)result_str, (char*)correct_str);
	if(hash_match == 0){
		sprintf(*logStr, "SUCCESS");
	}else{
		sprintf(*logStr, "FAILED\n \t\tEXPECTED: %s\n \t\tRECEIVED: %s", correct_str, result_str);
	}
	return;
}

__host__ void testMerkleHash(WORKLOAD * t_load, BYTE * test_str, BYTE * correct_str, int test_size, char ** logStr){
	BYTE result_str[65];
	BYTE correct_hex[32];
	int hash_match;

	for(int i = 0; i < test_size; i++){
		encodeWord(&test_str[i*64], &t_load->buffer_h[i*8], 64);
	}
	cudaMemcpyAsync(t_load->buffer_d, t_load->buffer_h, HASH_SIZE*test_size, cudaMemcpyHostToDevice, t_load->stream);
	int tree_size = pow(2.0, ceil(log2((double)test_size)));

	// MERKLE WORKFLOW RESULTS
	merkleKernel_workflow<<<1, MERKLE_THREADS, 0, t_load->stream>>>(t_load->buffer_d, t_load->block_d, t_load->basestate_d, test_size, tree_size);
	cudaMemcpyAsync(t_load->hash_h, &t_load->block_d[9], HASH_SIZE, cudaMemcpyDeviceToHost, t_load->stream);
	cudaMemcpyAsync(t_load->block_h, t_load->block_d, BLOCK_SIZE, cudaMemcpyDeviceToHost, t_load->stream);
	cudaMemcpyAsync(t_load->basestate_h, t_load->basestate_d, HASH_SIZE, cudaMemcpyDeviceToHost, t_load->stream);
	cudaDeviceSynchronize();

	// COMPARE BASE STATE CALCULATION:
	printf("\n\nBLOCK: ");
	printWords(t_load->block_h, 20);
	printf("\nHASH: ");
	printWords(t_load->hash_h, 8);
	printf("\nBASE: ");
	printWords(t_load->basestate_h, 8);

	// Compare results
	decodeWord(t_load->hash_h, result_str, 8);
	encodeHex(correct_str, correct_hex, 64);
	hash_match = strcmp((char*)result_str, (char*)correct_str);
	if(hash_match == 0){
		sprintf(*logStr, "SUCCESS");
	}else{
		sprintf(*logStr, "FAILED\n \t\tEXPECTED: %s\n \t\tRECEIVED: %s", correct_str, result_str);
	}
	return;
}

// TEST FUNCTION FOR IMPROVED MINING KERNEL, WHICH IS ACCELERATED WITH THE USE OF
// PRECOMPUTED BLOCK HASHING CONSTANTS AND LOWER MEMORY USAGE
__host__ void miningBenchmarkTest(int num_workers){
  // INITIALIZE BENCHMARK VARIABLES
	WORKLOAD * t_load;
	t_load = (WORKLOAD*)malloc(sizeof(WORKLOAD));
	allocWorkload(0, t_load, 1);
  char logResult[1000];
	float worker_time, block_time, thread_time;

	// INITIALIZE BENCHMARK PROFILING DOMAIN
	char stream_name[50];
	sprintf(stream_name, "BENCHMARK STREAM");
	NAME_STREAM(t_load->stream, stream_name);
	#ifdef USE_NVTX
		DOMAIN_HANDLE handle;
	#else
		int handle = 0;
	#endif
	DOMAIN_CREATE(handle, "BENCHMARK TEST");
	PUSH_DOMAIN(handle, "BENCHMARK TEST", -2, 0, 0);

	// INITIALIZE CONSTANTS FOR USE IN THE MINING KERNEL
	int * iterations_h;
	int total_iterations = 0;
	int * iterations_d;
	iterations_h = (int*)malloc(sizeof(int));
	cudaMalloc((void **) &iterations_d, sizeof(int));

	WORD * time_h;
	cudaStream_t tStream;
	initTime(&tStream, &time_h);

	cudaEventRecord(t_load->t_start, t_load->stream);

	// SET TARGET DIFFICULTY
	t_load->block_h[18] = START_DIFF;
	getDifficulty(t_load);

	srand(time(0));
	for(int j = 0; j < BENCHMARK_LOOPS; j++){
		// CREATE RANDOM TEST BLOCK
	  for(int i = 0; i < 17; i++){
				t_load->block_h[i] = (((rand() % 255) & 0xFF) << 24) | (((rand() % 255) & 0xFF) << 16) | (((rand() % 255) & 0xFF) << 8) | ((rand() % 255) & 0xFF);
	  }
		t_load->block_h[0] = 0x01000000;
		t_load->block_h[17] = getTime();
		t_load->block_h[18] = START_DIFF;
		t_load->block_h[19] = 0x00000000;

		cudaMemcpyAsync(t_load->block_d, t_load->block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, t_load->stream);
		calculateFirstState(t_load->basestate_h, t_load->block_h);
		cudaMemcpyToSymbolAsync(block_const, t_load->basestate_h, HASH_SIZE, 0, cudaMemcpyHostToDevice, t_load->stream);
		cudaMemsetAsync(t_load->flag, 0, sizeof(int), t_load->stream);
		cudaMemsetAsync(iterations_d, 0, sizeof(int), t_load->stream);

		LAUNCH_BENCHMARK_TEST(NUM_WORKERS, t_load->id, t_load->stream, t_load->block_d, t_load->hash_d, t_load->hash_byte, t_load->flag, iterations_d);
		// UPDATE TIMING VARIABLE
		while(cudaStreamQuery(t_load->stream) != 0){
			updateTime(&tStream, time_h, handle);
		}

		cudaMemcpyAsync(iterations_h, iterations_d, sizeof(int), cudaMemcpyDeviceToHost, t_load->stream);
		cudaMemcpyAsync(t_load->block_h, t_load->block_d, BLOCK_SIZE, cudaMemcpyDeviceToHost, t_load->stream);
		cudaMemcpyAsync(t_load->hash_h, t_load->hash_d, HASH_SIZE, cudaMemcpyDeviceToHost, t_load->stream);
		total_iterations += *iterations_h;
		cudaStreamSynchronize(t_load->stream);
		printf("\n\nBLOCK SOLUTION found in %d iterations \n", *iterations_h);
		printWords(t_load->block_h, 20);
		printf("RESULT: ");
		printWords(t_load->hash_h, 8);
	}
	cudaEventRecord(t_load->t_stop, t_load->stream);
	cudaDeviceSynchronize();
	POP_DOMAIN(handle);
	freeTime(&tStream, &time_h);
	cudaEventElapsedTime(&t_load->t_result, t_load->t_start, t_load->t_stop);
	printf("TOTAL ITERATIONS PASSED: %i\n", total_iterations);
	printf("WORKER_BLOCKS: %i\n", WORKER_BLOCKS);
	printf("NUM THREADS: %i\n\n", NUM_THREADS);

	long long int all_iterations = 0;
	all_iterations = ((long long int)total_iterations)*((long long int)NUM_THREADS);
	printf("ALL ITERATIONS: %lld \n", all_iterations);

	worker_time = ((all_iterations)/(t_load->t_result*1000));
	block_time = worker_time/WORKER_BLOCKS;
	thread_time = (block_time*1000)/NUM_THREADS;

	sprintf(logResult, "\n****************************NEW MINING BENCHMARK ANALYSIS FOR %i WORKER CHAINS****************************\n\
	TOTAL TIME: %f\n\
	WORKER HASHRATE:\t %.3f MH/s\n\
	BLOCK HASHRATE:\t %.3f MH/s\n\
	THREAD HASHRATE:\t %.3f KH/s\n\
	**********************************************************************************************\n\
	", num_workers, t_load->t_result, worker_time, block_time, thread_time);
	printLog(logResult);
	DOMAIN_DESTROY(handle);

	free(iterations_h);
	cudaFree(iterations_d);
	freeWorkload(t_load);
  return;
}


// IMPROVED MINING KERNEL BENCHMARK TEST FUNCTION
// THIS TEST USES MULTIPLE COMPLEMENTARY KERNELS TO SIMULATE A REALISTIC WORKLOAD
// ADDITIONAL OUTPUTS USED FOR PYTHON GRAPHING SCRIPT
__host__ void miningBenchmarkTest_full(int num_workers){
  // INITIALIZE BENCHMARK VARIABLES
	WORKLOAD * t_load;
	t_load = (WORKLOAD*)malloc(sizeof(WORKLOAD));
	allocWorkload(0, t_load, 1);

	char out_location[30];

	if(MULTILEVEL == 1){
	  sprintf(out_location, "outputs/benchtest/results_%i_pchains", num_workers);
	}else{
	  sprintf(out_location, "outputs/benchtest/results_%i_chains", num_workers);
	}

	initializeBenchmarkOutfile(t_load->outFile, out_location, num_workers);

	// COMPLEMENT WORKLOAD
	WORKLOAD * c_load;
	WORKLOAD * c_workload;
	c_workload = (WORKLOAD*)malloc(sizeof(WORKLOAD)*(num_workers-1));
	for(int i = 0; i < (num_workers-1); i++){
		// ALLOCATE WORKLOAD INNER VARIABLES
		allocWorkload(i+1, &c_workload[i], WORKER_BUFFER_SIZE);
		POP_DOMAIN(w_handle[i]); // END WORKER ALLOCATION RANGE
	}

	char logResult[1000];
	float worker_time, block_time, thread_time;
	//float complement_time;

	// INITIALIZE BENCHMARK PROFILING DOMAIN
	char stream_name[50];
	sprintf(stream_name, "BENCHMARK STREAM");
	NAME_STREAM(t_load->stream, stream_name);
	#ifdef USE_NVTX
		DOMAIN_HANDLE handle;
	#else
		int handle = 0;
	#endif
	DOMAIN_CREATE(handle, "BENCHMARK TEST");
	PUSH_DOMAIN(handle, "BENCHMARK TEST", -2, 0, 0);

	// INITIALIZE CONSTANTS FOR USE IN THE MINING KERNEL
	int * iterations_h;
	int total_iterations = 0;
	int * iterations_d;
	iterations_h = (int*)malloc(sizeof(int));
	cudaMalloc((void **) &iterations_d, sizeof(int));

	// INITIALIZE CONSTANTS FOR USE IN THE COMPLEMENT MINING KERNEL
	int * c_iterations_h;
	int c_total_iterations = 0;
	int * c_iterations_d;
	int * c_iterations_ptr;

	c_iterations_h = (int*)malloc(sizeof(int));
	cudaMalloc((void **) &c_iterations_d, sizeof(int)*(num_workers-1));

	WORD * time_h;
	cudaStream_t tStream;
	initTime(&tStream, &time_h);

	// SET TARGET DIFFICULTY
	t_load->block_h[18] = START_DIFF;
	getDifficulty(t_load);

	printf("STARTING WORKLOAD SIMULATION\n");

	for(int i = 0; i < (num_workers-1); i++){
		c_load = &c_workload[i];
		c_iterations_ptr = &c_iterations_d[i];
		// SET HIGH COMPLEMENT TARGET DIFFICULTY
		c_load->block_h[18] = 0x1a00ffff;
		getDifficulty(c_load);
		cudaEventRecord(c_load->t_start, c_load->stream);

		srand(time(0));
		// SET COMPLEMENT WORKLOAD
		for(int i = 0; i < 17; i++){
				c_load->block_h[i] = (((rand() % 255) & 0xFF) << 24) | (((rand() % 255) & 0xFF) << 16) | (((rand() % 255) & 0xFF) << 8) | ((rand() % 255) & 0xFF);
		}
		c_load->block_h[0] = 0x01000000;
		c_load->block_h[17] = getTime();
		c_load->block_h[18] = 0x1a00ffff;
		c_load->block_h[19] = 0x00000000;

		// CHANGED FIXME SET FOR C LOAD
		cudaMemcpyAsync(c_load->block_d, c_load->block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, c_load->stream);
		calculateFirstState(c_load->basestate_h, c_load->block_h);
		cudaMemcpyToSymbolAsync(block_const, c_load->basestate_h, HASH_SIZE, HASH_SIZE*c_load->id, cudaMemcpyHostToDevice, c_load->stream);
		cudaMemsetAsync(c_load->flag, 0, sizeof(int), c_load->stream);
		cudaMemsetAsync(c_iterations_d, 0, sizeof(int), c_load->stream);

		LAUNCH_BENCHMARK_TEST(NUM_WORKERS, c_load->id, c_load->stream, c_load->block_d, c_load->hash_d, c_load->hash_byte, c_load->flag, c_iterations_ptr);
	}

	cudaEventRecord(t_load->t_start, t_load->stream);
	printf("************************\nSTARTING BENCHMARK LOOPS\n************************\n");
	for(int j = 0; j < BENCHMARK_LOOPS; j++){
		// CREATE RANDOM TEST BLOCK
	  for(int i = 0; i < 17; i++){
				t_load->block_h[i] = (((rand() % 255) & 0xFF) << 24) | (((rand() % 255) & 0xFF) << 16) | (((rand() % 255) & 0xFF) << 8) | ((rand() % 255) & 0xFF);
	  }
		t_load->block_h[0] = 0x01000000;
		t_load->block_h[17] = getTime();
		t_load->block_h[18] = 0x1d00ffff;
		t_load->block_h[19] = 0x00000000;

		cudaMemcpyAsync(t_load->block_d, t_load->block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, t_load->stream);
		calculateFirstState(t_load->basestate_h, t_load->block_h);
		cudaMemcpyToSymbolAsync(block_const, t_load->basestate_h, HASH_SIZE, 0, cudaMemcpyHostToDevice, t_load->stream);
		cudaMemsetAsync(t_load->flag, 0, sizeof(int), t_load->stream);
		cudaMemsetAsync(iterations_d, 0, sizeof(int), t_load->stream);

		LAUNCH_BENCHMARK_TEST(NUM_WORKERS, t_load->id, t_load->stream, t_load->block_d, t_load->hash_d, t_load->hash_byte, t_load->flag, iterations_d);
		// UPDATE TIMING VARIABLE
		while(cudaStreamQuery(t_load->stream) != 0){
			updateTime(&tStream, time_h, handle);
		}

		cudaMemcpyAsync(iterations_h, iterations_d, sizeof(int), cudaMemcpyDeviceToHost, t_load->stream);
		cudaMemcpyAsync(t_load->block_h, t_load->block_d, BLOCK_SIZE, cudaMemcpyDeviceToHost, t_load->stream);
		cudaMemcpyAsync(t_load->hash_h, t_load->hash_d, HASH_SIZE, cudaMemcpyDeviceToHost, t_load->stream);
		total_iterations += *iterations_h;
		cudaStreamSynchronize(t_load->stream);
		printf("\n\nBLOCK SOLUTION found in %d iterations \n", *iterations_h);
		printWords(t_load->block_h, 20);
		printf("RESULT: ");
		printWords(t_load->hash_h, 8);
	}
	cudaEventRecord(t_load->t_stop, t_load->stream);
	printf("Finished Testing, waiting for GPU to finish processing\n");

	for(int i = 0; i < (num_workers-1); i++){
		c_load = &c_workload[i];
		cudaMemcpyAsync(c_load->flag, t_load->flag, sizeof(int), cudaMemcpyDeviceToDevice, t_load->stream);
		cudaEventRecord(c_load->t_stop, c_load->stream);
	}

	cudaDeviceSynchronize();

	for(int i = 0; i < (num_workers-1); i++){
		c_iterations_ptr = &c_iterations_d[i];
		cudaMemcpyAsync(c_iterations_h, c_iterations_ptr, sizeof(int), cudaMemcpyDeviceToHost, c_load->stream);
		c_total_iterations += *c_iterations_h;
	}

	cudaDeviceSynchronize();
	printf("Processing finished, compiling results\n");
	POP_DOMAIN(handle);
	freeTime(&tStream, &time_h);
	cudaEventElapsedTime(&t_load->t_result, t_load->t_start, t_load->t_stop);

	for(int i = 0; i < (num_workers-1); i++){
		c_load = &c_workload[i];
		cudaEventElapsedTime(&c_load->t_result, c_load->t_start, c_load->t_stop);

		// CHANGED ADDED 11-21
		printf("Worker %i Elapsed Time: %f \n", c_load->id, c_load->t_result);
		//complement_time += c_load->t_result;
	}

	// These may be useful for future graphs
	//printf("Complement Iterations: %i \n", c_total_iterations);
	//long long int all_c_iterations = 0;
	//all_c_iterations = ((long long int)c_total_iterations)*((long long int)NUM_THREADS);
	//complement_time = ((all_c_iterations)/(complement_time*1000));
	//printf("COMPLEMENT HASHRATE: \t %.3f MH/s \n", complement_time);

	printf("TOTAL ITERATIONS PASSED: %i\n", total_iterations);
	printf("WORKER_BLOCKS: %i\n", WORKER_BLOCKS);
	printf("NUM THREADS: %i\n\n", NUM_THREADS);

	long long int all_iterations = 0;
	all_iterations = ((long long int)total_iterations)*((long long int)NUM_THREADS);
	printf("ALL ITERATIONS: %lld \n", all_iterations);

	worker_time = ((all_iterations)/(t_load->t_result*1000));
	block_time = worker_time/WORKER_BLOCKS;
	thread_time = (block_time*1000)/NUM_THREADS;

	sprintf(logResult, "\n****************************MINING BENCHMARK ANALYSIS FOR %i WORKER CHAINS****************************\n\
	NUM BLOCKS: %i\n\
	NUM THREADS: %i\n\
	TOTAL ITERATIONS: %i\n\
	TOTAL TIME: %f\n\n\
	WORKER HASHRATE:\t %.3f MH/s\n\
	BLOCK HASHRATE:\t %.3f MH/s\n\
	THREAD HASHRATE:\t %.3f KH/s\n\
	**********************************************************************************************\n\
	", num_workers, WORKER_BLOCKS, NUM_THREADS, total_iterations, t_load->t_result, worker_time, block_time, thread_time);

	printf("PRINTING TO LOG FILE\n");
	printLog(logResult);
	printf("FINISHED PRINTING TO LOG FILE\n");

	// PRINT PRIMARY DATA TO A FILE
	if(t_load->inFile = fopen(t_load->outFile, "w")){
		printf("OPENED FILE %s\n", t_load->outFile);
		fprintf(t_load->inFile, "%s\n", logResult);
		printf("PRINTED TO FILE %s\n", t_load->outFile);
		fclose(t_load->inFile);
		printf("CLOSED FILE %s\n", t_load->outFile);

	}
	else{
		printf("WORKER %i OUTPUT FILE: %s NOT FOUND", num_workers, t_load->outFile);
	}

	printf("FINISHED PRINTING TO OUTPUT FILE ")

	DOMAIN_DESTROY(handle);

	printf("FINISHED DOMAIN DESTROY");

	// CHANGED FREE COMPLEMENT VARIABLES

	free(c_iterations_h);
	cudaFree(c_iterations_d);
	for(int i = 0; i < (num_workers-1); i++){
		// FREE WORKLOAD INNER VARIABLES
		freeWorkload( &c_workload[i]);
	}
	free(c_workload);

	free(iterations_h);
	cudaFree(iterations_d);
	freeWorkload(t_load);
	free(t_load);
  return;
}





__host__ void colorTest(int num_colors, int num_palettes){
	START_PROFILE;
	// INITIALIZE PROFILING DOMAINS
	char range_name[50];
	#ifdef USE_NVTX
		DOMAIN_HANDLE test_handle;
	#endif
	DOMAIN_CREATE(test_handle, "COLOR PALETTE TEST");

	for(int i = 0; i < num_palettes; i++){
		sprintf(range_name, "PALETTE %i", i);
		PUSH_DOMAIN(test_handle, range_name, -2, 0, 0);
		for(int j = 0; j < num_colors; j++){
			sprintf(range_name, "COLOR %i", j);
			PUSH_DOMAIN(test_handle, range_name, -2, i, j);
			POP_DOMAIN(test_handle);
		}
		POP_DOMAIN(test_handle);
	}
	DOMAIN_DESTROY(test_handle);
	unsigned int color = 0x80;
	for(int i = 0; i < 12; i++){
		printf("0xff%06x, ", color);
		color *= 2 ;
	}
	STOP_PROFILE;
}


/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************  _______________________________________________________________________________________________________________________________________________  ************/
/************  |    __  __   ______   __  __    ____    _____   __     __    ______   _    _   _   _    _____   _______   _____    ____    _   _    _____    |  ************/
/************  |   |  \/  | |  ____| |  \/  |  / __ \  |  __ \  \ \   / /   |  ____| | |  | | | \ | |  / ____| |__   __| |_   _|  / __ \  | \ | |  / ____|   |  ************/
/************  |   | \  / | | |__    | \  / | | |  | | | |__) |  \ \_/ /    | |__    | |  | | |  \| | | |         | |      | |   | |  | | |  \| | | (___     |  ************/
/************  |   | |\/| | |  __|   | |\/| | | |  | | |  _  /    \   /     |  __|   | |  | | | . ` | | |         | |      | |   | |  | | | . ` |  \___ \    |  ************/
/************  |   | |  | | | |____  | |  | | | |__| | | | \ \     | |      | |      | |__| | | |\  | | |____     | |     _| |_  | |__| | | |\  |  ____) |   |  ************/
/************  |   |_|  |_| |______| |_|  |_|  \____/  |_|  \_\    |_|      |_|       \____/  |_| \_|  \_____|    |_|    |_____|  \____/  |_| \_| |_____/    |  ************/
/************  |_____________________________________________________________________________________________________________________________________________|  ************/
/************                                                                                                                                                   ************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/************************************************************************MEMORY ALLOCATION FUNCTIONS************************************************************************/
__host__ void allocWorkload(int id, WORKLOAD * load, int buffer_size){
	// INITIALIZE BASIC VARIABLES
	load->id = id;
	load->readErr = 0;
	load->blocks = 0;

	load->diff_level = 1;
	load->alive = 1;

	// INIT TIMING TO ZERO
	load->t_result = 0.0;
	load->t_diff = 0.0;

	cudaStreamCreate(&load->stream);

	cudaEventCreate(&load->t_start);
	cudaEventCreate(&load->t_stop);
	cudaEventCreate(&load->t_diff_start);
	cudaEventCreate(&load->t_diff_stop);

	// ALLOCATE TARGET VARIABLE
	load->target = (WORD*)malloc(HASH_SIZE);

	// Allocate Mining Flag
	cudaMalloc((void **) &load->flag, sizeof(int));

	// ALLOCATE BYTE HASH FOR MINING KERNEL EFFICIENCY
	cudaMalloc((void **) &load->hash_byte, HASH_SIZE_BYTE);

	// MERKEL HASHING VARIABLE WORDS
	load->block_h = (WORD *)malloc(BLOCK_SIZE);
	cudaMalloc((void **) &load->block_d, BLOCK_SIZE);

	// MERKEL HASHING VARIABLES
	load->buffer_h = (WORD*)malloc(HASH_SIZE*(buffer_size));
	cudaMalloc((void **) &load->buffer_d, HASH_SIZE*(buffer_size));

	// MERKEL HASHING VARIABLE WORDS
	load->hash_h = (WORD*)malloc(HASH_SIZE);
	cudaMalloc((void **) &load->hash_d, HASH_SIZE);

	// CONSTANT PARTIAL HASH INPUT FOR MINER
	load->basestate_h = (WORD*)malloc(HASH_SIZE);
	cudaMalloc((void **) &load->basestate_d, HASH_SIZE);

	// MAXIMUM SIZE FOR THE MERKLE BUFFER
	load->buff_size = buffer_size;

	// CURRENT NUMBER OF BLOCKS IN THE BUFFER
	load->buff_blocks = 0;
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*************************************************************************MEMORY FREEING FUNCTIONS**************************************************************************/
__host__ void freeWorkload(WORKLOAD * load){
	// DESTROY CUDA STREAMS AND EVENTS
	cudaStreamDestroy(load->stream);

	cudaEventDestroy(load->t_start);
	cudaEventDestroy(load->t_stop);
	cudaEventDestroy(load->t_diff_start);
	cudaEventDestroy(load->t_diff_stop);

	// FREE WORKING MEMORY
	free(load->target);
	cudaFree(load->flag);

	cudaFree(load->hash_byte);

	free(load->block_h);
	cudaFree(load->block_d);

	free(load->buffer_h);
	cudaFree(load->buffer_d);

	free(load->hash_h);
	cudaFree(load->hash_d);

	free(load->basestate_h);
	cudaFree(load->basestate_d);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/************************************************************************CUDA MANAGEMENT FUNCTIONS**************************************************************************/
__host__ void createCudaVars(cudaEvent_t * timing1, cudaEvent_t * timing2, cudaStream_t * stream){
  cudaEventCreate(timing1);
  cudaEventCreate(timing2);

	// TEST EVENT FLAGS (FIXES TIME UPDATE BUG, BUT NO TIMING STATISTICS AVAILABLE )
//	cudaEventCreateWithFlags(timing1, cudaEventDisableTiming);
//	cudaEventCreateWithFlags(timing2, cudaEventDisableTiming);

//  cudaStreamCreate(stream);
	cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking); //Create the stream such that it may run concurrently with the default stream, lower priority than timing stream
}
__host__ void destroyCudaVars(cudaEvent_t * timing1, cudaEvent_t * timing2, cudaStream_t * stream){
  cudaEventDestroy(*timing1);
  cudaEventDestroy(*timing2);
  cudaStreamDestroy(*stream);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/************************************************************************TIME MANAGEMENT FUNCTIONS**************************************************************************/
// CREATE AND FREE FUNCTIONS FOR UPDATING THE DEVICE TIME
__host__ void initTime(cudaStream_t * tStream, WORD ** time_h){
  *time_h = (WORD *)malloc(sizeof(WORD));
  cudaStreamCreateWithPriority(tStream, cudaStreamNonBlocking, -1);
  updateTime(tStream, *time_h, 0);
}
__host__ void freeTime(cudaStream_t * tStream, WORD ** time_h){
  free(*time_h);
  cudaStreamDestroy(*tStream);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/************************************************************************TIME MANAGEMENT FUNCTIONS**************************************************************************/



/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/***************  __________________________________________________________________________________________________________________________________________  **************/
/***************  |    __  __   _____   _   _   _____   _   _    _____     ______   _    _   _   _    _____   _______   _____    ____    _   _    _____    |  **************/
/***************  |   |  \/  | |_   _| | \ | | |_   _| | \ | |  / ____|   |  ____| | |  | | | \ | |  / ____| |__   __| |_   _|  / __ \  | \ | |  / ____|   |  **************/
/***************  |   | \  / |   | |   |  \| |   | |   |  \| | | |  __    | |__    | |  | | |  \| | | |         | |      | |   | |  | | |  \| | | (___     |  **************/
/***************  |   | |\/| |   | |   | . ` |   | |   | . ` | | | |_ |   |  __|   | |  | | | . ` | | |         | |      | |   | |  | | | . ` |  \___ \    |  **************/
/***************  |   | |  | |  _| |_  | |\  |  _| |_  | |\  | | |__| |   | |      | |__| | | |\  | | |____     | |     _| |_  | |__| | | |\  |  ____) |   |  **************/
/***************  |   |_|  |_| |_____| |_| \_| |_____| |_| \_|  \_____|   |_|       \____/  |_| \_|  \_____|    |_|    |_____|  \____/  |_| \_| |_____/    |  **************/
/***************  |________________________________________________________________________________________________________________________________________|  **************/
/***************                                                                                                                                              **************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/**********************************************************************BLOCK INITIALIZATION FUNCTIONS***********************************************************************/
__host__ void initializeBlockHeader(BYTE * block, BYTE * version, BYTE * prevBlock, BYTE * merkleRoot, BYTE * time_b, BYTE * target, BYTE * nonce){
  for(int i = 0; i < 4; i++){
    block[i] = version[i];
  }
  for(int i = 0; i < 32; i++){
    block[i + 4] = prevBlock[i];
  }
  for(int i = 0; i < 32; i++){
    block[i + 36] = merkleRoot[i];
  }
  for(int i = 0; i < 4; i++){
    block[i + 68] = time_b[i];
  }
  for(int i = 0; i < 4; i++){
    block[i + 72] = target[i];
  }
  for(int i = 0; i < 4; i++){
    block[i + 76] = nonce[i];
  }
  return;
}

__host__ void initializeBlockHeader(WORD * block, WORD version, WORD * prevBlock, WORD * merkleRoot, WORD time_b, WORD target, WORD nonce){
	block[0] = version;
  for(int i = 0; i < 8; i++){
    block[i + 1] = prevBlock[i];
  }
  for(int i = 0; i < 8; i++){
    block[i + 9] = merkleRoot[i];
  }
	block[17] = time_b;
	block[18] = target;
	block[19] = nonce;
  return;
}

__host__ void initializeWorkerBlock(WORKLOAD * load){
  WORD prevBlock[8], word_time;             // Previous Block and time vars
  WORD version = 0x01000000;      // Default Version
	WORD diff_bits = START_DIFF;
	WORD nonce = 0x00000000;		// Starting Nonce
  for(int i = 0; i < 8; i++){
    prevBlock[i] = 0x00000000;
  }
  word_time = getTime();
  initializeBlockHeader(load->block_h, version, prevBlock, load->buffer_h, word_time, diff_bits, nonce);
}

__host__ void initializeParentBlock(WORD * pBlock_h){
	WORD prevBlock[8], hash[8], word_time;             // Previous Block and time vars
  WORD version = 0x01000000;      // Default Version
	WORD diff_bits = START_DIFF;
//	WORD diff_bits = 0x1c00ffff; // Starting Difficulty
	WORD nonce = 0x00000000;		// Starting Nonce
  for(int i = 0; i < 8; i++){
		hash[i] = 0x00000000;
    prevBlock[i] = 0x00000000;
  }
  word_time = getTime();
  initializeBlockHeader(pBlock_h, version, prevBlock, hash, word_time, diff_bits, nonce);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/**************************************************************************MINING UPDATE FUNCTIONS**************************************************************************/
// UPDATE WORKER BLOCK WITH THE PREVIOUS HASH VALUE AND A NEW HASH FROM THE INPUT FILE
// FIXME DEPRECATED. Replaced with updateBlock_load, kept for now as backup
__host__ int updateBlock(FILE * inFile, WORD * block_h, WORD * hash_h, WORD * buffer_h){
  int errEOF = 0;
	for(int i = 0; i < 8; i++){
    block_h[i + 1] = hash_h[i];
  }
  errEOF = readNextHash(inFile, buffer_h);
  for(int i = 0; i < 8; i++){
    block_h[i + 9] = buffer_h[i];
  }
	block_h[17] = getTime();
  return errEOF;
}

// UPDATE WORKER BLOCK WITH THE PREVIOUS HASH VALUE AND A NEW HASH FROM THE INPUT FILE
__host__ int updateBlock_load(WORKLOAD * load){
	WORD * buff_ptr;
	for(int i = 0; i < 8; i++){
    load->block_h[i + 1] = load->hash_h[i];
  }
	for(; load->buff_blocks < load->buff_size; load->buff_blocks++){
		buff_ptr = &(load->buffer_h[8*load->buff_blocks]);
		load->readErr = readNextHash(load->inFile, buff_ptr);
		if(load->readErr == 1){
			break;
		}
	}
  //load->readErr= readNextHash(load->inFile, load->buffer_h);
  for(int i = 0; i < 8; i++){
    load->block_h[i + 9] = load->buffer_h[i];
  }
	load->block_h[17] = getTime();
  return load->readErr;
}

// UPDATE BLOCK PREVIOUS HASH TO THE GIVEN HASH
__host__ void updateParentHash(WORD * block_h, WORD * hash_h){
  for(int i = 0; i < 8; i++){
    block_h[i + 1] = hash_h[i];
  }
	block_h[17] = getTime();
  return;
}

// UPDATE DIFFICULTY BY DECREASING THE LARGEST TARGET BYTE BY 1
// NEW UPDATE INCLUDES VARIABLES FOR DIFFICULTY SCALING AND PRESET DIFFICULTY BITS
__host__ void updateDifficulty(WORD * block_h, int diff_level){
	char debugOut[100];
  int new_pow = 0x00;
  int new_diff = 0x000000;
  new_pow = START_POW -(((diff_level*DIFF_SCALING)+DIFFICULTY_BITS)/0xFF);
  new_diff = 0x00FFFF - ((((diff_level*DIFF_SCALING)+DIFFICULTY_BITS)%0xFF)<<8);
	sprintf(debugOut, "UPDATE DIFFICULTY: START: 0x%02x%06x  | NEW: 0x%02x%06x \n ", START_POW, START_BITS, new_pow, new_diff);
	printDebug((const char*)debugOut);
	block_h[18] = (new_pow << 24) | new_diff;
}

// UPDATE THE CURRENT TIME ON DEVICE IN CASE OF NONCE OVERFLOW
__host__ void updateTime(cudaStream_t * tStream, WORD * time_h, DOMAIN_HANDLE prof_handle){
	WORD old_time = *time_h;
	*time_h = time(0);
	if(old_time != *time_h){ // Time has changed, update device memory
//		cudaError_t time_err;
		#ifdef USE_NVTX
		printf("UPDATING...");
		PUSH_DOMAIN(prof_handle, "T_UPDATE", -1, 1, 0);
		cudaMemcpyToSymbolAsync(time_const, time_h, sizeof(WORD), 0, cudaMemcpyHostToDevice, *tStream);
	//	cudaMemcpyToSymbol(time_const, time_h, sizeof(WORD), 0, cudaMemcpyHostToDevice);
		cudaStreamSynchronize(*tStream);
		printf("HOST TIME UPDATED: %08x\n", *time_h);

		POP_DOMAIN(prof_handle);
		#else
//		printf("UPDATING...");
		cudaMemcpyToSymbolAsync(time_const, time_h, sizeof(WORD), 0, cudaMemcpyHostToDevice, *tStream);
//		cudaMemcpyToSymbol(time_const, time_h, sizeof(WORD), 0, cudaMemcpyHostToDevice);
//		printf("\nTIME STATUS: [CODE: %i]:(%s: %s) \n", time_err, cudaGetErrorName(time_err), cudaGetErrorString(time_err));
//		time_err = cudaStreamQuery(*tStream);
//		printf("\nSTREAM STATUS: [CODE: %i]:(%s: %s) \n", time_err, cudaGetErrorName(time_err), cudaGetErrorString(time_err));

//		cudaStreamSynchronize(*tStream);
//		printf("HOST TIME UPDATED: %08x\n", *time_h);
		#endif
	}
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/**************************************************************************MINING GETTER FUNCTIONS**************************************************************************/
// GET THE CURRENT TIME IN SECONDS SINCE THE LAST EPOCH (1970)
__host__ WORD getTime(void){
  return time(0);
}

__host__ void getDifficulty(WORKLOAD * load){
	char logOut[300];
	char debugOut[300];
	char chain_id[20];

	BYTE target_bytes[32];
	BYTE block_target[4];
	block_target[0] = (load->block_h[18] >> 24) & 0x000000FF;
	block_target[1] = (load->block_h[18] >> 16) & 0x000000FF;
	block_target[2] = (load->block_h[18] >> 8) & 0x000000FF;
	block_target[3] = (load->block_h[18]) & 0x000000FF;

	// FIXME CREATE VERSION WITH WORD INPUT AND NO BYTE OUTPUT
  calculateMiningTarget(block_target, target_bytes, load->target);
  load->difficulty = calculateDifficulty(block_target);

	// USE OLD TARGET CALCULATION FOR PRINTABLE BYTES
	load->target_len = calculateTarget(block_target, target_bytes);

	cudaMemcpyToSymbolAsync(target_const, load->target, HASH_SIZE, HASH_SIZE*load->id, cudaMemcpyHostToDevice, load->stream);

	BYTE target_str[100];
	decodeHex(target_bytes, target_str, load->target_len);
	if(load->id == 0){
		sprintf(chain_id, "PARENT");
	}else{
		sprintf(chain_id, "WORKER %i", load->id);
	}
	sprintf(debugOut, "BLOCK TARGET: %08x , LENGTH: %i\n        TARGET VALUE: %s\n", load->block_h[18], load->target_len, (char*)target_str);
	sprintf(logOut, "NEW DIFFICULTY %s: %lf", chain_id, load->difficulty);
	printLog((const char*)logOut);
	printDebug((const char*)debugOut);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/************************************************************************MINING CALCULATION FUNCTIONS***********************************************************************/
// GET THE MINING DIFFICULTY FROM THE GIVEN BITS, RETURN DIFFICULTY AS A DOUBLE
__host__ double calculateDifficulty(BYTE * bits){
  // FIRST BYTE FOR LEADING ZEROS, REST FOR TARGET VALUE
  int start_pow = 0x1d;
	//int start_pow = START_POW; 		// FOR USE IF USING A CUSTOM TARGET FOR DIFFICULTY LEVEL 1
  int start_diff = 0x00ffff;
  int bit_pow = bits[0];
  int bit_diff = (((unsigned int)bits[1]) << 16) + (((unsigned int)bits[2]) << 8) + ((unsigned int)bits[3]);
  float diff_coef = log((float)start_diff / (float)bit_diff) + (start_pow - bit_pow)*log(256);
  double difficulty = exp(diff_coef);
  return difficulty;
}
// CALCULATE NEW TARGET VALUE, RETURN TARGET LENGTH
__host__ int calculateTarget(BYTE * bits, BYTE * target){
  // FIRST BYTE DETERMINES LEADING ZEROS
  // DIFFICULTY MODIFIED TO REDUCE INITIAL COMPUTATION TIME
  int padding = (32 - bits[0]);
  int length = (padding + 3);
  for(int i = 0; i < 32; i++){
    if(i < padding){
      target[i] = 0x00;
    }else if(i < padding + 3){
      target[i] = bits[i - padding + 1];
    }else{
      target[i] = 0x00;
    }
  }
  return length;
}

// CALCULATE NEW TARGET VALUE IN WORDS, RETURN TARGET LENGTH IN NUMBER OF WORDS
// REVERSE USUAL BYTE ORDER, 0xFF PADDING INSTEAD OF 0x00
__host__ int calculateMiningTarget(BYTE * bits, BYTE * target_bytes, WORD * target){
  // FIRST BYTE DETERMINES TRAILING ZEROS
  // DIFFICULTY MODIFIED TO REDUCE INITIAL COMPUTATION TIME
  int padding = (32 - bits[0]);
  int length = (padding + 3);
	BYTE reverse_bits[3];
	reverse_bits[0] = bits[3];
	reverse_bits[1] = bits[2];
	reverse_bits[2] = bits[1];
	// COMPUTE BYTES FIRST
	for(int i = 0; i < 32; i++){
		if(i < 32-length){
			target_bytes[i] = 0xFF;
		}else if(i < 32 - padding){
			target_bytes[i] = reverse_bits[i - (29 - padding)];
		}else{
			target_bytes[i] = 0x00;
		}
	}
	for(int i = 0; i< 8; i++){
		target[i] = (target_bytes[i*4] << 24) | (target_bytes[i*4+1] << 16) | (target_bytes[i*4+2] << 8) | (target_bytes[i*4+3]);
	}
  return length;
}

// FULL MESSAGE SCHEDULE COMPUTATION USING FIRST 16 WORDS
// [NOT RECOMMENDED FOR USE DUE TO HIGH MEMORY USAGE (2KB)]
__host__ void calculateSchedule(WORD m[]){
	m[16] = SIG1(m[14]) + m[9] + SIG0(m[1]) + m[0];
	m[17] = SIG1(m[15]) + m[10] + SIG0(m[2]) + m[1];
	m[18] = SIG1(m[16]) + m[11] + SIG0(m[3]) + m[2];
	m[19] = SIG1(m[17]) + m[12] + SIG0(m[4]) + m[3];
	m[20] = SIG1(m[18]) + m[13] + SIG0(m[5]) + m[4];
	m[21] = SIG1(m[19]) + m[14] + SIG0(m[6]) + m[5];
	m[22] = SIG1(m[20]) + m[15] + SIG0(m[7]) + m[6];
	m[23] = SIG1(m[21]) + m[16] + SIG0(m[8]) + m[7];
	m[24] = SIG1(m[22]) + m[17] + SIG0(m[9]) + m[8];
	m[25] = SIG1(m[23]) + m[18] + SIG0(m[10]) + m[9];
	m[26] = SIG1(m[24]) + m[19] + SIG0(m[11]) + m[10];
	m[27] = SIG1(m[25]) + m[20] + SIG0(m[12]) + m[11];
	m[28] = SIG1(m[26]) + m[21] + SIG0(m[13]) + m[12];
	m[29] = SIG1(m[27]) + m[22] + SIG0(m[14]) + m[13];
	m[30] = SIG1(m[28]) + m[23] + SIG0(m[15]) + m[14];
	m[31] = SIG1(m[29]) + m[24] + SIG0(m[16]) + m[15];
	m[32] = SIG1(m[30]) + m[25] + SIG0(m[17]) + m[16];
	m[33] = SIG1(m[31]) + m[26] + SIG0(m[18]) + m[17];
	m[34] = SIG1(m[32]) + m[27] + SIG0(m[19]) + m[18];
	m[35] = SIG1(m[33]) + m[28] + SIG0(m[20]) + m[19];
	m[36] = SIG1(m[34]) + m[29] + SIG0(m[21]) + m[20];
	m[37] = SIG1(m[35]) + m[30] + SIG0(m[22]) + m[21];
	m[38] = SIG1(m[36]) + m[31] + SIG0(m[23]) + m[22];
	m[39] = SIG1(m[37]) + m[32] + SIG0(m[24]) + m[23];
	m[40] = SIG1(m[38]) + m[33] + SIG0(m[25]) + m[24];
	m[41] = SIG1(m[39]) + m[34] + SIG0(m[26]) + m[25];
	m[42] = SIG1(m[40]) + m[35] + SIG0(m[27]) + m[26];
	m[43] = SIG1(m[41]) + m[36] + SIG0(m[28]) + m[27];
	m[44] = SIG1(m[42]) + m[37] + SIG0(m[29]) + m[28];
	m[45] = SIG1(m[43]) + m[38] + SIG0(m[30]) + m[29];
	m[46] = SIG1(m[44]) + m[39] + SIG0(m[31]) + m[30];
	m[47] = SIG1(m[45]) + m[40] + SIG0(m[32]) + m[31];
	m[48] = SIG1(m[46]) + m[41] + SIG0(m[33]) + m[32];
	m[49] = SIG1(m[47]) + m[42] + SIG0(m[34]) + m[33];
	m[50] = SIG1(m[48]) + m[43] + SIG0(m[35]) + m[34];
	m[51] = SIG1(m[49]) + m[44] + SIG0(m[36]) + m[35];
	m[52] = SIG1(m[50]) + m[45] + SIG0(m[37]) + m[36];
	m[53] = SIG1(m[51]) + m[46] + SIG0(m[38]) + m[37];
	m[54] = SIG1(m[52]) + m[47] + SIG0(m[39]) + m[38];
	m[55] = SIG1(m[53]) + m[48] + SIG0(m[40]) + m[39];
	m[56] = SIG1(m[54]) + m[49] + SIG0(m[41]) + m[40];
	m[57] = SIG1(m[55]) + m[50] + SIG0(m[42]) + m[41];
	m[58] = SIG1(m[56]) + m[51] + SIG0(m[43]) + m[42];
	m[59] = SIG1(m[57]) + m[52] + SIG0(m[44]) + m[43];
	m[60] = SIG1(m[58]) + m[53] + SIG0(m[45]) + m[44];
	m[61] = SIG1(m[59]) + m[54] + SIG0(m[46]) + m[45];
	m[62] = SIG1(m[60]) + m[55] + SIG0(m[47]) + m[46];
	m[63] = SIG1(m[61]) + m[56] + SIG0(m[48]) + m[47];
	return;
}

// HOST FUNCTION FOR PRECOMPUTING THE FIRST STATE CONSTANT
// (FASTER ALTERNATIVE TO SENDING BLOCK OR SCHEDULE FOR SPEEDUP)
__host__ void calculateFirstState(WORD state[], WORD base[]){
	WORD a, b, c, d, e, f, g, h, i, t1, t2;
	WORD m[64];
	for(i = 0; i < 16; i++){
		m[i] = base[i];
	}
	calculateSchedule(m);

	a = 0x6a09e667;
	b = 0xbb67ae85;
	c = 0x3c6ef372;
	d = 0xa54ff53a;
	e = 0x510e527f;
	f = 0x9b05688c;
	g = 0x1f83d9ab;
	h = 0x5be0cd19;

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k_host[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	state[0] = a + 0x6a09e667;
	state[1] = b + 0xbb67ae85;
	state[2] = c + 0x3c6ef372;
	state[3] = d + 0xa54ff53a;
	state[4] = e + 0x510e527f;
	state[5] = f + 0x9b05688c;
	state[6] = g + 0x1f83d9ab;
	state[7] = h + 0x5be0cd19;
}
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

 /***************************************************************************************************************************************************************************/
 /***************************************************************************************************************************************************************************/
 /***************  __________________________________________________________________________________________________________________________________________   *************/
 /***************  |    _  __  ______   _____    _   _   ______   _          ______   _    _   _   _    _____   _______   _____    ____    _   _    _____    |  *************/
 /***************  |   | |/ / |  ____| |  __ \  | \ | | |  ____| | |        |  ____| | |  | | | \ | |  / ____| |__   __| |_   _|  / __ \  | \ | |  / ____|   |  *************/
 /***************  |   | ' /  | |__    | |__) | |  \| | | |__    | |        | |__    | |  | | |  \| | | |         | |      | |   | |  | | |  \| | | (___     |  *************/
 /***************  |   |  <   |  __|   |  _  /  | . ` | |  __|   | |        |  __|   | |  | | | . ` | | |         | |      | |   | |  | | | . ` |  \___ \    |  *************/
 /***************  |   | . \  | |____  | | \ \  | |\  | | |____  | |____    | |      | |__| | | |\  | | |____     | |     _| |_  | |__| | | |\  |  ____) |   |  *************/
 /***************  |   |_|\_\ |______| |_|  \_\ |_| \_| |______| |______|   |_|       \____/  |_| \_|  \_____|    |_|    |_____|  \____/  |_| \_| |_____/    |  *************/
 /***************  |_________________________________________________________________________________________________________________________________________|  *************/
 /***************                                                                                                                                               *************/
 /***************************************************************************************************************************************************************************/
 /***************************************************************************************************************************************************************************/

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/**************************************************************************INPUT GENERATION KERNEL**************************************************************************/
__host__ void launchGenHash(WORD ** hash_hf, WORD ** hash_df, WORD ** seed_h, WORD ** seed_d, size_t size_hash){
  cudaMemcpy(*seed_d, *seed_h, HASH_SIZE, cudaMemcpyHostToDevice);
  genHashKernel<<<MAX_BLOCKS, NUM_THREADS>>>(*hash_df, *seed_d, MAX_BLOCKS);
  cudaDeviceSynchronize();
  cudaMemcpy(*hash_hf, *hash_df, size_hash, cudaMemcpyDeviceToHost);
}
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*****************************************************************************MERKLE TREE KERNEL****************************************************************************/
// FIXME DEPRECATED. No longer used, kept as backup/reference
__host__ void launchMerkle(WORKLOAD * load){
  cudaMemcpyAsync(load->buffer_d, load->buffer_h, HASH_SIZE*load->buff_size, cudaMemcpyHostToDevice, load->stream);
	cudaMemcpyAsync(load->block_d, load->block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, load->stream);  // COPY OVER CURRENT BLOCK
	int tree_size = pow(2.0, ceil(log2((double)load->buff_blocks)));
	merkleKernel<<<1, MERKLE_THREADS, 0, load->stream>>>(load->buffer_d, &load->block_d[9], load->buff_blocks,  tree_size);
	cudaMemcpyAsync(load->block_h, load->block_d, BLOCK_SIZE, cudaMemcpyDeviceToHost, load->stream);
}
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*******************************************************************************MINING KERNEL*******************************************************************************/

// LAUNCH MINER KERNEL ON AN INDEPENDENT STREAM USING THE SPECIFIED NUMBER OF BLOCKS
// FIXME DEPRECATED. No longer used in main code, slot for removal
__host__ void launchMiner(WORKLOAD * load){
//	int num_blocks = (load->id == 0) ? PARENT_BLOCKS:WORKER_BLOCKS;
  cudaMemcpyAsync(load->block_d, load->block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, load->stream);
	cudaMemsetAsync(load->flag, 0, sizeof(int), load->stream);

	// COMPUTE THE CONSTANT PARTIAL HASH FOR THE FIRST 64 BYTES
	calculateFirstState(load->basestate_h, load->block_h);
	cudaMemcpyToSymbolAsync(block_const, load->basestate_h, HASH_SIZE, HASH_SIZE*load->id, cudaMemcpyHostToDevice, load->stream);
/*
	if(load->id == 0){
		LAUNCH_MINER(PARENT_BLOCKS, load->id, load->stream, load->block_d, load->hash_d, load->hash_byte, load->flag);
	} else{
		LAUNCH_MINER(WORKER_BLOCKS, load->id, load->stream, load->block_d, load->hash_d, load->hash_byte, load->flag);
	}
*/
	if(load->id == 0){
		LAUNCH_MINER(0, load->id, load->stream, load->block_d, load->hash_d, load->hash_byte, load->flag);
	} else{
		LAUNCH_MINER(NUM_WORKERS, load->id, load->stream, load->block_d, load->hash_d, load->hash_byte, load->flag);
	}
}

// LOAD MINER RESULTS BACK FROM THE GPU USING ASYNCHRONOUS STREAMING
__host__ void returnMiner(WORKLOAD * load){
  cudaMemcpyAsync(load->block_h, load->block_d, BLOCK_SIZE, cudaMemcpyDeviceToHost, load->stream);
	cudaMemcpyAsync(load->hash_h, load->hash_d, HASH_SIZE, cudaMemcpyDeviceToHost, load->stream);
}

/***************************************************************************************************************************************************************************/
/***********************************************************************MULTISTREAM WORKFLOW FUNCTION***********************************************************************/
/***************************************************************************************************************************************************************************/

// TODO Clean up workflow, clear old or irrelevant comments
// BASE FUNCTION TO COORDINATE NON-BLOCKING OPERATIONS INTO VARIOUS STREAMS
__host__ void launchWorkflow(WORKLOAD * load){
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*******************************************************************************PREREQUISITES*******************************************************************************/
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	// PREREQUISITES:
	//	BUFFER_H MUST BE FILLED WITH SOME DATA PRIOR TO STARTING
	//	(MAY BE BEST TO USE A FLAG TO INDICATE WHEN THE BUFFER IS READY)
	// 	BLOCK_H NEEDS THE PREVIOUS HASH TO BE COPIED TO BYTE[4-36] OR WORD[1-9] AND TIME NEEDS TO BE UP TO DATE
	//		(SHOULD BE DONE AFTER THE PREVIOUS BLOCK IS WRITTEN TO THE FILE, COULD SPEED THIS UP BY SENDING A COPY TO ANOTHER CPU CORE FOR WRITING)
	//		IN A MULTICORE CASE, ANOTHER CORE CAN WRITE TO FILE WHILE THE BUFFER IS COPIED H2D. NEW BLOCK CAN THEN BE SET AND COPIED AFTER THE BUFFER COPY IS COMPLETE (UNLESS COPY BLOCKS SOME OTHER FUNCTIONS)
	// EX FUNCTION DEPENDENCIES:
	// initializeHash(&w_load[i]); // CREATES FILE, READ FIRST HASH
	//initializeWorkerBlock(&w_load[i]);
	//initializeParentBlock(p_load->block_h);
	//getDifficulty(p_load);

	// PARENT: COPY CONTENTS OF BUFFER BLOCKS INTO BUFFER_H
	// WORKER: READ IN CONTENTS OF NEXT BUFFER_H
	// NOTE: READING IN FOR WORKER TO BUFFER_H CAN BEGIN AS SOON AS THE MEMORY COPY FROM BUFFER_H TO BUFFER_D COMPLETES
	//				SIMILAR SITUATION FOR THE PARENT. MAY BE EASIER TO STORE WORKER RESULTS DIRECTLY INTO THE PARENT BUFFER TO AVOID FUTURE DELAYS
	//				IE, IF QUERY PARENT COPY EVENT == TRUE, WRITE TO BUFFER, ELSE WAIT OR COPY TO A BUFFER
	//				BETTER: COPY TO OVERFLOW BUFFER IF P_BUFFER_H == BUSY, ELSE WRITE DIRECTLY INTO BUFFER_H
	//				>	WORKER CAN OPERATE ON THE SAME PRINCIPLE, READING INTO A SEPARATE BUFFER UNTIL THE WORKER BUFFER_H IS READY
	//				> UPON RECEIVING A SIGNAL, THE OVERFLOW IS COPIED INTO BUFFER_H. COULD ALSO BE DONE WITH A CALLBACK

/*------------------------------------------------------------------------------------||------------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/********************************************************************************MERKLE LAUNCH*******************************************************************************/
/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
	//printf("LAUNCHING WORKER %i", load->id);

	cudaEventRecord(load->t_start, load->stream);
/*----------------------------------------------------------------------------MERKLE MEMCPY H2D-----------------------------------------------------------------------------*/
	// COPY BUFFER H2D (MUST BE READY TO COPY)

	// COPY BLOCK H2D (PREPARED EARLIER ON)
	cudaMemcpyAsync(load->block_d, load->block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, load->stream);  // COPY OVER CURRENT BLOCK
	//cudaMemcpyAsync(load->block_d, load->block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, load->stream);  // COPY OVER CURRENT BLOCK
	// TREE SIZE CAN BE PRECOMPUTED PRIOR TO BUFFER WRITE
	int tree_size = pow(2.0, ceil(log2((double)load->buff_blocks)));
	// MUST BE PERFORMED AFTER PREVIOUS KERNEL HAS FINISHED, PLACE AFTER BUFFER CPY TO AVOID BLOCKING
	cudaMemsetAsync(load->flag, 0, sizeof(int), load->stream);

//	printf("\nW[%i]\tSTART BUFFER COPY\n", load->id);

	// NOTE Prints the merkle tree for each worker, which is useful, but also a huge mess
	//printMerkle(load);


	cudaMemcpyAsync(load->buffer_d, load->buffer_h, HASH_SIZE*load->buff_size, cudaMemcpyHostToDevice, load->stream);
//	printf("\nW[%i]\tSTART MERKLE WITH %i BLOCKS AND %i TREE SIZE\n", load->id, load->buff_blocks, tree_size);
/*-----------------------------------------------------------------------------MERKLE HASH TREE-----------------------------------------------------------------------------*/
	// FIXME RUN COMPUTATION FOR BASESTATE AND UPDATE BLOCK TIME HERE.
//	merkleKernel<<<1, MERKLE_THREADS, 0, load->stream>>>(load->buffer_d, &load->block_d[9], load->buff_blocks,  tree_size);
	merkleKernel_workflow<<<1, MERKLE_THREADS, 0, load->stream>>>(load->buffer_d, load->block_d, load->basestate_d, load->buff_blocks,  tree_size);
	load->buff_blocks = 0;
//	printf("\nW[%i]\tCOPY BACK BLOCK_D\n", load->id);
/*-------------------------------------------------------------------------------MERKLE RETURN------------------------------------------------------------------------------*/
	// BLOCK IS ONLY NECCESSARY WHEN USING A CALLBACK TO LOG THE CURRENT STATE
	cudaMemcpyAsync(load->block_h, load->block_d, BLOCK_SIZE, cudaMemcpyDeviceToHost, load->stream);


	// LOG MINER START (PRINT TIME AND HASH BEING SOLVED)
	// TODO IMPLEMENT AS A CALLBACK
	//logStart(p_load->id, p_load->blocks+1, &p_load->block_h[9]);
	logStart(load->id, load->blocks+1, &load->block_h[9]); // TODO Callback after merkle

/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*********************************************************************************MINER LAUNCH*******************************************************************************/
/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------MERKLE MEMCPY H2D-----------------------------------------------------------------------------*/
	// ALREADY DONE IF MERKLE IS USED...
	//cudaMemcpyAsync(load->block_d, load->block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, load->stream);


/*------------------------------------------------------------------------MERKLE BASESTATE COMPUTE--------------------------------------------------------------------------*/
	// COMPUTE THE CONSTANT PARTIAL HASH FOR THE FIRST 64 BYTES
	// FIXME MOVE THIS PART TO THE MERKLE KERNEL IF POSSIBLE
	// WOULD REQUIRE AN ADDITIONAL WRITE TO HOST SO THAT BASESTATE CAN BE SET IN CONSTANT MEMORY BY THE HOST
	// IDEA START SYMBOLIC COPY ASYNC, AND TRY TO INTEGRATE A CALL BACK THAT LOGS THE STARTING CONDITION WHILE THE H2D TRANSFER TAKES PLACE
	//calculateFirstState(load->basestate_h, load->block_h);
//	printf("\nW[%i]\tSTART SYMBOL COPY\n", load->id);
/*-------------------------------------------------------------------------COPY BASESTATE TO SYMBOL-------------------------------------------------------------------------*/
	cudaMemcpyToSymbolAsync(block_const, load->basestate_d, HASH_SIZE, HASH_SIZE*load->id, cudaMemcpyDeviceToDevice, load->stream);
//	printf("W[%i]\tSTART MINER\n", load->id);
/*---------------------------------------------------------------------------MINER KERNEL FUNCTION--------------------------------------------------------------------------*/
/*
	// MINER KERNEL, DEPENDENT ON THE COMPLETION OF THE MERKLE HASH AND SYMBOLIC COPY
	if(load->id == 0){
		LAUNCH_MINER(PARENT_BLOCKS, load->id, load->stream, load->block_d, load->hash_d, load->hash_byte, load->flag);
	} else{
		LAUNCH_MINER(WORKER_BLOCKS, load->id, load->stream, load->block_d, load->hash_d, load->hash_byte, load->flag);
	}
*/
	if(load->id == 0){
		LAUNCH_MINER(0, load->id, load->stream, load->block_d, load->hash_d, load->hash_byte, load->flag);
	} else{
		LAUNCH_MINER(NUM_WORKERS, load->id, load->stream, load->block_d, load->hash_d, load->hash_byte, load->flag);
	}
//	printf("W[%i]\tRETURN MINER\n", load->id);
	// MINER RETURN
/*----------------------------------------------------------------------------MINER KERNEL RETURN---------------------------------------------------------------------------*/
	// UPON MINER COMPLETION, WRITE BACK RESULTS, PRINT, AND UPDATE BLOCK FOR THE NEXT HASH
//	cudaMemcpyAsync(load->block_h, load->block_d, BLOCK_SIZE, cudaMemcpyDeviceToHost, load->stream);
//	cudaMemcpyAsync(load->hash_h, load->hash_d, HASH_SIZE, cudaMemcpyDeviceToHost, load->stream);

	//cudaEventRecord(load->t_stop, load->stream);
//	printf("W[%i]\tFINISH\n", load->id);
/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*******************************************************************************POST PROCESSING******************************************************************************/
/*--------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/****************************
CALLBACK TEST September 2019
NOTE: Both methods cause CPU Stall

*/
// BLOCKING TESTS
//cudaEventRecord(load->t_stop, load->stream); // Event Record

//cudaStreamAddCallback(load->stream, MyCallback, load, 0); // Callback

//cudaHostFn_t fn = myHostNodeCallback;
//cudaLaunchHostFunc(load->stream, fn, load);	// Host function launch

// Callback test
//cudaStreamAddCallback(load->stream, MyCallback, (void*)callback_temp, 0);

// Host function test
//cudaLaunchHostFunc( cudaStream_t stream, cudaHostFn_t fn, void* userData);

/*-----------------------------------------------------------------------------PARENT POSTPROCESS---------------------------------------------------------------------------*/
// COPY BACK DATA, RECORD TIME, PRINT TO FILE, AND UPDATE HASH
//returnMiner(p_load);
//cudaEventSynchronize(p_load->t_stop);
//cudaEventElapsedTime(&p_load->t_result, p_load->t_start, p_load->t_stop);
//printOutputFile(bfilename, p_load->block_h, p_load->hash_h, p_load->blocks, p_load->t_result, p_load->difficulty, -1, 1);
//updateParentHash(p_load->block_h, p_load->hash_h);

/*-----------------------------------------------------------------------------WORKER POSTPROCESS---------------------------------------------------------------------------*/
// CALCULATE TIMING, PRINT TO OUTPUT FILE
//cudaEventRecord(w_ptr->t_stop, w_ptr->stream);
//cudaEventSynchronize(w_ptr->t_stop);
//cudaEventElapsedTime(&w_ptr->t_result, w_ptr->t_start, w_ptr->t_stop);
//printOutputFile(w_ptr->outFile, w_ptr->block_h, w_ptr->hash_h, w_ptr->blocks, w_ptr->t_result, w_ptr->difficulty, i, 1);

// LOAD PARENT BUFFER IF WORKER
//p_load->buffer_h[p_load->buff_blocks*8 + j] = w_ptr->hash_h[j];

// INCREMENT DIFFICULTY IF THE LIMIT HAS BEEN REACHED (PRINT IF TARGET HAS BEEN REACHED)
// IF DIFF TIMER NOT YET RECORDED, RECORD EVENT NOW, THEN PRINT
//printDifficulty(w_ptr->outFile, w_ptr->id, w_ptr->difficulty, w_ptr->t_diff, (w_ptr->blocks-(w_ptr->diff_level-1)*DIFFICULTY_LIMIT));

// IF TARGET NOT REACHED, INCREMENT DIFFICULTY, RECORD DIFF START EVENT
// updateDifficulty(w_ptr->block_h, w_ptr->diff_level); getDifficulty(w_ptr);

// IF TARGET NOT YET REACHED, UPDATE BLOCK (WRITE HASH BACK, MUST BE DONE AFTER DATA IS SENT FOR WRITING)
//errEOF[i] = updateBlock(w_ptr->inFile, w_ptr->block_h, w_ptr->hash_h, w_ptr->buffer_h);

// START TIMER, AND BEGIN NEXT BLOCK
// cudaEventRecord(w_ptr->t_start, w_ptr->stream);
// logStart(w_ptr->id, (w_ptr->blocks)+1, w_ptr->buffer_h); launchMiner(w_ptr);


}


/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/********  ______________________________________________________________________________________________________________________________________________________  *********/
/********  |   _    _   _______   _____   _        _____   _______  __     __    ______   _    _   _   _    _____   _______   _____    ____    _   _    _____   |  *********/
/********  |  | |  | | |__   __| |_   _| | |      |_   _| |__   __| \ \   / /   |  ____| | |  | | | \ | |  / ____| |__   __| |_   _|  / __ \  | \ | |  / ____|  |  *********/
/********  |  | |  | |    | |      | |   | |        | |      | |     \ \_/ /    | |__    | |  | | |  \| | | |         | |      | |   | |  | | |  \| | | (___    |  *********/
/********  |  | |  | |    | |      | |   | |        | |      | |      \   /     |  __|   | |  | | | . ` | | |         | |      | |   | |  | | | . ` |  \___ \   |  *********/
/********  |  | |__| |    | |     _| |_  | |____   _| |_     | |       | |      | |      | |__| | | |\  | | |____     | |     _| |_  | |__| | | |\  |  ____) |  |  *********/
/********  |   \____/     |_|    |_____| |______| |_____|    |_|       |_|      |_|       \____/  |_| \_|  \_____|    |_|    |_____|  \____/  |_| \_| |_____/   |  *********/
/********  |____________________________________________________________________________________________________________________________________________________|  *********/
/********                                                                                                                                                          *********/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*************************************************************************HEX CONVERSION FUNCTIONS**************************************************************************/
// CONVERT THE INPUT TEXT STRING OF HALF-BYTES INTO HEX BYTE VALUES
__host__ void encodeHex(BYTE * str, BYTE * hex, int len){
  //  int len_s = strlen(str);
    for(int i = 0; i < len; i+=2){
      char temp[3];
      sprintf(temp, "0%c%c", str[i], str[i+1]);
      hex[(i == 0?0 : i/2)] = (BYTE)strtoul(temp, NULL, 16);
    }
    return;
}
__host__ void encodeWord(BYTE * str, WORD * hex, int len){
  //  int len_s = strlen(str);
    for(int i = 0; i < len; i+=8){
      char temp[9];
      sprintf(temp, "0%c%c%c%c%c%c%c%c", str[i], str[i+1], str[i+2], str[i+3], str[i+4], str[i+5], str[i+6], str[i+7]);
      hex[(i == 0?0 : i/8)] = (WORD)strtoul(temp, NULL, 16);
    }
    return;
}
// CONVERT HEX BYTE VALUES INTO A HUMAN READABLE STRING
__host__ void decodeHex(BYTE * hex, BYTE * str, int len){
  char temp[3];
  for(int i = 0; i < len; i+=1){
    sprintf(temp, "%03x", hex[i]);
    str[i*2] = temp[1];
    str[i*2+1] = temp[2];
  }
  str[len*2] = '\0';
  return;
}

// CONVERT HEX BYTE VALUES INTO A HUMAN READABLE STRING
__host__ void decodeWord(WORD * hex, BYTE * str, int len){
  char temp[9];
  for(int i = 0; i < len; i++){
    sprintf(temp, "%09x", hex[i]);
    str[i*8] = temp[1];
    str[i*8+1] = temp[2];
		str[i*8+2] = temp[3];
		str[i*8+3] = temp[4];
		str[i*8+4] = temp[5];
		str[i*8+5] = temp[6];
		str[i*8+6] = temp[7];
		str[i*8+7] = temp[8];
  }
  str[len*8] = '\0';
  return;
}
// PRINT A HEX VALUE TO THE CONSOLE
__host__ void printHex(BYTE * hex, int len){
  char temp[3];
  BYTE total[len*2+1];
  for(int i = 0; i < len; i+=1){
    sprintf(temp, "%03x", hex[i]);
    total[i*2] = temp[1];
    total[i*2+1] = temp[2];
  }
  total[len*2] = '\0';
  printf("%s\n", total);
  return;
}
// PRINT A HEX VALUE TO A FILE
__host__ void printHexFile(FILE * outfile, BYTE * hex, int len){
  char temp[3];
  BYTE total[len*2+1];
  for(int i = 0; i < len; i+=1){
    sprintf(temp, "%03x", hex[i]);
    total[i*2] = temp[1];
    total[i*2+1] = temp[2];
  }
  total[len*2] = '\0';
  fprintf(outfile,"%s\n", total);
  return;
}

// PRINT WORDS OF LENGTH LEN TO THE CONSOLE
__host__ void printWords(WORD * hash, int len){
	for(int i = 0; i < len; i++){
		printf("%08x", hash[i]);
	}
	printf("\n");
}

// NOTE Debugging function to print merkle tree
__host__ void printMerkle(WORKLOAD * load){//WORD * buffer_h, int buff_blocks, int block_num){
	printf("PRINTING BLOCK %i CONTENTS:  \n", load->blocks+1);
	char merkle_debug[50+WORKER_BUFFER_SIZE*100];
	char hash_entry[80];
	BYTE temp_hash[65];
	sprintf(merkle_debug, "BLOCK %i CONTENTS:  \n", load->blocks+1);
	for(int i = 0; i < load->buff_blocks; i++){
		decodeWord(&(load->buffer_h[i*8]), temp_hash, 8);
		//printf("%08x\n", load->buffer_h[i]);
		sprintf(hash_entry, "%i\t%s\n", i, (char*)temp_hash);
		strcat(merkle_debug, hash_entry);
	}
	// PRINT PARENT BLOCK CONTENTS
	printDebug(merkle_debug);
}

__host__ void host_convertHash_Word2Byte(WORD * in, BYTE* out){
	#pragma unroll 4
	for (int i = 0; i < 4; ++i) {
		out[i]      = (in[0] >> (24 - i * 8)) & 0x000000ff;
		out[i + 4]  = (in[1] >> (24 - i * 8)) & 0x000000ff;
		out[i + 8]  = (in[2] >> (24 - i * 8)) & 0x000000ff;
		out[i + 12] = (in[3] >> (24 - i * 8)) & 0x000000ff;
		out[i + 16] = (in[4] >> (24 - i * 8)) & 0x000000ff;
		out[i + 20] = (in[5] >> (24 - i * 8)) & 0x000000ff;
		out[i + 24] = (in[6] >> (24 - i * 8)) & 0x000000ff;
		out[i + 28] = (in[7] >> (24 - i * 8)) & 0x000000ff;
	}
}

__host__ void host_convertHash_Byte2Word(BYTE * in, WORD* out, int len){
	for (int i = 0; i < len; ++i) {
		out[i] = (in[i*4] << 24) | (in[i*4+1] << 16) | (in[i*4+2] << 8) | (in[i*4+3]);
	}
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*************************************************************************STATUS LOGGING FUNCTIONS**************************************************************************/
// FUNCTION TO PRINT LOG MESSAGES WITH TIMESTAMP
__host__ void printLog(const char * msg){
  time_t c_time = time(NULL);
  struct tm *ptm = localtime(&c_time);
  printf("[LOG]-(%02d:%02d:%02d):%s\n",ptm->tm_hour, ptm->tm_min, ptm->tm_sec, msg);
}
// FUNCTION TO PRINT MESSAGES ONLY WHEN DEBUG == 1
__host__ void printDebug(const char * msg){
  if(DEBUG == 1){
    printf("[DEBUG]:%s\n", msg);
  }
}
// FUNCTION TO PRINT ERROR MESSAGES
__host__ void printError(const char * msg){
  printf("\n/*****************************************************************/\n[ERROR]:%s\n/*****************************************************************/\n", msg);
}
// FUNCTION TO PRINT MINER STARTING MESSAGES
__host__ void logStart(int workerID, int block, WORD * start_hash){
  char name[20];
  if(workerID == 0){
    sprintf(name, "PARENT");
  } else{
    sprintf(name, "WORKER %i", workerID);
  }
  char logMessage[50];
  BYTE hash[65];
  decodeWord(start_hash, hash, 8);
  sprintf(logMessage,"%s STARTED MINING BLOCK %i\n      ROOT: %s\n", name, block, (char*)hash);
  printLog(logMessage);
}
// PRINT FUNCTION TO SHOW THE CURRENT MINING PROGRESS
__host__ int printProgress(int mining_state, int multilevel,int num_workers,int pchain_blocks, int *chain_blocks){
    char outStr[100] = "\r";
    char tempStr[10] = "";
    int next_state = 0;
    switch (mining_state) {
      case 0:
        strcat(outStr, " | ");
        next_state = 1;
      break;
      case 1:
        strcat(outStr, " / ");
        next_state = 2;
      break;
      case 2:
        strcat(outStr, " - ");
        next_state = 3;
      break;
      case 3:
        strcat(outStr, " \\ ");
        next_state = 0;
      break;
      default:
        next_state = 0;
      break;
    }
    strcat(outStr, " MINING:{");
    if(multilevel){
      sprintf(tempStr, "P[%i]|", pchain_blocks+1);
      strcat(outStr, tempStr);
    }
    sprintf(tempStr, "W[%i", chain_blocks[0]+1);
    strcat(outStr, tempStr);
    for(int i = 1; i < num_workers; i++){
      sprintf(tempStr, " | %i", chain_blocks[i]+1);
      strcat(outStr, tempStr);
    }
    strcat(outStr, "]}\r");
    printf("%s",outStr);
    fflush(stdout);
    return next_state;
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/**************************  ___________________________________________________________________________________________________________________  **************************/
/**************************  |    _____       __   ____        ______   _    _   _   _    _____   _______   _____    ____    _   _    _____    |  **************************/
/**************************  |   |_   _|     / /  / __ \      |  ____| | |  | | | \ | |  / ____| |__   __| |_   _|  / __ \  | \ | |  / ____|   |  **************************/
/**************************  |     | |      / /  | |  | |     | |__    | |  | | |  \| | | |         | |      | |   | |  | | |  \| | | (___     |  **************************/
/**************************  |     | |     / /   | |  | |     |  __|   | |  | | | . ` | | |         | |      | |   | |  | | | . ` |  \___ \    |  **************************/
/**************************  |    _| |_   / /    | |__| |     | |      | |__| | | |\  | | |____     | |     _| |_  | |__| | | |\  |  ____) |   |  **************************/
/**************************  |   |_____| /_/      \____/      |_|       \____/  |_| \_|  \_____|    |_|    |_____|  \____/  |_| \_| |_____/    |  **************************/
/**************************  |_________________________________________________________________________________________________________________|  **************************/
/**************************                                                                                                                       **************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/***************************************************************************INPUT FILE FUNCTIONS****************************************************************************/
// CHANGED Reads in numerous hashes to fill the buffer for each worker
// CREATE OR READ INPUT FILES FOR EACH WORKER, READ FIRST HASH VALUE
// RETURN OPENED INPUT FILES AND ERROR FLAG
__host__ int initializeHash(WORKLOAD * load){
  char filename[20], logOut[100];
  int Err = 0;
	WORD * buff_ptr;
  sprintf(filename, "inputs/chain_input%d.txt", load->id);
  if(load->inFile = fopen(filename, "r")){
      sprintf(logOut,"READING DATA FROM INPUT FILE '%s'",filename);
      printDebug((const char*)logOut);
			for(; load->buff_blocks < load->buff_size; load->buff_blocks++){
				buff_ptr = &(load->buffer_h[8*load->buff_blocks]);
				load->readErr = readNextHash(load->inFile, buff_ptr);
				if(load->readErr == 1){
					break;
				}
			}
  }else{
      sprintf(logOut,"INPUT FILE '%s' NOT FOUND, GENERATING FILE",filename);
      printDebug((const char*)logOut);
      // USE GPU TO CREATE RANDOMLY GENERATED INPUT FILES
			initializeInputFile(load->inFile, filename);
      if(load->inFile = fopen(filename, "r")){
          sprintf(logOut,"INPUT FILE '%s' CREATED SUCCESSFULLY!", filename);
          printDebug((const char*)logOut);
					for(; load->buff_blocks < load->buff_size; load->buff_blocks++){
						buff_ptr = &(load->buffer_h[8*load->buff_blocks]);
						load->readErr = readNextHash(load->inFile, buff_ptr);
						if(load->readErr == 1){
							break;
						}
					}
					//load->buff_blocks = 1;
      }else{
        printError("INPUT FILE STILL COULDN'T BE ACCESSED, ABORTING!!!");
        load->readErr = 1;
      }
  }
  if(load->readErr == 1){
    sprintf(logOut,"INPUT FILE '%s' COULD NOT BE READ!!!",filename);
    printError((const char*)logOut);
    Err = 1;
  }
  return Err;
}

// CREATE A NEW INPUT FILE, CALL KERNEL TO GENERATE RANDOM INPUT HASHES
__host__ void initializeInputFile(FILE * inFile, char * filename){
  // ALLOCATE SPACE FOR HASHES
  WORD *hash_hf, *hash_df;
  size_t size_hash = NUM_THREADS * MAX_BLOCKS * HASH_SIZE;
  hash_hf = (WORD *) malloc(size_hash);
  cudaMalloc((void **) &hash_df, size_hash);

  // ALLOCATE SPACE FOR SEED VALUES
  WORD *seed_h, *seed_d;
  seed_h = (WORD*)malloc(HASH_SIZE);
  cudaMalloc((void **) &seed_d, HASH_SIZE);

  // CREATE NEW INPUT FILE
  FILE *file_out;
  char status[100], file_log[100];;
  if(file_out = fopen(filename, "w")){
    sprintf(file_log,"CREATED NEW INPUT FILE '%s'\n", filename);
    printDebug((const char*)file_log);
    fclose(file_out);
  } else{
    sprintf(file_log,"FILE '%s' COULD NOT BE CREATED", filename);
    printError((const char*)file_log);
  }

  srand(time(0));
  for(int j = 0; j < INPUT_LOOPS; j++){
    // CREATE RANDOM SEEDS
    for(int i = 0; i < 8; i++){
        seed_h[i] = (((rand() % 255) & 0xFF) << 24) | (((rand() % 255) & 0xFF) << 16) | (((rand() % 255) & 0xFF) << 8) | ((rand() % 255) & 0xFF);
    }
    // GENERATE NEW SET OF HASHES AND APPEND TO INPUT FILE
    launchGenHash(&hash_hf, &hash_df, &seed_h, &seed_d, size_hash);
    sprintf(status, "FINISHED INPUT GENERATION LOOP %i of %i", j, INPUT_LOOPS);
    printDebug((const char*)status);
    printInputFile(hash_hf, filename, MAX_BLOCKS, NUM_THREADS);
  }
  printDebug((const char*)"FINISHED GENERATING INPUT HASHES");
  free(seed_h);
  cudaFree(seed_d);
  free(hash_hf);
  cudaFree(hash_df);
  return;
}

// APPEND A SET OF HASHES TO THE SPECIFIED INPUT FILE
__host__ void printInputFile(WORD * hash_f, char * filename, int blocks, int threads){
  FILE *file_out;
	WORD * hash_ptr;
  int count = 0;
  // PARSE HASHES AND PRINT TO FILE
  if(file_out = fopen(filename, "a")){
    for(int i=0; i < blocks; i++){
        for(int j = 0; j < threads; j++){
					hash_ptr = &hash_f[i*threads + j*8];
          fprintf(file_out, "%08x%08x%08x%08x%08x%08x%08x%08x\n", hash_ptr[0],hash_ptr[1],hash_ptr[2],hash_ptr[3],hash_ptr[4],hash_ptr[5],hash_ptr[6],hash_ptr[7]);
          count++;
        }
    }
    char logmsg[50];
    sprintf(logmsg, "ADDING %i HASHES TO INPUT FILE '%s'\n", count, filename);
    printLog((const char*)logmsg);
    fclose(file_out);
  }
  else{
    char input_err[100];
    sprintf(input_err, "INPUT FILE '%s' COULD NOT BE CREATED!!!", filename);
    printError((const char*)input_err);
  }
}

// READ THE NEXT HASH FROM THE GIVEN INPUT FILE
__host__ int readNextHash(FILE * inFile, WORD * hash_h){
    int readErr = 0;
    BYTE inputBuffer[65];
    if(!fscanf(inFile, "%s", inputBuffer)){
      printError((const char*)"READ IN FAILED!!!!!");
      readErr = 1;
    }
    else {
      encodeWord(inputBuffer, hash_h, 64);
    }
    return readErr;
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/***************************************************************************OUTPUT FILE FUNCTIONS***************************************************************************/
// CREATE OUTPUT FILES FOR EACH WORKER, AND OUTPUT DIRECTORY IF NECCESSARY
__host__ int initializeOutfile(char * outFile, char * out_dir_name, int worker_id){
  printDebug((const char*)"BEGIN OUTPUT INITIALIZATION");
  int readErr = 0; char logOut[100]; FILE * output;
  mkdir("outputs", ACCESSPERMS);
  mkdir(out_dir_name, ACCESSPERMS);
  sprintf(outFile, "%s/outputs_%d.txt", out_dir_name, worker_id);
  if(output = fopen(outFile, "w")){
    sprintf(logOut,"FOUND WORKER %i OUTPUT FILE: %s.",worker_id, outFile);
    fprintf(output, "WORKER CHAIN %i OUTPUT FILE\nFORMAT:\n BLOCK_HEADER#: \n HASH_SOLUTION: \n CORRECT_NONCE: \n COMPUTATION_TIME: 0 \t\t BLOCK_DIFFICULTY: 0 \n\n", worker_id);
		fclose(output);
  }
  else{
      sprintf(logOut,"WORKER %i OUTPUT FILE: %s NOT FOUND",worker_id, outFile);
      readErr = 1;
  } printDebug((const char*)logOut);
  return readErr;
}

// CREATE PARENT OUTPUT FILES FOR INPUT HASHES AND SOLVED PARENT BLOCKS
__host__ int initializeParentOutputs(char * bfilename, char * hfilename){
  int writeErr = 0;
  FILE * pblocks, * phashes;
  char logOut[100];
  if(pblocks = fopen(bfilename, "w")){
    sprintf(logOut,"FOUND PARENT OUTPUT BLOCK FILE %s, READING DATA.", bfilename);
    fprintf(pblocks, "PARENT CHAIN BLOCK OUTPUT FILE\nFORMAT:\n BLOCK_HEADER#: \n HASH_SOLUTION: \n CORRECT_NONCE: \n COMPUTATION_TIME: \t\t BLOCK_DIFFICULTY:\n\n");
    fclose(pblocks);
  }else{
      sprintf(logOut,"BLOCK OUTPUT FILE '%s' NOT FOUND", bfilename);
      writeErr = 1;
  } printDebug((const char*)logOut);
  if(phashes= fopen(hfilename, "w")){
    sprintf(logOut,"FOUND PARENT OUTPUT HASH FILE %s, READING DATA.", hfilename);
    fprintf(phashes, "PARENT CHAIN HASH OUTPUT FILE\nFORMAT:\n PARENT_BLOCK_HEADER#: \n HASH_SOLUTION: \n CORRECT_NONCE: \n COMPUTATION_TIME: \t\t BLOCK_DIFFICULTY:\n\n");
    fclose(phashes);
  }else{
      sprintf(logOut,"HASH OUTPUT FILE '%s' NOT FOUND", hfilename);
      writeErr = 1;
  } printDebug((const char*)logOut);
  return writeErr;
}

// CREATE BENCHMARK OUTPUT FILES FOR EACH WORKER, AND OUTPUT DIRECTORY IF NECCESSARY
__host__ int initializeBenchmarkOutfile(char * outFile, char * out_dir_name, int worker_id){
  printDebug((const char*)"BEGIN OUTPUT INITIALIZATION");
  int readErr = 0; char logOut[100]; FILE * output;
  mkdir("outputs", ACCESSPERMS);
	mkdir("outputs/benchtest", ACCESSPERMS);
  mkdir(out_dir_name, ACCESSPERMS);
  sprintf(outFile, "%s/benchmark_%i_threads.txt", out_dir_name, NUM_THREADS);

  if(output = fopen(outFile, "w")){
		sprintf(logOut,"FOUND WORKER %i OUTPUT FILE: %s.",worker_id, outFile);
		fclose(output);
  }
  else{
      sprintf(logOut,"WORKER %i OUTPUT FILE: %s NOT FOUND",worker_id, outFile);
      readErr = 1;
  } printDebug((const char*)logOut);
  return readErr;
}

// PRINT TOTAL TIMING RESULTS FOR A GIVEN DIFFICULTY (OR BLOCK)
__host__ void printDifficulty(char* diff_file, int worker_num, double difficulty, float diff_time, int num_blocks){
  float avg_time = diff_time/(float)num_blocks;
  char name[20];
  char printOut[200];
  if(worker_num < 1){
    if(worker_num == 0){ // hfilename: PRINTING BUFFER FILL TIME
      sprintf(name, "PARENT_BUFFER");
    }else{ // bfilename: PRINTING PARENT DIFFICULTY BLOCK STATS
      sprintf(name, "PARENT_BLOCK");
    }
  } else{
    sprintf(name, "WORKER%i", worker_num);
  }
  sprintf(printOut, "%s DIFFICULTY_STATISTICS:\tTOTAL_TIME: %f\tAVG_TIME: %f\tDIFFICULTY: %lf\n ", name, diff_time, avg_time, difficulty);
  printLog(printOut);
  // PRINT TO FILE
  FILE * outFile;
  if(outFile = fopen(diff_file, "a")){
    fprintf(outFile, "%s\n ", printOut);
    fclose(outFile);
  }
}
// PRINT TOTAL TIMING RESULTS FOR A GIVEN DIFFICULTY (OR BLOCK)
__host__ void printErrorTime(char* err_file, char *err_msg, float err_time){
  char printOut[500];
  time_t c_time = time(NULL);
  struct tm *ptm = localtime(&c_time);
  sprintf(printOut, "\n[ERROR]-(%02d:%02d:%02d): TIME: %f \t MSG: %s\n ",ptm->tm_hour, ptm->tm_min, ptm->tm_sec, err_time,err_msg);
  printDebug(printOut);
  // PRINT TO FILE
  FILE * outFile;
  if(outFile = fopen(err_file, "a")){
    fprintf(outFile, "%s\n ", printOut);
    fclose(outFile);
  }
}
// PRINT BLOCK SOLUTIONS TO FILE AND CONSOLE IF SELECTED
__host__ void printOutputFile(char * outFileName, WORD * block_h, WORD * hash_f, int block, float calc_time, double difficulty, int id, int log_flag){
    char printOut[1000];
    char logOut[1000];
    char name[20];
    // Get chain name by ID
    if(id+1 == 0){
      sprintf(name, "[PARENT]");
    } else{
      sprintf(name, "WORKER %i", id+1);
    }
    // SET FILL FOR NAME PADDING
    int fill = (block < 1)? 1 : floor(1+log10(block));
    int fill_l = floor((float)(56-fill)/2)-(1 + fill%2);
    int fill_r = ceil((float)(56-fill)/2)-1;
    char stars1[30] = "", stars2[30] = "";
    for(int i = 0; i < fill_r; i++){
      if(i<=fill_r){
        strcat(stars1, "*");
      }
      if(i<=fill_l){
        strcat(stars2, "*");
      }
    } // SET SPACE FILL FOR TIME/DIFFICULTY PADDING
    int time_pad, diff_pad;
    if(calc_time < 1){
      time_pad = 1;
    }else{
      time_pad = 1+floor(log10(calc_time));
      diff_pad = 1 + floor(log10(difficulty));
    }
    char time_space[100] = "", diff_space[100] = "";
    for(int i = 0; i < (21 - time_pad); i++){
      strcat(time_space, " ");
    }
    for(int i = 0; i < (21 - diff_pad); i++){
      strcat(diff_space, " ");
    }
    // GET STRING VALUES OF BLOCK SOLUTION
    BYTE block_str[2][90], hash_str[65], nonce_str[10];
		decodeWord(block_h, block_str[0], 10);
		decodeWord(&(block_h[10]), block_str[1], 10);
    decodeWord(hash_f, hash_str, 8);
    decodeWord(&(block_h[19]), nonce_str, 1);

sprintf(logOut, "%s SOLVED BLOCK %i \n      HASH: %s\n", name, block, hash_str);
sprintf(printOut, "\n________________________________________________________________________________\n\
%s-%s FINISHED BLOCK %i %s|\n\
BLOCK_HEADER:___________________________________________________________________|\n%s|\n%s|\n\
********************************************************************************|\n\
HASH: %s          |\n\
NONCE: 0x%s                                                               |\n\
BLOCK_TIME: %f%sDIFFICULTY: %lf%s|\n\
________________________________________________________________________________|\n", stars1, name, block,stars2,block_str[0],block_str[1], hash_str, nonce_str, calc_time, time_space, difficulty, diff_space);

  // FLAG TO DETERMINE IF PRINT SHOULD BE LOGGED
  if(log_flag == 1){
    printLog(logOut);
    printDebug(printOut);
  }
  // PRINT TO FILE
  FILE * outFile;
  if(outFile = fopen(outFileName, "a")){
    fprintf(outFile, "%s\n ", printOut);
    fclose(outFile);
  }
  else{
    char err_out[50];
    sprintf(err_out, "COULDN'T PRINT TO OUTPUT FILE '%s'", outFileName);
    printError(err_out);
  }
}


/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
/***********  _________________________________________________________________________________________________________________________________________________________________   ***********/
/***********  |                                                                                                                                                                |  ***********/
/***********  |     /$$$$$$  /$$        /$$$$$$  /$$$$$$$   /$$$$$$  /$$             /$$$$$$$$ /$$   /$$ /$$   /$$  /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$  /$$$$$$    |  ***********/
/***********  |    /$$__  $$| $$       /$$__  $$| $$__  $$ /$$__  $$| $$            | $$_____/| $$  | $$| $$$ | $$ /$$__  $$|__  $$__/|_  $$_/ /$$__  $$| $$$ | $$ /$$__  $$   |  ***********/
/***********  |   | $$  \__/| $$      | $$  \ $$| $$  \ $$| $$  \ $$| $$            | $$      | $$  | $$| $$$$| $$| $$  \__/   | $$     | $$  | $$  \ $$| $$$$| $$| $$  \__/   |  ***********/
/***********  |   | $$ /$$$$| $$      | $$  | $$| $$$$$$$ | $$$$$$$$| $$            | $$$$$   | $$  | $$| $$ $$ $$| $$         | $$     | $$  | $$  | $$| $$ $$ $$|  $$$$$$    |  ***********/
/***********  |   | $$|_  $$| $$      | $$  | $$| $$__  $$| $$__  $$| $$            | $$__/   | $$  | $$| $$  $$$$| $$         | $$     | $$  | $$  | $$| $$  $$$$ \____  $$   |  ***********/
/***********  |   | $$  \ $$| $$      | $$  | $$| $$  \ $$| $$  | $$| $$            | $$      | $$  | $$| $$\  $$$| $$    $$   | $$     | $$  | $$  | $$| $$\  $$$ /$$  \ $$   |  ***********/
/***********  |   |  $$$$$$/| $$$$$$$$|  $$$$$$/| $$$$$$$/| $$  | $$| $$$$$$$$      | $$      |  $$$$$$/| $$ \  $$|  $$$$$$/   | $$    /$$$$$$|  $$$$$$/| $$ \  $$|  $$$$$$/   |  ***********/
/***********  |    \______/ |________/ \______/ |_______/ |__/  |__/|________/      |__/       \______/ |__/  \__/ \______/    |__/   |______/ \______/ |__/  \__/ \______/    |  ***********/
/***********  |________________________________________________________________________________________________________________________________________________________________|  ***********/
/***********                                                                                                                                                                      ***********/
/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/


/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/****************************************************************************HASH TEST FUNCTIONS****************************************************************************/

// MINING BENCHMARK TEST FUNCTION
template <int blocks, int id>  // CHANGED TEMPLATE TO DIFFERENTIATE TARGET CONSTANTS
__global__ void miningBenchmarkKernel(WORD * block_d, WORD * result_d, BYTE * hash_d, int * flag_d, int * total_iterations){
	int success = 0, i = 0, j=0;
	int write = 0;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int inc_size = blocks*NUM_THREADS;			// SAVES 8 REGISTERS
	unsigned int max_iteration = (0xffffffff / inc_size)+1;

	// THREADS SHARE FIRST 64 BYTES, SET IN CONSTANT MEMORY
	// EACH THREAD HAS ITS OWN VARIABLE FOR TOP 16 BYTES
	// ALLOCATED ON SHARED MEMORY TO FREE UP REGISTER USAGE FOR HASHING

	__shared__ WORD unique_data[NUM_THREADS][4];
	WORD * unique_ptr = unique_data[threadIdx.x];

	// ID based addressing for constants
	WORD * base = &(block_const[id*8]);
	WORD * target = &(target_const[id*8]);

	// HARDWARE DEBUGGING, ONLY ACTIVE IF DEV_DEBUG >= 3
	// DOESN'T ADD TO MEMORY USAGE
	DEVICE_DEBUG(if(threadIdx.x == 0){printf("W [%i| %i]: [SM: %i | WARP: %i]\n", id, blockIdx.x, get_smid(), get_warpid());})

	WORD state_ptr[8];

	atomicExch(&(unique_ptr[0]), block_d[16]);
	atomicExch(&(unique_ptr[1]), block_d[17]);
	atomicExch(&(unique_ptr[2]), block_d[18]);

	#pragma unroll 1
	do{
		if(*flag_d == 0){ // reduces regs to 32
			#pragma unroll 1
			for(i = 1, atomicExch(&(unique_ptr[3]), idx);
					i <= max_iteration; // Iterations in max block size
					i++, atomicAdd(&(unique_ptr[3]), inc_size)){

					success = sha256_blockHash(unique_ptr, base, state_ptr, target);

					if(success == 1){
						write = atomicCAS(flag_d, 0, 1);
						if(write == 0){
							convertHash_Word2Byte(state_ptr, hash_d); // 32 regs with write
							for(j = 0; j < 8; j++){
								result_d[j] = state_ptr[j];
							}
							// CHANGED ADDS TO MEMORY USAGE, BREAKING BENCHMARK TEST
							// INCREASES BENCHMARK REGISTER USAGE, CAUSING A STALL WHEN THE HIGH DIFFICULTY WORKLOAD SIMULATION IS STARTED
							//DEVICE_PRINT_SOLN("THREAD: [%i,%i] FOUND BLOCK ON ITERATION %i.\n", threadIdx.x, blockIdx.x, i);
							//DEVICE_PRINT_SOLN("STATE %08x%08x%08x%08x", state_ptr[0], state_ptr[1], state_ptr[2], state_ptr[3]);
							//DEVICE_PRINT_SOLN("%08x%08x%08x%08x.\n\n", state_ptr[4], state_ptr[5], state_ptr[6], state_ptr[7]);
							block_d[16] = unique_ptr[0];
							block_d[17] = unique_ptr[1];
							block_d[18] = unique_ptr[2];
							block_d[19] = unique_ptr[3];
						}
					}
					if(*flag_d > 0){
						break;
					}
			} // END FOR LOOP
			if(threadIdx.x == 0){
			 atomicAdd(total_iterations, i);
			}
			atomicExch(&(unique_ptr[1]), time_const);
			// NOTE CALLED TO SHOW THAT DEVICE IS STILL FUNCTIONING DURING SLOWER DESIGN RUNS
			DEVICE_TIME("NEW TIME %08x\n", time_const);
		}
	}while(*flag_d == 0);

}	// FINISH TEST BENCHMARK

__global__ void hashTestMiningKernel(WORD * test_block, WORD * result_block, int * success){
	WORD uniquedata[4][4];
	uniquedata[threadIdx.x][0] = test_block[16];
	uniquedata[threadIdx.x][1] = test_block[17];
	uniquedata[threadIdx.x][2] = test_block[18];
	uniquedata[threadIdx.x][3] = test_block[19];

	__shared__ WORD state[4][8];

	WORD base[8];
	WORD target[8];

	#pragma unroll 8
	for(int i = 0; i < 8; i++){
		base[i] = block_const[i];
		target[i] = target_const[i];
	}

	*success = sha256_blockHash(uniquedata[0], base, state[0], target);
	for(int i = 0; i < 8; i++){
		result_block[i] = state[threadIdx.x][i];
	}

	//TEST HARDWARE LOGGING FUNCTIONS
	printf("HARDWARE DEBUG: [SM: %i | WARP: %i| LANE: %i]\n", get_smid(), get_warpid(), get_laneid());

	return;
}

template <int sel>
__global__ void hashTestDoubleKernel(WORD * test_block, WORD * result_block){
	int i;
	__shared__ WORD hash_result[8];
	__shared__ WORD data_in[16];

	if(sel == 32){
		#pragma unroll 8
		for(i = 0; i < 8; i++){
			data_in[i] = test_block[i];
		}
		sha256_merkleHash_32B(data_in, hash_result);
	}else if(sel == 64){
		#pragma unroll 16
		for(i = 0; i < 16; i++){
			data_in[i] = test_block[i];
		}
		sha256_merkleHash_64B(data_in, hash_result);
	}
	#pragma unroll 8
	for(i = 0; i < 16; i++){
		result_block[i] = hash_result[i];
	}
	return;
}

 /*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
 /***************************************************************************HASH MINING FUNCTIONS***************************************************************************/
__global__ void genHashKernel(WORD * hash_df, WORD * seed, int num_blocks){
	WORD unique_data = (WORD)(threadIdx.x + blockIdx.x * blockDim.x);
  int offset = 8*threadIdx.x + blockIdx.x * blockDim.x;

  WORD seed_hash[8];
  #pragma unroll 7
  for(int i = 0; i < 7; i++){
    seed_hash[i] = seed[i];
  }

  seed_hash[7] = unique_data;

	sha256_merkleHash_32B(seed_hash, &hash_df[offset]);
}

template <int blocks, int id>
__global__ void minerKernel(WORD * block_d, WORD * result_d, BYTE * hash_d, int * flag_d){
	int success = 0, i = 0, j=0;
	int write = 0;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int inc_size = blocks*NUM_THREADS;			// SAVES 8 REGISTERS
	unsigned int max_iteration = (0xffffffff / inc_size)+1;

	// THREADS SHARE FIRST 64 BYTES, SET IN CONSTANT MEMORY
	// EACH THREAD HAS ITS OWN VARIABLE FOR TOP 16 BYTES
	// ALLOCATED ON SHARED MEMORY TO FREE UP REGISTER USAGE FOR HASHING

	__shared__ WORD unique_data[NUM_THREADS][4];
	WORD * unique_ptr = unique_data[threadIdx.x];

	// HARDWARE DEBUGGING
	DEVICE_DEBUG(if(threadIdx.x == 0){printf("W [%i| %i]: [SM: %i | WARP: %i]\n", id, blockIdx.x, get_smid(), get_warpid());})

	// ADDS ADDITIONAL REGISTERS (8 REGS EACH)
//	WORD * block_ptr = &(block_const[block_offset]);
	WORD * block_ptr = &(block_const[id*8]);
	WORD * target_ptr = &(target_const[id*8]);

	WORD state_ptr[8];

	atomicExch(&(unique_ptr[0]), block_d[16]);
	atomicExch(&(unique_ptr[1]), block_d[17]);
	atomicExch(&(unique_ptr[2]), block_d[18]);

	#pragma unroll 1
	do{
		if(*flag_d == 0){ // reduces regs to 32
			#pragma unroll 1
			for(i = 1, atomicExch(&(unique_ptr[3]), idx);
					i <= max_iteration; // Iterations in max block size
					i++, atomicAdd(&(unique_ptr[3]), inc_size)){

					success = sha256_blockHash(unique_ptr, block_ptr, state_ptr, target_ptr);

					if(success == 1){
						write = atomicCAS(flag_d, 0, 1);
						if(write == 0){
							convertHash_Word2Byte(state_ptr, hash_d); // 32 regs with write
							for(j = 0; j < 8; j++){
								result_d[j] = state_ptr[j];
							}
							//printf("FOUND HASH SOLUTION! %08x\n", state_ptr[0]);
							DEVICE_PRINT_SOLN("THREAD: [%i,%i] FOUND BLOCK ON ITERATION %i.\n", threadIdx.x, blockIdx.x, i);
							DEVICE_PRINT_SOLN("STATE %08x%08x%08x%08x", state_ptr[0], state_ptr[1], state_ptr[2], state_ptr[3]);
							DEVICE_PRINT_SOLN("%08x%08x%08x%08x.\n\n", state_ptr[4], state_ptr[5], state_ptr[6], state_ptr[7]);
							block_d[16] = unique_ptr[0];
							block_d[17] = unique_ptr[1];
							block_d[18] = unique_ptr[2];
							block_d[19] = unique_ptr[3];
						}
					}
					if(*flag_d > 0){
						break;
					}
			} // END FOR LOOP
			atomicExch(&(unique_ptr[1]), time_const);
			DEVICE_TIME("NEW TIME %08x\n", time_const);
		}
	}while(*flag_d == 0);

}	// FINISH TEST BENCHMARK

// NOTE: Deprecated. May produce incorrect results due to lack of synchronization
__global__ void merkleKernel(WORD * pHash_d, WORD * block_d, int buffer_blocks,  int tree_size){
	// surface height is constant

  // Shared memory for sharing hash results
  __shared__ WORD local_mem_in[MERKLE_THREADS][16];
  __shared__ WORD local_mem_out[MERKLE_THREADS][8];

	WORD * local_in;
	WORD * local_out;
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int offset = idx * 8;
	int mid = 1;

	if(threadIdx.x < MERKLE_THREADS){
		local_in = local_mem_in[threadIdx.x];
		local_out = local_mem_out[threadIdx.x];

		if(threadIdx.x < buffer_blocks){
		 	sha256_merkleHash_32B(&pHash_d[offset], local_out);
			//DEVICE_PRINT_SOLN("INIT THREAD %i HASH: %08x%08x%08x%08x\n", threadIdx.x, local_out[0], local_out[1], local_out[2], local_out[3]);
			for(int i = 2; i <= tree_size; i*=2){
	      if(threadIdx.x % i == 0){
					mid = i/2;
	        if(threadIdx.x + mid < buffer_blocks){
	          #pragma unroll 8
	          for(int j = 0; j < 8; j++){
	            local_in[j] = local_out[j];
	            local_in[8+j] = local_mem_out[threadIdx.x+mid][j];
	          }
					}else{ // HASH TOGETHER DUPLICATES FOR UNMATCHED BRANCHES
	          #pragma unroll 8
	          for(int j = 0; j < 8; j++){
	            local_in[j] = local_out[j];
	            local_in[8+j]= local_out[j];
	          }
	        }
					sha256_merkleHash_64B(local_in, local_out);
					//DEVICE_PRINT_SOLN("ROUND %i THREAD %i HASH: %08x%08x%08x%08x\n", i, threadIdx.x, local_out[0], local_out[1], local_out[2], local_out[3]);
	      }
	    } //END FOR LOOP
			if(threadIdx.x == 0){
				#pragma unroll 8
				for(int i = 0; i < 8; i++){
					block_d[i] = local_out[i];
				}
			}
		}	// END BUFFER IF
	} // END IF
}

//*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
//*************************************************************************WORKFLOW MINING FUNCTIONS*************************************************************************/
// CHANGED Added new merkleKernel for workers which stores results on the device side, eliminating the need for extra memory transfers and host side computations
// IDENTICAL TO MERKLE KERNEL, WITH A FEW EXCEPTIONS TO REDUCE HOST MEMORY TRANSFERS AND COMPUTATION
// WRITES TO THE ENTIRE BLOCK (TO INCLUDE UPDATED TIME)
__global__ void merkleKernel_workflow(WORD * pHash_d, WORD * block_d, WORD * basestate_d, int buffer_blocks,  int tree_size){
	// surface height is constant
  // Shared memory for sharing hash results
  __shared__ WORD local_mem_in[MERKLE_THREADS][16];
  __shared__ WORD local_mem_out[MERKLE_THREADS][8];

	WORD * local_in;
	WORD * local_out;
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int offset = idx * 8;
	int mid = 1;

	if(threadIdx.x < MERKLE_THREADS){
		local_in = local_mem_in[threadIdx.x];
		local_out = local_mem_out[threadIdx.x];

		if(threadIdx.x < buffer_blocks){
		 	sha256_merkleHash_32B(&pHash_d[offset], local_out);
			//DEVICE_PRINT_SOLN("INIT THREAD %i HASH: %08x%08x%08x%08x\n", threadIdx.x, local_out[0], local_out[1], local_out[2], local_out[3]);
			// FIXME Debugging for merkle mechanics
			//printf("Round 1: Thread %i \t Warp %i \t Lane %i \n", threadIdx.x, get_warpid(), get_laneid());
			//printf("INIT THREAD %i HASH: %08x%08x%08x%08x\n", threadIdx.x, local_out[0], local_out[1], local_out[2], local_out[3]);

			for(int i = 2; i <= tree_size; i*=2){
				// CHANGED 10/6 added sync to prevent race conditions
				__syncthreads();	// Needed to prevent race conditions on shared memory
	      if(threadIdx.x % i == 0){
					mid = i/2;
	        if(threadIdx.x + mid < buffer_blocks){
	          #pragma unroll 8
	          for(int j = 0; j < 8; j++){
	            local_in[j] = local_out[j];
	            local_in[8+j] = local_mem_out[threadIdx.x+mid][j];
	          }
					}else{ // HASH TOGETHER DUPLICATES FOR UNMATCHED BRANCHES
	          #pragma unroll 8
	          for(int j = 0; j < 8; j++){
	            local_in[j] = local_out[j];
	            local_in[8+j]= local_out[j];
	          }
	        }

					sha256_merkleHash_64B(local_in, local_out);
					//DEVICE_PRINT_SOLN("ROUND %i THREAD %i HASH: %08x%08x%08x%08x\n", i, threadIdx.x, local_out[0], local_out[1], local_out[2], local_out[3]);
					// FIXME Debugging for results per round
					//printf("Round %i: Thread %i \t Warp %i \t Lane %i \n", i, threadIdx.x, get_warpid(), get_laneid());
	      }
	    } //END FOR LOOP
			if(threadIdx.x == 0){
				// BLOCK[0] = VERSION, [1-8] = PREVIOUS HEADER HASH
				// MERKLE ROOT STORED IN BLOCK[9-16]
				// TIME IS STORED IN BLOCK[17] (18=DIFF, 19=NONCE)
				#pragma unroll 8
				for(int i = 0; i < 8; i++){
					block_d[i+9] = local_out[i];
				}
				block_d[17] = time_const;
				sha256_merkleHash_base(block_d, basestate_d);
				/*
				sha256_merkleHash_base(block_d, local_out);
				#pragma unroll 8
				for(int i = 0; i < 8; i++){
					basestate_d[i] = local_out[i];
				}
				printState(basestate_d);
				*/
				//printf("FINISHED MERKLE HASHING!!!\n");
			}
		}	// END BUFFER IF
	} // END IF
}

/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
/***********  _________________________________________________________________________________________________________________________________________________________________   ***********/
/***********  |                                                                                                                                                               |   ***********/
/***********  |    /$$$$$$$  /$$$$$$$$ /$$    /$$ /$$$$$$  /$$$$$$  /$$$$$$$$       /$$$$$$$$ /$$   /$$ /$$   /$$  /$$$$$$  /$$$$$$$$ /$$$$$$  /$$$$$$  /$$   /$$  /$$$$$$    |   ***********/
/***********  |   | $$__  $$| $$_____/| $$   | $$|_  $$_/ /$$__  $$| $$_____/      | $$_____/| $$  | $$| $$$ | $$ /$$__  $$|__  $$__/|_  $$_/ /$$__  $$| $$$ | $$ /$$__  $$   |   ***********/
/***********  |   | $$  | $$| $$$$$   |  $$ / $$/  | $$  | $$      | $$$$$         | $$$$$   | $$  | $$| $$ $$ $$| $$         | $$     | $$  | $$  | $$| $$ $$ $$|  $$$$$$    |   ***********/
/***********  |   | $$  \ $$| $$      | $$   | $$  | $$  | $$  \__/| $$            | $$      | $$  | $$| $$$$| $$| $$  \__/   | $$     | $$  | $$  \ $$| $$$$| $$| $$  \__/   |   ***********/
/***********  |   | $$  | $$| $$__/    \  $$ $$/   | $$  | $$      | $$__/         | $$__/   | $$  | $$| $$  $$$$| $$         | $$     | $$  | $$  | $$| $$  $$$$ \____  $$   |   ***********/
/***********  |   | $$  | $$| $$        \  $$$/    | $$  | $$    $$| $$            | $$      | $$  | $$| $$\  $$$| $$    $$   | $$     | $$  | $$  | $$| $$\  $$$ /$$  \ $$   |   ***********/
/***********  |   | $$$$$$$/| $$$$$$$$   \  $/    /$$$$$$|  $$$$$$/| $$$$$$$$      | $$      |  $$$$$$/| $$ \  $$|  $$$$$$/   | $$    /$$$$$$|  $$$$$$/| $$ \  $$|  $$$$$$/   |   ***********/
/***********  |   |_______/ |________/    \_/    |______/ \______/ |________/      |__/       \______/ |__/  \__/ \______/    |__/   |______/ \______/ |__/  \__/ \______/    |   ***********/
/***********  |_______________________________________________________________________________________________________________________________________________________________|   ***********/
/***********                                                                                                                                                                      ***********/
/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/
/********************************************************************************************************************************************************************************************/

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*************************************************************************DEVICE UTILITY FUNCTIONS**************************************************************************/

__device__ void printHash(BYTE * hash){
  printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x \n", hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7], hash[8], hash[9],\
  hash[10], hash[11], hash[12], hash[13], hash[14], hash[15], hash[16], hash[17], hash[18], hash[19],\
  hash[20], hash[21], hash[22], hash[23], hash[24], hash[25], hash[26], hash[27], hash[28], hash[29], hash[30], hash[31]);
}
__device__ void printBlock(BYTE * hash){
	printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n", \
	hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7], hash[8], hash[9],\
  hash[10], hash[11], hash[12], hash[13], hash[14], hash[15], hash[16], hash[17], hash[18], hash[19],\
  hash[20], hash[21], hash[22], hash[23], hash[24], hash[25], hash[26], hash[27], hash[28], hash[29],\
	hash[30], hash[31]);

	printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n", \
	hash[32], hash[33], hash[34], hash[35], hash[36], hash[37], hash[38], hash[39],\
	hash[40], hash[41], hash[42], hash[43], hash[44], hash[45], hash[46], hash[47], hash[48], hash[49],\
	hash[50], hash[51], hash[52], hash[53], hash[54], hash[55], hash[56], hash[57], hash[58], hash[59],\
	hash[60], hash[61], hash[62], hash[63]);

	printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x\n", \
	hash[64], hash[65], hash[66], hash[67], hash[68], hash[69],\
	hash[70], hash[71], hash[72], hash[73], hash[74], hash[75], hash[76], hash[77], hash[78], hash[79]);
}

__device__ void printState(WORD * hash){
	printf("%08x%08x%08x%08x%08x%08x%08x%08x\n",hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7]);
}

__device__ void printBlockW(WORD * hash){
	printf("%08x%08x%08x%08x%08x%08x%08x%08x",hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7]);
	printf("%08x%08x%08x%08x%08x%08x%08x%08x", hash[8], hash[9], hash[10], hash[11], hash[12], hash[13], hash[14], hash[15]);
	printf("%08x%08x%08x%08x\n\n", hash[16], hash[17], hash[18], hash[19]);
}

__device__ __inline__ void convertHash_Word2Byte(WORD * in, BYTE* out){
	#pragma unroll 4
	for (int i = 0; i < 4; ++i) {
		out[i]      = (in[0] >> (24 - i * 8)) & 0x000000ff;
		out[i + 4]  = (in[1] >> (24 - i * 8)) & 0x000000ff;
		out[i + 8]  = (in[2] >> (24 - i * 8)) & 0x000000ff;
		out[i + 12] = (in[3] >> (24 - i * 8)) & 0x000000ff;
		out[i + 16] = (in[4] >> (24 - i * 8)) & 0x000000ff;
		out[i + 20] = (in[5] >> (24 - i * 8)) & 0x000000ff;
		out[i + 24] = (in[6] >> (24 - i * 8)) & 0x000000ff;
		out[i + 28] = (in[7] >> (24 - i * 8)) & 0x000000ff;
	}
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/************************************************************************MESSAGE SCHEDULE FUNCTIONS*************************************************************************/
// OPTIMIZED MEMORY SCHEDULE COMPUTATION USING A REDUCED 16 WORD STATE
// OPERATIONS ARE IDENTICAL TO THE PREVIOUS FUNCTION, EXCEPT MOD 16
// TO REDUCE THE OVERALL MEMORY USAGE
__device__ __inline__ void scheduleExpansion_short( WORD m[]){
	m[0] += SIG1(m[14]) + m[9] + SIG0(m[1]);
	m[1] += SIG1(m[15]) + m[10] + SIG0(m[2]);
	m[2] += SIG1(m[0]) + m[11] + SIG0(m[3]);
	m[3] += SIG1(m[1]) + m[12] + SIG0(m[4]);
	m[4] += SIG1(m[2]) + m[13] + SIG0(m[5]);
	m[5] += SIG1(m[3]) + m[14] + SIG0(m[6]);
	m[6] += SIG1(m[4]) + m[15] + SIG0(m[7]);
	m[7] += SIG1(m[5]) + m[0] + SIG0(m[8]);
	m[8] += SIG1(m[6]) + m[1] + SIG0(m[9]);
	m[9] += SIG1(m[7]) + m[2] + SIG0(m[10]);
	m[10] += SIG1(m[8]) + m[3] + SIG0(m[11]);
	m[11] += SIG1(m[9]) + m[4] + SIG0(m[12]);
	m[12] += SIG1(m[10]) + m[5] + SIG0(m[13]);
	m[13] += SIG1(m[11]) + m[6] + SIG0(m[14]);
	m[14] += SIG1(m[12]) + m[7] + SIG0(m[15]);
	m[15] += SIG1(m[13]) + m[8] + SIG0(m[0]);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/************************************************************************PARTIAL TRANSFORM FUNCTIONS************************************************************************/

__device__ __inline__ void sha256_hashQuarter(WORD state[8], WORD m[], int offset){
	int i;
	WORD t1, t2;
	// UNROLLED LOOP
	#pragma unroll 4
	for(i = 0; i < 16; i+=4){
		t1 = GET_T1(state[4],state[5],state[6],state[7], k_s[offset][i], m[i]);
		t2 = GET_T2(state[0],state[1],state[2]);
		state[7] = state[3] + t1;
		state[3] = t1 + t2;

		t1 = GET_T1(state[7],state[4],state[5],state[6], k_s[offset][i+1], m[i+1]);
		t2 = GET_T2(state[3],state[0],state[1]);
		state[6] = state[2] + t1;
		state[2] = t1 + t2;

		t1 = GET_T1(state[6],state[7],state[4],state[5], k_s[offset][i+2], m[i+2]);
		t2 = GET_T2(state[2],state[3],state[0]);
		state[5] = state[1] + t1;
		state[1] = t1 + t2;

		t1 = GET_T1(state[5],state[6],state[7],state[4], k_s[offset][i+3], m[i+3]);
		t2 = GET_T2(state[1],state[2],state[3]);
		state[4] = state[0] + t1;
		state[0] = t1 + t2;
	}
}

__device__ __inline__ void sha256_hashSingle(WORD * base, WORD * state, WORD * m){
	int i;
	#pragma unroll 8
	for(i=0; i < 8; i++){
		state[i] = base[i];
	}

	sha256_hashQuarter(state, m, 0);
	scheduleExpansion_short(m);
	sha256_hashQuarter(state, m, 1);
	scheduleExpansion_short(m);
	sha256_hashQuarter(state, m, 2);
	scheduleExpansion_short(m);
	sha256_hashQuarter(state, m, 3);

	#pragma unroll 8
	for(i=0; i < 8; i++){
		state[i] += base[i];
	}
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*************************************************************************FULL TRANSFORM FUNCTIONS**************************************************************************/

// DEFAULT TRANSFORM FUNCTION, ASSUMES MESSAGE SCHEDULE HAS BEEN COMPUTED
// UNIQUE FUNCTION TO PERFORM DOUBLE HASH (80B | 32B) AND TARGET COMPARISON WITHOUT SHA256 STATE
__device__ __inline__ int sha256_blockHash(WORD * uniquedata, WORD * base, WORD * state, WORD * target){
	int i;
	WORD m[16];
	// Finish the remainder of the first hash
	#pragma unroll 4
	for(i = 0; i < 4; i++){
		m[i] = uniquedata[i];
	}
	#pragma unroll 12
	for(i=4; i<16; i++){
		m[i] = msgSchedule_80B[i];
	}
	sha256_hashSingle(base, state, m);
	// Double hash the 32 bit state
	#pragma unroll 8
	for(i=0; i<8; i++){
		m[i] = state[i];
	}
	#pragma unroll 8
	for(i=8; i<16; i++){
		m[i] = msgSchedule_32B[i];
	}
	sha256_hashSingle(i_state, state, m);
	return (COMPARE(state[0],target[0]) & COMPARE(state[1],target[1]) & COMPARE(state[2],target[2]) & COMPARE(state[3],target[3]) & COMPARE(state[4],target[4]) & COMPARE(state[5],target[5]) & COMPARE(state[6],target[6]) & COMPARE(state[7],target[7]));
}

// UNIQUE FUNCTION TO PERFORM DOUBLE HASH (64B | 32B) FROM WORDS WITHOUT SHA256 STATE
// USED FOR HASHING INPUT DATA OR FOR THE SECONDARY MERKLE HASH STEPS
__device__ __inline__ void sha256_merkleHash_64B(WORD hash_data[16], WORD * state){
	int i;
	WORD m[16];
	WORD state_i[8];

	#pragma unroll 16
	for(i = 0; i < 16; i++){
		m[i] = hash_data[i];
	}
	sha256_hashSingle(i_state, state, m);

	#pragma unroll 8
	for(i=0; i < 8; i++){
		state_i[i] = state[i];
	}
	sha256_hashQuarter(state, msgSchedule_64B_s[0], 0);
	sha256_hashQuarter(state, msgSchedule_64B_s[1], 1);
	sha256_hashQuarter(state, msgSchedule_64B_s[2], 2);
	sha256_hashQuarter(state, msgSchedule_64B_s[3], 3);

	#pragma unroll 8
	for(i=0; i<8; i++){
		m[i] = state[i] + state_i[i];
	}
	#pragma unroll 8
	for(i=8; i<16; i++){
		m[i] = msgSchedule_32B[i];
	}
	sha256_hashSingle(i_state, state, m);
	return;
}

// UNIQUE FUNCTION TO PERFORM DOUBLE HASH (32B | 32B) FROM WORDS WITHOUT SHA256 STATE
// USED FOR HASHING INPUT DATA OR FOR THE FIRST MERKLE HASH STEP
__device__ __inline__ void sha256_merkleHash_32B(WORD * hash_data, WORD * state){
	int i;
	WORD m[16];
	// Perform the first 32B hash
	#pragma unroll 8
	for(i = 0; i < 8; i++){
		m[i] = hash_data[i];
	}
	#pragma unroll 8
	for(i=8; i<16; i++){
		m[i] = msgSchedule_32B[i];
	}
	sha256_hashSingle(i_state, state, m);

	// Double hash the 32 bit state
	#pragma unroll 8
	for(i=0; i<8; i++){
		m[i] = state[i];
	}
	#pragma unroll 8
	for(i=8; i<16; i++){
		// USE COPY TO REDUCE REG USAGE, 48 REGS IF NOT USED
		m[i] = msgSchedule_32B_cpy[i];
	}
	sha256_hashSingle(i_state, state, m);
	return;
}

// SHORT FUNCTION TO CALCULATE THE CONSTANT MINING BASE ON THE DEVICE
__device__ __inline__ void sha256_merkleHash_base(WORD * hash_data, WORD * state){
	int i;
	WORD m[16];

	#pragma unroll 16
	for(i = 0; i < 16; i++){
		m[i] = hash_data[i];
	}
	sha256_hashSingle(i_state, state, m);
	return;
}

// IDEA Callback like this can be used to queue work after mining procedures
/*
// Additions September 2019
// CUDA Callback function example
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *load){
    //printf("Callback Success %d\n", (int)load);
		printf("Callback Success!!!!!\n");
		printf("Worker: %d\n", ((WORKLOAD*)load)->id);
		// These CUDA functions will not work in a callback (might work if different stream is used)
//		cudaEventRecord(((WORKLOAD*)load)->t_stop, ((WORKLOAD*)load)->stream);
//		cudaEventSynchronize(((WORKLOAD*)load)->t_stop);
//		cudaEventElapsedTime(&(((WORKLOAD*)load)->t_result), ((WORKLOAD*)load)->t_start, ((WORKLOAD*)load)->t_stop);
//		printf("Callback Time: %f\n\n", ((WORKLOAD*)load)->t_result);
}

//CUDA host function callback example
void CUDART_CB myHostNodeCallback(void *load) {
	printf("Callback Success!!!!!\n");
	printf("Worker: %d\n", ((WORKLOAD*)load)->id);
	/*
  // Check status of GPU after stream operations are done
  callBackData_t *tmp = (callBackData_t *)(data);
  // checkCudaErrors(tmp->status);
  double *result = (double *)(tmp->data);
  char *function = (char *)(tmp->fn_name);
  printf("[%s] Host callback final reduced sum = %lf\n", function, *result);
  *result = 0.0;  // reset the result
	*/
//}





/**------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/**************************************************************************DEVICE DEBUG FUNCTIONS***************************************************************************/
// NOTE These functions are for device debugging, providing a query to obtain Lane, Warp, and SM information from a thread

// Returns current multiprocessor the thread is running on
static __device__ __inline__ uint32_t get_smid(){
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

// Returns current warp the thread is running in
static __device__ __inline__ uint32_t get_warpid(){
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;
}

// Returns current lane the thread is executing in
static __device__ __inline__ uint32_t get_laneid(){
  uint32_t laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}
