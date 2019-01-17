// ECE 677
// Term Project
// Due: December 6, 2018
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
     │   │   ├───testHash
		 │   │   ├───testMiningHash
     │   │   ├───hostBenchmarkTest
		 │   │   ├───miningBenchmarkKernel
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
     │   │   │   ├───initializeWorkerBlocks
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
     │   │   │   ├───getDifficulty
		 │   │       └───getMiningDifficulty
     │   │   │
     │   │   └───CALCULATIONS
     │   │       ├───calculateDifficulty
     │   │       ├───calculateTarget
		 │   │       └───calculateMiningTarget
     │   │
     │   ├───KERNELS
     │   │   ├───genHashKernel
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
     │   ├───genTestHashes
     │   ├───minerKernel
     │   └───getMerkleRoot
     │
     └───DEVICE_FUNCTIONS
         ├───printHash
				 ├───printBlock
				 ├───printSplitBlock
         ├───sha256_transform
         ├───sha256_init
         ├───sha256_update
         ├───sha256_final
         ├───sha256_final_target
				 ├───sha256_mining_transform
				 ├───sha256_mining_transform_short
				 ├───scheduleExpansion
				 ├───scheduleExpansion_short
				 └───sha256_blockHash_TEST
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

typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} SHA256_CTX;

/***************************************************************************************************************************************************************************/
/****************************************************************************MACRO DEFINITIONS******************************************************************************/
/***************************************************************************************************************************************************************************/

#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

#define SHFTCMP(x, y, n) (((x >> n) & 0x000000ff) <= ((y >> n) & 0x000000ff))
#define COMPARE(x, y) (SHFTCMP(x,y,24) & SHFTCMP(x,y,16) & SHFTCMP(x,y,8) & SHFTCMP(x,y,0))

/***************************************************************************************************************************************************************************/
/**************************************************************************CONSTANT DEFINITIONS*****************************************************************************/
/***************************************************************************************************************************************************************************/
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest
#define BLOCK_SIZE sizeof(BYTE)*80 			// SIZE OF EACH BLOCK IN BYTES
#define BASE_SIZE sizeof(BYTE)*64				// SIZE OF BLOCK BASE IN BYTES
#define HASH_SIZE sizeof(BYTE)*32				// SIZE OF HASH IN BYTES
#define NONCE_SIZE sizeof(BYTE)*4				// SIZE OF NONCE

#define TARGET_C_SIZE sizeof(WORD)*8		// SIZE OF CONSTANT TARGET (WORDS)
#define BLOCK_C_SIZE sizeof(WORD)*16			// SIZE OF CONSTANT BLOCK (WORDS)

#define MAX_WORKERS 16 // 16 WORKERS MAX BASED ON MAX BLOCK SIZE
#define BLOCK_CONST_SIZE (MAX_WORKERS+1)*16
#define TARGET_CONST_SIZE (MAX_WORKERS+1)*8


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

// PRECOMPUTED SCHEDULE PADDING VALUES FOR 32 BYTE BLOCK HASH
__constant__ WORD msgSchedule_32B[16] = {
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000000,
	0x80000000, 0x00000000, 0x00000000, 0x00000000,
	0x00000000, 0x00000000, 0x00000000, 0x00000100
};



/*----------------------------------------------------------------------------CONSTANT SYMBOLS-----------------------------------------------------------------------------*/
// TEST CONSTANTS
//__constant__ BYTE test_unique_c[16];
__constant__ WORD test_basemsg_c[16];
__constant__ WORD test_target_c[8];

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

// neon
// pink 		green 		blue   orange
//ff4080		80ff40   40bfff  ff8040

// order ff4080  ff8040  40ff80 80ff40  4080ff  8040ff

 // TODO SET SPECIAL CASE FOR MINING, DIFFICULTY IS GRAY SCALE, BLOCKS PROCEED FROM A LIGHT SHADE, UP TO DARK


	const int num_colors = sizeof(colors[0])/sizeof(uint32_t);	// COLORS PER PALETTE
	const int num_palettes = sizeof(colors)/(sizeof(uint32_t)*num_colors); 																// TOTAL NUMBER OF COLOR PALETTES

//	const int num_palettes = 4; 			// TOTAL NUMBER OF COLOR PALETTES
//	const int num_colors = 12;		// COLORS PER PALETTE

	#define NUM_PALETTES num_palettes
	#define NUM_COLORS num_colors

	// TEST TO SEE IF PROFILING MACRO WAS PASSED IN
	#define PRINT_MACRO printf("MACRO PASSED SUCCESSFULLY!!\n\n")
	#define START_PROFILE cudaProfilerStart()
	#define STOP_PROFILE cudaProfilerStop()

	#define NAME_STREAM(stream, name) { \
		if(PROFILER == 1){ \
			nvtxNameCuStreamA(stream, name); \
		}\
	}

	// DEFAULT RANGE MANAGEMENT FUNCTIONS
	#define PUSH_RANGE(name,cid) { \
		if(PROFILER == 1){ \
			int color_id = cid; \
			color_id = color_id%num_colors;\
			nvtxEventAttributes_t eventAttrib = {0}; \
			eventAttrib.version = NVTX_VERSION; \
			eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
			eventAttrib.colorType = NVTX_COLOR_ARGB; \
			eventAttrib.color = colors[0][color_id]; \
			eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
			eventAttrib.message.ascii = name; \
			nvtxRangePushEx(&eventAttrib); \
		}}

	#define POP_RANGE if(PROFILER == 1){nvtxRangePop();}

	// DOMAIN MANAGEMENT FUNCTIONS
	#define DOMAIN_HANDLE nvtxDomainHandle_t

	#define DOMAIN_CREATE(handle, name){ \
			if(PROFILER == 1){ \
				handle = nvtxDomainCreateA(name); \
	}}

	#define DOMAIN_DESTROY(handle){ \
			if(PROFILER == 1){ \
				nvtxDomainDestroy(handle); \
	}}

	// ID specifies color related pattern, send -2 for time, -1 for parent
	#define PUSH_DOMAIN(handle, name, id, level, cid) { \
		if(PROFILER == 1){ \
			int worker_id = id; \
			int color_id = cid; \
			int palette_id = level; \
			worker_id = worker_id%num_colors; \
			color_id = color_id%num_colors;\
			palette_id = palette_id%num_palettes; \
			uint32_t color = colors[palette_id][color_id]; \
			if(id > -1){			\
				if(level == 2){   	\
				/*	color = color ^ ~colors[3][worker_id];		*/								\
				}												\
			}							\
			/*ADD IF STATEMENT HERE FOR ID*/ \
			nvtxEventAttributes_t eventAttrib = {0}; \
			eventAttrib.version = NVTX_VERSION; \
			eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
			eventAttrib.colorType = NVTX_COLOR_ARGB; \
			eventAttrib.color = color; \
			eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64; \
			eventAttrib.payload.llValue = level; \
			eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
			eventAttrib.message.ascii = name; \
			nvtxDomainRangePushEx(handle, &eventAttrib); \
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

// FIXME SEPARATE VARIABLES BY TYPE
/***************************************************************************************************************************************************************************/
/****************************************************************************GLOBAL VARIABLES*******************************************************************************/
/***************************************************************************************************************************************************************************/
#define AVAILABLE_BLOCKS 20 	// TOTAL NUMBER OF POSSIBLE CONCURRENT BLOCKS
#define NUM_THREADS 1024
#define MAX_BLOCKS 16					// MAXIMUM NUMBER OF BLOCKS TO BE ALLOCATED FOR WORKERS
#define PARENT_BLOCK_SIZE 16
#define DIFFICULTY_LIMIT 32

//#define TARGET_DIFFICULTY 256
//#define TARGET_DIFFICULTY 1024
int TARGET_DIFFICULTY = 1;

#define TARGET_BLOCKS DIFFICULTY_LIMIT*TARGET_DIFFICULTY
//#define TARGET_BLOCKS 2

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

//#define WORKER_BLOCKS ((MULTILEVEL == 1) ? MAX_BLOCKS: AVAILABLE_BLOCKS)/NUM_WORKERS
#define WORKER_BLOCKS MAX_BLOCKS/NUM_WORKERS
#define PARENT_BLOCKS AVAILABLE_BLOCKS-MAX_BLOCKS



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
__host__ void testHash(BYTE * test_str, BYTE * correct_str, BYTE * test_h, BYTE * test_d, BYTE * result_h, BYTE * result_d, int test_size, int double_hash, char ** logStr);
__host__ void testMiningHash(BYTE * test_str, BYTE * correct_str, BYTE * test_h, BYTE * test_d, BYTE * result_h, BYTE * result_d, int test_size, BYTE diff_pow, char ** logStr);
__host__ void hostBenchmarkTest(int num_workers);
__host__ void miningBenchmarkTest(int num_workers);
__host__ void colorTest(int num_colors, int num_palettes);

// TODO ADD TESTING CORES HERE
/*-----------------------------------------------------------------------------------||------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/************************************************************************MEMORY MANAGEMENT FUNCTIONS************************************************************************/
/***************************************************************************************************************************************************************************/
/*---------------------------------------------------------------------------MEMORY ALLOCATION-----------------------------------------------------------------------------*/
__host__ void allocWorkerMemory(int num_workers, BYTE ** hash_h, BYTE ** hash_d, BYTE ** block_h, BYTE ** block_d);
__host__ void allocParentMemory(BYTE ** pHash_h, BYTE ** pHash_d, BYTE ** pBlock_h, BYTE ** pBlock_d, BYTE ** pRoot_h, BYTE ** pRoot_d, BYTE ** pHash_out_h, BYTE ** pHash_out_d, BYTE ** pHash_merkle_h, BYTE ** pHash_merkle_d);
__host__ void allocMiningMemory(BYTE ** target_h, BYTE ** target_d, BYTE ** nonce_h, BYTE ** nonce_d, int ** flag_d);
__host__ void allocFileStrings(char * str[], int num_workers);
/*----------------------------------------------------------------------------MEMORY FREEING-------------------------------------------------------------------------------*/
__host__ void freeWorkerMemory(int num_workers, BYTE ** hash_h, BYTE ** hash_d, BYTE ** block_h, BYTE ** block_d);
__host__ void freeParentMemory(BYTE ** pHash_h, BYTE ** pHash_d, BYTE ** pBlock_h, BYTE ** pBlock_d, BYTE ** pRoot_h, BYTE ** pRoot_d, BYTE ** pHash_out_h, BYTE ** pHash_out_d, BYTE ** pHash_merkle_h, BYTE ** pHash_merkle_d);
__host__ void freeMiningMemory(BYTE ** target_h, BYTE ** target_d, BYTE ** nonce_h, BYTE ** nonce_d, int ** flag_d);
__host__ void freeFileStrings(char * str[], int num_workers);
/*-------------------------------------------------------------------------CUDA/TIMING MANAGEMENT--------------------------------------------------------------------------*/
__host__ void createCudaVars(cudaEvent_t * timing1, cudaEvent_t * timing2, cudaStream_t * stream);
__host__ void destroyCudaVars(cudaEvent_t * timing1, cudaEvent_t * timing2, cudaStream_t * stream);
__host__ void initTime(cudaStream_t * tStream, WORD ** time_h);
__host__ void freeTime(cudaStream_t * tStream, WORD ** time_h);
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/***********************************************************************MINING MANAGEMENT FUNCTIONS*************************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------MINING INITIALIZATION---------------------------------------------------------------------------*/
__host__ void initializeBlockHeader(BYTE * block, BYTE * version, BYTE * prevBlock, BYTE * merkleRoot, BYTE * time_b, BYTE * target, BYTE * nonce);
__host__ void initializeWorkerBlocks(BYTE ** hash_h, BYTE ** block_h, int num_workers);
__host__ void initializeParentBlock(BYTE * pBlock_h);
/*-----------------------------------------------------------------------------MINING UPDATES------------------------------------------------------------------------------*/
__host__  int updateBlock(FILE * inFile, BYTE * block_h, BYTE * hash_h);
__host__ void updateParentRoot(BYTE * block_h, BYTE * hash_h);
__host__ void updateParentHash(BYTE * block_h, BYTE * hash_h);
__host__ void updateDifficulty(BYTE * block_h, int diff_level);
__host__ void updateTime(cudaStream_t * tStream, WORD * time_h, DOMAIN_HANDLE prof_handle);
/*-----------------------------------------------------------------------------MINING GETTERS------------------------------------------------------------------------------*/
__host__ void getTime(BYTE * byte_time);
__host__ void getDifficulty(BYTE * block_h, BYTE ** target, int * target_length, double * difficulty, int worker_num);
// VARIANT TO GET TARGET AS WORDS INSTEAD OF BYTES, REVERSE BYTE ORDER
__host__ void getMiningDifficulty(BYTE * block_h, WORD ** target, int * target_length, double * difficulty, int worker_num);
__host__ void setMiningDifficulty(cudaStream_t * stream, BYTE * block_h, int worker_num);
/*---------------------------------------------------------------------------MINING CALCULATIONS---------------------------------------------------------------------------*/
__host__ double calculateDifficulty(BYTE * bits);
__host__ int calculateTarget(BYTE * bits, BYTE * target);
__host__ int calculateMiningTarget(BYTE * bits, BYTE * target_bytes, WORD * target);
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/************************************************************************KERNEL MANAGEMENT FUNCTIONS************************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------INPUT GENERATION KERNEL-------------------------------------------------------------------------*/
__host__ void genHashKernel(BYTE ** hash_hf, BYTE ** hash_df, BYTE ** seed_h, BYTE ** seed_d, size_t size_hash, size_t size_seed);
/*----------------------------------------------------------------------------MERKLE TREE KERNEL---------------------------------------------------------------------------*/
__host__ void launchMerkle(cudaStream_t * stream, BYTE ** merkle_d, BYTE ** root_d, BYTE ** merkle_h, BYTE ** root_h, int ** flag_d,  int * flag_h, int buffer_size);
/*------------------------------------------------------------------------------MINING KERNEL------------------------------------------------------------------------------*/
__host__ void launchMiner(int kernel_id, cudaStream_t * stream, BYTE ** block_d, BYTE ** hash_d, BYTE ** nonce_d, BYTE ** block_h, BYTE ** hash_h, BYTE ** nonce_h, BYTE ** target_d, int ** flag_d,  int * flag_h, int * target_length);
__host__ void returnMiner(cudaStream_t * stream, BYTE ** block_d, BYTE ** hash_d, BYTE ** nonce_d, BYTE ** block_h, BYTE ** hash_h, BYTE ** nonce_h);
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/*****************************************************************************UTILITY FUNCTIONS*****************************************************************************/
/***************************************************************************************************************************************************************************/
/*------------------------------------------------------------------------HEX CONVERSION FUNCTIONS-------------------------------------------------------------------------*/
__host__ void encodeHex(BYTE * str, BYTE * hex, int len);
__host__ void decodeHex(BYTE * hex, BYTE * str, int len);
__host__ void printHex(BYTE * hex, int len);
__host__ void printHexFile(FILE * outfile, BYTE * hex, int len);
/*------------------------------------------------------------------------STATUS LOGGING FUNCTIONS-------------------------------------------------------------------------*/
__host__ void printLog(const char* msg);
__host__ void printDebug(const char * msg);
__host__ void printError(const char * msg);
__host__ void logStart(int workerID, int block, BYTE * start_hash);
__host__ int printProgress(int mining_state, int multilevel,int num_workers,int pchain_blocks, int *chain_blocks);
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/*************************************************************************I/O MANAGEMENT FUNCTIONS**************************************************************************/
/***************************************************************************************************************************************************************************/
/*--------------------------------------------------------------------------INPUT FILE FUNCTIONS---------------------------------------------------------------------------*/
__host__ int initializeHashes(FILE ** inFiles, int num_workers, BYTE ** hash_h);
__host__ void initializeInputFile(FILE * inFile, char * filename);
__host__ void printInputFile(BYTE *hash_f, char * filename, int blocks, int threads);
__host__ int readNextHash(FILE * inFile, BYTE * hash_h);
/*--------------------------------------------------------------------------OUTPUT FILE FUNCTIONS--------------------------------------------------------------------------*/
__host__ int initializeOutputs(char * outFiles[], char * out_dir_name, int num_workers);
__host__ int initializeParentOutputs(char * bfilename, char * hfilename);
__host__ void printDifficulty(char* diff_file, int worker_num, double difficulty, float time, int num_blocks);
__host__ void printErrorTime(char* err_file, char *err_msg, float err_time);
__host__ void printOutputFile(char * outFileName, BYTE * block_h, BYTE * hash_f, BYTE * nonce_h, int block, float calc_time, double difficulty, int id, int log_out);
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

/*---------------------------------------------------------------------------HASH TEST FUNCTIONS---------------------------------------------------------------------------*/
__global__ void cudaTest(void);
// FIXME ADD MORE TEST KERNELS HERE
__global__ void benchmarkKernel(BYTE * block_d);
__global__ void miningBenchmarkKernel(BYTE * block_d, BYTE * hash_d, int * flag_d, int * total_iterations);
__global__ void hashTestKernel(BYTE * test_block, BYTE * result_block, int size);
__global__ void hashTestMiningKernel(BYTE * test_block, BYTE * result_block, int * success);

/*-----------------------------------------------------------------------------MINING FUNCTIONS----------------------------------------------------------------------------*/
__global__ void genTestHashes(BYTE * hash_df, BYTE * seed, int num_blocks);
__global__ void minerKernel(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, BYTE * target, int * flag_d, int compare);
__global__ void minerKernel_new(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, int * flag_d, int block_offset, int target_offset);
__global__ void getMerkleRoot(BYTE * pHash_d, BYTE * pRoot_d, int buffer_blocks);
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
__device__ void printHash(BYTE * hash);
__device__ void printBlock(BYTE * hash);
__device__ void printSplitBlock(BYTE * hash, BYTE * split);

/*-----------------------------------------------------------------------------SHA256 FUNCTIONS----------------------------------------------------------------------------*/

__device__ void sha256_transform(SHA256_CTX *ctx, const BYTE data[]);
__device__ void sha256_init(SHA256_CTX *ctx);
__device__ void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);
__device__ void sha256_final(SHA256_CTX *ctx, BYTE hash[]);
__device__ int sha256_final_target(SHA256_CTX *ctx, BYTE hash[], const BYTE target[], const int compare);

/*-----------------------------------------------------------------------OPTIMIZED HASHING FUNCTIONS-----------------------------------------------------------------------*/
__device__ void sha256_mining_transform(WORD state[], WORD m[]);
__device__ void sha256_mining_transform_short(WORD state[], WORD m[]);

__device__ __inline__ void scheduleExpansion(WORD m[]);
__device__ __inline__ void scheduleExpansion_short( WORD m[]);

__device__ int sha256_blockHash_TEST(WORD uniquedata[], BYTE hash[], WORD basedata[], WORD target[]);

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/************************************************************************END FUNCTION DECLARATIONS**************************************************************************/
/***************************************************************************************************************************************************************************/



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
\t  -diff #   \t\t STARTING DIFFICULTY MODIFIER AS AN INTEGER, HIGHER VALUES ARE MORE DIFFICULT [-3 MINIMUM, 0 NORMAL, 26 MAXIMUM] (DEFAULT: -1)\n", argv[0]);
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
//			colorTest(NUM_COLORS, NUM_PALETTES);
    }
    // RUN BENCHMARK TEST FOR DEVICE PERFORMANCE
    if(bench_flag == 1){
      printf("BENCHMARK TESTING SELECTED!!!!!\n");
//      hostBenchmarkTest(NUM_WORKERS);
			printf("\nBLOCK MINING BENCHMARK TESTING:\n");
			miningBenchmarkTest(NUM_WORKERS);
    }
    // START MINING IF DRY RUN IS NOT SELECTED
    if(dry_run == 0){
      // TODO CHECK FOR PROFILER ENABLED, INCLUDE LOGGING OF ENABLED SETTINGS
      hostCoreProcess(NUM_WORKERS, MULTILEVEL);
			cudaDeviceReset();  //
    } else{
      printLog("MINING DISABLED FOR DRY RUN TESTING. NOW EXITING...\n\n");
    }
  }

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
/*----------------------------MAIN VARIABLES-----------------------------*/
    BYTE * block_h[num_workers];    // Host storage for current block
    BYTE * block_d[num_workers];    // Device storage for current block
    BYTE * hash_h[num_workers];     // Host storage for result hash
    BYTE * hash_d[num_workers];     // Device storage for hashes
/*----------------------------CUDA VARIABLES-----------------------------*/
    float timingResults[num_workers];
    cudaEvent_t t1[num_workers], t2[num_workers];
    cudaStream_t streams[num_workers];

    // GET TOTAL TIME WORKER SPENT ON A DIFFICULTY LEVEL
    float diff_timing[num_workers];
    cudaEvent_t diff_t1[num_workers], diff_t2[num_workers];
/*---------------------------IO FILE VARIABLES---------------------------*/
    FILE * inFiles[num_workers];
    char * outFiles[num_workers];
    char worker_diff_file[50];
/*----------------------------MINING VARIABLES---------------------------*/
    BYTE * target_h[num_workers];
    BYTE * target_d[num_workers];
    BYTE * nonce_h[num_workers];
    BYTE * nonce_d[num_workers];
    int * flag_d[num_workers];
		int live_stream[num_workers];

    int chain_blocks[num_workers];
    int diff_level[num_workers];
    int errEOF[num_workers];
    int target_length[num_workers];
    double difficulty[num_workers];

	/*---------------------------MEMORY ALLOCATION---------------------------*/

	    allocWorkerMemory(num_workers, hash_h, hash_d, block_h, block_d);
	    allocFileStrings(outFiles, num_workers);
			for(int i = 0; i < num_workers; i++){
				allocMiningMemory(&target_h[i], &target_d[i], &nonce_h[i], &nonce_d[i], &flag_d[i]);
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
    BYTE * pBlock_h;                // Host storage for current block
    BYTE * pBlock_d;                // Device storage for current block
    BYTE * pRoot_h;                 // Host storage for Merkle Root
    BYTE * pRoot_d;                 // Device storage for Merkle Root
    BYTE * pHash_out_h;             // Host storage for result hash
    BYTE * pHash_out_d;             // Device storage for hashes
    BYTE * pHash_merkle_h;          // Host storage for Merkle Tree
    BYTE * pHash_merkle_d;          // Device storage for Merkle Tree
    BYTE * pHash_h[PARENT_BLOCK_SIZE]; // Host buffer for parent hashes
    BYTE * pHash_d[PARENT_BLOCK_SIZE]; // Device buffer for worker hashes
/*-------------------------PARENT CUDA VARIABLES--------------------------*/
    float ptimingResults;
    cudaEvent_t p1, p2;
    cudaStream_t pStream;

    // GET TOTAL TIME PARENT SPENT ON A DIFFICULTY LEVEL
    float pdiff_timing;
    cudaEvent_t diff_p1, diff_p2;

    // GET TIME NEEDED TO CREATE EACH PARENT BLOCK
    float pbuff_timing;
    double pbuff_diffSum = 0;
    cudaEvent_t buff_p1, buff_p2;
/*------------------------PARENT IO FILE VARIABLES-------------------------*/
    char bfilename[50];
    char hfilename[50];
/*------------------------PARENT MINING VARIABLES--------------------------*/
    BYTE * ptarget_h;
    BYTE * ptarget_d;
    BYTE * pnonce_h;
    BYTE * pnonce_d;
    int * pflag_d;
    int * flag_h;
		int live_pstream;

    int worker_record[PARENT_BLOCK_SIZE];
    double pdifficulty;
    int ptarget_length;
    int parentFlag=0;
    int pbuffer_blocks=0;
    int pchain_blocks=0;
    int pdiff_level = 1;
/*-----------------------------------------------------------------------*/
/**************************PARENT INITIALIZATION**************************/
    if(multilevel == 1){
					createCudaVars(&p1, &p2, &pStream);
					NAME_STREAM(pStream, stream_name);
      /*---------------------------MEMORY ALLOCATION---------------------------*/
          allocParentMemory(pHash_h, pHash_d, &pBlock_h, &pBlock_d, &pRoot_h, &pRoot_d, &pHash_out_h, &pHash_out_d, &pHash_merkle_h, &pHash_merkle_d);
					allocMiningMemory(&ptarget_h, &ptarget_d, &pnonce_h, &pnonce_d, &pflag_d);
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
initializeHashes(inFiles, num_workers, hash_h);
initializeWorkerBlocks(hash_h, block_h, num_workers);
initializeOutputs(outFiles, out_location, num_workers);
sprintf(worker_diff_file, "%s/workerDiffScale.txt",out_location);
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

    flag_h = (int *)malloc(sizeof(int));
    flag_h[0] = 0;
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
				createCudaVars(&t1[i], &t2[i], &streams[i]);
				NAME_STREAM(streams[i], stream_name);

        chain_blocks[i] = 0; diff_level[i] = 1; errEOF[i] = 0;
				sprintf(stream_name, "WORKER_%i", i);
				NAME_STREAM(streams[i], stream_name);
				live_stream[i] = 1;
        cudaEventCreate(&diff_t1[i]);
        cudaEventCreate(&diff_t2[i]);

        getDifficulty(block_h[i], &target_h[i], &target_length[i], &difficulty[i], i+1);
				setMiningDifficulty(&(streams[i]), block_h[i], i+1);
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
			initializeParentBlock(pBlock_h);
			initializeParentOutputs(bfilename, hfilename);
	/*------------------------CHAIN INITIALIZATION---------------------------*/
			sprintf(stream_name, "PARENT");
			NAME_STREAM(pStream, stream_name);
			live_pstream = 1;
			cudaEventCreate(&diff_p1);
			cudaEventCreate(&diff_p2);
			cudaEventCreate(&buff_p1);
			cudaEventCreate(&buff_p2);

			getDifficulty(pBlock_h, &ptarget_h, &ptarget_length, &pdifficulty, 0);
			setMiningDifficulty(&pStream, pBlock_h, 0);
			cudaMemcpyAsync(ptarget_d, ptarget_h, HASH_SIZE, cudaMemcpyHostToDevice, pStream);
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
        logStart(i, 1, hash_h[i]);
        cudaEventRecord(t1[i], streams[i]);
        cudaEventRecord(diff_t2[i], streams[i]);
        cudaEventRecord(diff_t1[i], streams[i]);
        launchMiner(i+1, &streams[i], &block_d[i],  &hash_d[i], &nonce_d[i], &block_h[i], &hash_h[i], &nonce_h[i], &target_d[i], &flag_d[i], flag_h, &target_length[i]);
        // SET EVENT TO RECORD AFTER KERNEL COMPLETION (BLOCKS OTHER STREAMS IF TOO MANY ARE SET)
//				printf("LAUNCHED MINER %i\n", i);
				cudaEventRecord(t2[i], streams[i]);
//				printf("SET EVENT %i\n", i);

				POP_DOMAIN(w_handle[i]); // POP START
				PUSH_DOMAIN(w_handle[i], "B", i, 2, 5);  // START BLOCKS
				// TODO START DIFFICULTY RANGE & BLOCK COUNT HERE
    }
    // START PARENT TIMERS
    if(multilevel == 1){
      cudaEventRecord(buff_p2, pStream);
      cudaEventRecord(buff_p1, pStream);
      cudaEventRecord(diff_p1, pStream);
    }
		POP_DOMAIN(t_handle); // END STREAM INITIALIZATION
    cudaEventRecord(g_time[1], g_timeStream);
		PUSH_DOMAIN(t_handle, "MINING", -2, 2, 5); // START MINING LOOP
    /*--------------------------------------------------------------------------------------------------------------------------------*/
    /********************************************BEGIN MINING UNTIL TARGET BLOCKS ARE FOUND********************************************/
    int block_total = 0;
    while(block_total < TARGET_BLOCKS || PROC_REMAINING != 0){
      updateTime(&tStream, time_h, t_handle);
      if(MINING_PROGRESS == 1){
        mining_state = printProgress(mining_state, multilevel, num_workers, pchain_blocks, chain_blocks);
      }
      // SET FLAG_TARGET TO 1
      if(block_total >= TARGET_BLOCKS && FLAG_TARGET == 0){
          FLAG_TARGET = 1;

					// END MINING SECTION, MOVE ON TO FINAL HASH
					for(int i = 0; i < num_workers; i++){
						POP_DOMAIN(w_handle[i]); // POP BLOCKS, REPLACE WITH FINAL
						PUSH_DOMAIN(w_handle[i], "FINAL", i, 2, 6);  // START FINAL MINING
					}
					POP_DOMAIN(t_handle); // END MINING LOOP
          cudaEventRecord(g_time[2], g_timeStream);
					PUSH_DOMAIN(t_handle, "FINAL", -2, 2, 6); // START FINAL LOOP


          printLog("\n\n**********************************************\nTARGET REACHED, FINISHING REMAINING PROCESSES*\n**********************************************\n\n");
      }
      /*--------------------------------------------------------------------------------------------------------------------------------*/
      /*******************************************LOOP OVER MINERS TO CHECK STREAM COMPLETION********************************************/
      for(int i = 0; i < num_workers; i++){

        if(multilevel == 1){  // CHECK PARENT MINER COMPLETION STATUS IF MULTILEVEL
					if(live_pstream == 1){ // MAKE SURE PARENT STREAM IS ALIVE BEFORE CHECKING IT
	          if(cudaStreamQuery(pStream) == 0 && parentFlag == 1){   // PARENT CHAIN RESULTS ARE READY, PROCESS OUTPUTS AND PRINT
	            // processParent
	            pchain_blocks++;
	            returnMiner(&pStream, &pBlock_d,  &pHash_out_d, &pnonce_d, &pBlock_h,  &pHash_out_h, &pnonce_h);
	            cudaEventSynchronize(p2);
	            cudaEventElapsedTime(&ptimingResults, p1, p2);
	            printOutputFile(bfilename, &pBlock_h[0], pHash_out_h, pnonce_h, pchain_blocks, ptimingResults, pdifficulty, -1, 1);
	            updateParentHash(pBlock_h, pHash_out_h);
	            parentFlag = 0;
							POP_DOMAIN(p_handle); // POP THE PREVIOUS BLOCK
	          }
	          // PARENT CHAIN IS STILL PROCESSING LAST BLOCK, WAIT FOR COMPLETION
	          else if(parentFlag == 1 && pbuffer_blocks == PARENT_BLOCK_SIZE){
	                cudaError_t pErr = cudaStreamQuery(pStream);
	                char alert_buf_full[1000];
	                char alert_start[150] = "\n***********************************************************************\nALERT: PARENT BUFFER IS FULL AND PREVIOUS BLOCK IS NOT YET FINISHED!!!*\n";
	                char alert_end[150] = "BLOCKING UNTIL MINING RESOURCES ARE AVAILABLE...                      *\n***********************************************************************\n";
	                sprintf(alert_buf_full, "%sPARENT STREAM STATUS: [CODE: %i]:(%s: %s)*\n%s", alert_start, pErr, cudaGetErrorName(pErr), cudaGetErrorString(pErr), alert_end);
	                printDebug(alert_buf_full);
	                cudaEventRecord(errStart, errStream);
	                cudaEventRecord(buff_p2, pStream);
	                for(int j = 0; j < num_workers; j++){
	                  cudaEventSynchronize(diff_t2[i]);
	                  cudaEventElapsedTime(&diff_timing[i], diff_t1[i], diff_t2[i]);
	                }
	                // WAIT FOR PARENT TO FINISH, THEN RETRIEVE RESULTS
	                while(cudaStreamQuery(pStream) != 0){
	                  updateTime(&tStream, time_h, t_handle);
	                  if(MINING_PROGRESS == 1){
	                    mining_state = printProgress(mining_state, multilevel, num_workers, pchain_blocks, chain_blocks);
	                  }
	                  // MONITOR WORKER TIMING WHILE WAITING
	                  for(int j = 0; j < num_workers; j++){
											if(live_stream[i] == 1){ // ONLY CHECK LIVING WORKERS
		                    if((cudaStreamQuery(streams[i]) == cudaSuccess && diff_timing[i] <= 0) && (chain_blocks[i] >= diff_level[i] * DIFFICULTY_LIMIT || FLAG_TARGET == 1)){
		                        cudaEventRecord(diff_t2[i], streams[i]);
		                        cudaEventSynchronize(diff_t2[i]);
		                        cudaEventElapsedTime(&diff_timing[i], diff_t1[i], diff_t2[i]);
		                    }
											}
	                  }
	                }
	                cudaEventRecord(errFinish, errStream);
	                cudaStreamSynchronize(errStream);
	                cudaEventElapsedTime(&err_time, errStart, errFinish);
	                printErrorTime(error_filename, (char*)"PARENT BUFFER IS FULL AND PREVIOUS BLOCK IS NOT YET FINISHED!!!", err_time);

	                pchain_blocks++;
	                returnMiner(&pStream, &pBlock_d,  &pHash_out_d, &pnonce_d, &pBlock_h,  &pHash_out_h, &pnonce_h);
	                cudaEventSynchronize(p2);
	                cudaEventElapsedTime(&ptimingResults, p1, p2);
	                printOutputFile(bfilename, &pBlock_h[0], pHash_out_h, pnonce_h, pchain_blocks, ptimingResults, pdifficulty, -1, 1);
	                updateParentHash(pBlock_h, pHash_out_h);
	                parentFlag = 0;
									POP_DOMAIN(p_handle); // POP THE PREVIOUS BLOCK
	          }
	          // PARENT BUFFER IS READY, EXIT FOR LOOP TO BEGIN PARENT EXECUTION
	          if(pbuffer_blocks == PARENT_BLOCK_SIZE){
	              printDebug("NEW PARENT BLOCK IS READY!\n");
	              break;
	          }
					}
        } // END PARENT CHAIN MONITOR
        // PROCESS WORKER RESULTS AND START NEXT BLOCK IF THE TARGET HAS NOT BEEN MET
				if(live_stream[i] == 1){ // ONLY PROCEED IF THE STREAM ISN'T DEAD
	        if(cudaStreamQuery(streams[i]) == cudaSuccess && errEOF[i] != 1){
	          // UPDATE WORKER COUNTERS
	          chain_blocks[i]++;
	          block_total++;
	          // GET RESULTS AND TIME FOR PRINTING
	          returnMiner(&streams[i], &block_d[i],  &hash_d[i], &nonce_d[i], &block_h[i], &hash_h[i], &nonce_h[i]);
	          cudaEventSynchronize(t2[i]);
	          cudaEventElapsedTime(&timingResults[i], t1[i], t2[i]);
	          printOutputFile(outFiles[i], block_h[i], hash_h[i], nonce_h[i], chain_blocks[i], timingResults[i], difficulty[i], i, 1);
	          // PRINT TO PARENT HASH FILE AND ADD RESULTS TO PARENT BUFFER IF MULTILEVEL
						POP_DOMAIN(w_handle[i]); // POP CURRENT BLOCK

						if(multilevel == 1){
	            printOutputFile(hfilename, block_h[i], hash_h[i], nonce_h[i], chain_blocks[i], timingResults[i], difficulty[i], i, 0);
	            // COPY HASH TO THE PARENT BUFFER
	            for(int j = 0; j < 32; j++){
	              pHash_h[pbuffer_blocks][j] = hash_h[i][j];
	            }
	            worker_record[pbuffer_blocks] = i+1;
	            pbuff_diffSum+=difficulty[i];
	            pbuffer_blocks++;
	          }
	          // INCREMENT DIFFICULTY IF THE LIMIT HAS BEEN REACHED (PRINT IF TARGET HAS BEEN REACHED)
	          if(chain_blocks[i] >= diff_level[i] * DIFFICULTY_LIMIT || FLAG_TARGET == 1){
	            // PRINT DIFFICULTY BLOCK STATISTICS
	            cudaEventSynchronize(diff_t2[i]);
	            cudaEventElapsedTime(&diff_timing[i], diff_t1[i], diff_t2[i]);
	            if(diff_timing[i] <= 0){ // DIFF TIMER NOT YET RECORDED, RECORD EVENT NOW
	              cudaEventRecord(diff_t2[i], streams[i]);
	              cudaEventSynchronize(diff_t2[i]);
	              cudaEventElapsedTime(&diff_timing[i], diff_t1[i], diff_t2[i]);
	            }
	    //        cudaEventRecord(diff_t2[i], streams[i]);
	    //        cudaEventSynchronize(diff_t2[i]);
	    //        cudaEventElapsedTime(&diff_timing[i], diff_t1[i], diff_t2[i]);
	            printDifficulty(outFiles[i], i+1, difficulty[i], diff_timing[i], (chain_blocks[i]-(diff_level[i]-1)*DIFFICULTY_LIMIT));
	            // INCREMENT IF TARGET HASN'T BEEN REACHED
	            if(FLAG_TARGET == 0){
								POP_DOMAIN(w_handle[i]); // POP CURRENT DIFF
	              updateDifficulty(block_h[i], diff_level[i]);
	              getDifficulty(block_h[i], &target_h[i], &target_length[i], &difficulty[i], i+1);
								setMiningDifficulty(&(streams[i]), block_h[i], i+1);
	              cudaMemcpyAsync(target_d[i], target_h[i], HASH_SIZE, cudaMemcpyHostToDevice, streams[i]);
	              cudaEventRecord(diff_t1[i], streams[i]);
	              diff_level[i]++;
								PUSH_DOMAIN(w_handle[i], "DIFF", i, 2, 5);  // START NEW DIFF
	            }
	          }

	          // MINE NEXT BLOCK ON THIS WORKER IF TARGET HASN'T BEEN REACHED
	          if(FLAG_TARGET == 0){
							PUSH_DOMAIN(w_handle[i], "B", i, 2, 5);  // START NEXT BLOCK
	            errEOF[i] = updateBlock(inFiles[i], block_h[i], hash_h[i]);
	            if(errEOF[i] == 1){
	              char eof_str[20];
	              sprintf(eof_str, "WORKER %i INPUT EOF!", i+1);
	              printErrorTime(error_filename, eof_str, 0.0);
	            }
	            logStart(i, chain_blocks[i]+1, hash_h[i]);
	            cudaEventRecord(t1[i], streams[i]);
	            launchMiner(i+1, &streams[i], &block_d[i],  &hash_d[i], &nonce_d[i], &block_h[i], &hash_h[i], &nonce_h[i], &target_d[i],  &flag_d[i], flag_h, &target_length[i]);
	            cudaEventRecord(t2[i], streams[i]);
	          } else{ // EXECUTION COMPLETED, DELETE CUDA VARS TO PREVENT ADDITIONAL ENTRY INTO THIS CASE
							destroyCudaVars(&t1[i], &t2[i], &streams[i]);

							// END WORKER FINAL, START CLEANUP
							POP_DOMAIN(w_handle[i]); // POP DIFF
							POP_DOMAIN(w_handle[i]); // POP MINING
							PUSH_DOMAIN(w_handle[i], "CLEAN", i, 2, 9);  // START WORKER MINING

							live_stream[i] = 0; // INDICATES THAT THE STREAM IS DEAD, DONT CHECK IT ANY MORE!!
	            cudaEventDestroy(diff_t1[i]);
	            cudaEventDestroy(diff_t2[i]);
	            PROC_REMAINING--;
	          }
	        }
				}
      } // FOR LOOP END
      /*--------------------------------------------------------------------------------------------------------------------------------*/
      /**********************************************START PARENT MINING WHEN BUFFER IS FULL*********************************************/
      // PROC_REMAINING == 1 INDICATES THAT THIS IS THE FINAL ITERATION, MUST BE AT LEAST 1 BLOCK IN BUFFER FROM PRIOR WORKER BLOCKS
      if((multilevel == 1 && parentFlag == 0) && (pbuffer_blocks == PARENT_BLOCK_SIZE || PROC_REMAINING == 1)){
    //    if(pbuffer_blocks > 0){
          // COPY IN THE CURRENT BUFFER CONTENTS
          char merkle_debug[50+PARENT_BLOCK_SIZE*100];
          char hash_entry[80];
          BYTE temp_hash[65];

          sprintf(merkle_debug, "PARENT BLOCK %i CONTENTS:  \n", pchain_blocks+1);
          for(int i = 0; i < pbuffer_blocks; i++){
            for(int j = 0; j < 32; j++){
              pHash_merkle_h[i*32 + j] = pHash_h[i][j];
            }
            decodeHex(&(pHash_merkle_h[i*32]), temp_hash, 32);
            sprintf(hash_entry, "WORKER %i\t%s\n", worker_record[i], (char*)temp_hash);
            strcat(merkle_debug, hash_entry);
          }
          // PRINT PARENT BLOCK CONTENTS
          printDebug(merkle_debug);
          // PARENT DIFFICULTY SCALING
          if(pchain_blocks >= pdiff_level * DIFFICULTY_LIMIT){ // Increment difficulty
						POP_DOMAIN(p_handle); // POP THE PREVIOUS DIFFICULTY
            cudaEventRecord(diff_p2, pStream);
            cudaEventSynchronize(diff_p2);
            cudaEventElapsedTime(&pdiff_timing, diff_p1, diff_p2);
            printDifficulty(bfilename, -1, pdifficulty, pdiff_timing, (pchain_blocks-(pdiff_level-1)*DIFFICULTY_LIMIT));

            updateDifficulty(pBlock_h, pdiff_level);
            getDifficulty(pBlock_h, &ptarget_h, &ptarget_length, &pdifficulty, 0);
						setMiningDifficulty(&pStream, pBlock_h, 0);
            cudaMemcpyAsync(ptarget_d, ptarget_h, HASH_SIZE, cudaMemcpyHostToDevice, pStream);
            cudaEventRecord(diff_p1, pStream);
            pdiff_level++;
						PUSH_DOMAIN(p_handle, "DIFF", -1, 0, 5); // PUSH NEW DOMAIN
          }
					PUSH_DOMAIN(p_handle, "B", -1, 2, 5); // START NEXT BLOCK
          cudaEventRecord(p1, pStream);
          launchMerkle(&pStream, &pHash_merkle_d,  &pRoot_d, &pHash_merkle_h,  &pRoot_h, &pflag_d, flag_h, pbuffer_blocks);
          updateParentRoot(pBlock_h, pRoot_h);

          cudaEventSynchronize(buff_p2);
          cudaEventElapsedTime(&pbuff_timing, buff_p1, buff_p2);
          if(pbuff_timing <= 0){ // NEW BUFFER TIMER NOT YET RECORDED, RECORD EVENT NOW
            cudaEventRecord(buff_p2, pStream);
            cudaEventSynchronize(buff_p2);
            cudaEventElapsedTime(&pbuff_timing, buff_p1, buff_p2);
          }

//          cudaEventRecord(buff_p2, pStream);
//          cudaEventSynchronize(buff_p2);
//          cudaEventElapsedTime(&pbuff_timing, buff_p1, buff_p2);
          pbuff_diffSum /= pbuffer_blocks;
          printDifficulty(hfilename, 0, pbuff_diffSum, pbuff_timing, pbuffer_blocks);
          pbuff_diffSum = 0;
          cudaEventRecord(buff_p1, pStream);

          logStart(-1, pchain_blocks+1, pRoot_h);
          launchMiner(0, &pStream, &pBlock_d,  &pHash_out_d, &pnonce_d, &pBlock_h,  &pHash_out_h, &pnonce_h, &ptarget_d, &pflag_d, flag_h, &ptarget_length);
          cudaEventRecord(p2, pStream);
          pbuffer_blocks = 0;
          parentFlag = 1;

          // FINAL ITERATION, WAIT FOR PARENT STREAM TO FINISH
          if(PROC_REMAINING == 1){
            while(cudaStreamQuery(pStream) != 0){
              updateTime(&tStream, time_h, t_handle);
              if(MINING_PROGRESS == 1){
                mining_state = printProgress(mining_state, multilevel, num_workers, pchain_blocks, chain_blocks);
              }
            }
            pchain_blocks++;
            returnMiner(&pStream, &pBlock_d,  &pHash_out_d, &pnonce_d, &pBlock_h,  &pHash_out_h, &pnonce_h);
            cudaEventSynchronize(p2);
            cudaEventElapsedTime(&ptimingResults, p1, p2);
            printOutputFile(bfilename, &pBlock_h[0], pHash_out_h, pnonce_h, pchain_blocks, ptimingResults, pdifficulty, -1, 1);
            updateParentHash(pBlock_h, pHash_out_h);
            parentFlag = 0;
						POP_DOMAIN(p_handle); // POP THE PREVIOUS BLOCK

            cudaEventRecord(diff_p2, pStream);
            cudaEventSynchronize(diff_p2);
            cudaEventElapsedTime(&pdiff_timing, diff_p1, diff_p2);
            printDifficulty(bfilename, -1, pdifficulty, pdiff_timing, (pchain_blocks-(pdiff_level-1)*DIFFICULTY_LIMIT));
            // CLEAN UP PARENT CUDA VARS, SET PROCS_REMAINING TO ZERO TO EXIT
            destroyCudaVars(&p1, &p2, &pStream);

						// FINISH PARENT, MOVE ON TO CLEANUP
						POP_DOMAIN(p_handle); //POP DIFF
						POP_DOMAIN(p_handle); //POP MINING
						PUSH_DOMAIN(p_handle, "CLEAN", -1, 2, 9);

						live_pstream = 0;
            cudaEventDestroy(diff_p1);
            cudaEventDestroy(diff_p2);
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
    freeFileStrings(outFiles, num_workers);
    destroyCudaVars(&errStart, &errFinish, &errStream);
    for(int i = 0; i < num_workers; i++){
      fclose(inFiles[i]);
    }
    /*--------------------------------------------------------------------------------------------------------------------------------*/
    /*******************************************************FREE MINING VARIABLES******************************************************/
    printDebug((const char*)"FREEING MINING MEMORY");
    free(flag_h);
    freeTime(&tStream, &time_h);
    if(multilevel == 1){
      freeMiningMemory(&ptarget_h, &ptarget_d, &pnonce_h, &pnonce_d, &pflag_d);
    }
    for(int i = 0; i < num_workers; i++){
      freeMiningMemory(&target_h[i], &target_d[i], &nonce_h[i], &nonce_d[i], &flag_d[i]);
    }
    /*--------------------------------------------------------------------------------------------------------------------------------*/
    /*************************************************FREE PARENT AND WORKER VARIABLES*************************************************/
    printDebug((const char*)"FREEING WORKER MEMORY");
    freeWorkerMemory(num_workers, hash_h, hash_d, block_h, block_d);

		// DESTROY WORKER PROFILING DOMAINS
		for(int i = 0; i < num_workers; i++){
			POP_DOMAIN(w_handle[i]);POP_DOMAIN(w_handle[i]);
			DOMAIN_DESTROY(w_handle[i]);
		}
		if(multilevel == 1){
      printDebug((const char*)"FREEING PARENT MEMORY");
      freeParentMemory(pHash_h, pHash_d, &pBlock_h, &pBlock_d, &pRoot_h, &pRoot_d, &pHash_out_h, &pHash_out_d, &pHash_merkle_h, &pHash_merkle_d);

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

  return;
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*******************************************************************************TEST FUNCTIONS******************************************************************************/
__host__ void hostFunctionalTest(void){
  printf("STARTING FUNCTIONAL TEST\n");
	// INITIALIZE BENCHMARK VARIABLES
  BYTE * test_h;
  BYTE * test_d;
	BYTE * result_h;
	BYTE * result_d;
	// STORE DIFF_REDUCE TO BE SET LATER
	int temp_reduce = DIFF_REDUCE;
	DIFF_REDUCE = 0;

	BYTE test_str[161];
	BYTE correct_str[65];

	int logSize = 500;
  char logResult[4000];
	char * logStr;
	char logMsg[logSize];

  // Allocate test block memory
  test_h = (BYTE *)malloc(BLOCK_SIZE);
  cudaMalloc((void **) &test_d, BLOCK_SIZE);
	result_h = (BYTE *)malloc(HASH_SIZE);
	cudaMalloc((void **) &result_d, HASH_SIZE);

	// Prepare logging variables
  logStr = (char*)malloc(sizeof(char) * logSize);
	strcpy(logResult, "\n****************************HASHING FUNCTIONAL TESTS****************************\n");

	// INITIALIZE TEST PROFILING DOMAIN
	#ifdef USE_NVTX
		DOMAIN_HANDLE handle;
	#endif
	DOMAIN_CREATE(handle, "FUNCTIONAL TESTS");

	// HASH TESTS
	// Simple input 'abcd'
	PUSH_DOMAIN(handle, "SIMPLE TEST", -2, 0, 0);
		strcpy((char*)test_str, "61626364");
		strcpy((char*)correct_str, "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589");
		testHash(test_str, correct_str, test_h, test_d, result_h, result_d, 4, 0, &logStr);
		sprintf(logMsg, "BASIC TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
		strcat(logResult, logMsg);
	POP_DOMAIN(handle);

	// 32 BYTE MESSAGE
	PUSH_DOMAIN(handle, "32B TEST", -2, 1, 1);
		strcpy((char*)test_str, "1979507de7857dc4940a38410ed228955f88a763c9cccce3821f0a5e65609f56");
		strcpy((char*)correct_str, "928e8c1f694fc888316690b3c05573c226785344941bed6016909aefb07ecb6d");
		testHash(test_str, correct_str, test_h, test_d, result_h, result_d, 32, 0, &logStr);
		sprintf(logMsg, "32-BYTE TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
		strcat(logResult, logMsg);
	POP_DOMAIN(handle);

	// 64 BYTE MESSAGE
	PUSH_DOMAIN(handle, "64B TEST", -2, 2, 2);
		strcpy((char*)test_str, "0100000000000000000000000000000000000000000000000000000000000000000000001979507de7857dc4940a38410ed228955f88a763c9cccce3821f0a5e");
		strcpy((char*)correct_str, "8e8ce198ef7f22243d9ed05b336b49a8051003a45c5e746ae2d7965d9d93b072");
		testHash(test_str, correct_str, test_h, test_d, result_h, result_d, 64, 0, &logStr);
		sprintf(logMsg, "64-BYTE TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
		strcat(logResult, logMsg);
	POP_DOMAIN(handle);

	// 80 BYTE MESSAGE
	PUSH_DOMAIN(handle, "80B TEST", -2, 3, 3);
		strcpy((char*)test_str, "0100000000000000000000000000000000000000000000000000000000000000000000001979507de7857dc4940a38410ed228955f88a763c9cccce3821f0a5e65609f565c2ffb291d00ffff01004912");
		strcpy((char*)correct_str, "c45337946ef4402f6bf49e03039ca5d1dcf5edb5f885110fdb3f2e690d2ccb35");
		testHash(test_str, correct_str, test_h, test_d, result_h, result_d, 80, 0, &logStr);
		sprintf(logMsg, "BLOCK TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
		strcat(logResult, logMsg);
	POP_DOMAIN(handle);

	// 80 BYTE MESSAGE (DOUBLE HASH)
	PUSH_DOMAIN(handle, "80B MINING TEST", -2, 4, 4);
		PUSH_DOMAIN(handle, "DEFAULT HASH", -2, 4, 0);
			strcpy((char*)test_str, "0100000000000000000000000000000000000000000000000000000000000000000000001979507de7857dc4940a38410ed228955f88a763c9cccce3821f0a5e65609f565c2ffb291d00ffff01004912");
			strcpy((char*)correct_str, "265a66f42191c9f6b26a1b9d4609d76a0b5fdacf9b82b6de8a3b3e904f000000");
			testHash(test_str, correct_str, test_h, test_d, result_h, result_d, 80, 1, &logStr);
			sprintf(logMsg, "DOUBLE HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		// NEW DOUBLE HASH FUNCTION
		PUSH_DOMAIN(handle, "ACCEL HASH", -2, 4, 1);
			testMiningHash(test_str, correct_str, test_h, test_d, result_h, result_d, 80, 0x1e, &logStr);
			sprintf(logMsg, "NEW DOUBLE HASH TEST: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
	POP_DOMAIN(handle);




	// VARIOUS DIFFICULTIES TEST
	PUSH_DOMAIN(handle, "DIFFICULTY TEST", -2, 5, 5);
		// 2 ZEROS (DIFFICULTY: 0x2000ffff)
		PUSH_DOMAIN(handle, "D=0x2000ffff", -2, 5, 0);
			strcpy((char*)test_str, "01000000a509fafcf42a5f42dacdf8f4fb89ff525c0ee3acb0d68ad364f2794f2d8cd1007d750847aac01636528588e2bccccb01a91b0b19524de666fdfaa4cfad669fcd5c39b1141d00ffff00005cc0");
			strcpy((char*)correct_str, "d1bca1de492c24b232ee591a1cdf16ecd8c51400d4da49a97f9536f27b286e00");
			testMiningHash(test_str, correct_str, test_h, test_d, result_h, result_d, 80, 0x20, &logStr);
			sprintf(logMsg, "DIFFICULTY TEST 1 [0x2000ffff]: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		// 4 ZEROS (DIFFICULTY: 0x1f00ffff)
		PUSH_DOMAIN(handle, "D=0x1f00ffff", -2, 5, 1);
			strcpy((char*)test_str, "010000008e2e5fd95b75846393b579f7368ebbee8ca593ed574dd877b4255e1385cd0000286e0824b41e054a6afea14b0b4588017895ace8f9cc4837279074e238462cd75c340d171d00ffff0002043d");
			strcpy((char*)correct_str, "fbbb3f2adadd66d9d86cdacc735f99edece886faed7a0fbc17594da445820000");
			testMiningHash(test_str, correct_str, test_h, test_d, result_h, result_d, 80, 0x1f, &logStr);
			sprintf(logMsg, "DIFFICULTY TEST 2 [0x1f00ffff]: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		// 6 ZEROS (DIFFICULTY: 0x1e00ffff)
		PUSH_DOMAIN(handle, "D=0x1e00ffff", -2, 5, 2);
			strcpy((char*)test_str, "010000000298ff1c6d24d9f04ed441ce3f3a4b695d7fdb8cc13bc7f7417a68a44b000000d49d1c71552793e1d9182ab63ca5fe8d23f2711ecb26f7b0f9ad931c5980aadb5c340d521c00ffff020caca2");
			strcpy((char*)correct_str, "46b26c30b35175ecb88ddbe08f2d56070f616b2d6f302ef334286fc575000000");
			testMiningHash(test_str, correct_str, test_h, test_d, result_h, result_d, 80, 0x1e, &logStr);
			sprintf(logMsg, "DIFFICULTY TEST 3 [0x1e00ffff]: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		// 8 ZEROS (DIFFICULTY: 0x1d00ffff)
		PUSH_DOMAIN(handle, "D=0x1d00ffff", -2, 5, 3);
			strcpy((char*)test_str, "01000000ac44a5ddb3c7a252ab2ea9278ab4a27a5fd88999ff192d5f6e86f66b000000009984a9337cf3852ef758d5f8baf090700c89133ba9c19e27f39b465942d8e7465c3440bd1b00ffffdba51c5e");
			strcpy((char*)correct_str, "30498d768dba64bd6b1455ae358fefa3217096449f05800b61e2e93b00000000");
			testMiningHash(test_str, correct_str, test_h, test_d, result_h, result_d, 80, 0x1d, &logStr);
			sprintf(logMsg, "DIFFICULTY TEST 4 [0x1d00ffff]: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
		// 16 ZEROS (DIFFICULTY: 0x1900ffff)
		PUSH_DOMAIN(handle, "D=0x1900ffff", -2, 5, 4);
			strcpy((char*)test_str, "0100000081cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a308000000000000e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0f1fc122bc7f5d74df2b9441a42a14695");
			strcpy((char*)correct_str, "1dbd981fe6985776b644b173a4d0385ddc1aa2a829688d1e0000000000000000");
			testMiningHash(test_str, correct_str, test_h, test_d, result_h, result_d, 80, 0x19, &logStr);
			sprintf(logMsg, "DIFFICULTY TEST 5 [0x1900ffff]: \nINPUT: %s \n \t%s\n\n", test_str, logStr);
			strcat(logResult, logMsg);
		POP_DOMAIN(handle);
	POP_DOMAIN(handle);

	// DESTROY FUNCTIONAL TEST DOMAIN
	DOMAIN_DESTROY(handle);

	strcat(logResult, "********************************************************************************\n\n");
  printLog(logResult);

	// RETURN DIFF_REDUCE TO ITS ORIGINAL VALUE
	DIFF_REDUCE = temp_reduce;

	free(test_h);
  cudaFree(test_d);
	free(result_h);
	cudaFree(result_d);
  return;

}

__host__ void testHash(BYTE * test_str, BYTE * correct_str, BYTE * test_h, BYTE * test_d, BYTE * result_h, BYTE * result_d, int test_size, int double_hash, char ** logStr){
	BYTE result_str[65];
	BYTE correct_hex[32];
	int hash_match;
	cudaStream_t test_stream;
	cudaStreamCreate(&test_stream);

	// ADD NAME TO STREAM
	char stream_name[50];
	sprintf(stream_name, "TEST STREAM %i", TEST_COUNT);
	TEST_COUNT++;
	NAME_STREAM(test_stream, stream_name);

	memset(test_h, 0, BLOCK_SIZE);
	cudaMemcpyAsync(result_d, test_h, HASH_SIZE, cudaMemcpyHostToDevice, test_stream);
	encodeHex(test_str, test_h, test_size*2);

	cudaMemcpyAsync(test_d, test_h, BLOCK_SIZE, cudaMemcpyHostToDevice, test_stream);
	hashTestKernel<<<1, 1, 0, test_stream>>>(test_d, result_d, test_size);
	cudaMemcpyAsync(result_h, result_d, HASH_SIZE, cudaMemcpyDeviceToHost, test_stream);
	cudaDeviceSynchronize();

	if(double_hash == 1){
		cudaMemcpyAsync(test_d, result_d, HASH_SIZE, cudaMemcpyHostToDevice, test_stream);
		hashTestKernel<<<1, 1, 0, test_stream>>>(test_d, result_d, 32);
		cudaMemcpyAsync(result_h, result_d, HASH_SIZE, cudaMemcpyDeviceToHost, test_stream);
	}
	cudaDeviceSynchronize();
	cudaStreamDestroy(test_stream);

	// Compare results
	decodeHex(result_h, result_str, 32);
	encodeHex(correct_str, correct_hex, 64);
	hash_match = strcmp((char*)result_str, (char*)correct_str);
	if(hash_match == 0){
		strcpy(*logStr, "SUCCESS");
	}else{
		sprintf(*logStr, "FAILED\n \t\tEXPECTED: %s\n \t\tRECEIVED: %s", correct_str, result_str);
	}
	return;
}

__host__ void testMiningHash(BYTE * test_str, BYTE * correct_str, BYTE * test_h, BYTE * test_d, BYTE * result_h, BYTE * result_d, int test_size, BYTE diff_pow, char ** logStr){
	BYTE result_str[65];
	BYTE correct_hex[32];
	int hash_match;
	cudaStream_t test_stream;
	cudaStreamCreate(&test_stream);

	double t_difficulty;
	int t_target_length;
	int * success_h;
	int * success_d;

	success_h = (int*)malloc(sizeof(int));
	cudaMalloc((void **) &success_d, sizeof(int));

	// SET DIFFICULTY (pow 0x1d -> target 00000000FFFF )
	test_h[72] = diff_pow;
	test_h[73] = 0x00;
	test_h[74] = 0xff;
	test_h[75] = 0xff;

	WORD * target_h;
	target_h = (WORD*)malloc(TARGET_C_SIZE);
	getMiningDifficulty(test_h, &target_h, &t_target_length, &t_difficulty, 0);
	printf("TARGET: %08x%08x%08x%08x%08x%08x%08x%08x\n\n", target_h[0], target_h[1], target_h[2], target_h[3], target_h[4], target_h[5], target_h[6], target_h[7]);
	cudaMemcpyToSymbolAsync(test_target_c, target_h, TARGET_C_SIZE, 0, cudaMemcpyHostToDevice, test_stream);

	// ADD NAME TO STREAM
	char stream_name[50];
	sprintf(stream_name, "TEST STREAM %i", TEST_COUNT);
	TEST_COUNT++;
	NAME_STREAM(test_stream, stream_name);

	memset(test_h, 0, BLOCK_SIZE);
	cudaMemcpyAsync(result_d, test_h, HASH_SIZE, cudaMemcpyHostToDevice, test_stream);
	encodeHex(test_str, test_h, test_size*2);

	cudaMemcpyAsync(test_d, test_h, BLOCK_SIZE, cudaMemcpyHostToDevice, test_stream);

	WORD basemsg_h[16];
	// SET SYMBOL FOR BASE BLOCK
	for(int i = 0; i < 16; i++){
		basemsg_h[i] = (test_h[i*4] << 24) | (test_h[i*4+1] << 16) | (test_h[i*4+2] << 8) | (test_h[i*4+3]);
	}
	cudaMemcpyToSymbolAsync(test_basemsg_c, basemsg_h, sizeof(WORD)*16, 0, cudaMemcpyHostToDevice, test_stream);

	hashTestMiningKernel<<<1, 1, 0, test_stream>>>(test_d, result_d, success_d);
	cudaMemcpyAsync(result_h, result_d, HASH_SIZE, cudaMemcpyDeviceToHost, test_stream);
	cudaMemcpyAsync(success_h, success_d, sizeof(int), cudaMemcpyDeviceToHost, test_stream);
	cudaDeviceSynchronize();
	cudaStreamDestroy(test_stream);

	// Compare results
	decodeHex(result_h, result_str, 32);
	encodeHex(correct_str, correct_hex, 64);
	hash_match = strcmp((char*)result_str, (char*)correct_str);
	if(hash_match == 0){
		sprintf(*logStr, "SUCCESS, TARGET MET VALUE: %i", *success_h);
	}else{
		sprintf(*logStr, "FAILED, TARGET MET VALUE: %i\n \t\tEXPECTED: %s\n \t\tRECEIVED: %s", *success_h, correct_str, result_str);
	}

	free(success_h);
	cudaFree(success_d);

	free(target_h);

	return;
}


__host__ void hostBenchmarkTest(int num_workers){
  // INITIALIZE BENCHMARK VARIABLES
  BYTE * test_block_h;
  BYTE * test_block_d;
  char logResult[1000];
  float bench_time, worker_time, block_time, thread_time;
  cudaEvent_t bench_s, bench_f;
  cudaStream_t bench_stream;

  createCudaVars(&bench_s, &bench_f, &bench_stream);

	// INITIALIZE BENCHMARK PROFILING DOMAIN
	char stream_name[50];
	sprintf(stream_name, "BENCHMARK STREAM");
	NAME_STREAM(bench_stream, stream_name);
	#ifdef USE_NVTX
		DOMAIN_HANDLE handle;
	#endif
	DOMAIN_CREATE(handle, "BENCHMARK TEST");
	PUSH_DOMAIN(handle, "BENCHMARK TEST", -2, 0, 0);

  // Allocate test block memory
  test_block_h = (BYTE *)malloc(BLOCK_SIZE);
  cudaMalloc((void **) &test_block_d, BLOCK_SIZE);

  // CREATE RANDOM TEST BLOCK
  srand(time(0));
  for(int i = 0; i < 80; i++){
      test_block_h[i] = (rand() % 255) & 0xFF;
  }

  cudaEventRecord(bench_s, bench_stream);
  cudaMemcpyAsync(test_block_d, test_block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, bench_stream);
  benchmarkKernel<<<WORKER_BLOCKS, NUM_THREADS, 0, bench_stream>>>(test_block_d);
  cudaEventRecord(bench_f, bench_stream);

  cudaDeviceSynchronize();
	POP_DOMAIN(handle);

  cudaEventElapsedTime(&bench_time, bench_s, bench_f);

  worker_time = 0xEFFFFFFF/(bench_time/1000);
  block_time = worker_time/WORKER_BLOCKS;
  thread_time = block_time/NUM_THREADS;

  sprintf(logResult, "\n/****************************BENCHMARK ANALYSIS FOR %i WORKER CHAINS****************************/\n\
  TOTAL TIME: %f\n\
  WORKER HASHES PER SECOND: %f\n\
  BLOCK HASHES PER SECOND: %f \n\
  THREAD HASHES PER SECOND: %f \n\
  /**********************************************************************************************/\n\
  ", num_workers, bench_time, worker_time, block_time, thread_time);
  printLog(logResult);

	// DESTROY BENCHMARK TEST DOMAIN
	DOMAIN_DESTROY(handle);

  destroyCudaVars(&bench_s, &bench_f, &bench_stream);
  free(test_block_h);
  cudaFree(test_block_d);
  return;
}


// TEST FUNCTION FOR IMPROVED MINING KERNEL, WHICH IS ACCELERATED WITH THE USE OF
// PRECOMPUTED BLOCK HASHING CONSTANTS AND LOWER MEMORY USAGE
__host__ void miningBenchmarkTest(int num_workers){
  // INITIALIZE BENCHMARK VARIABLES
  BYTE * test_block_h;
  BYTE * test_block_d;

	BYTE * test_hash_h;
	BYTE * test_hash_d;

	double t_difficulty;
	int t_target_length;
  char logResult[1000];
  float bench_time;
	float worker_time, block_time, thread_time;
  cudaEvent_t bench_s, bench_f;
  cudaStream_t bench_stream;

  createCudaVars(&bench_s, &bench_f, &bench_stream);

	// INITIALIZE BENCHMARK PROFILING DOMAIN
	char stream_name[50];
	sprintf(stream_name, "BENCHMARK STREAM");
	NAME_STREAM(bench_stream, stream_name);
	#ifdef USE_NVTX
		DOMAIN_HANDLE handle;
	#else
		int handle = 0;
	#endif
	DOMAIN_CREATE(handle, "BENCHMARK TEST");
	PUSH_DOMAIN(handle, "BENCHMARK TEST", -2, 0, 0);

  // Allocate test block memory
  test_block_h = (BYTE *)malloc(BLOCK_SIZE);
  cudaMalloc((void **) &test_block_d, BLOCK_SIZE);

	test_hash_h = (BYTE *)malloc(HASH_SIZE);
	cudaMalloc((void **) &test_hash_d, HASH_SIZE);

	// INITIALIZE CONSTANTS FOR USE IN THE MINING KERNEL
	WORD * target_h;
	target_h = (WORD*)malloc(TARGET_C_SIZE);
	WORD * basemsg_h;
	basemsg_h = (WORD *)malloc(BLOCK_C_SIZE);
	int * test_flag;
	cudaMalloc((void **) &test_flag, sizeof(int));


	int * iterations_h;
	int total_iterations = 0;
	iterations_h = (int*)malloc(sizeof(int));
	int * iterations_d;
	cudaMalloc((void **) &iterations_d, sizeof(int));
	WORD * time_h;
	cudaStream_t tStream;
	initTime(&tStream, &time_h);

	cudaEventRecord(bench_s, bench_stream);

	// SET TARGET DIFFICULTY
	test_block_h[72] = 0x1d;
	test_block_h[73] = 0x00;
	test_block_h[74] = 0xff;
	test_block_h[75] = 0xff;
	getMiningDifficulty(test_block_h, &target_h, &t_target_length, &t_difficulty, 0);
	cudaMemcpyToSymbolAsync(test_target_c, target_h, TARGET_C_SIZE, 0, cudaMemcpyHostToDevice, bench_stream);
	srand(time(0));
//	for(int j = 0; j < 10; j++){
		// CREATE RANDOM TEST BLOCK
	  for(int i = 0; i < 80; i++){
	      test_block_h[i] = (rand() % 255) & 0xFF;
	  }
		cudaMemcpyAsync(test_block_d, test_block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, bench_stream);

		// INITIALIZE MESSAGE SCHEDULE WITH CONSTANT BASE BLOCK, NO EXTRA REGISTERS USED!!
		for(int i = 0; i < 16; i++){
			basemsg_h[i] = (test_block_h[i*4] << 24) | (test_block_h[i*4+1] << 16) | (test_block_h[i*4+2] << 8) | (test_block_h[i*4+3]);
		}
		cudaMemcpyToSymbolAsync(test_basemsg_c, basemsg_h, BLOCK_C_SIZE, 0, cudaMemcpyHostToDevice, bench_stream);
		cudaMemsetAsync(test_flag, 0, sizeof(int), bench_stream);
		cudaMemsetAsync(iterations_d, 0, sizeof(int), bench_stream);

	  miningBenchmarkKernel<<<WORKER_BLOCKS, NUM_THREADS, 0, bench_stream>>>(test_block_d, test_hash_d, test_flag, iterations_d);

		// UPDATE TIMING VARIABLE
		while(cudaStreamQuery(bench_stream) != 0){
			updateTime(&tStream, time_h, handle);
		}

		cudaMemcpyAsync(iterations_h, iterations_d, sizeof(int), cudaMemcpyDeviceToHost, bench_stream);
		cudaMemcpyAsync(test_block_h, test_block_d, BLOCK_SIZE, cudaMemcpyDeviceToHost, bench_stream);
		cudaMemcpyAsync(test_hash_h, test_hash_d, HASH_SIZE, cudaMemcpyDeviceToHost, bench_stream);
		total_iterations += *iterations_h;
	//	printf("FINSHED BLOCK %i IN %i ITERATIONS! \n HASH: ", j, *iterations_h);
		printHex(test_hash_h, 32);
		printf("\n\n");
//	}

	cudaEventRecord(bench_f, bench_stream);
  cudaDeviceSynchronize();
	POP_DOMAIN(handle);

	freeTime(&tStream, &time_h);

  cudaEventElapsedTime(&bench_time, bench_s, bench_f);
	printf("TOTAL ITERATIONS PASSED: %i\n", total_iterations);
	printf("WORKER_BLOCKS: %i\n", WORKER_BLOCKS);
	printf("NUM THREADS: %i\n\n", NUM_THREADS);

	long long int all_iterations = 0;
	all_iterations = ((long long int)total_iterations)*((long long int)WORKER_BLOCKS)*((long long int)NUM_THREADS);
	printf("ALL ITERATIONS: %lld \n", all_iterations);

  worker_time = ((all_iterations)/(bench_time*1000));
  block_time = worker_time/WORKER_BLOCKS;
  thread_time = (block_time*1000)/NUM_THREADS;

  sprintf(logResult, "\n/****************************NEW MINING BENCHMARK ANALYSIS FOR %i WORKER CHAINS****************************/\n\
  TOTAL TIME: %f\n\
  WORKER HASHRATE:\t %.3f MH/s\n\
  BLOCK HASHRATE:\t %.3f MH/s\n\
  THREAD HASHRATE:\t %.3f KH/s\n\
  /**********************************************************************************************/\n\
  ", num_workers, bench_time, worker_time, block_time, thread_time);
  printLog(logResult);

	// DESTROY BENCHMARK TEST DOMAIN
	DOMAIN_DESTROY(handle);

  destroyCudaVars(&bench_s, &bench_f, &bench_stream);
  free(test_block_h);
  cudaFree(test_block_d);
	free(test_hash_h);
	cudaFree(test_hash_d);
	free(iterations_h);
	cudaFree(iterations_d);
	cudaFree(test_flag);

	free(target_h);
	free(basemsg_h);
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
__host__ void allocWorkerMemory(int num_workers, BYTE ** hash_h, BYTE ** hash_d, BYTE ** block_h, BYTE ** block_d){
    for(int i = 0; i < num_workers; i++){
      // Allocate hash memory
      hash_h[i] = (BYTE *)malloc(HASH_SIZE);
      cudaMalloc((void **) &hash_d[i], HASH_SIZE);
      // Allocate block memory
      block_h[i] = (BYTE *)malloc(BLOCK_SIZE);
      cudaMalloc((void **) &block_d[i], BLOCK_SIZE);
    }
}
__host__ void allocParentMemory(BYTE ** pHash_h, BYTE ** pHash_d, BYTE ** pBlock_h, BYTE ** pBlock_d, BYTE ** pRoot_h, BYTE ** pRoot_d, BYTE ** pHash_out_h, BYTE ** pHash_out_d, BYTE ** pHash_merkle_h, BYTE ** pHash_merkle_d){
    for(int i = 0; i < PARENT_BLOCK_SIZE; i++){
      // Allocate hash memory
      pHash_h[i] = (BYTE *)malloc(HASH_SIZE);
      cudaMalloc((void **) &pHash_d[i], HASH_SIZE);
    }
    // Allocate parent root hash memory
    *pRoot_h = (BYTE *)malloc(HASH_SIZE);
    cudaMalloc((void **) pRoot_d, HASH_SIZE);
    // Allocate parent block memory
    *pBlock_h = (BYTE *)malloc(BLOCK_SIZE);
    cudaMalloc((void **) pBlock_d, BLOCK_SIZE);
    // Allocate parent output hash memory
    *pHash_out_h = (BYTE *)malloc(HASH_SIZE);
    cudaMalloc((void **) pHash_out_d, HASH_SIZE);
    // Allocate parent merkle hash memory
    *pHash_merkle_h = (BYTE *)malloc(HASH_SIZE*(PARENT_BLOCK_SIZE));
    cudaMalloc((void **) pHash_merkle_d, HASH_SIZE*(PARENT_BLOCK_SIZE));
}
__host__ void allocMiningMemory(BYTE ** target_h, BYTE ** target_d, BYTE ** nonce_h, BYTE ** nonce_d, int ** flag_d){
  // Allocate root hash memory
  *target_h = (BYTE *)malloc(HASH_SIZE);
  cudaMalloc((void **) target_d, HASH_SIZE);
  // Allocate block memory
  *nonce_h = (BYTE *)malloc(NONCE_SIZE);
  cudaMalloc((void **) nonce_d, NONCE_SIZE);
  // Allocate Mining Flag
  cudaMalloc((void **) flag_d, sizeof(int));
}
__host__ void allocFileStrings(char ** str, int num_workers){
  for(int i = 0; i < num_workers; i++){
    str[i] = (char*)malloc(sizeof(char)*50);
  }
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*************************************************************************MEMORY FREEING FUNCTIONS**************************************************************************/
__host__ void freeWorkerMemory(int num_workers, BYTE ** hash_h, BYTE ** hash_d, BYTE ** block_h, BYTE ** block_d){
  for(int i = 0; i < num_workers; i++){
    // Free worker blocks
    free(block_h[i]);
    cudaFree(block_d[i]);
    // Free worker hashes
    free(hash_h[i]);
    cudaFree(hash_d[i]);
  }
}
__host__ void freeParentMemory(BYTE ** pHash_h, BYTE ** pHash_d, BYTE ** pBlock_h, BYTE ** pBlock_d, BYTE ** pRoot_h, BYTE ** pRoot_d,  BYTE ** pHash_out_h, BYTE ** pHash_out_d, BYTE ** pHash_merkle_h, BYTE ** pHash_merkle_d){
  for(int i = 0; i < PARENT_BLOCK_SIZE; i++){
    // Free hash memory
    free(pHash_h[i]);
    cudaFree(pHash_d[i]);
  }
  // Free root hash memory
  free(*pRoot_h);
  cudaFree(*pRoot_d);
  // Free parent block memory
  free(*pBlock_h);
  cudaFree(*pBlock_d);
  // Free parent output hash memory
  free(*pHash_out_h);
  cudaFree(*pHash_out_d);
  // Free parent merkle hash memory
  free(*pHash_merkle_h);
  cudaFree(*pHash_merkle_d);
}
__host__ void freeMiningMemory(BYTE ** target_h, BYTE ** target_d, BYTE ** nonce_h, BYTE ** nonce_d, int ** flag_d){
  // Free mining target
  free(*target_h);
  cudaFree(*target_d);
  // Free mining nonce
  free(*nonce_h);
  cudaFree(*nonce_d);
  // Free mining flag
  cudaFree(*flag_d);
}
__host__ void freeFileStrings(char ** str, int num_workers){
  for(int i = 0; i < num_workers; i++){
    free(str[i]);
  }
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

__host__ void initializeWorkerBlocks(BYTE ** hash_h, BYTE ** block_h, int num_workers){
  BYTE prevBlock[32], byte_time[4];             // Previous Block and time vars
  BYTE version[4] = {0x01,0x00,0x00,0x00};      // Default Version
  BYTE diff_bits[4] = {0x1d, 0x00, 0xff, 0xff}; // Starting Difficulty
  BYTE nonce[4] = {0x00, 0x00, 0x00, 0x00};     // Starting Nonce
  for(int i = 0; i < 32; i++){
    prevBlock[i] = 0x00;
  }
  getTime(byte_time);
  for(int i = 0; i < num_workers; i++){
    initializeBlockHeader(block_h[i], version, prevBlock, hash_h[i], byte_time, diff_bits, nonce);
  }
}

__host__ void initializeParentBlock(BYTE * pBlock_h){
  BYTE prevBlock[32], hash[32], byte_time[4];
  BYTE version[4] = {0x01,0x00,0x00,0x00};
  BYTE diff_bits[4] = {0x1d, 0x00, 0xff, 0xff};
  BYTE nonce[4] = {0x00, 0x00, 0x00, 0x00};
  // Use zero value previous block
  for(int i = 0; i < 32; i++){
    prevBlock[i] = 0x00;
    hash[i] = 0x00;
  }
  getTime(byte_time);
  initializeBlockHeader(pBlock_h, version, prevBlock, hash, byte_time, diff_bits, nonce);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/**************************************************************************MINING UPDATE FUNCTIONS**************************************************************************/
// UPDATE WORKER BLOCK WITH THE PREVIOUS HASH VALUE AND A NEW HASH FROM THE INPUT FILE
__host__ int updateBlock(FILE * inFile, BYTE * block_h, BYTE * hash_h){
  int errEOF = 0;
  for(int i = 0; i < 32; i++){
    block_h[i + 4] = hash_h[i];
  }
  errEOF = readNextHash(inFile, hash_h);
  for(int i = 0; i < 32; i++){
    block_h[i + 36] = hash_h[i];
  }
  BYTE byte_time[4];
  getTime(byte_time);
  for(int i = 0; i < 4; i++){
    block_h[i + 68] = byte_time[i];
  }
  return errEOF;
}
// UPDATE BLOCK MERKLE ROOT TO THE GIVEN HASH
__host__ void updateParentRoot(BYTE * block_h, BYTE * hash_h){
  for(int i = 0; i < 32; i++){
    block_h[i + 36] = hash_h[i];
  }
  BYTE byte_time[4];
  getTime(byte_time);
  for(int i = 0; i < 4; i++){
    block_h[i + 68] = byte_time[i];
  }
  return;
}
// UPDATE BLOCK PREVIOUS HASH TO THE GIVEN HASH
__host__ void updateParentHash(BYTE * block_h, BYTE * hash_h){
  for(int i = 0; i < 32; i++){
    block_h[i + 4] = hash_h[i];
  }
  BYTE byte_time[4];
  getTime(byte_time);
  for(int i = 0; i < 4; i++){
    block_h[i + 68] = byte_time[i];
  }
  return;
}
// UPDATE DIFFICULTY BY DECREASING THE LARGEST TARGET BYTE BY 1
__host__ void updateDifficulty(BYTE * block_h, int diff_level){
  int start_pow = 0x1d;
  int start_diff = 0x00ffff;
  int new_pow = 0x00;
  int new_diff = 0x000000;
  new_pow = start_pow-((diff_level)/0xFF);
  new_diff = start_diff - (((diff_level)%0xFF)<<8);
  block_h[72] = new_pow;
  block_h[73] = new_diff  >> 16;
  block_h[74] = new_diff  >> 8;
  block_h[75] = new_diff;
}
// UPDATE THE CURRENT TIME ON DEVICE IN CASE OF NONCE OVERFLOW
__host__ void updateTime(cudaStream_t * tStream, WORD * time_h, DOMAIN_HANDLE prof_handle){
	WORD old_time = *time_h;
	*time_h = time(0);
//  getTime(time_h);
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
		printf("UPDATING...");
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
__host__ void getTime(BYTE * byte_time){
  int current_time = time(0);
  byte_time[0] = (BYTE)(current_time >> 24) & 0xFF;
  byte_time[1] = (BYTE)(current_time >> 16) & 0xFF;
  byte_time[2] = (BYTE)(current_time >> 8) & 0xFF;
  byte_time[3] = (BYTE)(current_time & 0xFF);
}
// UPDATE THE TARGET AND DIFFICULTY VARIABLES WITH THE BLOCK DIFFICULTY BITS
__host__ void getDifficulty(BYTE * block_h, BYTE ** target, int * target_length, double * difficulty, int worker_num){
  char logOut[100];
  char debugOut[100];
  BYTE block_target[] = {block_h[72], block_h[73], block_h[74], block_h[75]};
  *target_length = calculateTarget(block_target, *target);
  *difficulty = calculateDifficulty(block_target);
  BYTE target_str[100];
  decodeHex(*target, target_str, *target_length);
  char chain_id[20];
  if(worker_num == 0){
    sprintf(chain_id, "PARENT");
  }else{
    sprintf(chain_id, "WORKER %i", worker_num);
  }
  sprintf(debugOut, "BLOCK TARGET: %02x %02x %02x %02x\n        TARGET VALUE: %s\n", block_target[0], block_target[1], block_target[2], block_target[3], (char*)target_str);
  sprintf(logOut, "NEW DIFFICULTY %s: %lf", chain_id, *difficulty);
  printLog((const char*)logOut);
  printDebug((const char*)debugOut);
}

__host__ void getMiningDifficulty(BYTE * block_h, WORD ** target, int * target_length, double * difficulty, int worker_num){
  char logOut[100];
  char debugOut[100];
	BYTE * target_bytes;
	target_bytes = (BYTE*)malloc(HASH_SIZE);
  BYTE block_target[] = {block_h[72], block_h[73], block_h[74], block_h[75]};
  *target_length = calculateMiningTarget(block_target, target_bytes, *target);
  *difficulty = calculateDifficulty(block_target);
  BYTE target_str[100];
  decodeHex(target_bytes, target_str, *target_length*4);
  char chain_id[20];
  if(worker_num == 0){
    sprintf(chain_id, "PARENT");
  }else{
    sprintf(chain_id, "WORKER %i", worker_num);
  }
  sprintf(debugOut, "BLOCK TARGET: %02x %02x %02x %02x\n        TARGET VALUE: %s\n", block_target[0], block_target[1], block_target[2], block_target[3], (char*)target_str);
  sprintf(logOut, "NEW DIFFICULTY %s: %lf", chain_id, *difficulty);
  printLog((const char*)logOut);
  printDebug((const char*)debugOut);
	free(target_bytes);
}

__host__ void setMiningDifficulty(cudaStream_t * stream, BYTE * block_h, int worker_num){
	WORD target[8];

	BYTE target_bytes[32];
  BYTE block_target[] = {block_h[72], block_h[73], block_h[74], block_h[75]};

  calculateMiningTarget(block_target, target_bytes, target);
  calculateDifficulty(block_target);

	cudaMemcpyToSymbolAsync(target_const, target, TARGET_C_SIZE, TARGET_C_SIZE*worker_num, cudaMemcpyHostToDevice, *stream);
}
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/************************************************************************MINING CALCULATION FUNCTIONS***********************************************************************/
// GET THE MINING DIFFICULTY FROM THE GIVEN BITS, RETURN DIFFICULTY AS A DOUBLE
__host__ double calculateDifficulty(BYTE * bits){
  // FIRST BYTE FOR LEADING ZEROS, REST FOR TARGET VALUE
  int start_pow = 0x1d;
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
  int padding = (32 - bits[0])+DIFF_REDUCE;
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
  int padding = (32 - bits[0])+DIFF_REDUCE;
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
__host__ void genHashKernel(BYTE ** hash_hf, BYTE ** hash_df, BYTE ** seed_h, BYTE ** seed_d, size_t size_hash, size_t size_seed){
  cudaMemcpy(*seed_d, *seed_h, size_seed, cudaMemcpyHostToDevice);
  genTestHashes<<<MAX_BLOCKS, NUM_THREADS>>>(*hash_df, *seed_d, MAX_BLOCKS);
  cudaDeviceSynchronize();
  cudaMemcpy(*hash_hf, *hash_df, size_hash, cudaMemcpyDeviceToHost);
}
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*****************************************************************************MERKLE TREE KERNEL****************************************************************************/
__host__ void launchMerkle(cudaStream_t * stream, BYTE ** merkle_d, BYTE ** root_d, BYTE ** merkle_h, BYTE ** root_h, int ** flag_d,  int * flag_h, int buffer_size){
  cudaMemcpyAsync(*merkle_d, *merkle_h, HASH_SIZE*buffer_size, cudaMemcpyHostToDevice, *stream);
  cudaMemcpyAsync(*flag_d, flag_h, sizeof(int), cudaMemcpyHostToDevice, *stream);
  getMerkleRoot<<<2, NUM_THREADS, 0, *stream>>>(*merkle_d, *root_d, buffer_size);
  cudaMemcpyAsync(*root_h, *root_d, HASH_SIZE, cudaMemcpyDeviceToHost, *stream);
}
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*******************************************************************************MINING KERNEL*******************************************************************************/
// LAUNCH MINER KERNEL ON AN INDEPENDENT STREAM USING THE SPECIFIED NUMBER OF BLOCKS
__host__ void launchMiner(int kernel_id, cudaStream_t * stream, BYTE ** block_d, BYTE ** hash_d, BYTE ** nonce_d, BYTE ** block_h, BYTE ** hash_h, BYTE ** nonce_h, BYTE ** target_d, int ** flag_d,  int * flag_h, int * target_length){
	int num_blocks = (kernel_id == 0) ? PARENT_BLOCKS:WORKER_BLOCKS;
  cudaMemcpyAsync(*block_d, *block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, *stream);
  cudaMemcpyAsync(*flag_d, flag_h, sizeof(int), cudaMemcpyHostToDevice, *stream);

	// CONSTANT MEMORY UPDATES FOR IMPROVED EFFICIENCY
	//FIXME Move mining difficulty elsewhere, may be best to load constant when updating difficulty

	WORD basemsg_hw[16];
	//basemsg_hw = (WORD *)malloc(sizeof(WORD)*16);

	// INITIALIZE MESSAGE SCHEDULE WITH CONSTANT BASE BLOCK, NO EXTRA REGISTERS USED!!
	for(int i = 0; i < 16; i++){
		basemsg_hw[i] = ((*block_h)[i*4] << 24) | ((*block_h)[i*4+1] << 16) | ((*block_h)[i*4+2] << 8) | ((*block_h)[i*4+3]);
	}
//	cudaMemcpyToSymbolAsync(block_const, basemsg_hw, sizeof(WORD)*16, kernel_id*16, cudaMemcpyHostToDevice, *stream);
	cudaMemcpyToSymbolAsync(block_const, basemsg_hw, BLOCK_C_SIZE, BLOCK_C_SIZE*kernel_id, cudaMemcpyHostToDevice, *stream);
//	free(basemsg_hw);
	int block_offset = kernel_id*16;
	int target_offset = kernel_id*8;

//  minerKernel<<<num_blocks,NUM_THREADS, 0, *stream>>>(*block_d, *hash_d, *nonce_d, *target_d, *flag_d, *target_length);
//	printf("WAITING FOR KERNEL MEMORY COPY TO HOST \n");
//	cudaStreamSynchronize(*stream);
//	printf("COPY COMPLETE, STARTING KERNEL...\n");


	minerKernel_new<<<num_blocks,NUM_THREADS, 0, *stream>>>(*block_d, *hash_d, *nonce_d, *flag_d, block_offset, target_offset);
}
// LOAD MINER RESULTS BACK FROM THE GPU USING ASYNCHRONOUS STREAMING
__host__ void returnMiner(cudaStream_t * stream, BYTE ** block_d, BYTE ** hash_d, BYTE ** nonce_d, BYTE ** block_h, BYTE ** hash_h, BYTE ** nonce_h){
  cudaMemcpyAsync(*block_h, *block_d, BLOCK_SIZE, cudaMemcpyDeviceToHost, *stream);
  cudaMemcpyAsync(*hash_h, *hash_d, HASH_SIZE, cudaMemcpyDeviceToHost, *stream);
  cudaMemcpyAsync(*nonce_h, *nonce_d, NONCE_SIZE, cudaMemcpyDeviceToHost, *stream);
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
__host__ void logStart(int workerID, int block, BYTE * start_hash){
  char name[20];
  if(workerID+1 == 0){
    sprintf(name, "PARENT");
  } else{
    sprintf(name, "WORKER %i", workerID+1);
  }
  char logMessage[50];
  BYTE hash[65];
  decodeHex(start_hash, hash, 32);
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
// CREATE OR READ INPUT FILES FOR EACH WORKER, READ FIRST HASH VALUE
// RETURN OPENED INPUT FILES AND ERROR FLAG
__host__ int initializeHashes(FILE ** inFiles, int num_workers, BYTE ** hash_h){
  char filename[20], logOut[100];
  int readErr = 0;
  int Err = 0;
  for(int i = 0; i < num_workers; i++){
    sprintf(filename, "inputs/chain_input%d.txt", i);
    if(inFiles[i] = fopen(filename, "r")){
        sprintf(logOut,"READING DATA FROM INPUT FILE '%s'",filename);
        printDebug((const char*)logOut);
        readErr = readNextHash(inFiles[i], hash_h[i]);
    }else{
        sprintf(logOut,"INPUT FILE '%s' NOT FOUND, GENERATING FILE",filename);
        printDebug((const char*)logOut);
        // USE GPU TO CREATE RANDOMLY GENERATED INPUT FILES
        initializeInputFile(inFiles[i], filename);
        if(inFiles[i] = fopen(filename, "r")){
            sprintf(logOut,"INPUT FILE '%s' CREATED SUCCESSFULLY!", filename);
            printDebug((const char*)logOut);
            readErr = readNextHash(inFiles[i], hash_h[i]);
        }else{
          printError("INPUT FILE STILL COULDN'T BE ACCESSED, ABORTING!!!");
          readErr = 1;
        }
    }
    if(readErr == 1){
      sprintf(logOut,"INPUT FILE '%s' COULD NOT BE READ!!!",filename);
      printError((const char*)logOut);
      readErr = 0;
      Err = 1;
    }
  }
  return Err;
}
// CREATE A NEW INPUT FILE, CALL KERNEL TO GENERATE RANDOM INPUT HASHES
__host__ void initializeInputFile(FILE * inFile, char * filename){
  // ALLOCATE SPACE FOR HASHES
  BYTE *hash_hf, *hash_df;
  size_t size_hash = NUM_THREADS * MAX_BLOCKS *(32 * sizeof(BYTE));
  hash_hf = (BYTE *) malloc(size_hash);
  cudaMalloc((void **) &hash_df, size_hash);

  // ALLOCATE SPACE FOR SEED VALUES
  BYTE *seed_h, *seed_d;
  size_t size_seed = (30 * sizeof(BYTE));
  seed_h = (BYTE *)malloc(size_seed);
  cudaMalloc((void **) &seed_d, size_seed);

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
    for(int i = 0; i < 30; i++){
        seed_h[i] = (rand() % 255) & 0xFF;
    }
    // GENERATE NEW SET OF HASHES AND APPEND TO INPUT FILE
    genHashKernel(&hash_hf, &hash_df, &seed_h, &seed_d, size_hash, size_seed);
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
__host__ void printInputFile(BYTE * hash_f, char * filename, int blocks, int threads){
  FILE *file_out;
  char debugmsg[100+blocks*threads*65];
  BYTE hash_str[65];
  int count = 0;
  sprintf(debugmsg, "HASHES ADDED:\n");
  // PARSE HASHES AND PRINT TO FILE
  if(file_out = fopen(filename, "a")){
    for(int i=0; i < blocks; i++){
        for(int j = 0; j < threads; j++){
          decodeHex(&hash_f[i*threads + j*32], hash_str, 32);
          strcat(debugmsg, (const char*)hash_str);
          strcat(debugmsg, "\n");
          fprintf(file_out, "%s\n", hash_str);
          count++;
        }
    }
    char logmsg[50];
    sprintf(logmsg, "ADDING %i HASHES TO INPUT FILE '%s'\n", count, filename);
    printLog((const char*)logmsg);
    printDebug((const char*)debugmsg);
    fclose(file_out);
  }
  else{
    char input_err[100];
    sprintf(input_err, "INPUT FILE '%s' COULD NOT BE CREATED!!!", filename);
    printError((const char*)input_err);
  }
}
// READ THE NEXT HASH FROM THE GIVEN INPUT FILE
__host__ int readNextHash(FILE * inFile, BYTE * hash_h){
    int readErr = 0;
    BYTE inputBuffer[65];
    if(!fscanf(inFile, "%s", inputBuffer)){
      printError((const char*)"READ IN FAILED!!!!!");
      readErr = 1;
    }
    else {
      encodeHex(inputBuffer, hash_h, 64);
    }
    return readErr;
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/***************************************************************************OUTPUT FILE FUNCTIONS***************************************************************************/
// CREATE OUTPUT FILES FOR EACH WORKER, AND OUTPUT DIRECTORY IF NECCESSARY
__host__ int initializeOutputs(char * outFiles[], char * out_dir_name, int num_workers){
  printDebug((const char*)"BEGIN OUTPUT INITIALIZATION");
  int readErr = 0; char logOut[100]; FILE * output;
  mkdir("outputs", ACCESSPERMS);
  mkdir(out_dir_name, ACCESSPERMS);
  for(int i = 0; i < num_workers; i++){
    sprintf(outFiles[i], "%s/outputs_%d.txt", out_dir_name, i);
    if(output = fopen(outFiles[i], "w")){
      sprintf(logOut,"FOUND WORKER %i OUTPUT FILE: %s.",i, outFiles[i]);
      fprintf(output, "WORKER CHAIN %i OUTPUT FILE\nFORMAT:\n BLOCK_HEADER#: \n HASH_SOLUTION: \n CORRECT_NONCE: \n COMPUTATION_TIME: 0 \t\t BLOCK_DIFFICULTY: 0 \n\n", i);
    }
    else{
        sprintf(logOut,"WORKER %i OUTPUT FILE: %s NOT FOUND",i, outFiles[i]);
        readErr = 1;
    } printDebug((const char*)logOut);
    fclose(output);
  }
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
__host__ void printOutputFile(char * outFileName, BYTE * block_h, BYTE * hash_f, BYTE * nonce_h, int block, float calc_time, double difficulty, int id, int log_flag){
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
    decodeHex(block_h, block_str[0], 40);
    decodeHex(&(block_h[40]), block_str[1], 40);
    decodeHex(hash_f, hash_str, 32);
    decodeHex(nonce_h, nonce_str, 4);

sprintf(logOut, "%s SOLVED BLOCK %i \n      HASH: %s\n", name, block, hash_str);
sprintf(printOut, "\n________________________________________________________________________________\n\
%s-%s FINISHED BLOCK %i-%s|\n\
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

__global__ void cudaTest(void){
	//SHA256_CTX ctx;
//	printf("MERKLE ROOT COPY TEST PRINT: \n");
		printf("THREAD %i WORKING\n", threadIdx.x);

//	printf("MERKLE ROOT COPY TEST FINISHED\n");
}


__global__ void benchmarkKernel(BYTE * block_d){
    SHA256_CTX thread_ctx;
 	  unsigned int nonce = 0x00000000;
 	  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
 		int inc_size = blockDim.x * gridDim.x;
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

 		while(nonce < 0xEFFFFFFF){
 	    threadBlock[76] = (BYTE)(nonce >> 24) & 0xFF;
 	    threadBlock[77] = (BYTE)(nonce >> 16) & 0xFF;
 	    threadBlock[78] = (BYTE)(nonce >> 8) & 0xFF;
 	    threadBlock[79] = (BYTE)(nonce & 0xFF);

 	    sha256_init(&thread_ctx);
 	    sha256_update(&thread_ctx, threadBlock, 64);
 	    sha256_update(&thread_ctx, &(threadBlock[64]), 16);
 			sha256_final(&thread_ctx, hash_t_i);

 	    sha256_init(&thread_ctx);
 	  	sha256_update(&thread_ctx, hash_t_i, 32);
 	  	sha256_final(&thread_ctx, hash_t_f);

 			nonce += inc_size;
 	  }
 }

// BENCHMARK NEW SHA256 FUNCTION
// TODO ADD MORE ITERATIONS WITH TIMING TO INCREASE ACCURACY
__global__ void miningBenchmarkKernel(BYTE * block_d, BYTE * hash_d, int * flag_d, int * total_iterations){
	  int success = 0, i = 0, j=0;
	  unsigned int nonce = 0x00000000;
		unsigned int iteration = 0;
		unsigned int threadId = threadIdx.x;
		unsigned int inc_size = blockDim.x * gridDim.x;
		unsigned int idx = threadId + blockIdx.x * blockDim.x;

		nonce += idx;
		unsigned int max_iteration = 0xffffffff / inc_size;

		// THREADS SHARE FIRST 64 BYTES, SET IN CONSTANT MEMORY
		// EACH THREAD HAS ITS OWN VARIABLE FOR TOP 16 BYTES
		// ALLOCATED ON SHARED MEMORY TO FREE UP REGISTER USAGE FOR HASHING
		__shared__ WORD uniqueBlock[4096];
		WORD * unique_ptr = &(uniqueBlock[threadId*4]);

		// COMPUTE UNIQUE PORTION OF THE BLOCK HERE INSTEAD OF INSIDE THE LOOP FOR SPEEDUP
		unique_ptr[0] = (block_d[64] << 24) | (block_d[65] << 16) | (block_d[66] << 8) | (block_d[67]);
		unique_ptr[1] = (block_d[68] << 24) | (block_d[69] << 16) | (block_d[70] << 8) | (block_d[71]);
		unique_ptr[2] = (block_d[72] << 24) | (block_d[73] << 16) | (block_d[74] << 8) | (block_d[75]);
		unique_ptr[3] = (block_d[76] << 24) | (block_d[77] << 16) | (block_d[78] << 8) | (block_d[79]);

		while(flag_d[0] == 0){
//		while(nonce < 0xEFFFFFFF){
			if(iteration < max_iteration){
				iteration++;
			}else{ // UPDATE TIME
				if(idx == 0){
					printf("NEW TIME %08x\n\n", time_const);
					*total_iterations += iteration;
				}
				iteration = 0;
				unique_ptr[1] = time_const;
			}

			unique_ptr[3] = nonce;
			success = sha256_blockHash_TEST(unique_ptr, hash_d, test_basemsg_c, test_target_c);


			if(success == 0){
				nonce += inc_size;
			}else{
				flag_d[0] = 1;
				for(i = 0, j = 64; i < 4; i++, j+=4){
					block_d[j] = (unique_ptr[i] >> 24) & 0x000000FF;
					block_d[j+1] = (unique_ptr[i] >> 16) & 0x000000FF;
					block_d[j+2] = (unique_ptr[i] >> 8) & 0x000000FF;
					block_d[j+3] = (unique_ptr[i]) & 0x000000FF;
				}
				break;
			}
	  } // END WHILE LOOP
		if(idx == 0){
			*total_iterations += iteration;
		}
}

__global__ void hashTestKernel(BYTE * test_block, BYTE * result_block, int size){
	SHA256_CTX thread_ctx;

	sha256_init(&thread_ctx);
	sha256_update(&thread_ctx, test_block, size);
	sha256_final(&thread_ctx, result_block);

	return;
}

__global__ void hashTestMiningKernel(BYTE * test_block, BYTE * result_block, int * success){
	WORD uniquedata[4];
	uniquedata[0] = (test_block[64] << 24) | (test_block[65] << 16) | (test_block[66] << 8) | (test_block[67]);
	uniquedata[1] = (test_block[68] << 24) | (test_block[69] << 16) | (test_block[70] << 8) | (test_block[71]);
	uniquedata[2] = (test_block[72] << 24) | (test_block[73] << 16) | (test_block[74] << 8) | (test_block[75]);
	uniquedata[3] = (test_block[76] << 24) | (test_block[77] << 16) | (test_block[78] << 8) | (test_block[79]);
	*success = sha256_blockHash_TEST(uniquedata, result_block, test_basemsg_c, test_target_c);
	return;
}

 /*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
 /***************************************************************************HASH MINING FUNCTIONS***************************************************************************/

__global__ void genTestHashes(BYTE * hash_df, BYTE * seed, int num_blocks){
  SHA256_CTX ctx;
  BYTE block = (BYTE)(blockIdx.x & 0xFF);
  BYTE thread = (BYTE)(threadIdx.x & 0xFF);
  int offset = 32*threadIdx.x + blockIdx.x * blockDim.x;

  BYTE seed_hash[32];
  #pragma unroll 30
  for(int i = 0; i < 30; i++){
    seed_hash[i] = seed[i];
  }

  seed_hash[30] = block;
  seed_hash[31] = thread;

  sha256_init(&ctx);
  sha256_update(&ctx, seed_hash, 32);
  sha256_final(&ctx, &hash_df[offset]);
}

__global__ void minerKernel(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, BYTE * target, int * flag_d, int compare){
  int success = 0;

  unsigned int nonce = 0x00000000;
  int iteration = 0;
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  SHA256_CTX thread_ctx;
  int inc_size = blockDim.x * gridDim.x;
  int max_iteration = 0xffffffff / inc_size;
  nonce += idx;

	// THREADS SHARE FIRST 64 BYTES, COPY FOR EACH BLOCK
	__shared__ BYTE baseBlock[64];
	if(threadIdx.x < 64){
		#pragma unroll 64
		for(int i = 0; i < 64; i++){
			baseBlock[threadIdx.x] = block_d[threadIdx.x];
		}
	}

	// EACH THREAD HAS ITS OWN VARIABLE FOR TOP 16 BYTES
	BYTE uniqueBlock[16];
	#pragma unroll 16
	for(int i = 0; i < 16; i++){
		uniqueBlock[i] = block_d[i+64];
	}
/*
	if(idx == 0){
			printf("COMPARE BASEBLOCK TO ORIGINAL: \n");
			printf("ORIGINAL: \n");
			printBlock(block_d);
			printf("SPLIT BLOCK: \n");
			printSplitBlock(baseBlock, uniqueBlock);
	}
*/
	// SYNCHRONIZE TO ENSURE SHARED MEMORY IS READY
	//__syncthreads();

	// FIXME ONE OF THESE CAN BE REMOVED IF HASH UPDATE IS ONLY CALLED ONCE
	// Try to use shared memory for intermediate value
//	__shared__ BYTE local_mem_in[parent_block_size][64];
//	__shared__ BYTE hash_t_i[NUM_THREADS][32];
  BYTE hash_t_i[32];
  BYTE hash_t_f[32];
  #pragma unroll 32
  for(int i = 0; i < 32; i++){
//    hash_t_i[threadIdx.x][i] = 0x00;
		hash_t_i[i] = 0x00;
    hash_t_f[i] = 0x00;
  }

  while(flag_d[0] == 0){
    if(iteration < max_iteration){
      iteration++;
    }else{ // UPDATE TIME
      iteration = 0;
			uniqueBlock[4] = (BYTE)(time_const >> 24) & 0xFF;
			uniqueBlock[5] = (BYTE)(time_const >> 16) & 0xFF;
			uniqueBlock[6] = (BYTE)(time_const >> 8) & 0xFF;
			uniqueBlock[7] = (BYTE)(time_const & 0xFF);
			/*
      uniqueBlock[4] = time_d[0];
      uniqueBlock[5] = time_d[1];
      uniqueBlock[6] = time_d[2];
      uniqueBlock[7] = time_d[3];
*/
			if(idx == 0){
        printf("NEW TIME %08x\n\n", time_const);
      }
    }

    uniqueBlock[12] = (BYTE)(nonce >> 24) & 0xFF;
    uniqueBlock[13] = (BYTE)(nonce >> 16) & 0xFF;
    uniqueBlock[14] = (BYTE)(nonce >> 8) & 0xFF;
    uniqueBlock[15] = (BYTE)(nonce & 0xFF);

    sha256_init(&thread_ctx);
    sha256_update(&thread_ctx, baseBlock, 64);
    sha256_update(&thread_ctx, uniqueBlock, 16);
    sha256_final(&thread_ctx, hash_t_i);

    sha256_init(&thread_ctx);
		sha256_update(&thread_ctx, hash_t_i, 32);
    success = sha256_final_target(&thread_ctx, hash_t_f, target, compare);

    if(success == 0){
      nonce += inc_size;
    }else{
      flag_d[0] = 1;
      nonce_f[0] = uniqueBlock[12];
      nonce_f[1] = uniqueBlock[13];
      nonce_f[2] = uniqueBlock[14];
      nonce_f[3] = uniqueBlock[15];
      block_d[76] = uniqueBlock[12];
      block_d[77] = uniqueBlock[13];
      block_d[78] = uniqueBlock[14];
      block_d[79] = uniqueBlock[15];

      #pragma unroll 16
      for(int i = 0; i < 16; i++){
        block_d[i+64] = uniqueBlock[i];
      }
      #pragma unroll 32
      for(int i = 0; i < 32; i++){
        hash_d[i] = hash_t_f[i];
      }
      break;
    }
  }
}


__global__ void minerKernel_new(BYTE * block_d, BYTE * hash_d, BYTE * nonce_f, int * flag_d, int block_offset, int target_offset){
	int success = 0, i = 0, j = 0;
	unsigned int nonce = 0x00000000;
	unsigned int iteration = 0;
	unsigned int threadId = threadIdx.x;
	unsigned int inc_size = blockDim.x * gridDim.x;
	unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

	// POINTERS TO BLOCK AND TARGET CONSTANTS
	WORD * block_ptr = &(block_const[block_offset]);
	WORD * target_ptr = &(target_const[target_offset]);

	nonce += idx;
	unsigned int max_iteration = 0xffffffff / inc_size;

	// THREADS SHARE FIRST 64 BYTES, SET IN CONSTANT MEMORY
	// EACH THREAD HAS ITS OWN VARIABLE FOR TOP 16 BYTES
	// ALLOCATED ON SHARED MEMORY TO FREE UP REGISTER USAGE FOR HASHING
	__shared__ WORD uniqueBlock[4096];
	WORD * unique_ptr = &(uniqueBlock[threadId*4]);
	//FIXME PERFORMANCE WOULD MOST LIKELY IMPROVE IF block_d WAS ONLY USED HERE
	// COMPUTE UNIQUE PORTION OF THE BLOCK HERE INSTEAD OF INSIDE THE LOOP FOR SPEEDUP
	unique_ptr[0] = (block_d[64] << 24) | (block_d[65] << 16) | (block_d[66] << 8) | (block_d[67]);  	// END OF PREVIOUS HASH (CONSTANT)
	unique_ptr[1] = (block_d[68] << 24) | (block_d[69] << 16) | (block_d[70] << 8) | (block_d[71]); 	// CURRENT TIME ON THE BLOCK (UPDATED WHEN ALL NONCES ARE TRIED)
	unique_ptr[2] = (block_d[72] << 24) | (block_d[73] << 16) | (block_d[74] << 8) | (block_d[75]);		// BLOCK DIFFICULTY (CONSTANT)
	unique_ptr[3] = (block_d[76] << 24) | (block_d[77] << 16) | (block_d[78] << 8) | (block_d[79]);		// NONCE (UNIQUE PER THREAD, UPDATED EACH ITERATION)

  while(flag_d[0] == 0){
    if(iteration < max_iteration){
      iteration++;
    }else{ // UPDATE TIME
      iteration = 0;
			unique_ptr[1] = time_const;
			if(idx == 0){
        printf("NEW TIME %08x\n\n", time_const);
      }
    }

		unique_ptr[3] = nonce;
		success = sha256_blockHash_TEST(unique_ptr, hash_d, block_ptr, target_ptr);

    if(success == 0){
			nonce += inc_size;
    }else{
      flag_d[0] = 1;
			// A NEW VARIABLE SHOULD BE USED TO STORE BLOCK, WHICH MAY REDUCE REG USAGE. NONCE IS UNNECCESSARY
			nonce_f[0] = (unique_ptr[3] >> 24) & 0x000000FF;
			nonce_f[1] = (unique_ptr[3] >> 16) & 0x000000FF;
			nonce_f[2] = (unique_ptr[3] >> 8) & 0x000000FF;
			nonce_f[3] = (unique_ptr[3]) & 0x000000FF;
			for(i = 0, j = 64; i < 4; i++, j+=4){
				block_d[j] = (unique_ptr[i] >> 24) & 0x000000FF;
				block_d[j+1] = (unique_ptr[i] >> 16) & 0x000000FF;
				block_d[j+2] = (unique_ptr[i] >> 8) & 0x000000FF;
				block_d[j+3] = (unique_ptr[i]) & 0x000000FF;
			}
      break;
    }
  }
}  // end new kernel

__global__ void getMerkleRoot(BYTE * pHash_d, BYTE * pRoot_d, int buffer_blocks){
  SHA256_CTX ctx;
  // Shared memory for sharing hash results
  __shared__ BYTE local_mem_in[PARENT_BLOCK_SIZE][64];
  __shared__ BYTE local_mem_out[PARENT_BLOCK_SIZE][32];
  int tree_size = pow(2.0, ceil(log2((double)buffer_blocks)));
	volatile int idx = threadIdx.x; // REDUCES REGISTER USAGE FROM 57 TO 32

  // SET UP HASH TREE THREADS
  if(idx < buffer_blocks){
    //SET UP UNIQUE THREADS
    for(int i = 0; i < 32; i++){
      local_mem_in[idx][i] = pHash_d[idx*32+i];
    }

    // Calculate first hash, store in shared memory
    sha256_init(&ctx);
    sha256_update(&ctx, local_mem_in[idx], 32);
    sha256_final(&ctx, local_mem_out[idx]);

    #pragma unroll 32
    for(int i = 0; i < 32; i++){
      local_mem_in[idx][i] = local_mem_out[idx][i];
    }

    sha256_init(&ctx);
    sha256_update(&ctx, local_mem_in[idx], 32);
    sha256_final(&ctx, local_mem_out[idx]);

    // Sequential hash reduction
    // First iteration 0 = 0|1	2=2|3 	4=4|5		6=6|7
    // Second iteration 0 = (0|1)|(2|3) 	4=(4|5)|(6|7)
    // Third iteration 0 = ((0|1)|(2|3))|((4|5)|(6|7)), etc...
    // Progressively loop to combine hashes
    for(int i = 2; i <= tree_size; i*=2){
      if(idx % i == 0){
        int mid = i/2;
        if(idx + mid < buffer_blocks){
          #pragma unroll 32
          for(int j = 0; j < 32; j++){
            local_mem_in[idx][j] = local_mem_out[idx][j];
            local_mem_in[idx][32+j]= local_mem_out[idx+mid][j];
          }
        }else{ // HASH TOGETHER DUPLICATES FOR UNMATCHED BRANCHES
          #pragma unroll 32
          for(int j = 0; j < 32; j++){
            local_mem_in[idx][j] = local_mem_out[idx][j];
            local_mem_in[idx][32+j]= local_mem_out[idx][j];
          }
        }
        sha256_init(&ctx);
        sha256_update(&ctx, local_mem_in[idx], 64);
        sha256_final(&ctx, local_mem_out[idx]);

        #pragma unroll 32
        for(int j = 0; j < 32; j++){
          local_mem_in[idx][j] = local_mem_out[idx][j];
        }

        sha256_init(&ctx);
        sha256_update(&ctx, local_mem_in[idx], 32);
        sha256_final(&ctx, local_mem_out[idx]);
      }
    }
    // All values coalesce into thread 0 shared memory space, and then get read back
    if(idx == 0){
      for(int i = 0; i < 32; i++){
        pRoot_d[i] = local_mem_out[0][i];
      }
    }
  }
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
//TODO ADD DEVICE FUNCTIONS

__device__ void printHash(BYTE * hash){
  printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x \n", hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7], hash[8], hash[9],\
  hash[10], hash[11], hash[12], hash[13], hash[14], hash[15], hash[16], hash[17], hash[18], hash[19],\
  hash[20], hash[21], hash[22], hash[23], hash[24], hash[25], hash[26], hash[27], hash[28], hash[29], hash[30], hash[31]);
}
__device__ void printBlock(BYTE * hash){
/*
  printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x \n\
	", hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7], hash[8], hash[9],\
  hash[10], hash[11], hash[12], hash[13], hash[14], hash[15], hash[16], hash[17], hash[18], hash[19],\
  hash[20], hash[21], hash[22], hash[23], hash[24], hash[25], hash[26], hash[27], hash[28], hash[29],\
	hash[30], hash[31], hash[32], hash[33], hash[34], hash[35], hash[36], hash[37], hash[38], hash[39],\
	hash[40], hash[41], hash[42], hash[43], hash[44], hash[45], hash[46], hash[47], hash[48], hash[49],\
	hash[50], hash[51], hash[52], hash[53], hash[54], hash[55], hash[56], hash[57], hash[58], hash[59],\
	hash[60], hash[61], hash[62], hash[63], hash[64], hash[65], hash[66], hash[67], hash[68], hash[69],\
	hash[70], hash[71], hash[72], hash[73], hash[74], hash[75], hash[76], hash[77], hash[78], hash[79]);
*/
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

__device__ void printSplitBlock(BYTE * hash, BYTE * split){
	/*
  printf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x \n\
	", hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7], hash[8], hash[9],\
  hash[10], hash[11], hash[12], hash[13], hash[14], hash[15], hash[16], hash[17], hash[18], hash[19],\
  hash[20], hash[21], hash[22], hash[23], hash[24], hash[25], hash[26], hash[27], hash[28], hash[29],\
	hash[30], hash[31], hash[32], hash[33], hash[34], hash[35], hash[36], hash[37], hash[38], hash[39],\
	hash[40], hash[41], hash[42], hash[43], hash[44], hash[45], hash[46], hash[47], hash[48], hash[49],\
	hash[50], hash[51], hash[52], hash[53], hash[54], hash[55], hash[56], hash[57], hash[58], hash[59],\
	hash[60], hash[61], hash[62], hash[63],\
	split[0], split[1], split[2], split[3], split[4], split[5], split[6], split[7], split[8], split[9], split[10], split[11], split[12], split[13], split[14], split[15]);
*/
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
	split[0], split[1], split[2], split[3], split[4], split[5], split[6], split[7], split[8], split[9], split[10], split[11], split[12], split[13], split[14], split[15]);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/******************************************************************************SHA256 FUNCTIONS*****************************************************************************/

__device__ void sha256_transform(SHA256_CTX *ctx, const BYTE data[])
{
//	WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
	WORD a, b, c, d, e, f, g, h, i, j, t1, t2;
	// FORCE MESSAGE SCHEDULE TO GLOBAL MEMORY, FREE UP 8+ REGISTERS
	volatile WORD m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4) // TODO use accelerated addition, break into parts
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
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

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}

__device__ void sha256_init(SHA256_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

__device__ void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len)
{
	WORD i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			sha256_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

__device__ void sha256_final(SHA256_CTX *ctx, BYTE hash[])
{
	WORD i;

	i = ctx->datalen;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	sha256_transform(ctx, ctx->data);

	// Since this implementation uses little endian byte ordering and SHA uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
	for (i = 0; i < 4; ++i) {
		hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
	}
}

// USE VOLATILE TARGET AND COMPARE TO FORCE GLOBAL STORAGE
__device__ int sha256_final_target(SHA256_CTX *ctx, BYTE hash[], const BYTE target[], const int compare)
{
	WORD i;

	i = ctx->datalen;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	sha256_transform(ctx, ctx->data);

	// Since this implementation uses little endian byte ordering and SHA uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
	for (i = 0; i < 4; ++i) {
		hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
	}

	int success = 1;

	for(int i = 0; i < compare; i++){
		if(hash[31 - i] > target[i]){
			success = 0;
			break;
		}
	}

	// Store little endian ordering for bytewise comparison against the desired target
	return success;
}


/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/
/************************  ______________________________________________________________________________________________________________________  *************************/
/************************  |   _    _           _____ _    _     ____  _____ _______ _____ __  __ _____ ______      _______ _____ ____  _   _   |  *************************/
/************************  |  | |  | |   /\    / ____| |  | |   / __ \|  __ \__   __|_   _|  \/  |_   _|___  /   /\|__   __|_   _/ __ \| \ | |  |  *************************/
/************************  |  | |__| |  /  \  | (___ | |__| |  | |  | | |__) | | |    | | | \  / | | |    / /   /  \  | |    | || |  | |  \| |  |  *************************/
/************************  |  |  __  | / /\ \  \___ \|  __  |  | |  | |  ___/  | |    | | | |\/| | | |   / /   / /\ \ | |    | || |  | | . ` |  |  *************************/
/************************  |  | |  | |/ ____ \ ____) | |  | |  | |__| | |      | |   _| |_| |  | |_| |_ / /__ / ____ \| |   _| || |__| | |\  |  |  *************************/
/************************  |  |_|  |_/_/    \_\_____/|_|  |_|   \____/|_|      |_|  |_____|_|  |_|_____/_____/_/    \_\_|  |_____\____/|_| \_|  |	 *************************/
/************************  |____________________________________________________________________________________________________________________|  *************************/
/************************                                                                                                                          *************************/
/***************************************************************************************************************************************************************************/
/***************************************************************************************************************************************************************************/

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/****************************************************************************TRANSFORM FUNCTIONS****************************************************************************/

// DEFAULT TRANSFORM FUNCTION, ASSUMES MESSAGE SCHEDULE HAS BEEN COMPUTED
__device__ void sha256_mining_transform(WORD state[], WORD m[]){
	WORD a, b, c, d, e, f, g, h, i, t1, t2;

	a = state[0];
	b = state[1];
	c = state[2];
	d = state[3];
	e = state[4];
	f = state[5];
	g = state[6];
	h = state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
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

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

// MODIFIED TRANSFORM FOR REDUCED MESSAGE SCHEDULE MEMORY SPACE
// INPUT SCHEDULE MUST CONTAIN FIRST 16 WORDS, THE REST ARE COMPUTED WITHIN THE TRANSFORM FUNCTION
__device__ void sha256_mining_transform_short(WORD state[], WORD m[]){
	WORD a, b, c, d, e, f, g, h, i, t1, t2;

	a = state[0];
	b = state[1];
	c = state[2];
	d = state[3];
	e = state[4];
	f = state[5];
	g = state[6];
	h = state[7];

	// FIRST QUARTER USES PRECOMPUTED MESSAGE SCHEDULE
	for (i = 0; i < 16; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
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

	scheduleExpansion_short(m);

	for (i = 0; i < 16; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[16+i] + m[i];
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

	scheduleExpansion_short(m);

	for (i = 0; i < 16; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[32+i] + m[i];
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

	scheduleExpansion_short(m);

	for (i = 0; i < 16; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[48+i] + m[i];
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

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}


/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/************************************************************************MESSAGE SCHEDULE FUNCTIONS*************************************************************************/

// FULL MESSAGE SCHEDULE COMPUTATION USING FIRST 16 WORDS
// [NOT RECOMMENDED FOR USE DUE TO HIGH MEMORY USAGE (2KB)]
__device__ __inline__ void scheduleExpansion(WORD m[]){
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

// OPTIMIZED MEMORY SCHEDULE COMPUTATION USING A REDUCED 16 WORD STATE
// OPERATIONS ARE IDENTICAL TO THE PREVIOUS FUNCTION, EXCEPT MOD 16
// TO REDUCE THE OVERALL MEMORY USAGE
__device__ __inline__ void scheduleExpansion_short( WORD m[]){
	m[0] = SIG1(m[14]) + m[9] + SIG0(m[1]) + m[0];
	m[1] = SIG1(m[15]) + m[10] + SIG0(m[2]) + m[1];
	m[2] = SIG1(m[0]) + m[11] + SIG0(m[3]) + m[2];
	m[3] = SIG1(m[1]) + m[12] + SIG0(m[4]) + m[3];
	m[4] = SIG1(m[2]) + m[13] + SIG0(m[5]) + m[4];
	m[5] = SIG1(m[3]) + m[14] + SIG0(m[6]) + m[5];
	m[6] = SIG1(m[4]) + m[15] + SIG0(m[7]) + m[6];
	m[7] = SIG1(m[5]) + m[0] + SIG0(m[8]) + m[7];
	m[8] = SIG1(m[6]) + m[1] + SIG0(m[9]) + m[8];
	m[9] = SIG1(m[7]) + m[2] + SIG0(m[10]) + m[9];
	m[10] = SIG1(m[8]) + m[3] + SIG0(m[11]) + m[10];
	m[11] = SIG1(m[9]) + m[4] + SIG0(m[12]) + m[11];
	m[12] = SIG1(m[10]) + m[5] + SIG0(m[13]) + m[12];
	m[13] = SIG1(m[11]) + m[6] + SIG0(m[14]) + m[13];
	m[14] = SIG1(m[12]) + m[7] + SIG0(m[15]) + m[14];
	m[15] = SIG1(m[13]) + m[8] + SIG0(m[0]) + m[15];
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/**********************************************************************ACCELERATED HASHING FUNCTIONS************************************************************************/


// UNIQUE FUNCTION TO PERFORM DOUBLE HASH (80B | 32B) AND TARGET COMPARISON WITHOUT SHA256 STATE
// ONLY UPDATE HASH ON SUCCESS, TRIPLE THE DEFAULT MINING SPEED
// THIS IS A TEST FUNCTION, A FULL FUNCTION WILL NEED TO ACCESS UNIQUE CONSTANT VARIABLES
__device__ int sha256_blockHash_TEST(WORD uniquedata[], BYTE hash[], WORD basedata[], WORD target[]){
	int i;
	int success = 1;
	WORD state[8];    // NEW STATE VARIABLE
	WORD m[16]; 		// MESSAGE SCHEDULE

	// INIT FIRST HASH STATES USING CONSTANT
	state[0] = i_state[0];
	state[1] = i_state[1];
	state[2] = i_state[2];
	state[3] = i_state[3];
	state[4] = i_state[4];
	state[5] = i_state[5];
	state[6] = i_state[6];
	state[7] = i_state[7];


	for(i=0; i < 16; i++){
		m[i] = basedata[i];
	}

	sha256_mining_transform_short(state, m);


	// COMPUTE SECOND MESSAGE SCHEDULE, LOAD FIRST 16 BYTES FROM uniquedata
	m[0] = uniquedata[0];
	m[1] = uniquedata[1];
	m[2] = uniquedata[2];
	m[3] = uniquedata[3];
	// LOAD REMAINING SCHEDULE WITH PRECOMPUTED PADDING VALUES FOR 80 BYTE BLOCK HASH
	for(i=4; i<16; i++){
		m[i] = msgSchedule_80B[i];
	}

	sha256_mining_transform_short(state, m);

	// DOUBLE HASH
	// STORE RESULTS IN THE MESSAGE SCHEDULE, FIRST HALF OF SCHEDULE IS THE PREVIOUS STATE
	m[0] = state[0];
	m[1] = state[1];
	m[2] = state[2];
	m[3] = state[3];
	m[4] = state[4];
	m[5] = state[5];
	m[6] = state[6];
	m[7] = state[7];
	// LOAD REMAINING SCHEDULE WITH PRECOMPUTED PADDING VALUES FOR 32 BYTE HASH
	for(i=8; i<16; i++){
		m[i] = msgSchedule_32B[i];
	}

	// REINITIALIZE STATE VARIABLES
	state[0] = i_state[0];
	state[1] = i_state[1];
	state[2] = i_state[2];
	state[3] = i_state[3];
	state[4] = i_state[4];
	state[5] = i_state[5];
	state[6] = i_state[6];
	state[7] = i_state[7];

	sha256_mining_transform_short(state, m);

	// COMPARE TARGET AGAINST RESULTING STATES
	success = (COMPARE(state[0],target[0]) & COMPARE(state[1],target[1]) & COMPARE(state[2],target[2]) & COMPARE(state[3],target[3]) & COMPARE(state[4],target[4]) & COMPARE(state[5],target[5]) & COMPARE(state[6],target[6]) & COMPARE(state[7],target[7]));


	// CONVERT FROM LITTLE ENDIAN TO BIG ENDIAN BYTE ORDER WHEN THE TARGET IS MET
	if(success == 0){
		// FOR BRANCH PREDICTION
	} else{
		for (i = 0; i < 4; ++i) {
			hash[i]      = (state[0] >> (24 - i * 8)) & 0x000000ff;
			hash[i + 4]  = (state[1] >> (24 - i * 8)) & 0x000000ff;
			hash[i + 8]  = (state[2] >> (24 - i * 8)) & 0x000000ff;
			hash[i + 12] = (state[3] >> (24 - i * 8)) & 0x000000ff;
			hash[i + 16] = (state[4] >> (24 - i * 8)) & 0x000000ff;
			hash[i + 20] = (state[5] >> (24 - i * 8)) & 0x000000ff;
			hash[i + 24] = (state[6] >> (24 - i * 8)) & 0x000000ff;
			hash[i + 28] = (state[7] >> (24 - i * 8)) & 0x000000ff;
		}
	}

	return success;
}
