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
     │   │   ├───benchmarkTest
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
     │   │   │   └───getDifficulty
     │   │   │
     │   │   └───CALCULATIONS
     │   │       ├───calculateDifficulty
     │   │       └───calculateTarget
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
     ├───GLOBAL_FUNCTIONS FIXME
     └───DEVICE_FUNCTIONS FIXME

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

#include "cuda_sha.h"
#include <cuda.h>

/***************************************************************************************************************************************************************************/
/*****************************************************************************TYPE DEFINITIONS******************************************************************************/
/***************************************************************************************************************************************************************************/

typedef unsigned char BYTE;


/***************************************************************************************************************************************************************************/
/****************************************************************************MACRO DEFINITIONS******************************************************************************/
/***************************************************************************************************************************************************************************/

// TODO



/***************************************************************************************************************************************************************************/
/**************************************************************************CONSTANT DEFINITIONS*****************************************************************************/
/***************************************************************************************************************************************************************************/

#define BLOCK_SIZE sizeof(BYTE)*80
#define HASH_SIZE sizeof(BYTE)*32
#define NONCE_SIZE sizeof(BYTE)*4



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

#define NUM_THREADS 256
#define MAX_BLOCKS 16
#define PARENT_BLOCK_SIZE 16
#define DIFFICULTY_LIMIT 32

//#define TARGET_DIFFICULTY 256
#define TARGET_DIFFICULTY 1024
//#define TARGET_DIFFICULTY 2

//#define TARGET_BLOCKS DIFFICULTY_LIMIT*TARGET_DIFFICULTY
#define TARGET_BLOCKS 30

// INPUTS GENERATED = LOOPS * NUM_THREADS * NUM_BLOCKS
#define INPUT_LOOPS 25

// Exponentially reduce computation time, 0 is normal, positive values up to 3 drastically reduce difficulty
#define DIFF_REDUCE 1

// INITIALIZE DEFAULT GLOBAL VARIABLES FOR COMMAND LINE OPTIONS
// INFORMATIVE COMMAND OPTIONS
int DEBUG = 0;            // DEBUG DISABLED BY DEFAULT
int PROFILER = 0;         // PROFILER SWITCH, DISABLED BY DEFAULT
int MINING_PROGRESS = 0;  // MINING PROGRESS INDICATOR DISABLED BY DEFAULT (ONLY ENABLE IF NOT SAVING CONSOLE OUTPUT TO A FILE, OTHERWISE THE STATUS WILL OVERTAKE THE WRITTEN OUTPUT)

// ARCHITECTURE COMMAND OPTIONS
int MULTILEVEL = 0;       // MULTILEVEL ARCHITECTURE DISABLED BY DEFAULT
int NUM_WORKERS = 1;      // NUMBER OF WORKERS 1 BY DEFAULT

// MINING COMMAND OPTIONS
// FIXME: ADD NUM_THREADS, MAX_BLOCKS, OPTIMIZE_BLOCKS, etc. here


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
__host__ void benchmarkTest(int num_workers);

// FIXME Move this with global functions
__global__ void cudaTest(void);
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
__host__ void initTime(cudaStream_t * tStream, BYTE ** time_h, BYTE ** time_d);
__host__ void freeTime(cudaStream_t * tStream, BYTE ** time_h, BYTE ** time_d);
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
__host__ void updateTime(cudaStream_t * tStream, BYTE * time_h, BYTE * time_d);
/*-----------------------------------------------------------------------------MINING GETTERS------------------------------------------------------------------------------*/
__host__ void getTime(BYTE * byte_time);
__host__ void getDifficulty(BYTE * block_h, BYTE ** target, int * target_length, double * difficulty, int worker_num);
/*---------------------------------------------------------------------------MINING CALCULATIONS---------------------------------------------------------------------------*/
__host__ double calculateDifficulty(BYTE * bits);
__host__ int calculateTarget(BYTE * bits, BYTE * target);
/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/

/***************************************************************************************************************************************************************************/
/************************************************************************KERNEL MANAGEMENT FUNCTIONS************************************************************************/
/***************************************************************************************************************************************************************************/
/*-------------------------------------------------------------------------INPUT GENERATION KERNEL-------------------------------------------------------------------------*/
__host__ void genHashKernel(BYTE ** hash_hf, BYTE ** hash_df, BYTE ** seed_h, BYTE ** seed_d, size_t size_hash, size_t size_seed);
/*----------------------------------------------------------------------------MERKLE TREE KERNEL---------------------------------------------------------------------------*/
__host__ void launchMerkle(cudaStream_t * stream, BYTE ** merkle_d, BYTE ** root_d, BYTE ** merkle_h, BYTE ** root_h, int ** flag_d,  int * flag_h, int buffer_size);
/*------------------------------------------------------------------------------MINING KERNEL------------------------------------------------------------------------------*/
__host__ void launchMiner(int num_blocks, cudaStream_t * stream, BYTE ** block_d, BYTE ** hash_d, BYTE ** nonce_d, BYTE ** block_h, BYTE ** hash_h, BYTE ** nonce_h, BYTE ** target_d, BYTE ** time_d, int ** flag_d,  int * flag_h, int * target_length);
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

// TODO ADD GLOBAL FUNCTIONS HERE


/***************************************************************************************************************************************************************************/
/************************************  _______________________________________________________________________________________________  ************************************/
/************************************  |    ___   ___ __   __ ___  ___  ___   ___  _   _  _  _   ___  _____  ___  ___   _  _  ___    |  ************************************/
/************************************  |   |   \ | __|\ \ / /|_ _|/ __|| __| | __|| | | || \| | / __||_   _||_ _|/ _ \ | \| |/ __|   |  ************************************/
/************************************  |   | |) || _|  \ V /  | || (__ | _|  | _| | |_| || .` || (__   | |   | || (_) || .` |\__ \   |  ************************************/
/************************************  |   |___/ |___|  \_/  |___|\___||___| |_|   \___/ |_|\_| \___|  |_|  |___|\___/ |_|\_||___/   |  ************************************/
/************************************  |_____________________________________________________________________________________________|  ************************************/
/************************************                                                                                                   ************************************/
/***************************************************************************************************************************************************************************/

// TODO ADD DEVICE FUNCTION DECLARATIONS HERE


/***************************************************************************************************************************************************************************/
/************************************************************************END FUNCTION DECLARATIONS**************************************************************************/
/***************************************************************************************************************************************************************************/









// HOST INITIALIZATION, BEGIN WITH PARSING COMMAND LINE ARGUMENTS
int main(int argc, char *argv[]){
  // IMPROVED COMMAND LINE ARGUMENT PARSING
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
      printf("\nARGUMENT %i: %s\n", i, arg_in);
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
              printf("%s   fatal:  OPTION '-w' EXPECTS A POSITIVE INTEGER ARGUMENT, RECEIVED '%s' INSTEAD\n\n", argv[0], argv[i+1]);
              err_flag = 1;
              break;
            }
          } else{
            printf("%s   fatal:  ARGUMENT EXPECTED AFTER '-w'\n\n", argv[0]);
            err_flag = 1;
            break;
          }
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
\t  -w #   \t\t NUMBER OF WORKER CHAINS AS A POSITIVE INTEGER (DEFAULT: 1)\n", argv[0]);
  }
  // RUN THE SELECTED IMPLEMENTATION(S)
  else{
    // RUN DEVICE QUERY TO SEE AVAILABLE RESOURCES
    if(query_flag == 1){
      hostDeviceQuery();
    }
    // RUN FUNCTIONAL TEST FOR THE HASHING FUNCTIONS
    if(test_flag == 1){
      printf("FUNCTIONAL TESTING SELECTED!!!!! (TODO: ADD FUNC TEST FOR SINGLE/DOUBLE SHA256 & MERKLE ROOT)\n\n");
    //  hostTestProcess();
    }
    // RUN BENCHMARK TEST FOR DEVICE PERFORMANCE
    if(bench_flag == 1){
      printf("BENCHMARK TESTING SELECTED!!!!!\n");
      benchmarkTest(NUM_WORKERS);
    }
    // START MINING IF DRY RUN IS NOT SELECTED
    if(dry_run == 0){
      // TODO CHECK FOR PROFILER ENABLED, INCLUDE LOGGING OF ENABLED SETTINGS
      hostCoreProcess(NUM_WORKERS, MULTILEVEL);
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

/*----------------------------GLOBAL TIMING VARIABLES-----------------------------*/
float total_time[6];
cudaStream_t g_timeStream;
cudaEvent_t g_timeStart, g_timeFinish;
createCudaVars(&g_timeStart, &g_timeFinish, &g_timeStream);

cudaEvent_t g_time[4];
for(int i = 0; i < 4; i++){
  cudaEventCreate(&g_time[i]);
}
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
/*********************************************************WORKER CREATION**********************************************************/
/**********************************************************************************************************************************/

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

    int chain_blocks[num_workers];
    int diff_level[num_workers];
    int errEOF[num_workers];
    int target_length[num_workers];
    double difficulty[num_workers];
/*-----------------------------------------------------------------------*/
/******************************INITIALIZATION*****************************/
/*---------------------------MEMORY ALLOCATION---------------------------*/
    allocWorkerMemory(num_workers, hash_h, hash_d, block_h, block_d);
    allocFileStrings(outFiles, num_workers);
/*-------------------------BLOCK INITIALIZATION--------------------------*/
    initializeHashes(inFiles, num_workers, hash_h);
    initializeWorkerBlocks(hash_h, block_h, num_workers);
    initializeOutputs(outFiles, out_location, num_workers);
    sprintf(worker_diff_file, "%s/workerDiffScale.txt",out_location);
/*------------------------THREAD INITIALIZATION---------------------------*/
    for(int i = 0; i < num_workers; i++){
        chain_blocks[i] = 0; diff_level[i] = 1; errEOF[i] = 0;
        createCudaVars(&t1[i], &t2[i], &streams[i]);
        cudaEventCreate(&diff_t1[i]);
        cudaEventCreate(&diff_t2[i]);
        allocMiningMemory(&target_h[i], &target_d[i], &nonce_h[i], &nonce_d[i], &flag_d[i]);
        getDifficulty(block_h[i], &target_h[i], &target_length[i], &difficulty[i], i+1);
    }
/*------------------------------------------------------------------------*/
/**************************************************************************/

/**********************************************************************************************************************************/
/*********************************************************PARENT CREATION**********************************************************/
/**********************************************************************************************************************************/

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
      /*---------------------------MEMORY ALLOCATION---------------------------*/
          allocParentMemory(pHash_h, pHash_d, &pBlock_h, &pBlock_d, &pRoot_h, &pRoot_d, &pHash_out_h, &pHash_out_d, &pHash_merkle_h, &pHash_merkle_d);
      /*-------------------------BLOCK INITIALIZATION--------------------------*/
          sprintf(bfilename, "outputs/results_%i_pchains/pBlockOutputs.txt",num_workers);
          sprintf(hfilename, "outputs/results_%i_pchains/pHashOutputs.txt",num_workers);
          initializeParentBlock(pBlock_h);
          initializeParentOutputs(bfilename, hfilename);
      /*------------------------CHAIN INITIALIZATION---------------------------*/
          createCudaVars(&p1, &p2, &pStream);
          cudaEventCreate(&diff_p1);
          cudaEventCreate(&diff_p2);
          cudaEventCreate(&buff_p1);
          cudaEventCreate(&buff_p2);
          allocMiningMemory(&ptarget_h, &ptarget_d, &pnonce_h, &pnonce_d, &pflag_d);
          getDifficulty(pBlock_h, &ptarget_h, &ptarget_length, &pdifficulty, 0);
          cudaMemcpyAsync(ptarget_d, ptarget_h, HASH_SIZE, cudaMemcpyHostToDevice, pStream);
    }
/*-------------------------FLAG INITIALIZATION----------------------------*/
    BYTE * time_h;
    BYTE * time_d;
    cudaStream_t tStream;
    initTime(&tStream, &time_h, &time_d);

    flag_h = (int *)malloc(sizeof(int));
    flag_h[0] = 0;
    int FLAG_TARGET = 0;
    int PROC_REMAINING = num_workers+multilevel;

    int mining_state;

/*------------------------------------------------------------------------*/
/**************************************************************************/

/**********************************************************************************************************************************/
/********************************************************MINING LOOP BEGIN*********************************************************/
/**********************************************************************************************************************************/
cudaEventRecord(g_time[0], g_timeStream);
/*--------------------------------------------------------------------------------------------------------------------------------*/
/**************************************************INITIALIZE ASYNCHRONOUS STREAMS*************************************************/

    for(int i = 0; i < num_workers; i++){
        logStart(i, 1, hash_h[i]);
        cudaEventRecord(t1[i], streams[i]);
        cudaEventRecord(diff_t2[i], streams[i]);
        cudaEventRecord(diff_t1[i], streams[i]);
        cudaMemcpyAsync(target_d[i], target_h[i], HASH_SIZE, cudaMemcpyHostToDevice, streams[i]);
        launchMiner(MAX_BLOCKS/num_workers, &streams[i], &block_d[i],  &hash_d[i], &nonce_d[i], &block_h[i], &hash_h[i], &nonce_h[i], &target_d[i], &time_d, &flag_d[i], flag_h, &target_length[i]);
        // SET EVENT TO RECORD AFTER KERNEL COMPLETION
        cudaEventRecord(t2[i], streams[i]);
    }
    // START PARENT TIMERS
    if(multilevel == 1){
      cudaEventRecord(buff_p2, pStream);
      cudaEventRecord(buff_p1, pStream);
      cudaEventRecord(diff_p1, pStream);
    }
    cudaEventRecord(g_time[1], g_timeStream);
    /*--------------------------------------------------------------------------------------------------------------------------------*/
    /********************************************BEGIN MINING UNTIL TARGET BLOCKS ARE FOUND********************************************/
    int block_total = 0;
    while(block_total < TARGET_BLOCKS || PROC_REMAINING != 0){
      updateTime(&tStream, time_h, time_d);
      if(MINING_PROGRESS == 1){
        mining_state = printProgress(mining_state, multilevel, num_workers, pchain_blocks, chain_blocks);
      }
      // SET FLAG_TARGET TO 1
      if(block_total >= TARGET_BLOCKS && FLAG_TARGET == 0){
          FLAG_TARGET = 1;
          cudaEventRecord(g_time[2], g_timeStream);
          printLog("\n\n**********************************************\nTARGET REACHED, FINISHING REMAINING PROCESSES*\n**********************************************\n\n");
      }
      /*--------------------------------------------------------------------------------------------------------------------------------*/
      /*******************************************LOOP OVER MINERS TO CHECK STREAM COMPLETION********************************************/
      for(int i = 0; i < num_workers; i++){

        if(multilevel == 1){  // CHECK PARENT MINER COMPLETION STATUS IF MULTILEVEL
          if(cudaStreamQuery(pStream) == 0 && parentFlag == 1){   // PARENT CHAIN RESULTS ARE READY, PROCESS OUTPUTS AND PRINT
            // processParent
            pchain_blocks++;
            returnMiner(&pStream, &pBlock_d,  &pHash_out_d, &pnonce_d, &pBlock_h,  &pHash_out_h, &pnonce_h);
            cudaEventSynchronize(p2);
            cudaEventElapsedTime(&ptimingResults, p1, p2);
            printOutputFile(bfilename, &pBlock_h[0], pHash_out_h, pnonce_h, pchain_blocks, ptimingResults, pdifficulty, -1, 1);
            updateParentHash(pBlock_h, pHash_out_h);
            parentFlag = 0;
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
                  updateTime(&tStream, time_h, time_d);
                  if(MINING_PROGRESS == 1){
                    mining_state = printProgress(mining_state, multilevel, num_workers, pchain_blocks, chain_blocks);
                  }
                  // MONITOR WORKER TIMING WHILE WAITING
                  for(int j = 0; j < num_workers; j++){
                    if((cudaStreamQuery(streams[i]) == cudaSuccess && diff_timing[i] <= 0) && (chain_blocks[i] >= diff_level[i] * DIFFICULTY_LIMIT || FLAG_TARGET == 1)){
                        cudaEventRecord(diff_t2[i], streams[i]);
                        cudaEventSynchronize(diff_t2[i]);
                        cudaEventElapsedTime(&diff_timing[i], diff_t1[i], diff_t2[i]);
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
          }
          // PARENT BUFFER IS READY, EXIT FOR LOOP TO BEGIN PARENT EXECUTION
          if(pbuffer_blocks == PARENT_BLOCK_SIZE){
              printDebug("NEW PARENT BLOCK IS READY!\n");
              break;
          }
        } // END PARENT CHAIN MONITOR
        // PROCESS WORKER RESULTS AND START NEXT BLOCK IF THE TARGET HAS NOT BEEN MET
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
              updateDifficulty(block_h[i], diff_level[i]);
              getDifficulty(block_h[i], &target_h[i], &target_length[i], &difficulty[i], i+1);
              cudaMemcpyAsync(target_d[i], target_h[i], HASH_SIZE, cudaMemcpyHostToDevice, streams[i]);
              cudaEventRecord(diff_t1[i], streams[i]);
              diff_level[i]++;
            }
          }

          // MINE NEXT BLOCK ON THIS WORKER IF TARGET HASN'T BEEN REACHED
          if(FLAG_TARGET == 0){
            errEOF[i] = updateBlock(inFiles[i], block_h[i], hash_h[i]);
            if(errEOF[i] == 1){
              char eof_str[20];
              sprintf(eof_str, "WORKER %i INPUT EOF!", i+1);
              printErrorTime(error_filename, eof_str, 0.0);
            }
            logStart(i, chain_blocks[i]+1, hash_h[i]);
            cudaEventRecord(t1[i], streams[i]);
            launchMiner(MAX_BLOCKS/num_workers, &streams[i], &block_d[i],  &hash_d[i], &nonce_d[i], &block_h[i], &hash_h[i], &nonce_h[i], &target_d[i], &time_d, &flag_d[i], flag_h, &target_length[i]);
            cudaEventRecord(t2[i], streams[i]);
          } else{ // EXECUTION COMPLETED, DELETE CUDA VARS TO PREVENT ADDITIONAL ENTRY INTO THIS CASE
            destroyCudaVars(&t1[i], &t2[i], &streams[i]);
            cudaEventDestroy(diff_t1[i]);
            cudaEventDestroy(diff_t2[i]);
            PROC_REMAINING--;
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
            cudaEventRecord(diff_p2, pStream);
            cudaEventSynchronize(diff_p2);
            cudaEventElapsedTime(&pdiff_timing, diff_p1, diff_p2);
            printDifficulty(bfilename, -1, pdifficulty, pdiff_timing, (pchain_blocks-(pdiff_level-1)*DIFFICULTY_LIMIT));

            updateDifficulty(pBlock_h, pdiff_level);
            getDifficulty(pBlock_h, &ptarget_h, &ptarget_length, &pdifficulty, 0);
            cudaMemcpyAsync(ptarget_d, ptarget_h, HASH_SIZE, cudaMemcpyHostToDevice, pStream);
            cudaEventRecord(diff_p1, pStream);
            pdiff_level++;
          }
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
          launchMiner(4, &pStream, &pBlock_d,  &pHash_out_d, &pnonce_d, &pBlock_h,  &pHash_out_h, &pnonce_h, &ptarget_d, &time_d, &pflag_d, flag_h, &ptarget_length);
          cudaEventRecord(p2, pStream);
          pbuffer_blocks = 0;
          parentFlag = 1;

          // FINAL ITERATION, WAIT FOR PARENT STREAM TO FINISH
          if(PROC_REMAINING == 1){
            while(cudaStreamQuery(pStream) != 0){
              updateTime(&tStream, time_h, time_d);
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

            cudaEventRecord(diff_p2, pStream);
            cudaEventSynchronize(diff_p2);
            cudaEventElapsedTime(&pdiff_timing, diff_p1, diff_p2);
            printDifficulty(bfilename, -1, pdifficulty, pdiff_timing, (pchain_blocks-(pdiff_level-1)*DIFFICULTY_LIMIT));
            // CLEAN UP PARENT CUDA VARS, SET PROCS_REMAINING TO ZERO TO EXIT
            destroyCudaVars(&p1, &p2, &pStream);
            cudaEventDestroy(diff_p1);
            cudaEventDestroy(diff_p2);
            cudaEventDestroy(buff_p1);
            cudaEventDestroy(buff_p2);
            PROC_REMAINING--;
          }
      }
    } // WHILE LOOP END
    cudaEventRecord(g_time[3], g_timeStream);
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
    freeTime(&tStream, &time_h, &time_d);
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
    if(multilevel == 1){
      printDebug((const char*)"FREEING PARENT MEMORY");
      freeParentMemory(pHash_h, pHash_d, &pBlock_h, &pBlock_d, &pRoot_h, &pRoot_d, &pHash_out_h, &pHash_out_d, &pHash_merkle_h, &pHash_merkle_d);
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
__global__ void cudaTest(void){
	//SHA256_CTX ctx;
//	printf("MERKLE ROOT COPY TEST PRINT: \n");
		printf("THREAD %i WORKING\n", threadIdx.x);

//	printf("MERKLE ROOT COPY TEST FINISHED\n");
}

__host__ void benchmarkTest(int num_workers){
  // INITIALIZE BENCHMARK VARIABLES
  BYTE * test_block_h;
  BYTE * test_block_d;
  char logResult[1000];
  float bench_time, worker_time, block_time, thread_time;
  cudaEvent_t bench_s, bench_f;
  cudaStream_t bench_stream;

  createCudaVars(&bench_s, &bench_f, &bench_stream);


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
  benchKernel<<<MAX_BLOCKS/num_workers, NUM_THREADS, 0, bench_stream>>>(test_block_d);
  //TODO ADD KERNEL CALL HERE

  cudaEventRecord(bench_f, bench_stream);

  cudaDeviceSynchronize();

  cudaEventElapsedTime(&bench_time, bench_s, bench_f);

  worker_time = 0x0FFFFFFF/(bench_time/1000);
  block_time = worker_time/(MAX_BLOCKS/num_workers);
  thread_time = block_time/NUM_THREADS;





  sprintf(logResult, "\n/****************************BENCHMARK ANALYSIS FOR %i WORKER CHAINS****************************/\n\
  TOTAL TIME: %f\n\
  WORKER HASHES PER SECOND: %f\n\
  BLOCK HASHES PER SECOND: %f \n\
  THREAD HASHES PER SECOND: %f \n\
  /**********************************************************************************************/\n\
  ", num_workers, bench_time, worker_time, block_time, thread_time);
  printLog(logResult);

  destroyCudaVars(&bench_s, &bench_f, &bench_stream);
  free(test_block_h);
  cudaFree(test_block_d);
  return;
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
  cudaStreamCreate(stream);
}
__host__ void destroyCudaVars(cudaEvent_t * timing1, cudaEvent_t * timing2, cudaStream_t * stream){
  cudaEventDestroy(*timing1);
  cudaEventDestroy(*timing2);
  cudaStreamDestroy(*stream);
}

/*-------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/************************************************************************TIME MANAGEMENT FUNCTIONS**************************************************************************/
// CREATE AND FREE FUNCTIONS FOR UPDATING THE DEVICE TIME
__host__ void initTime(cudaStream_t * tStream, BYTE ** time_h, BYTE ** time_d){
  *time_h = (BYTE *)malloc(4*sizeof(BYTE));
  cudaMalloc((void **) time_d, 4*sizeof(BYTE));
  cudaStreamCreate(tStream);
  updateTime(tStream, *time_h, *time_d);
}
__host__ void freeTime(cudaStream_t * tStream, BYTE ** time_h, BYTE ** time_d){
  free(*time_h);
  cudaFree(*time_d);
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
__host__ void updateTime(cudaStream_t * tStream, BYTE * time_h, BYTE * time_d){
  getTime(time_h);
  cudaMemcpyAsync(time_d, time_h, 4*sizeof(BYTE), cudaMemcpyHostToDevice, *tStream);
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
  int padding = (32 - bits[0])-DIFF_REDUCE;
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
__host__ void launchMiner(int num_blocks, cudaStream_t * stream, BYTE ** block_d, BYTE ** hash_d, BYTE ** nonce_d, BYTE ** block_h, BYTE ** hash_h, BYTE ** nonce_h, BYTE ** target_d, BYTE ** time_d, int ** flag_d,  int * flag_h, int * target_length){
//  printf("STARTING MINER MEMORY TRANSFER!\n");
  cudaMemcpyAsync(*block_d, *block_h, BLOCK_SIZE, cudaMemcpyHostToDevice, *stream);
  cudaMemcpyAsync(*flag_d, flag_h, sizeof(int), cudaMemcpyHostToDevice, *stream);
//  printf("STARTING MINER WITH %i BLOCKS AND %i THREADS!!!\n", num_blocks, NUM_THREADS);
  minerKernel<<<num_blocks,NUM_THREADS, 0, *stream>>>(*block_d, *hash_d, *nonce_d, *target_d, *time_d, *flag_d, *target_length, MAX_BLOCKS);
//  cudaTest<<<num_blocks, NUM_THREADS>>>();
//  cudaDeviceSynchronize();
//  printf("FINISHED MINER!\n");
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
 // TODO ADD GLOBAL FUNCTIONS




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
