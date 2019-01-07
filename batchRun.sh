# Baseline single chain
./cuda_miner --debug --benchmark -w 1 -t 525 | tee outputs/results_1_chains/console.out

# Mirror base parent chain
./cuda_miner --debug --benchmark --multi -w 1 -t 525 | tee outputs/results_1_pchains/console.out

# Higher difficulty parent chains to account for slower difficulty scaling per block 
./cuda_miner --debug --benchmark --multi -w 2 -t 1050 | tee outputs/results_2_pchains/console.out
./cuda_miner --debug --benchmark --multi -w 4 -t 2100 | tee outputs/results_4_pchains/console.out
./cuda_miner --debug --benchmark --multi -w 8 -t 4200 | tee outputs/results_8_pchains/console.out
