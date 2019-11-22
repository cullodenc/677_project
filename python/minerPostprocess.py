# MinerPostprocess
# Created 10-16-2019
# Description: This file contains the functions needed for postprocessing of the miner output data
# Scope: These functions operate on a single design run, analyzing multiple workers and comparing data between them

# TODO: Functions to be added:
# DONE- Input Catcher: parse arguments such as postprocessing steps and other options (file locations, merkle sizes, etc)
# DONE- Worker Loader: Parse the output files and pull out essential information
# DONE- Worker Verifier: Compare the miner produced output to expected values, flag any inconsistencies
# Worker Data Storage: Store mining data in a spreadsheet for organization and extrapolation
# Worker Analysis: Manipulate stored data to calculate essential statistics needed for performance evaluation
# Worker Visualization: Use stored data to generate graphs and other visualization aides

#TODO: Use this file for later batch analysis when larger groups of results need to be analyzed
#TODO: Add command arguments to only run specific parts of the process, such as only verification, or analysis and graphing
#TODO dataWriter.py - Should contain functions needed to store data in a spreadsheet, possibly analysis and graphing features as well

import sys

import minerThread
from minerThread import WorkerThread
from minerThread import ParentThread

# FIXME May or may not need fancy logging
# LOGGING STUFF
#import logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.debug('This is a log message.')


def main():
	if (len(sys.argv) < 3):
		print "Usage: python minerPostprocess.py [-w <int>] [-wTree <int>] [-pTree <int>] [--multi] [--silent]"
		sys.exit()

	#print "Python postprocessing script for " + sys.argv[1] + " workers and multilevel == "+ str(sys.argv[2])
	workers = 1
	wTreeSize=64
	pTreeSize=16
	multi = 0
	silent = 0
	skip = 0
	error=0

	# Better input parsing
	for i in range(1, len(sys.argv)):
		if(skip == 1):
			skip = 0
			continue
		elif(sys.argv[i] == '-w'):
			workers = sys.argv[i+1]
			skip = 1
		elif(sys.argv[i] == '-wTree'):
			wTreeSize = sys.argv[i+1]
			skip = 1
		elif(sys.argv[i] == '-pTree'):
			pTreeSize = sys.argv[i+1]
			skip = 1
		elif(sys.argv[i] == '--multi'):
			multi = 1
		elif(sys.argv[i] == '--silent'):
			silent=1
		else:
			error=1
			print "ERROR: Unknown input "+ sys.argv[i] + " detected"


	if(error == 0):

		if(multi == 1):
			print "["+str(workers)+" Chains | Multilevel]"
		else:
			print "["+str(workers)+" Chains]"

		total_errors = processMiner(int(workers), int(multi), int(silent), int(wTreeSize), int(pTreeSize))
		print "Total Errors: "+ str(total_errors) + "\n"
	else:
		print "Postprocessing aborted due to 1 or more errors"

	return;

# END main()

#TODO USE to run routine on each thread
def processMiner(total_workers, multilevel, silent, wTreeSize, pTreeSize):
	minerThreads = []  # Store threads for this miner
	total_errors = 0
	parentThread = None
	for i in range(1, total_workers+1):
		#print "TESTING WORKER "+ str(i)
		if(silent!=1):print "***************************\nTESTING WORKER "+ str(i) +"\n***************************\n"
		minerThreads.append(WorkerThread(i, total_workers, multilevel, wTreeSize))
		minerThreads[i-1].load()
		minerThreads[i-1].verify()
		minerThreads[i-1].analysis()
		if(silent!=1):print "--------------------------------------------------------------------------------------\n"
		total_errors+= minerThreads[i-1].errors
		# LOAD WORKER
		# VERIFY WORKER
		# STORE DATA
		#checkInputs(total_workers, i)

	if(multilevel == 1):
		#print "TODO Add parent thread verification"
		if(silent!=1):print "***************************\nTESTING PARENT \n***************************\n"
		parentThread = ParentThread(total_workers, minerThreads, pTreeSize)
		parentThread.load()
		parentThread.verify()
		if(silent!=1):print "--------------------------------------------------------------------------------------\n"
		total_errors+=parentThread.errors
	return total_errors;

# END processMiner()

main()
#END Postprocessing
