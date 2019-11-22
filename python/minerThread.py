# MinerThread
# Created 10-16-2019
# Description: This file contains the functions needed for processing of a single miner thread
# Scope: These functions operate on the data for a single miner

import minerBlock
import minerParser

from minerBlock import WorkerBlock
from minerParser import WorkerParser
from minerParser import ParentParser

class MinerThread:
    def __init__(self, total_workers, multilevel, tree_size):
        self.total_workers = total_workers
        self.tree_size = tree_size
        self.errors = 0
        self.time = 0
        self.blockchain = []
        self.label = ""

        if multilevel == 1:
            self.out_dir="outputs/results_"+str(total_workers)+"_pchains/"
        else:
            self.out_dir="outputs/results_"+str(total_workers)+"_chains/"
        return;

    def load(self):
        #print "Loading data for worker "+str(self.worker_number)
        in_file=open(self.in_filename, "r")
        out_file=open(self.out_filename, "r")

        #parser = WorkerParser(self.tree_size, self.worker_number)
        parser = self.createParser()
        parser.setInFile(in_file)
        outlines = out_file.readlines()

    	for ol in outlines: # Parse output file to get computed block headers
    	   parser.search(ol, self.blockchain)

        self.handleMerkle(parser)

    	in_file.close()
    	out_file.close()
        return;

    def verify(self):
    	self.errors = 0;
    	for block in self.blockchain:
    		self.errors += block.test()
        print str(self.label)+" CHAIN LENGTH: "+ str(len(self.blockchain))
    	#print "TOTAL ERRORS: " + str(self.errors)
        return;

    def analysis(self):
    	for block in self.blockchain:
    		self.time += float(block.getTime())
        print str(self.label)+" MINING TIME: "+ str(self.time) + " ms"
    	#print "TOTAL ERRORS: " + str(self.errors)
        return;


class WorkerThread(MinerThread):
    def __init__(self, worker_num, total_workers, multilevel, tree_size):
        MinerThread.__init__(self, total_workers, multilevel, tree_size)
        self.in_filename = "inputs/chain_input" + str(worker_num) + ".txt"
        self.out_filename= self.out_dir+"outputs_" + str(worker_num) + ".txt"
        self.worker_number = worker_num
        self.label = "WORKER "+ str(self.worker_number)
        return;

    def __del__(self):
        #print "Worker thread "+str(self.worker_number)+" deleted"
        return;

    def getBlock(self, index):
        return self.blockchain[index];

    def createParser(self):
        return WorkerParser(self.tree_size, self.worker_number);

    def handleMerkle(self, parser):
        for block in self.blockchain:
            block.setMerkle(self.tree_size, parser.in_file)
        return;




class ParentThread(MinerThread):
    def __init__(self, total_workers, wthreads, tree_size):
        MinerThread.__init__(self, total_workers, 1, tree_size)
        self.in_filename=self.out_dir+"pHashOutputs.txt"
        self.out_filename =self.out_dir+"pBlockOutputs.txt"
        self.worker_threads = wthreads          # link the worker threads under this parent thread
        self.label = "PARENT"
        return;

    def __del__(self):
        #print "Parent thread deleted"
        return;

    def createParser(self):
        return ParentParser(self.tree_size);

    def handleMerkle(self, parser):
        parser.buildMerkle(self.blockchain, self.worker_threads)
        return;
