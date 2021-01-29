

import minerThread
from minerThread import WorkerThread
from minerThread import ParentThread

import spreadsheetWriter
from spreadsheetWriter import SpreadsheetWriter

class MinerGroup:
    def __init__(self, total_workers, multilevel, wTreeSize, pTreeSize):
        self.minerThreads = []
    	self.total_errors = 0
    	self.parentThread = None
        self.mainChain = []

        self.workers = total_workers
        self.multilevel= multilevel
        self.wTreeSize= wTreeSize
        self.pTreeSize = pTreeSize

        self.name = str(total_workers) + "_Worker_Design"
        self.label = str(total_workers) + " Workers"
        return;

    def processMiner(self, silent):
    	#minerThreads = []  # Store threads for this miner
    	#total_errors = 0
    	#parentThread = None
    	spreadsheet_writer = SpreadsheetWriter(self.workers, self.multilevel) # TODO Add inputs for unique spreadsheet name


    	for i in range(1, self.workers+1):
    		#print "TESTING WORKER "+ str(i)
    		if(silent!=1):print "***************************\nTESTING WORKER "+ str(i) +"\n***************************\n"
    		self.minerThreads.append(WorkerThread(i, self.workers, self.multilevel, self.wTreeSize))
    		self.minerThreads[i-1].load()
    		self.minerThreads[i-1].verify()
    		self.minerThreads[i-1].analysis()
    		spreadsheet_writer.addWorksheet(self.minerThreads[i-1]);
    		spreadsheet_writer.writeResults(self.minerThreads[i-1]);


    		#for diff in minerThreads[i-1].diff_block:
    		#	print "TARGET " + str(diff.target) + " | TIME: " + str(diff.total_time)


    		if(silent!=1):print "--------------------------------------------------------------------------------------\n"
    		self.total_errors+= self.minerThreads[i-1].errors
    		# LOAD WORKER
    		# VERIFY WORKER
    		# STORE DATA
    		#checkInputs(total_workers, i)

    	spreadsheet_writer.addStatsheet(self.minerThreads);

    	#spreadsheet_writer.

    	if(self.multilevel == 1):
    		#print "TODO Add parent thread verification"
    		if(silent!=1):print "***************************\nTESTING PARENT \n***************************\n"
    		self.parentThread = ParentThread(self.workers, self.minerThreads, self.pTreeSize)
    		self.parentThread.load()
    		self.parentThread.verify()
    		if(silent!=1):print "--------------------------------------------------------------------------------------\n"
    		self.total_errors+=self.parentThread.errors
    	return self.total_errors;

    # END processMiner()

    def createComparison(self):
        for thread in self.minerThreads:
            for block in thread.blockchain:
                self.mainChain.append(block)
                self.mainChain[-1].count = len(self.mainChain)-1

        self.mainChain.sort(key=blockSort)

        for block_count, block in enumerate(self.mainChain):
            block.count = block_count


        return;

# Result sorting key funtion
def blockSort(MinerBlock):
	return int(MinerBlock.elapsed_time);
