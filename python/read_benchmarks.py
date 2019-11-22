import subprocess
import xlsxwriter
import sys
import os
import re


class BenchmarkResult:
	def __init__(self, num_workers):
		self.workers=num_workers;
		self.threads=None;
		self.blocks=None;
		self.block_iterations=None;
		self.time=None;
		self.hashrate=None;
		self.worker_rate=None;
		self.block_rate=None;
		self.thread_rate=None;
		return;

	def setHashrate(self, hashrate):
		self.hashrate = str(hashrate);
		return;

	def setBlocks(self, num_blocks):
		self.blocks = str(num_blocks);
		return;

	def setThreads(self, threads):
		self.threads = str(threads);
		return;

	def setIterations(self, num_iterations):
		self.block_iterations = str(num_iterations);
		self.thread_iterations = str(int(num_iterations) * int(self.threads));
		return;

	def setTime(self, time):
		self.time = str(time);
		return;

	def printResults(self):
		print("\t " +str(self.workers) +" \t | " + str(self.threads)+"   \t | " + str(self.blocks) + " \t\t | " +str(self.hashrate)+" (MH/s) \t | " +str(self.time)+ " (ms) \t | " + str(self.block_iterations))
		return;

	def printStats(self):
		self.worker_rate = (float(self.block_iterations) * float(self.threads))/(float(self.time)*1000)		# MH/s
		self.block_rate = float(self.worker_rate)/float(self.blocks)
		self.thread_rate = (float(self.block_rate)*1000)/float(self.threads)
		print( "\t"+str(self.workers) + " \t | \t " + str(self.worker_rate) + "      \t | \t " + str(self.block_rate) + "      \t | \t " + str(self.thread_rate))
		return;

# END BenchmarkResult


# Result sorting key funtion
def chainSort(BenchmarkResult):
	return int(BenchmarkResult.workers);

class Subchain:
	def __init__(self, num_threads, group):
		self.name= str(num_threads)+"_threads_"+str(group)
		self.threads=num_threads;
		self.benchmarks = [];
		self.worksheet = None;
		self.workbook = None;
		self.curr_row = 1;
		self.stat_titles = ["Workers per Miner", "Blocks per Worker", "Threads per Block", "Time (ms)", "Block Iterations", "Thread Iterations", "Hashrate per Worker (MH/s)", "Hashrate per Block (MH/s)", "Hashrate per Thread (KH/s)"]
		self.stat_map = {
					"workers" : {"col" : 0, "label": "Workers per Miner"},
					"blocks" : {"col" : 1, "label": "Blocks per Worker"},
					"threads" : {"col" : 2, "label": "Threads per Block"},
					"time" : {"col" : 3, "label": "Time (ms)"},
					"block_iterations": {"col" : 4, "label": "Block Iterations"},
					"thread_iterations": {"col" : 5, "label": "Thread Iterations"},
					"worker_rate": {"col" : 6, "label": "Hashrate (MH/s)"},
					"block_rate": {"col" : 7, "label": "Hashrate (MH/s)"},
					"thread_rate": {"col" : 8, "label": "Hashrate (KH/s)"}
					}
		return;

	def addWorksheet(self, workbook):
		self.workbook = workbook
		self.worksheet = workbook.add_worksheet(self.name)
		cell_format = workbook.add_format({'font_size':10})
		header_format = workbook.add_format({'bold': True, 'font_size':10})
		self.worksheet.set_column('A:I', 20, cell_format)
		self.worksheet.set_row( 0, 15, header_format)
		for col_num, data in enumerate(self.stat_titles):
			self.worksheet.write(0, col_num, data)
		return;

	def writeResults(self):
		count = 0;
		for row, res in enumerate(self.benchmarks):
			self.worksheet.write(row+self.curr_row, 0, int(res.workers))
			self.worksheet.write(row+self.curr_row, 1, int(res.blocks))
			self.worksheet.write(row+self.curr_row, 2, int(res.threads))
			self.worksheet.write(row+self.curr_row, 3, float(res.time))
			self.worksheet.write(row+self.curr_row, 4, int(res.block_iterations))
			self.worksheet.write(row+self.curr_row, 5, int(res.thread_iterations))
			self.worksheet.write(row+self.curr_row, 6, float(res.worker_rate))
			self.worksheet.write(row+self.curr_row, 7, float(res.block_rate))
			self.worksheet.write(row+self.curr_row, 8, float(res.thread_rate))
			count+=1

		self.curr_row += count;
		return;

	def addChart(self, category, values, title):
		self.chart = self.workbook.add_chart({'type': 'scatter', 'subtype':'smooth'})
		self.chart.set_title({'name': title})
		self.chart.set_x_axis({'name': [self.name, 0, (self.stat_map[category]["col"])]})
		self.chart.set_y_axis({'name': (self.stat_map[values]["label"])})
		self.chart.add_series({
			'name'			:	str(self.threads) + " Threads/Block",
			'categories'	:	[self.name, 1, (self.stat_map[category]["col"]), self.curr_row, (self.stat_map[category]["col"])],
			'values'		:	[self.name, 1, (self.stat_map[values]["col"]), self.curr_row, (self.stat_map[values]["col"])],
			#'data_labels': {'value':True, 'category': True},
			'marker':{'type':'automatic'}
			})

		self.worksheet.insert_chart('A'+str(self.curr_row+5), self.chart)
		self.curr_row += 15;
		return;


	def initChart(self, workbook, category, values, title):
		chart = workbook.add_chart({'type': 'scatter', 'subtype':'smooth'})
		chart.set_title({'name': title})
		chart.set_x_axis({'name': [self.name, 0, (self.stat_map[category]["col"])]})
		chart.set_y_axis({'name': (self.stat_map[values]["label"])})
		return chart;


	def appendChart(self, category, values, chart):
		chart.add_series({
			'name'			:	str(self.threads) + " Threads/Block",
			'categories'	:	[self.name, 1, (self.stat_map[category]["col"]), self.curr_row, (self.stat_map[category]["col"])],
			'values'		:	[self.name, 1, (self.stat_map[values]["col"]), self.curr_row, (self.stat_map[values]["col"])],
			#'data_labels': {'value':True, 'category': True},
			'marker':{'type':'automatic'}
			})
		return;

	def parseFile(self, pathname):
		resfile = open(pathname)
		reslines = resfile.readlines()
		for l in reslines:
			if re.search("WORKER HASHRATE:", l):
				self.benchmarks[-1].setHashrate(l.split()[2]);
			elif re.search("NUM BLOCKS:", l):
				self.benchmarks[-1].setBlocks(l.split()[2]);
			elif re.search("NUM THREADS:", l):
				self.benchmarks[-1].setThreads(l.split()[2]);
			elif re.search("TOTAL ITERATIONS:", l):
				self.benchmarks[-1].setIterations(l.split()[2]);
			elif re.search("TOTAL TIME:", l):
				self.benchmarks[-1].setTime(l.split()[2]);
		return;

# END Subchain

class ChainGroup:
	def __init__(self, name):
		self.chains = []
		self.name=name
		self.label=name.upper()
		return;

	def setSubchain(self, filename):
		found = 0
		for subchain in self.chains:
			if(subchain.threads == filename.split('_')[1]):
				current_chain = subchain
				found = 1

		if(found == 0):
			self.chains.append(Subchain(filename.split('_')[1], self.name))
			current_chain = self.chains[-1]

		return current_chain;

	def printResults(self):
		for sub in self.chains:
			sub.benchmarks.sort(key=chainSort)

		for chain in self.chains:
			print "\n\n---------------------------------------------"+str(self.label)+" CHAIN RESULTS---------------------------------------------"
			print("# WORKERS \t | # THREADS \t | # BLOCKS \t | HASHRATE (MH/s) \t | TIME (ms) \t\t | ITERATIONS")	# BENCHMARK RESULT TABLE FORMATTING
			for res in chain.benchmarks:
				res.printResults();

			print("\n# WORKERS \t | RATE PER WORKER (MH/s)\t | RATE PER BLOCK (MH/s)\t | RATE PER THREAD(KH/s)")	# HASHRATE TABLE FORMATTING
			for res in chain.benchmarks:
				res.printStats();

		return;

	def createCharts(self, workbook):
		start = 1;
		self.charts = [];

		for subchain in self.chains:
			if(start == 1):
				start = 0;
				self.charts.append(subchain.initChart(workbook, 'workers', 'worker_rate', "Hashrates per Worker"))
				self.charts.append(subchain.initChart(workbook, 'workers', 'block_rate', "Hashrates per Block"))
				self.charts.append(subchain.initChart(workbook, 'workers', 'thread_rate', "Hashrates per Thread"))

			row = 0
			col = 0
			threads = 0;
			subchain.addWorksheet(workbook);
			subchain.writeResults();
			subchain.addChart('workers', 'worker_rate', "Hashrates per Worker");
			subchain.addChart('workers', 'block_rate', "Hashrates per Block");
			subchain.addChart('workers', 'thread_rate', "Hashrates per Thread");

			subchain.appendChart('workers', 'worker_rate', self.charts[0]);
			subchain.appendChart('workers', 'block_rate', self.charts[1]);
			subchain.appendChart('workers', 'thread_rate', self.charts[2]);


		comparison_ws = workbook.add_worksheet(str(self.name)+"_level_comparison")
		c_row = 1
		for c in self.charts:
			comparison_ws.insert_chart('A'+str(c_row), c)
			c_row+=15
		return;

#END ChainGroup

# main benchmark parsing
if (len(sys.argv) < 2):
	print "Usage: python read_benchmarks.py [directories]"
	sys.exit()

errors = 0
dirs = []
topDirs = []
args = sys.argv[1:]

for a in args:
	if a[-1:] != '/':
		a = a + '/'

	if not os.path.isdir(a):
		errors += 1
		print "Error: '" + a + "' is not a directory."
	else:
		topDirs.append(a)

if errors > 0:
	sys.exit()

# Parse directory to get benchmark results
for d in topDirs:
	for dirName, subdirList, fileList in os.walk(d):
		if "results" in dirName:
			dirs.append(dirName + '/')


worker_group = ChainGroup("single")
parent_group = ChainGroup("multi")


# Search each output directory, organized by number of threads
for d in dirs:
    # Get each output file name
    for root, dirs, files in os.walk(d):
        for filename in files:
			if "benchmark" in filename:
				# Select the group for storing the data for this design
				if("pchains" in d):
					main_group = parent_group;
				else:
					main_group = worker_group;

				# Check to see if a subchain exists for this number of threads per block, if not create a new subchain
				current_chain = main_group.setSubchain(filename)
				current_chain.benchmarks.append(BenchmarkResult(d.split('_')[1]));

				# Parse the file and store the benchmark data
				pathname = os.path.realpath(d + filename)
				current_chain.parseFile(pathname)


#Print resulting benchmark lists
worker_group.printResults();
parent_group.printResults();

xlsx_dir = os.path.realpath("outputs/spreadsheets/")
if not os.path.exists(xlsx_dir):
    os.mkdir(xlsx_dir)

xlsx_file = xlsx_dir + "/benchmark_threads_per_block.xlsx"
workbook = xlsxwriter.Workbook(xlsx_file)

cell_format = workbook.add_format({'font_size': 10})
header_format = workbook.add_format({'bold': True, 'font_size':10})

worker_group.createCharts(workbook);
parent_group.createCharts(workbook);

workbook.close()

sys.exit()
