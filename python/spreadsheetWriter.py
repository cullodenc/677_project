
import os
import xlsxwriter

import minerThread
from minerThread import WorkerThread
from minerThread import ParentThread

from minerBlock import WorkerBlock
from minerParser import WorkerParser
from minerParser import ParentParser




class SpreadsheetWriter:
    def __init__(self, total_workers, multilevel):
        self.worksheets = []
        self.chartsheets = []
        self.total_workers = total_workers
        self.multilevel = multilevel
        self.initWorkbook();
        self.stat_map = {
        			"target" : {"col" : 0, "label": "Difficulty Target"},
        			"difficulty" : {"col" : 1, "label": "Difficulty"},
        			"total_time" : {"col" : 2, "label": "Total Time (ms)"},
        			"mean_time" : {"col" : 3, "label": "Average Time per Block (ms)"},
        			"std_dev": {"col" : 4, "label": "Standard Deviation"},
        			"variance": {"col" : 5, "label": "Variance"}#,
        			#"worker_rate": {"col" : 6, "label": "Hashrate (MH/s)"},
        			#"block_rate": {"col" : 7, "label": "Hashrate (MH/s)"},
        			#"thread_rate": {"col" : 8, "label": "Hashrate (KH/s)"}
        			}




        return;

    def initWorkbook(self):
        xlsx_dir = os.path.realpath("outputs/spreadsheets/mining/")
        if not os.path.exists(xlsx_dir):
            os.mkdir(xlsx_dir)

        if self.multilevel == 0:
            xlsx_file = xlsx_dir + "/miner_results_"+str(self.total_workers)+"_workers.xlsx"
        else:
            xlsx_file = xlsx_dir + "/miner_results_"+str(self.total_workers)+"_workers_multilevel.xlsx"

        self.workbook = xlsxwriter.Workbook(xlsx_file)
        return;

    def addWorksheet(self, thread):
        #print "Add plotting function here"
        self.worksheets.append(self.workbook.add_worksheet("Worker "+str(thread.worker_number)))
        cell_format = self.workbook.add_format({'font_size':10})
        header_format = self.workbook.add_format({'bold': True, 'font_size':10})
        self.worksheets[-1].set_column('A:I', 20, cell_format)
        self.worksheets[-1].set_row( 0, 15, header_format)
        return;

    def writeResults(self, thread):
        print "WRITING WORKER " + str(thread.worker_number)+": LENGTH " + str(len(thread.blockchain))
        self.stat_titles = ["Block Number", "Time (ms)", "Difficulty", "Elapsed Time (ms)"]
        for col_num, data in enumerate(self.stat_titles):
        	self.worksheets[-1].write(0, col_num, data)

        count = 0;
        self.curr_row = 1;



        for row, res in enumerate(thread.blockchain):
            #print "Round "+str(count)
            self.worksheets[-1].write(row+self.curr_row, 0, int(res.count))
            self.worksheets[-1].write(row+self.curr_row, 1, float(res.time))
            self.worksheets[-1].write(row+self.curr_row, 2, str(res.diff_target))
            self.worksheets[-1].write(row+self.curr_row, 3, str(res.elapsed_time))
            #self.worksheets[-1].write(row+self.curr_row, 3, str(res.hash).upper())
            #self.worksheet[-1].write(row+self.curr_row, 3, float(res.time))
            #self.worksheet[-1].write(row+self.curr_row, 4, int(res.block_iterations))
            #self.worksheet[-1].write(row+self.curr_row, 5, int(res.thread_iterations))
            #self.worksheet[-1].write(row+self.curr_row, 6, float(res.worker_rate))
            #self.worksheet[-1].write(row+self.curr_row, 7, float(res.block_rate))
            #self.worksheet[-1].write(row+self.curr_row, 8, float(res.thread_rate))
            count+=1

        self.curr_row += count;
        return;


    def createCharts(self, threads, sheetname):
    	start = 1;
    	self.charts = [];
        self.stat_titles = ["Difficulty Target", "Difficulty Value", "Total Time (ms)", "Average Time", "Standard Deviation", "Variance"]
        category = 'difficulty'
        values = 'mean_time'

        for thread in threads:
            if(start == 1):
            	start = 0;
            	self.charts.append(self.initChart('difficulty', 'mean_time', "Difficulty Scaling"))
            	#self.charts.append(subchain.initChart(workbook, 'workers', 'block_rate', "Hashrates per Block"))
            	#self.charts.append(subchain.initChart(workbook, 'workers', 'thread_rate', "Hashrates per Thread"))

            row = 0
            col = 0
            threads = 0;
            #subchain.addChartsheet(workbook);

            self.charts[-1].add_series({
            	'name'			:	"Worker "+str(thread.worker_number),
            	'categories'	:	[sheetname, 1, (self.stat_map[category]["col"]), len(thread.diff_block), (self.stat_map[category]["col"])],
            	'values'		:	[sheetname, 1, 1+((self.stat_map[values]["col"]-2)*self.total_workers)+thread.worker_number, len(thread.diff_block), 1+((self.stat_map[values]["col"]-2)*self.total_workers)+thread.worker_number],
            	#'data_labels': {'value':True, 'category': True},
            	'marker':{'type':'automatic'}
            	})
            print "Worker "+str(thread.worker_number)+ "Using column " + str(2+((self.stat_map[values]["col"]-2)*self.total_workers)+thread.worker_number)
            # Old values
            #'values'		:	[sheetname, 1, (self.stat_map[values]["col"]), len(thread.diff_block), (self.stat_map[values]["col"])],

            #2+((col_num-2) * self.total_workers)

    		#subchain.writeResults();
    		#subchain.addChart('workers', 'worker_rate', "Hashrates per Worker");
    		#subchain.addChart('workers', 'block_rate', "Hashrates per Block");
    		#subchain.addChart('workers', 'thread_rate', "Hashrates per Thread");

    		#subchain.appendChart('workers', 'worker_rate', self.charts[0]);
    		#subchain.appendChart('workers', 'block_rate', self.charts[1]);
    		#subchain.appendChart('workers', 'thread_rate', self.charts[2]);
    	comparison_ws = self.workbook.add_worksheet("difficulty_level_comparison")
    	c_row = 1
    	for c in self.charts:
    		comparison_ws.insert_chart('A'+str(c_row), c)
    		c_row+=15
    	return;





    def addStatsheet(self, threads):
        self.worksheets.append(self.workbook.add_worksheet("Difficulty_Stats"))
        cell_format = self.workbook.add_format({'font_size':10})
        header_format = self.workbook.add_format({'bold': True, 'font_size':10})
        self.worksheets[-1].set_column('A:Z', 20, cell_format)
        self.worksheets[-1].set_row( 0, 15, header_format)
        self.writeStatsheet(threads)
        self.createCharts(threads, "Difficulty_Stats")
        return;

    def writeStatsheet(self, threads):
        #self.stat_titles = ["Difficulty Block", "Total Time (ms)", "Average Time", "Standard Deviation", "Variance"]
        self.stat_titles = ["Difficulty Target", "Difficulty Value", "Total Time (ms)", "Average Time", "Standard Deviation", "Variance"]
        header_format = self.workbook.add_format({'bold': 1,'border': 5,'align': 'center','valign': 'vcenter'})

        single_col_format = self.workbook.add_format({'num_format': '0.00000', 'left': 5, 'right': 5, 'align': 'center','valign': 'vcenter'})
        middle_col_format = self.workbook.add_format({'num_format': '0.00000', 'left': 1, 'right': 1, 'align': 'right','valign': 'vcenter'})
        end_col_format = self.workbook.add_format({'num_format': '0.00000', 'left': 1, 'right': 5, 'align': 'right','valign': 'vcenter'})


        #header_format = self.workbook.add_format({'bold': True, 'font_size':10, 'align': 'center'})

        for col_num, data in enumerate(self.stat_titles):
            if(col_num < 2):
                self.worksheets[-1].merge_range(0, col_num, 1, col_num, data, header_format)
            else:
                if(self.total_workers == 1):
                    self.worksheets[-1].write(0, (col_num * self.total_workers), data, header_format)
                else:
                    self.worksheets[-1].merge_range(0, 2+((col_num-2) * self.total_workers), 0, 1+((col_num-1) * self.total_workers), data, header_format)

                for i in range(self.total_workers):
                    self.worksheets[-1].write(1, 2+((col_num-2) * self.total_workers)+i, "Worker " + str(i), header_format)


        self.curr_col = 2
        for thread in threads:
            if(thread.worker_number == self.total_workers):
                current_format = end_col_format
            else:
                current_format = middle_col_format

            print "WRITING WORKER " + str(thread.worker_number)+": LENGTH " + str(len(thread.blockchain))
            count = 0;
            self.curr_row = 2;
            for row, res in enumerate(thread.diff_block):
                # Write target and difficulty to the first 2 columns
                self.worksheets[-1].write(row+self.curr_row, 0, str(res.target), single_col_format)
                self.worksheets[-1].write(row+self.curr_row, 1, float(res.difficulty), single_col_format)


                self.worksheets[-1].write(row+self.curr_row, self.curr_col, float(res.total_time), current_format)
                self.worksheets[-1].write(row+self.curr_row, self.curr_col+(self.total_workers), float(res.mean), current_format)
                self.worksheets[-1].write(row+self.curr_row, self.curr_col+(2 * self.total_workers), float(res.std_dev), current_format)
                self.worksheets[-1].write(row+self.curr_row, self.curr_col+(3 * self.total_workers), float(res.variance), current_format)

                #self.worksheets[-1].write(row+self.curr_row, 3, str(res.hash).upper())
                #self.worksheet[-1].write(row+self.curr_row, 3, float(res.time))
                #self.worksheet[-1].write(row+self.curr_row, 4, int(res.block_iterations))
                #self.worksheet[-1].write(row+self.curr_row, 5, int(res.thread_iterations))
                #self.worksheet[-1].write(row+self.curr_row, 6, float(res.worker_rate))
                #self.worksheet[-1].write(row+self.curr_row, 7, float(res.block_rate))
                #self.worksheet[-1].write(row+self.curr_row, 8, float(res.thread_rate))
                count+=1
            self.curr_col+=1

        self.curr_row += count;
        return;




    def initChart(self, category, values, title):
    	chart = self.workbook.add_chart({'type': 'scatter', 'subtype':'smooth'})
    	chart.set_title({'name': title})
    	chart.set_x_axis({'name': ["Difficulty_Stats", 0, (self.stat_map[category]["col"])]})
    	chart.set_y_axis({'name': (self.stat_map[values]["label"])})
    	return chart;



    def addStatChart(self):


        return;


    def future_use(self):
        writer = SpreadsheetWriter()

        # TODO Create python spreadsheet writer
        # Initialize with functions of these threads

        xlsx_dir = os.path.realpath("outputs/spreadsheets/")
        if not os.path.exists(xlsx_dir):
        	os.mkdir(xlsx_dir)

        xlsx_file = xlsx_dir + "/benchmark_threads_per_block.xlsx"
        workbook = xlsxwriter.Workbook(xlsx_file)

        cell_format = workbook.add_format({'font_size': 10})
        header_format = workbook.add_format({'bold': True, 'font_size':10})

        #worker_group.createCharts(workbook);
        #parent_group.createCharts(workbook);

        workbook.close()
        return;





	def addWorksheet(self):
		self.worksheet = workbook.add_worksheet(self.name)
		cell_format = workbook.add_format({'font_size':10})
		header_format = workbook.add_format({'bold': True, 'font_size':10})
		self.worksheet.set_column('A:I', 20, cell_format)
		self.worksheet.set_row( 0, 15, header_format)
		for col_num, data in enumerate(self.stat_titles):
			self.worksheet.write(0, col_num, data)
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


	def appendChart(self, category, values, chart):
		chart.add_series({
			'name'			:	str(self.threads) + " Threads/Block",
			'categories'	:	[self.name, 1, (self.stat_map[category]["col"]), self.curr_row, (self.stat_map[category]["col"])],
			'values'		:	[self.name, 1, (self.stat_map[values]["col"]), self.curr_row, (self.stat_map[values]["col"])],
			#'data_labels': {'value':True, 'category': True},
			'marker':{'type':'automatic'}
			})
		return;




	def createCharts_cpy(self, workbook):
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
