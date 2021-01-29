
import os
import xlsxwriter

import minerThread
from minerThread import WorkerThread
from minerThread import ParentThread

from minerBlock import WorkerBlock
from minerParser import WorkerParser
from minerParser import ParentParser

import minerGroup
from minerGroup import MinerGroup


# FIXME Merge with spreadsheetwriter, further merges may be needed

class ComparisonWriter:
    def __init__(self, total_workers, multilevel):
        self.worksheets = []
        self.chartsheets = []
        self.total_workers = total_workers
        self.multilevel = multilevel
        self.initWorkbook();
        self.chart = None
        self.stat_map = {
        			"block_num" : {"col" : 0, "label": "Blocks Mined"},
                    "worker_num" : {"col" : 1, "label": "Worker Number"},
        			"difficulty" : {"col" : 2, "label": "Difficulty"},
        			"block_time" : {"col" : 3, "label": "Block Time (ms)"},
        			"total_time" : {"col" : 4, "label": "Elapsed Time (ms)"},
        			#"std_dev": {"col" : 4, "label": "Standard Deviation"},
        			#"variance": {"col" : 5, "label": "Variance"}#,
        			#"worker_rate": {"col" : 6, "label": "Hashrate (MH/s)"},
        			#"block_rate": {"col" : 7, "label": "Hashrate (MH/s)"},
        			#"thread_rate": {"col" : 8, "label": "Hashrate (KH/s)"}
        			}

        self.shortest_time = 0



        return;

    def initWorkbook(self):
        xlsx_dir = os.path.realpath("outputs/spreadsheets/mining/")
        if not os.path.exists(xlsx_dir):
            os.mkdir(xlsx_dir)

        if self.multilevel == 0:
            xlsx_file = xlsx_dir + "/miner_results_comparison.xlsx"
        else:
            xlsx_file = xlsx_dir + "/miner_results_multilevel_comparison.xlsx"

        self.workbook = xlsxwriter.Workbook(xlsx_file)
        return;

    def addWorksheet(self, thread):
        #print "Add plotting function here"
        self.worksheets.append(self.workbook.add_worksheet(thread.name))
        cell_format = self.workbook.add_format({'font_size':10})
        header_format = self.workbook.add_format({'bold': True, 'font_size':10})
        self.worksheets[-1].set_column('A:I', 20, cell_format)
        self.worksheets[-1].set_row( 0, 15, header_format)
        return;

    def writeResults(self, thread):
        print "WRITING DESIGN " + str(thread.workers)+": LENGTH " + str(len(thread.mainChain))
        self.stat_titles = ["Block Number", "Worker Number", "Difficulty", "Time (ms)", "Elapsed Time (ms)"]
        #TODO Add worker number, reorder columns
        for col_num, data in enumerate(self.stat_titles):
        	self.worksheets[-1].write(0, col_num, data)

        count = 0;
        self.curr_row = 1;



        for row, res in enumerate(thread.mainChain):
            #print "Round "+str(count)
            self.worksheets[-1].write(row+self.curr_row, 0, int(res.count))
            self.worksheets[-1].write(row+self.curr_row, 1, int(res.worker_number))
            self.worksheets[-1].write(row+self.curr_row, 2, str(res.diff_target))
            self.worksheets[-1].write(row+self.curr_row, 3, float(res.time))
            self.worksheets[-1].write(row+self.curr_row, 4, float(res.elapsed_time))
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


    def initChart(self, category, values, title):
    	self.chart = self.workbook.add_chart({'type': 'scatter', 'subtype':'smooth'})
    	self.chart.set_title({'name': title})
    	self.chart.set_x_axis({'name': (self.stat_map[category]["label"])})
    	self.chart.set_y_axis({'name': (self.stat_map[values]["label"])})
    	return;

    def appendChart(self, category, values, group):
        print "Adding series from " + str(group.name)
        self.chart.add_series({
    		'name'			:	group.label,
    		'categories'	:	[group.name, 1, (self.stat_map[category]["col"]), len(group.mainChain), (self.stat_map[category]["col"])],
    		'values'		:	[group.name, 1, (self.stat_map[values]["col"]), len(group.mainChain), (self.stat_map[values]["col"])],
    		#'data_labels': {'value':True, 'category': True},
    		#'marker':{'type':'automatic'},
            'marker':{'type':'none'},
            'line':{'width': 1}
    		})
    	return;

    def insertChart(self):
        chart_sheet = self.workbook.add_worksheet("Comparison_Chart")
        chart_sheet.insert_chart('A1', self.chart)
        return;
