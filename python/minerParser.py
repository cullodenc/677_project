# MinerParser
# Created 10-16-2019
# Description: This file contains the functions needed for scraping data from the miner output files
# Scope: These functions operate on the data for a single miner

import re			# Used for input file reading

import minerBlock
from minerBlock import WorkerBlock
from minerBlock import ParentBlock


class MinerParser:
	def __init__(self, tree_size):
		self.count = 0
		self.in_file = None
		self.tree_size = tree_size

		self.h_flag = 0
		self.header=""
		return;

	def setInFile(self, input_file):
		self.in_file = input_file
		return;

	def search(self, out_line, blocks):
		self.searchBlock(out_line, blocks)
		self.searchHash(out_line, blocks)
		self.searchTime(out_line, blocks)

	def searchHash(self, out_line, blocks):
		if re.search("HASH:", out_line):
			blocks[self.count-1].setHash(out_line.split()[1])
		return;

	def searchTime(self, out_line, blocks):
		if re.search("BLOCK_TIME:", out_line):
			blocks[self.count-1].setTime(out_line.split()[1])
		return;

	def searchBlock(self, out_line, blocks):
		if re.search("BLOCK_HEADER:", out_line):
			self.h_flag=2
		elif(self.h_flag==2):
			self.header = out_line.strip('|\n')
			self.h_flag-=1
		elif(self.h_flag==1):
			self.header +=out_line.strip('|\n')
			self.h_flag-=1
			self.addBlock(blocks)
			#print "Block " + str(self.count) + ": " + self.header
			self.count+=1
		return;


class WorkerParser(MinerParser):
	def __init__(self, tree_size, worker_number):
		MinerParser.__init__(self, tree_size)
		self.worker_number = worker_number
		return;

	def addBlock(self, blocks):
		blocks.append(WorkerBlock(self.count, self.header, self.worker_number))
		return;



class ParentParser(MinerParser):
	def __init__(self, tree_size):
		MinerParser.__init__(self, tree_size)
		self.worker_number = 0
		return;

	def addBlock(self, blocks):
		blocks.append(ParentBlock(self.count, self.header))
		return;

	def buildMerkle(self, blockchain, worker_threads):
		inlines = self.in_file.readlines()
		chain_length = len(blockchain)
		merkle_count = 0
		block_count = 0

		for il in inlines:
			if re.search("WORKER", il):
				worker_num = int(il.split()[1])
				block_num = int(il.split()[4])
				#print "Found Worker "+ str(worker_num)+"| Block "+ str(block_num)
				if merkle_count < self.tree_size:
					blockchain[block_count].appendTree(worker_threads[worker_num-1].getBlock(block_num-1))
					merkle_count += 1
				elif block_count < chain_length-1:
					merkle_count = 1
					block_count += 1
					blockchain[block_count].appendTree(worker_threads[worker_num-1].getBlock(block_num-1))
				else:
					break;		# PARENT CHAIN IS FULL, BREAK OUT OF INPUT PARSER LOOP

		for block in blockchain:
			#print "TESTING BLOCK " + str(block.count)+" SIZE "+str(block.getTreeSize())
			block.setMerkle()
		return;
