# MinerBlock
# Created 10-16-2019
# Description: This file contains the functions needed by a single block
# Scope: These functions operate on a single transaction block

import hashlib
import binascii
import math

class MinerBlock:
    def __init__(self, count, header):
        self.count = count						# BLOCK NUMBER
        self.header = header					# BLOCK HEADER
        self.root = header[72:136]				# BLOCK ROOT (ALGORITHM MERKLE)
        self.time = 0							# BLOCK COMPUTE TIME
        self.hash = ""							# VAR FOR VERIFICATION HASH
        self.merkle = ""						# VAR FOR VERIFICATION MERKLE
        self.label = ""
        return;

    def setHash(self, hash_in):					# SET BLOCK HASH VARIABLE
        self.hash = hash_in
        return;

    def getHash(self):
        return self.hash;

    def setTime(self, time):					# SET BLOCK COMPUTE TIME
        self.time = time
        return;

    def checkHash(self):						# CHECK THAT CORRECT HASH IS COMPUTED FROM MINING
        if(self.hash == str(doubleHash(self.header))):
            #print "HASH SUCCESS"
            #print "Expected: " + str(doubleHash(self.header))
            #print "Received: " + self.hash
            return 0;
        else:
            print self.label+"> HASH ERROR - BLOCK "+str(self.count)
            print "Expected: " + str(doubleHash(self.header))
            print "Received: " + self.hash + "\n"
            return 1;

    def checkMerkle(self):						# CHECK THAT CORRECT MERKLE ROOT IS USED IN MINING
    	if(self.root == self.merkle):
    		#print "MERKLE SUCCESS\nBLOCK "+str(self.count)
    		#print "Root: " + self.root
    		return 0;
    	else:
    		print self.label+"> MERKLE ERROR - BLOCK "+str(self.count)
    		print "Expected: " + self.merkle
    		print "Received: " + self.root + "\n"
    		return 1;


    def test(self):
        errors = 0;
        #print "Block "+str(self.count)+": took " + str(self.time) + "ms"
        errors += self.checkHash()
        errors += self.checkMerkle()
        return errors;



# Worker Block Object
class WorkerBlock(MinerBlock):
    def __init__(self, count, header, worker_number):
        MinerBlock.__init__(self, count, header)    # INHERIT FROM MinerBlock
        self.label = "[WORKER "+str(worker_number)+"]"
        return;

    def setMerkle(self, tree_size, input_file):			# CALCULATE MERKLE FROM INPUT FILE
        self.merkle = getMerkle(tree_size, input_file)
        return;



class ParentBlock(MinerBlock):
    def __init__(self, count, header):
        MinerBlock.__init__(self, count, header)    # INHERIT FROM MinerBlock
        self.tree = []                              # WORKER BLOCKS IN THE MERKLE TREE
        self.tree_size = 0                          # SIZE OF THE MERKLE TREE
        self.label = "[PARENT]"
        return;

    def appendTree(self, block):
        self.tree.append(block)
        self.tree_size+=1
        "Added to parent tree, size is now"+str(self.tree_size)
        return;

    def getTreeSize(self):
        return self.tree_size;

    def getTreeBlock(self, index):
        return self.tree[index];

    def setMerkle(self):			# CALCULATE MERKLE FROM INPUT FILE
        merkleTree = []
        if self.tree_size > 0:
            merkleSize=int(math.pow(2, math.ceil(math.log(self.tree_size,2))))
            # Build merkle tree for hashing
            for block in self.tree:
                merkleTree.append(block.getHash())

            self.merkle = calcMerkle(merkleTree, merkleSize, self.tree_size, 0)

        return;




def singleHash( input_block):
    hash_final = hashlib.sha256(binascii.unhexlify(input_block)).hexdigest()
    return hash_final;

def doubleHash( input_block):
    hash_temp = hashlib.sha256(binascii.unhexlify(input_block)).hexdigest()
    hash_final = hashlib.sha256(binascii.unhexlify(hash_temp)).hexdigest()
    return hash_final;


def calcMerkle(tree, size, treeSize, index):
	if(size==1):
		return doubleHash(tree[index]);
	elif(index+size/2 >= treeSize):
		#print "Clone: Merkle "+str(index)+" += Merkle "+str(index)
		return doubleHash(calcMerkle(tree, size/2, treeSize, index) + calcMerkle(tree, size/2, treeSize, index));
	else:
		#print "Combine: Merkle "+str(index)+" += Merkle "+str(index+size/2)
		return doubleHash(calcMerkle(tree, size/2, treeSize, index) + calcMerkle(tree, size/2, treeSize, index+size/2));


def getMerkle(tree_size, input_file):
    tree=[]
    for x in range(0,tree_size):
        tree.append(input_file.readline().strip('\n'))

    merkle_size=int(math.pow(2, math.ceil(math.log(tree_size,2))))

    return calcMerkle(tree, merkle_size, tree_size, 0);

def trackMerkle(tree_size, input_file):     # USED TO DEBUG MERKLE INPUTS
    tree=[]
    for x in range(0,tree_size):
        tree.append(input_file.readline().strip('\n'))
        print "Merkle "+str(x)+": "+str(tree[-1])

    return calcMerkle(tree, tree_size, 0);
