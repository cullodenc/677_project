import math
import binascii

class DiffBlock:
    def __init__(self, target, time):
        self.target = target
        self.difficulty = 0.0
        self.times = []
        self.total_time = 0
        self.count = 0
        self.mean = 0
        self.std_dev = 0
        self.variance = 0
        self.addTime(time)
        self.calcDifficulty()
        return;

    def addTime(self, time):
        self.times.append(time)
        self.total_time += float(time)
        self.count+=1
        self.mean = float(self.total_time/self.count)
        self.calcStats()
        return;

    def calcDifficulty(self):
        start_pow = 0x1d
        start_diff = 0x00ffff
        bit_pow = int(self.target[0:2], 16)
        bit_diff = int(self.target[2:8], 16)
        diff_coef = math.log(float(start_diff) / float(bit_diff)) + (start_pow - bit_pow)*math.log(256);
        self.difficulty = math.exp(diff_coef)
        return;

    def calcStats(self):
        total = 0.0;
        for t in self.times:
            total += math.pow(float(t) - float(self.mean), 2.0)

        self.variance = total/(len(self.times))
        self.std_dev = math.sqrt(self.variance)
        return;
