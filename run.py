import os
import sys
import random
import math
import string
import subprocess

# Hyperparameters
vectorSizes = [40, 50, 60]
windowSizes = [3, 5]
minWordFreqs = [10, 20]

# main
for vectorSize in vectorSizes:
	for windowSize in windowSizes:
		for min_word_freq in minWordFreqs:
			subprocess.Popen(["python","worker.py",str(vectorSize),str(windowSize),str(min_word_freq)])
