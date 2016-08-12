import os
import sys
import random
import string
import math
from gensim import corpora

import glob

import gensim
import gensim.models as gm

vector_size = int(float(sys.argv[1]))
window_size = int(float(sys.argv[2]))
dict_file = str(sys.argv[3])
model_file = str(sys.argv[4])
source_text = str(sys.argv[5])
save_path = str(sys.argv[6])
logfile = str(sys.argv[7])

# log = open(logfile,"a+")
# log.write("------Word2Vec Converter------\n")
# log.write("Vector size: " + str(vector_size) + "\n")
# log.write("Window size: " + str(window_size) + "\n")
# log.write("Dictionary being used: " + str(dict_file) + "\n")
# log.write("Model being used: " + str(model_file) + "\n")
# log.write("Processing file: " + str(source_text) + "\n")
# log.write("Data saved to: " + str(save_path) + "\n")
# log.flush()

tok = []
dictionary = corpora.Dictionary.load_from_text(dict_file)
model = gm.Word2Vec.load(model_file)

# converting words to vectors
def dataGen_3():
	print("Src2Vec: " + source_text + "\n")
	with open(source_text, "r") as txt:
		with open(save_path + "tmp_data.csv", "w+") as dat:
			# zero padding at the beginning
			for x in range(vector_size - 1):
				dat.write("0,")
			dat.write("0\n")
			for line in txt:
				sent = line.lower().split()
				for i in range(len(sent)):
					# looking up for vectors
					tok = model[ sent[i] ]
					# writing values of each dimension in the vector
					for j in range(vector_size - 1):
						dat.write(str(tok[j]))
						dat.write(",")
					dat.write(str(tok[vector_size - 1]))
					dat.write("\n")
				# zero padding at the end of each sentence
				for x in range(vector_size - 1):
					dat.write("0,")
				dat.write("0\n")
			dat.close()
		txt.close()

def dataGen_5():
	print("Src2Vec: " + source_text + "\n")
	with open(source_text, "r") as txt:
		with open(save_path + "tmp_data.csv", "w+") as dat:
			for x in range(vector_size - 1):
				dat.write("0,")
			dat.write("0\n")
			for x in range(vector_size - 1):
				dat.write("0,")
			dat.write("0\n")
			for line in txt:
				sent = line.lower().split()
				for i in range(len(sent)):
					tok = model[ sent[i] ]
					for j in range(vector_size - 1):
						dat.write(str(tok[j]))
						dat.write(",")
					dat.write(str(tok[vector_size - 1]))
					dat.write("\n")
				for x in range(vector_size - 1):
					dat.write("0,")
				dat.write("0\n")
				for x in range(vector_size - 1):
					dat.write("0,")
				dat.write("0\n")
			dat.close()
		txt.close()

if window_size == 3:
	dataGen_3()
elif window_size == 5:
	dataGen_5()
else:
	print("Window Size Error!")
	# log.write("Window Size Error!")

# log.write("------End Word2Vec Converter------\n")
# log.close()