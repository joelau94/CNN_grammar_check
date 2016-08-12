import os
import sys
import random
import string
import math
from gensim import corpora

import glob

import gensim
import gensim.models as gm

maxLine = int(float(sys.argv[1]))
file_name = str(sys.argv[2])	# "./nyt*.txt"
dict_file = str(sys.argv[3])
model_file = str(sys.argv[4])
save_path = str(sys.argv[5])
sharedFile = str(sys.argv[6])
logFile = str(sys.argv[7])

# log = open(logFile,"a+")
# log.write("------Generating Source Texts(with error) and Labels------\n")
# log.write("Total lines in one file: " + str(maxLine) + "\n")
# log.write("Processing files with name: " + str(head_file_name_match) + "*.txt\n")
# log.write("Dictionary being used: " + str(dict_file) + "\n")
# log.write("Model being used: " + str(model_file) + "\n")
# log.write("Shared value file: " + str(sharedFile) + "\n")
# log.write("Files saved to: " + str(save_path) + "\n")
# log.flush()

tok = []
dictionary = corpora.Dictionary.load_from_text(dict_file)
model = gm.Word2Vec.load(model_file)

lineCount = 0
fileCount = 0

# os.system("mkdir " + save_path)

# divide texts into files with fixed number of sentences(except the last file)
# and randomly replace a word with a wrong one in each sentence
print("Processing file: " + file_name + "\n")
# log.write("Processing file: " + file_name + "\n")
# iterating over files
with open(file_name, "r") as corr:
	# iterating over lines
	for line in corr:
		lineCount += 1
		# lineCount % maxLine == 1 indicates the start of a new file
		if lineCount%maxLine == 1:
			fileCount += 1
			err = open(save_path + str(fileCount) + "_source.txt", "w+")
			lbl = open(save_path + str(fileCount) + "_label.csv", "w+")
		tok = line.decode('utf-8').split()
		if len(tok):
			# replace a randomly chosen word in a sentence
			error_pos = random.randrange(0,len(tok),1)
			# with a randomly chosen word in dicitonary
			sub = random.randrange(0,max(dictionary.keys()))
			tok[error_pos] = dictionary.get(sub)
			for word in tok:
				err.write(word.encode('utf-8'))
				err.write(" ")
			err.write("\n")
			# then record the error position with labels
			for i in range(len(tok)):
				if i == error_pos :
					lbl.write("0\n")
				else:
					lbl.write("1\n")
		# lineCount % maxLine == 0 indicates reaching the maximum line count of a single file
		if lineCount%maxLine == 0:
			err.close()
			lbl.close()
	# for the last file, if it does not reach the maximum line count, close it
	if lineCount%maxLine != 0:
		err.close()
		lbl.close()
print("Total label file number: " + str(fileCount) + "\n")
# log.write("Total label file number: " + str(fileCount) + "\n")

# log.write("------End Generating Source Texts(with error) and Labels------\n")
# log.flush()
# log.close()

shared = open(sharedFile, "w+")
shared.write(str(fileCount))
shared.close()