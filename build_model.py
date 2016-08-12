import os
import sys
import random
import math
import string
import gensim
import gensim.models as gm
from gensim import corpora
from collections import defaultdict
import glob

window_size = int(float(sys.argv[1]))
vector_size = int(float(sys.argv[2]))
min_word_freq = int(float(sys.argv[3]))
corpus_path = str(sys.argv[4])
file_name_match = str(sys.argv[5])
save_path = str(sys.argv[6])
unk_src_path = str(sys.argv[7])
logfile = str(sys.argv[8])

log = open(logfile,"a+")
log.write("------ Word2Vec ------\n")
log.write("Window size: " + str(window_size) + "\n")
log.write("Vector size: " + str(vector_size) + "\n")
log.write("Minimum word frequency: " + str(min_word_freq) + "\n")
# log.write("Model trained on: " + str(filename) + "\n")
# log.write("Model saved to: " + str(save_path) + "\n")
log.flush()

# argument 'window' of Word2Vec: maximum distance from the word
one_side_window = int(math.floor(window_size/2))

os.system("mkdir " + unk_src_path)
file_names = glob.glob(corpus_path + file_name_match + "*.txt")

# calculating frequency of words
frequency = defaultdict(int)
for file_name in file_names:
	with open(file_name, "r") as raw:
		for line in raw:
			tok = line.decode('utf-8').lower().split()
			for word in tok:
				frequency[word] += 1
		raw.close()

# replacing non-frequent words with <unk>
tok = []
with open(unk_src_path + "unk_" + str(vector_size) + "_" + str(window_size) + "_" + str(min_word_freq) + ".txt", "w+") as txt:
	for file_name in file_names:	
		with open(file_name, "r") as raw:
			for line in raw:
				tok = line.decode('utf-8').split()
				if(len(tok)):
					for word in tok:
						if frequency[word.lower()] < min_word_freq:
							txt.write("<unk>".encode('utf-8'))
							txt.write(" ")
						else:
							txt.write(word.encode('utf-8'))
							txt.write(" ")
					txt.write("\n")
			raw.close()
	txt.close()

# build model and dictionary
# print("Building W2V Models ... ")
sentences = []
# with open("unk_" + str(vector_size) + "_" + str(window_size) + "_" + str(min_word_freq) + "_" + filename.split('/')[-1], "r") as corr:
sentences = []
with open(unk_src_path + "unk_" + str(vector_size) + "_" + str(window_size) + "_" + str(min_word_freq) + ".txt", "r") as corr:
	sentences = [[tok for tok in line.lower().split()] for line in corr]
	model = gm.Word2Vec(sentences, size=vector_size, window=one_side_window, min_count=min_word_freq, workers=4)
	model.save(save_path + str(vector_size)+"_"+str(window_size)+"_"+str(min_word_freq)+"_mdl")
	dictionary = corpora.Dictionary(sentences)
	dictionary.save_as_text(save_path + str(vector_size)+"_"+str(window_size)+"_"+str(min_word_freq)+"_dict")
	log.write("Dictionary Size: " + str(len(dictionary.values())) + "\n")
	corr.close()

# os.system("rm unk_" + str(vector_size) + "_" + str(window_size) + "_" + str(min_word_freq) + "_" + filename.split('/')[-1])
log.write("------ End Word2Vec ------\n")

log.close()