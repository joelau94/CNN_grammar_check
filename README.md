# CNN_grammar_check
Word-level grammatical error detection using simple CNN; including data processing, training and testing
Written with torch, it takes error-free texts as input, generates artificial error by randomly replacing one word in each sentence, and train with one-layer temporal convolution.

Usage:
1. make three directories under the root: "data", "LM", and "nnModel"
2. change corpus_path in worker.py into the directory containing error-free texts
3. change file_name_match in worker.py into file names you want to match in directory corpus_path
4. python run.py
