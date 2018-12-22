import numpy as np 
import random 

def generate_single_seq(length=30, min_len=5, max_len=10):
	'''
	generates a sequence of numbers of random length
	and inserts a sub-seqeunce of greater numbers at random place
	output:
	seqeunce of numbers, index of the start and end of the greater number subsequence
	'''
	seq_before = [random.randint(1, 5) for x in range(random.randint(min_len, max_len))]
	seq_during = [random.randint(6, 10) for x in range(random.randint(min_len, max_len))]
	seq_after = [random.randint(1, 5) for x in range(random.randint(min_len, max_len))]
	seq = seq_before + seq_during + seq_after 
	# pad zero for same input length
	seq = seq + ([0] * (length - len(seq)))
	return (seq, len(seq_before), len(seq_before)+len(seq_during)-1)

def generate_set_seq(N):
	# generate a set of N sequences of fixed lengths
	data = []
	starts = []
	ends = []
	for i in range(N):
		seq, ind_start, ind_end = generate_single_seq()
		data.append(seq)
		starts.append(ind_start)
		ends.appends(ind_end)
	return data, starts, ends 

def make_seq_data(n_samples, seq_len):
	# make sorting task data
	data, labels = []
	for _ in range(n_samples):
		input = np.random.permutation(range(seq_len)).to_list()
		target = sorted(range(len(input)), key=lambda k: input[k])
		data.append(input)
		labels.append(target)
	return data, labels