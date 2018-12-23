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


# generate sequence with multiple greater subsequences
def generate_single_seq_multiple(length=72, min_len=4, max_len=8, max_pointer=4):
	'''
	generates a sequence of numbers of random length
	and inserts a sub-seqeunce of greater numbers at random place
	output:
	seqeunce of numbers, index of the start and end of the greater number subsequence

	alternate: small, big, small, ...
	'''
	seq = []
	pos = []
	ptr_num = random.randint(1, max_pointer)
	for i in range(ptr_num):	
		temp_seq_small = [random.randint(1, 5) for x in range(random.randint(min_len, max_len))]
		temp_seq_big = [random.randint(6, 10) for x in range(random.randint(min_len, max_len))]
		seq += temp_seq_small
		pos1 = len(seq)
		seq += temp_seq_big
		pos2 = len(seq) - 1
		pos.extend([pos1, pos2])
	temp_seq_small = [random.randint(1, 5) for x in range(random.randint(min_len, max_len))]
	seq += temp_seq_small

	# pad to keep constant input seq length 
	seq = seq + ([0] * (length - len(seq)))
	return (seq, pos, ptr_num)

def generate_set_seq_multiple(N):
	# generate a set of N sequences of fixed lengths
	data = []    #(N, L)
	pos = []	 #(N, ptr*2)
	ptr_nums = []  #(N)
	for i in range(N):
		seq, seq_pos, seq_ptr = generate_single_seq_numtiple()
		data.append(seq)
		pos.append(seq_pos)
		ptr_nums.append(seq_ptr)
	return data, pos, ptr_nums














