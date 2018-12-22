# find the bounary of the greater subsequence
import torch 
from torch import optim 
import torch.nn.functional as F 
import numpy as np 
import generate_data
from utils import to_var 
from pointer_network import PointerNetwork 

total_size = 10000
weight_size = 256 
emb_size = 32
batch_size = 250
answer_seq_len = 2
n_epochs = 5 

dataset, starts, ends = generate_data.generate_set_seq(total_size)
targets = np.vstack((starts, ends)).T  #(total_size, M)  M=2
dataset = np.array(dataset)  #(total_size, L)

input_seq_len = dataset.shape[1]
inp_size = 11  # 0 to 10

input = to_var(torch.LongTensor(dataset))  #(N, L)
targets = to_var(torch.LongTensor(targets))  #(N, 2)

data_split = (int)(total_size*0.9)
train_X = input[ : data_split]
train_Y = targets[ : data_split]
test_X = input[data_split : ]
test_Y = targets[data_split : ]

def train(model, X, Y, batch_size, n_epochs):
	model.train()
	optimizer = optim.Adam(model.parameters())
	N = X.size(0)
	L = X.size(1)
	for epoch in range(n_epochs):
		for i in range(0, N-batch_size, batch_size):
			x = X[i : i+batch_size]  #(B, L)
			y = Y[i : i+batch_size]  #(B, M)

			probs = model(x)  #(B, M, L)
			outputs = probs.view(-1, L)  #(B*M, L)
			y = y.view(-1)  #(B*M)	
			loss = F.nll_loss(outputs, y)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if epoch % 2 == 0:
			print('epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))
			test(model, X, Y)

def test(model, X, Y):
	probs = model(X)  #(B, M, L)
	_v, indices = torch.max(probs, 2)  #(B, M)
	correct_count = sum([1 if torch.equal(ind,data, y.data) else 0 for ind, y in zip(indices, Y)])
	print('Acc: {:.2f}% ({}/{})'.format(correct_count/len(X)*100, correct_count, len(X)))

model = PointerNetwork(inp_size, emb_size, weight_size, answer_seq_len)
if torch.cuda.is_available():
	model.cuda()
train(model, trainX, trainY, batch_size, n_epochs)
print('----Test result---')
test(model, test_X, test_Y)









