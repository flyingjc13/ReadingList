# solving the sorting task
import torch
from torch import optim
import torch.nn.functional as F 
import generate_data
from utils import to_var 
from pointer_network import PointerNetwork 

total_size = 10000
weight_size = 256 
emb_size = 32 
batch_size = 250 
n_epochs = 5 

input_seq_len = 4 
input, targets = generate_data.make_seq_data(total_size, input_seq_len)
inp_size = input_seq_len 

# convert to torch tensors 
input = to_var(torch.LongTensor(input)) #(N, L)
targets = to_var(torch.LongTensor(targets)) #(N, L)

data_split = (int)(total_size*0.9)
train_X = input[ : data_split]
train_Y = targets[ : data_split]
test_X = input[data_split : ]
test_Y = targets[data_split : ]

def train(model, X, Y, batch_size, n_epcohs):
	model.train()
	optimizer = optim.Adam(model.parameters())
	N = X.size(0)
	L = X.size(1) # ans_seq_len(M) = inp_seq_len = L
	for epoch in range(n_epochs + 1):
		for i in range(0, N-batch_size, batch_size):
			x = X[i: i+batch_size] #(B, L)
			y = Y[i: i+batch_size] #(B, M)

			probs = model(x) #(B, M, L)
			outputs = probs.view(-1, L) #(B*M, L)
			y = y.view(-1) #(B*M)
			loss = F.nll_loss(outputs, y) #neg log likelihood loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if epoch % 2 == 0:
			print ("epoch: {}, loss: {:.5f}".format(epoch, loss.item()))
			test(model, X, Y)


def test(model, X, Y):
	# X: (B, L)  Y: (B, M)
	probs = model(X)  #(B, M, L)
	_v, indices = torch.max(probs, 2)  #(B, M)
	# show test examples 
	for i in range(len(indices)):
		print ("test", [v for v in X[i].data])
		print ("label", [v for v in Y[i].data])
		print ("pred", [v for v in indices[i].data])
	
	correct_count = sum([1 if torch.equal(ind.data, y.data) else 0 for ind,y in zip(indices, Y)])
	print ('Acc: {:.2f}% ({}/{})'.format(correct_count/len(X)*100, correct_count, len(X)))

model = PointerNetwork(inp_size, emb_size, weight_size, input_seq_len)
if torch.cuda.is_available():
	model.cuda()
train(model, train_X, train_Y, batch_size, n_epochs)
print ("-----Test Result-----")
test(model, test_X, test_Y)











