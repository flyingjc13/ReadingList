# adapted from: https://github.com/jojonki/Pointer-Networks
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from utils import to_var 

class PointerNetwork(nn.Module):
	def __init__(self, input_size, emb_size, weight_size, answer_seq_len, hidden_size=512, is_GRU=True):
		super(PointerNetwork, self).__init__()

		self.hidden_size = hidden_size 
		self.input_size = input_size 
		self.answer_seq_len = answer_seq_len 
		self.emb_size = emb_size 
		self.weight_size = weight_size 
		self.is_GRU = is_GRU

		self.emb = nn.Embedding(input_size, emb_size)
		if is_GRU:
			self.enc = nn.GRU(emb_size, hidden_size, batch_first=True)
			self.dec = nn.GRUCell(emb_size, hidden_size)
		else:
			self.enc = nn.LSTM(emb_size, hidden_size, batch_first=True)
			self.dec = nn.LSTMCell(emb_size, hidden_size)

		self.W1 = nn.Linear(hidden_size, weight_size, bias=False)
		self.W2 = nn.Linear(hidden_size, weight_size, bias=False)
		self.vt = nn.Linear(weight_size, 1, bias=False) 

	
	def forward(self, input):
		batch_size = input.size(0)
		input = self.emb(input) #(B, len, emb)

		encoder_states, hc = self.enc(input) #encoder_states: (B, L, hid)
		encoder_states = encoder_states.transpose(1, 0) #(L, B, hid)

		decoder_input = to_var(torch.zeros(batch_size, self.emb_size))
		hidden = to_var(torch.zeros([batch_size, self.hidden_size])) #(B, hid)
		cell_state = encoder_states[-1] #(B, hid)

		probs = []
		# decoding
		for i in range(self.answer_seq_len):
			if self.is_GRU:
				hidden = self.dec(decoder_input, hidden)
			else:
				hidden, cell_state = self.dec(decoder_input, (hidden, cell_state))

			# compute blended representation at each decoder time step
			blend1 = self.W1(encoder_states)	#(L, B, W)
			blend2 = self.W2(hidden)	#(B, W)
			# add blend2 to every step of encoder output (blend1)
			blend_sum = F.tanh(blend1 + blend2)	#(L, B, W)
			out = self.vt(blend_sum).squeeze() # (L, B)
			out = F.log_softmax(out.transpose(0, 1).contiguous(), -1) #(B, L)
			probs.append(out)
			# probs: (M, B, L) M: ans_seq_len

		probs = torch.stack(probs, dim=1)  #(B, M, L)

		return probs








