# adapted from: https://github.com/atulkum/pointer_summarizer

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.utils import pack_padded_sequence, pad_packed_sequence 
from data_util import config 
from numpy import random 

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(123)

'''
Well the original author wrote these functions to intialize weights,
but I personally find it fine to just use the default initialization
of LSTM and Dense layers.
'''

def init_lstm_wt(lstm):
	for names in lstm._all_weights:
		for name in names:
			if name.startswith('weight_'):
				wt = getattr(lstm, name)
				wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)
			elif name.startswith('bias_'):
				# set forget bias to 1 
				bias = getattr(lstm, name)
				n = bias.size(0)
				start, end = n//4, n//2 
				bias.data.fill_(0.)
				bias.data[start:end].fill_(1.)

def init_linear_wt(linear):
	linear.weight.data.normal_(std=config.trunc_norm_init_std)
	if linear.bias is not None:
		linear.bias.data.normal_(std=config.trunc_norm_init_std)

def init_wt_normal(wt):
	wt.data.normal_(std=config.trunc_norm_init_std)

def init_wt_unif(wt):
	wt.data.uniform_(-config.rand_unif_init_mag, config.rand_unif_init_mag)

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
		init_wt_normal(self.embedding_weight)

		self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
		init_lstm_wt(self.lstm)

		self.W_h = nn.Linear(config.hidden_dim*2, config.hidden_dim*2, bias=False)

	# seq should be sorted by len in decreasing order
	def forward(self, input, seq_lens):
		embedded = self.embedding(input)

		packed = pack_padded_sequence(embedded, seq_lens, batch_first=True)
		output, hidden = self.lstm(packed)

		encoder_outputs, _ = pad_packed_sequence(output, batch_first=True) 
		# shape (B, len, hidden_dim*2)
		encoder_outputs = encoder_outputs.contiguous()

		encoder_feature = encoder_outputs.view(-1, 2*config.hidden_dim) 
		# shape (B*len, hidden_dim*2)
		# output goes through a linear projection (same dim)
		encoder_feature = self.W_h(encoder_feature)

		return encoder_outputs, encoder_feature, hidden 

class ReduceState(nn.Module):
	# from 2*hid_dim to 1*hid_dim
	def __init__(self):
		super(ReduceState, self).__init__()

		self.reduce_h = nn.Linear(config.hidden_dim*2, config.hidden_dim)
		init_linear_wt(self.reduce_h)
		self.reduce_c = nn.Linear(config.hidden_dim*2, config.hidden_dim)
		init_linear_wt(self.reduce_c)

	def forward(self, hidden):
		h, c = hidden 
		# initial shape (2, batch, hid_dim)
		h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim*2)
		hidden_reduced_h = F.relu(self.reduce_h(h_in))
		c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim*2)
		hidden_reduced_c = F.relu(self.reduce_c(c_in))

		return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))
		# shape: (1, Batch, hid_dim)

class Attention(nn.Module):
	def __init__(self):
		super(Attention, self).__init__()
		if config.is_coverage:
			self.W_c = nn.Linear(1, config.hidden_dim*2, bias=False)
		self.decode_proj = nn.Linear(config.hidden_dim*2, config.hidden_dim*2)
		self.v = nn.Linear(config.hidden_dim*2, 1, bias=True)

	def forward(self, s_t_hat, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
		#t_k : len
		b, t_k, n = list(encoder_outputs.size())

		dec_fea = self.decoder_proj(s_t_hat) # (B, hid_dim*2)
		dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, t_k, n).contiguous() #(B, t_k, hid_dim*2)
		dec_fea_expanded = dec_fea.view(-1, n) #(B*t_k, hid_dim*2)

		att_features = encoder_feature + dec_fea_expanded 
		if config.is_coverage:
			converage_input = coverage.view(-1, 1) #(B*t_k, 1)
			coverage_feature = self.W_c(coverage_input) #*(B*t_k, hid_dim*2)
			att_features = att_features + coverage_feature 

		e = F.tanh(att_features) #(B*t_k, 2*hid_dim)
		scores = self.v(e)	#(B*t_k, 1)
		scores = socres.view(-1, t_k) #(B, t_k)

		# softmax then pad then normaize
		# could pad first then softmax as well
		attn_dist_ = F.softmax(scores, dim=1)*enc_padding_mask #(B, t_k)
		normalization_factor = attn_dist_.sum(1, keepdim=True)
		attn_dist = attn_dist_ / normalization_factor 

		## pad first then softmax:
		# scores = scores*enc_padding_mask
		# attn_dist_ = F.softmax(scores)
		# normalization_factor = attn_dist_.sum(1, keepdim=True)
		# attn_dist = attn_dist_ / normalization_factor

		attn_dist = attn_dist.unsqueeze(1) # (B, 1, t_k)
		# encoder_outputs: (B, t_k, 2*hid_dim)
		c_t = torch.bmm(attn_dist, encoder_outputs) #(B, 1, 2*hid_dim)
		c_t = c_t.view(-1, config.hidden_dim*2)	 #(B, hid_dim*2)
		# c_t: weighted context vector

		attn_dist = attn_dist.view(-1, t_k) # (B, t_k)

		if config.is_coverage:
			coverage = coverage.view(-1, t_k)
			''' 
			add the current attention to coverage
			so that the model remembers what parts have already 
			been covered
			'''
			coverage = coverage + attn_dist 

		return c_t, attn_dist, coverage 

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.attention_network = Attention()
		# decoder
		self.embedding = nn.Embedding(config.vocab_size, config.emb_dim)
		init_wt_normal(self.embedding_weight)

		self.x_context = nn.Linear(config.hidden_dim*2 + config.emb_dim)

		self.lstm = nn.LSTM(config.emb_dim, config.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
		init_lstm_wt(self.lstm)

		if config.pointer_gen:
			self.p_gen_linear = nn.Linear(config.hidden_dim*4 + config.emb_dim, 1)

		# p_vocab
		self.out1 = nn.Linear(config.hidden_dim*3, config.hidden_dim)
		self.out2 = nn.Linear(config.hidden_dim, config.vocab_size)
		init_linear_wt(self.out2)

	def forward(self, y_t_1, s_t_1, encoder_outputs, encoder_feature, enc_padding_mask,
				c_t_1, extra_zeros, enc_batch_extend_vocab, coverage, step):
		if not self.training and step == 0:
			h_decoder, c_decoder = s_t_1 
			s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
								 c_decoder.view(-1, config.hidden_dim)), 1) #(B, 2*hid_dim)
			c_t, _, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature, 
															enc_padding_mask, coverage)
			coverage = coverage_next 

		y_t_1_embd = self.embedding(y_t_1)
		x = self.x_context(torch.cat((c_t_1, y_t_1_embd), 1))
		# y_t_1, s_t_1, x are just one step
		# decode one at a time
		lstm_out, s_t = self.lstm(x.unsqueeze(1), s_t_1)

		h_decoder, c_decoder = s_t 
		s_t_hat = torch.cat((h_decoder.view(-1, config.hidden_dim),
							 c_decoder.view(-1, config.hidden_dim)), 1) #(B, 2*hid_dim)
		c_t, attn_dist, coverage_next = self.attention_network(s_t_hat, encoder_outputs, encoder_feature,
																enc_padding_mask, coverage)
		if self.training or step > 0:
			coverage = coverage_next 

		p_gen = None 
		if config.pointer_gen:
			p_gen_input = torch.cat((c_t, s_t_hat, x), 1) #(B, 2*2*hid_dim+emb_dim)
			p_gen = self.p_gen_linear(p_gen_input)
			p_gen = F.sigmoid(p_gen)

		output = torch.cat((lstm_out.view(-1, config.hidden_dim), c_t), 1) #(B, hidden_dim*3)
		output = self.out1(output) #(B, hid_dim)

		output = self.out2(output) #(B, vocab_size)
		vocab_dist = F.softmax(output, dim=1)

		if config.pointer_gen:
			vocacb_dist_ = p_gen * vocab_dist 
			# prob of copying
			attn_dist_ = (1-p_gen)*attn_dist

			if extra_zeros is not None:
				vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

			final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
		else:
			final_dist = vocab_dist 

		return final_dist, s_t, c_t, attn_dist, p_gen, coverage





## continue to finish this model
## add the transformer model as well 































