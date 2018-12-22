# adapted from: https://github.com/ofirpress/YouMayNotNeedAttention

import torch 
import torch.nn as nn 

# please check the original repo for embed_regularizer and locked_dropout
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout 

class RNNModel(nn.Module):
	# container module with an encoder, a recurrent module, a decoder

	# tie input and output embedding matrix can imporve performance
	def __init__(self, rnn_type='LSTM', ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, 
		dropouti=0.5, dropoute=0.1, wdrop=0, dropoutcomb=0.2, tie_weights=False):
		super(RNNModel, self).__init__()
		self.lockdrop = LockedDropout()

		# emn_dim: int(ninp/2)
		# this embedding is shared for both source and target language
		self.emb_dim = int(ninp/2)
		# so that the combine of input the preducter target have dimension ninp
		self.encoder = nn.Embedding(ntoken, emb_dim)
		assert rnn_type in ['LSTM'], 'use LSTM pls'
		if rnn_type == 'LSTM':
			#input is the combined w diemnsion ninp
			#in the paper n_hid == ninp
			self.rnns = [torch.nn.LSTM(ninp, nhid, 1, dropout=0) for l in range(nlayers)]

		self.rnns = torch.nn.ModuleList(self.rnns)
		self.decoder = nn.Linear(emb_dim, ntoken)
		self.combiner = nn.Linear(nhid, emb_dim)

		if tie_weights:
			self.decoder.weight = self.encoder.weight 

		self.init_weights()

		self.rnn_type = rnn_type 
		self.ninp = ninp 
		self.nhid = nhid 
		self.nlayers = nlayers 
		self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.dropoutcomb = dropoutcomb
        self.tie_weights = tie_weights

        size = 0
        for p in self.parameters():
        	size += p.nelement()
        	#print (p.size())
        print('Number of parameters: {:,}'.format(size))
        #print('Small model')

    # def reset(self):
    # 	if self.rnn_type == 'QRNN':
    # 		[r.reset() for i in self.rnns]

    def init_weights(self):
    	initrange = 0.1 
    	self.encoder.weight.data.uniform_(-initrange, initrange)
    	self.decoder.bias.data.fill_(0)
    	self.decider.weight.data.uniform_(-initrange, initrange)

    	self.combiner.bias.data.fill_(0)
    	self.combiner.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, prev_targets, hidden, return_h=False):
    	# for this step: take in current input and prev step predicted output/target
    	combined_targets = torch.cat((input.unsqueeze(-1), prev_targets.unqueeze(-1)), -1)
    	emb = embedded_dropout(self.encoder, combined_targets, dropout=self.dropoute if self.training else 0)
    	emb = emb.view(input.shape[0], input.shape[1], -1)
    	emb = self.lockdrop(emb, self.dropout)
    	combined = emb 

    	raw_output = combined 
    	new_hidden = []
    	raw_outputs = [] 
    	outputs = [] 
    	# hidden: initial hidden state (initialized as all zero) for each layer of LSTM
    	for l, rnn in enumerate(self.rnns):
    		raw_output, new_h = rnn(raw_output, hidden[l])
    		new_hidden.append(new_h)
    		raw_outputs.append(raw_output)
    		if l != self.nlayers - 1:
    			raw_output = self.lockdrop(raw_output, self.dropouth)
    			outputs.append(raw_output)
    	# use the generated hidden states as initial states for next batch
    	hidden = new_hidden 

    	# output of the final layer
    	output = self.lockdrop(raw_output, self.dropout)
    	outputs.append(output)

    	out_size_orig0 = output.size(0)
    	out_size_orig1 = output.size(1)

    	output_c = torch.tanh(self.combiner(output.view(output.size(0)*output.size(1), output.size(2))))
    	output_c = output_c.view(output.size(0), output.size(1), -1)

    	output_c_dropped = self.lockdrop(output_c, self.dropoutcomb)

    	decoded = self.decoder(output_c_dropped)
    	result = decoded.view(out_size_orig0, out_size_orig1, decoded.size(2))
    	if return_h:
    		return result, hidden, raw_outputs, outputs 
    	return result, hidden 

    def init_hidden(self, bsz):
    	weight = next(self.parameters()).data 
    	if self.rnn_type == 'LSTM';
    	return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]







