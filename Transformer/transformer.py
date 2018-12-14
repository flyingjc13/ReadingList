## Codes from HarvardNLP: http://nlp.seas.harvard.edu/2018/04/03/attention.html
## uses Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
%matplotlib inline


class EncoderDecoder(nn.Module):
	"""
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
    	super(EncoderDecoder, self).__init__()
    	self.encoder = encoder 
    	self.decoder = decoder 
    	self.src_embed = src_embed 
    	self.tgt_embed = tgt_embed 
    	self.generator = generator 

    def forward(self, src, tgt, src_mask, tgt_mask):
    	# Take in and process masked src and tgt seq 
    	return self.decode(self.encode(src, src_mask), src_mask, 
    						tgt, tgt_mask)

    def encode(self, src, src_mask):
    	return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
    	return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
	# Define standard linear + softmax generation step 
	def __init__(self, d_model, vocab):
		super(Generator, self).__init__()
		self.proj = nn.Linear(d_model, vocab)

	def forward(self, x):
		return F.log_softmax(self.proj(x), dim=-1)


## Encoder
# Produce N identical layers
def clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
	# core encoder is a stack of N layers
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, mask):
		# Pass the inpit and mask through each layer in turn
		for layer in self.layers:
			x = layer(x, mask)
		return self.norm(x)

# layer normalization
class LayerNorm(nn.Module):
	# a_2 and b_2 are learnable parameters (g and b in the paper)
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps eps 

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
	"""
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    Not sure if the order affects the performance?
    """
    def __init__(self, size, dropout):
    	super(SublayerConnection, self).__init__()
    	self.norm = LayerNorm(size)
    	self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
    	# Apply residual connection to any sublayer with the same size
    	return x + self.dropout(sublayer(self.norm(x)))
    	# return x + self.dropout(self.norm(sublayer(x)))
    	# return self.norm(x+self.dropout(sublayer(x)))

class EncoderLayer(nn.Module):
	# Encoder is made up of self-attn and feed-forward
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn 
		self.feed_forward = feed_forward 
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size 

	def forward(self, x, mask):
		# define the sublayers in one encoder layer
		x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)

## Decoder 
class Decoder(nn.Module):
	# N layer decoding with masking
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)

	def forward(self, x, memory, src_mask, tgt_mask):
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)

class DecoderLayer(nn.Module):
	# decoder is made of self-attn, src-attn, feed_forward
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size 
		self.self_attn = self_attn 
		self.src_attn = src_attn 
		self.feed_forward = feed_forward 
		self.sublayer = clones(SublayerConnection(size, dropout), 3)

	def forward(self, x, memory, src_mask, tgt_mask):
		m = memory 
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
		return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
	# mask out subsequent positions 
	attn_shape = (1, size, size)
	# triu: upper triangle of matrix
	subsequent_mask = np.triu(np.ones(atten_shape)m k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0 


# query: decoder state, key: encoder state, value: encoder state
# input shape: (nbatch, h, d_k)
def attention(query, key, value, mask=None, dropout=None):
	# compute scaled dot product attention 
	d_k = query.size(-1)
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
	if mask is not None:
		scores = scores.masked_fill(mask==0, -1e9)
	p_attn = F.softmax(scores, dim=-1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn 


class MultiHeadAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		super(MultiHeadAttention, self).__init__()
		assert d_model % h == 0
		# assume d_k == d_v
		self.d_k = d_model // h 
		self.h = h 
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None 
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, query, key, value, mask=None):
		if mask is not None:
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)

		# linear projections in batch from d_model => (h, d_k)
		# using the first 3 linear layers
		query, key, value = \
			[l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
			 for l,x in zip(self.linears, (query, key, value))]

		# apply attention on all the projected vectors in batch 
		x, self.attn = attention(query, key, value, mask=mask,
								dropout=self.dropout)

		# concat and apply a final linear
		x = x.transpose(1, 2).contiguous() \
			.view(nbatches, -1, self.h*self.d_k)

		return self.linears[-1](x)

## position-wise feed forward
## or: two 1D conv 
class PositionwiseFeedForward(nn.Module):
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
	# d_model: emb dim
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model 

	def forward(self.x):
		return self.lut(x) * math.sqrt(self.d_model)


## Positional Encoding 
## can learn relative positions
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# compute positional encoding
		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0))/d_model)
		pe[:, 0::2] = torch.sin(position*div_term)
		pe[:, 1::2] = torch.cos(position*div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + Variable(self.pe[:, :x.size(1)])
		return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, 
				d_model=512, d_ff=2048, h=8, dropout=0.1):
	# construct a model from hyperparameters 
	c = copy.deepcopy 
	attn = MultiHeadAttention(h, d_model)
	ff = PositionwiseFeedForward(d_model, d_ff, dropout)
	position = PositionalEncoding(d_model, dropout)
	model = EncoderDecoder(
		Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
		Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
		nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
		nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
		Generator(d_model, tgt_vocab))

	# glorot initialization
	for p in model.parameters():
		if p.dim() > 1:
			nn.init.xavier_uniform(p)
	return model 


## Btach + masking
class Batch:
	def __init__(self, src, trg=None, pad=0):
		self.src = src 
		self.src_mask = (src != pad).unsqueeze(-2)
		if trg is not None:
			self.trg = trg[:, :-1]
			self.trg_y = trg[:, 1:]
			self.trg_mask = self.make_std_mask(self.trg, pad)
			self.ntokens = (self.trg_y != pad).data.sum() 

	@staticmethod 
	def make_std_mask(tgt, pad):
		# create a mask to hide padding and future words 
		tgt_mask = (tgt!=pad).unsqueeze(-2)
		tgt_mask = tgt_mask & Variable(
			subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
		return tgt_mask 

## Training loop + logger
def run_epoch(data_iter, model, loss_compute):
	start = time.time()
	total_tokens = 0 
	total_loss = 0 
	tokens = 0
	for i, batch in enumerate(data_iter):
		out = model.forward(batch.src, batch,trg, batch.src_mask, batch.trg_mask)
		loss = loss_compute(out, batch.trg_y, batch.ntokens)
		total_loss += loss 
		total_tokens += batch.ntokens 
		tokens += batch.ntokens 
		if i%50 == 1:
			elapsed = time.time() - start 
			print ("Epoch Step: %d Loss: %f Tokens per Sec: %f"%
					(i, loss/batch.ntokens, tokens/elapsed))
			start = time.time()
			tokens = 0 
	return total_loss / total_tokens


## Training data and batching
global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
	# Keep augmenting batch and calculate total number of tokens + padding
	global max_src_in_batch, max_tgt_in_batch 
	if count == 1:
		max_src_in_batch = 0 
		max_tgt_in_batch = 0 
	max_src_in_batch = max(max_src_in_batch, len(new.src))
	max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg)+2)
	src_elements = count * max_src_in_batch 
	tgt_elements = count * max_tgt_in_batch 
	return max(src_elements, tgt_elements)


## Optimizer 
class NoamOpt:
	## learning rate: warm up then decrease 
	## following noam lr decay scheme
	def __init__(self, model_size, factor, warmup, optimizer):
		self.optimizer = optimizer 
		self._step = 0 
		self.warmup = warmup 
		self.factor = factor 
		self.model_size = model_size 
		self._rate = 0 

	def step(self):
		# update parameters and rate 
		self._step += 1
		rate = self.rate()
		for p in self.optimizer.param_groups:
			p['lr'] = rate 
		self._rate = rate 
		self.optimizer.step() 

	def rate(self, step=None):
		if step is None:
			step = self._step 
		# follow learning rate update formula
		return self.factor * \
			(self.model_size ** (-0.5) * 
			min(step**(-0.5), step*self.warmup**(-1.5)))

def get_std_opt(model):
	return NoamOpt(model.src_embed[0].d_model, 2, 4000, 
		torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


## Regularization
## Label Smoothing
'''
it hurts perplexity as the model learns to be unsure,
but it improves accuracy and BLEU score.
it starts to penalize the model if it gets very confident on a given choice
'''
class LabelSmoothing(nn.Module):
	def __init__(self, size, padding_idx, smoothing=0.0):
		super(LabelSmoothing, self).__init__()
		self.criterion = nn.KLDivLoss(size_average=False)
		self.padding_idx = padding_idx 
		self.confidence = 1.0 - smoothing 
		self.smoothing = smoothing 
		self.size = size 
		self.true_dist = None 

	def forward(self, x, target):
		assert x.size(1) == self.size 
		true_dist = x.data.clone()
		true_dist.fill_(self.smoothing / (self.size - 2))
		true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
		true_dist[:, self.padding_idx] = 0 
		mask = torch.nonzero(target.data == self.padding_idx)
		if mask.dim() > 0:
			true_dist.index_fill_(0, mask.squeeze(), 0.0)
		self.true_dist = true_dist 
		return self.criterion(x, Variable(true_dist, requires_grad=False))




## A First Example 
## Synthetic data:
'''
Given a random set of onput symbols,
generate back those same symbols.
'''
def data_gen(V, batch, nbatches):
	for i in range(nbatches):
		data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
		data[:, 0] = 1
		src = Variable(data, requires_grad=False)
		tgt = Variable(data, requires_grad=False)
		yield Batch(src, tgt, 0)

class SimpleLossCompute:
	def __init__(self, generator, criterion, opt=None):
		self.generator = generator 
		self.criterion = criterion 
		self.opt = opt 

	def __call__(self, x, y, norm):
		x = self.geenrator(x)
		loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
							  y.contiguous().view(-1)) / norm 
		loss.backward()
		if self.opt is not None:
			self.opt.step()
			self.opt.optimizer.zero_grad() 
		return loss.data[0]*norm 

## Train the simple copy task 
V = 11 
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
	torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
	model.train()
	run_epoch(data_gen(V, 30, 20), model, 
			  SimpleLossCompute(model.generator, criterion, model_opt))
	model.eval()
	print (run_epoch(data_gen(V, 30, 5), model, 
					 SimpleLossCompute(model.generator, criterion, None)))

def greedy_decode(model, src, src_mask, max_len, start_symbol):
	memory = model.encode(src, src_mask)
	ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
	prob = model.generator(out[:, -1])
	_, next_word = torch,max(prob, dim=1)
	next_word = next_word.data[0]
	ys = torch.cat([ys,
		torch.ones(1,1).type_as(src_data).fill_(next_word)], dim=1)
	return ys 

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]))
src_mask = Variable(torch.ones(1, 1, 10))
print (greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))



## NMT example
## To load data:
#!pip install torchtext spacy
#!python -m spacy download en
#!python -m spacy download de

from torchtext import data, datasets 

if True:
	import spacy 
	spacy_de = spacy.load('de')
	spacy_en = spacy.load('en')

	def tokenize_de(text):
		return [tok.text for tok in spacy_de.tokenizer(text)]

	def tokenizer_en(text):
		return [tok.text for tok in spacy_en.tokenizer(text)]

	BOS_WORD = '<s>'
	EOS_WORD = '</s>'
	BLANK_WORD = "<blank>"
	SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
	TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
					 eos_token=EOS_WORD, pad_token=BLANK_WORD)

	MAX_LEN = 100
	train, val, test = datasets.IWSLT.splits(
		exts=('.de', '.en'), fields=(SRC, TGT),
		filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
		len(vars(x)['trg']) <= MAX_LEN)
	MIN_FREQ = 2
	SRC.build_vocab(train.src, min_freq=MIN_FREQ)
	TRT.build_vocab(train.trg, min_freq=MIN_FREQ)

'''
Batching matters a ton for speed. 
We want to have very evenly divided batches, with absolutely minimal padding. 
To do this we have to hack a bit around the default torchtext batching. 
This code patches their default batching to make sure we search over enough sentences to find tight batches.
'''
class MyIterator(data.Iterator):
	def create_batches(self):
		if self.train:
			def pool(d, random_shuffler):
				for p in data.batch(d, self.batch_size*100):
					p_batch = data.batch(
						sorted(p, key=self.sort_key),
						self.batch_size, self.batch_size_fn)
					for b in random_shuffler(list(p_batch)):
						yield b 
			self.batches = pool(self.data(), self.random_shuffler)
		else:
			self.batches = []
			for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
				self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
	# fix order in torchtext to match ours 
	src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
	return Batch(src, trg, pad_idx)


## Multi-GPU Training
'''
This code implements multi-gpu word generation.
The idea is to split up word generation at training time into chunks 
to be processed in parallel across many different gpus. 
''' 
# Skip if not interested in multigpu.
class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."
    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, 
                                               devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size
        
    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, 
                                                devices=self.devices)
        out_scatter = nn.parallel.scatter(out, 
                                          target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, 
                                      target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.            
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize



## create model, criterion, optimizer, parallelization
devices = [0,1,2,3]
if True:
	pad_idx = TGT.vocab.stoi["<blank>"]
	model = make_model(len(SRC.vocab), len(TGT.vocab), N=6)
	model.cuda()
	criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
	criterion.cuda()
	BATCH_SIZE = 12000
	train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=0,
							repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
							batch_size_fn=batch_size_fn, train=True)
	valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=0,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
	model_par = nn.DataParallel(model, device_ids=devices)


if False:
	model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
						torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
	for epoch in range(10):
		model_par.train()
		run_epoch((rebatch(pad_idx, b) for b in train_iter),
				  model_par,
				  MultiGPULossCompute(model.generator, criterion, 
				  	devices=devices, opt=model_opt))
		model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter), 
                          model_par, 
                          MultiGPULossCompute(model.generator, criterion, 
                          devices=devices, opt=None))
        print(loss)
else:
	# load trained model
    model = torch.load("iwslt.pt")



## print some translations
for i, batch in enumerate(valid_iter):
    src = batch.src.transpose(0, 1)[:1]
    src_mask = (src != SRC.vocab.stoi["<blank>"]).unsqueeze(-2)
    out = greedy_decode(model, src, src_mask, 
                        max_len=60, start_symbol=TGT.vocab.stoi["<s>"])
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[i, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break


## Ensemble: average the last k checkpoints 
def average(model, models):
    "Average models into model"
    for ps in zip(*[m.params() for m in [model] + models]):
        p[0].copy_(torch.sum(*ps[1:]) / len(ps[1:]))


## Attention Visualization
tgt_sent = trans.split()
def draw(data, x, y, ax):
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax)
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Encoder Layer", layer+1)
    for h in range(4):
        draw(model.encoder.layers[layer].self_attn.attn[0, h].data, 
            sent, sent if h ==0 else [], ax=axs[h])
    plt.show()
    
for layer in range(1, 6, 2):
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    print("Decoder Self Layer", layer+1)
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], 
            tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()
    print("Decoder Src Layer", layer+1)
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    for h in range(4):
        draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], 
            sent, tgt_sent if h ==0 else [], ax=axs[h])
    plt.show()













