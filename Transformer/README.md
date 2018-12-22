## My Thoughts on Transformer

[Paper](https://arxiv.org/pdf/1706.03762.pdf)

Transformer is still an encoder-decoder structure, but it gets rid of RNN.
In the encoder, there are N identical layers stacked together, each consisting of a self-attention and a feed-forward NN sublayer.
In the decoder, apart from the same two sublayers as in the encoder, there is another attention layer that matches the alignment of output form the encoder and the output from the previous sublayer(self-attention) in the decoder. This attention layer has similar effect as the attention mechanism in the traditional seq2seq model. And my intuitive understanding of the need of self-attention is that it allows the model to learn what parts of the input/output are more important and should be carried to the next layers.

The attention is computed differently for transformer. In the traditional seq2seq, the compatibility function is computed using a feed-forward network with a single hidden layer (then use softmax to compute the weight and then a weighted sum of concatenated encoder hidden state). In the transformer, the weight is simply computed by a dot product and softmax, which is faster.

Another advantage of transformer is its multi-head attntion. It computes linear projections of Q, K, V h times, each with a different set of weights. Attention is computed on each of the h projections and then concatenated together and projected again to product the final attention. The advantage is that it exploits different ways of paying attention to different parts. In traditional seq2seq, there is only one attention to align the input and output. But in transformer, the model combines many different ways of alignment.

The disadvantage of not using RNN is that the model loses the ability to learn temporal(sequential) information. The transformer replaces it with position encoding where they use sine and cosine functions to learn the relative positions. BUT I personally suspect that this does not allow the model to learn long-term dependencies (compared to LSTM/GRU). If we are not so concerned about training speed and parallelization, perhaps replacing the positional encoding with LSTM/GRU could improve performance on tasks that need long-term dependencies. 


Update 1: It seems like the idea of combining LSTM and transformer has already been tested. Check this work(https://arxiv.org/pdf/1804.09849.pdf) by Chen et al.

In their RNMT+ model, the positional encoding, feed-forward neural network and self-attention has been replaced with BiLSTM (uniLSTM in decoder). This is pretty much like a stacked seq2seq but with multi-head attention between each encoder and decoder layer. There is indeed some improvement over transformer. I suspect that the difference will be bigger on tasks with more distant dependencies.

They also experimented with different hybrids of encoder and decoder from transformer and RNMT+. The conclusion is that the transformer encoder is better at feature extraction and RNMT+ decoder is better at conditional language modeling.

They further enhance the performance by combining two encoders together. They experimented with cascaded encoder (transformer encoder on top of RNMT+ encoder) and multi-column encoder (outputs from two encoder are concatenated).

So this is what you can do when you have a lot of GPUs :) 

Update 2: In this [paper](https://arxiv.org/pdf/1808.08946.pdf), the authors use experiments to show that transformer and CNNs do not outperform RNNs on tasks requiring long-range dependencies (e.g. aubject-verb agreement). However, transformer is better than CNNs and RNNs in feature extraction (e.g. word sense disambiguation).


