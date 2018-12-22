## My Thoughts on Eager Transaltion Model

[Paper](https://arxiv.org/pdf/1810.13409.pdf)

This model is simpler than existing machine translation models. It no longer uses the encoder-decoder structure and treats machine translation as a many-to-many RNN problem with stacked LSTM. 

It first aligns the tokens in source and target sentences and adds padding to avoid different lengths of input and output. 

The advantage is that it does not need to read the entire input sentence and then generates the output (that's why it's called eager translation).

In terms of performance, the model performs worse than seq2seq on short sentences but better than seq2seq on longer sentences. And it also converges slower than seq2seq. 

Some guesses on why it does not perform so well:
1) The model cannot learn well the alignment. Sometimes the source token and its corresponding target token are near to each other, but sometimes they can be far away, which is difficult to learn.

2) Sometimes the alignment does not necessarily reflect how the sentences are translated. It is possible that one source word gets translated to many target words or vice versa. The alignment can sometimes can very complicated. I think this is an important reason why multi-head attention in the transformer is very helpful. 

Imagine the case where the order of tokens is reversed in the source and target languages (i.e. first token of target language is translated from the last token of source language). Then the padded source sentence would have a lot of padding at the end and the target sentence would have a lot of padding at the front, which looks like an encoder-decoder combined together with one set of shared weights for LSTM. In this case, the advantage of eager translation is minimal (because it has to read the entire source sentence before starts generating real translation). 

So this eager translation model might only work well with language pairs that have roughly the same language order. (Would be interesting to see experiments on language pairs with very different word orders, e.g. SVO vs SOV.)

In summary, I find this work a neat idea but may not be powerful enough.

Update 1: Perhaps a possible way to incorporate this to attention-based models is to throw away the decoder in the transformer and replace the feed-forward NN (all or just the last layer's) with RNNs. Maybe self-attention could improve the performance.