## My Thoughts on Pointer Network

[Paper](https://arxiv.org/pdf/1506.03134.pdf)

The pointer network is a variation to the classical encoder-decoder structure.
The output of the decoder is not a distribution over an output dictionary but instead a distribution over the encoder steps. To compute the distribution is very similar to compute the attention where you compute a weighted sum of the current decoder hidden state and each encode state (at each timestep) and then do a linear projection for a score at each timestep. The distribution would be the softmax of this score over all timesteps of the encoder.

As a result, you can specify the decoder length. For example, you may want to find the start and end index of a particular subsequence in the input, then your decoder length would be 2.