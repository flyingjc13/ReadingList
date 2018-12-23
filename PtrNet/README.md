## My Thoughts on Pointer Network

[Paper](https://arxiv.org/pdf/1506.03134.pdf)

The pointer network is a variation to the classical encoder-decoder structure.
The output of the decoder is not a distribution over an output dictionary but instead a distribution over the encoder steps. To compute the distribution is very similar to compute the attention where you compute a weighted sum of the current decoder hidden state and each encode state (at each timestep) and then do a linear projection for a score at each timestep. The distribution would be the softmax of this score over all timesteps of the encoder.

As a result, you can specify the decoder length. For example, you may want to find the start and end index of a particular subsequence in the input, then your decoder length would be 2.

When I first read the paper, I thought the decoder length is fixed for pointer network. So I went on to code a modified pointer network that learns the decoder length. The experiments are documented in the notebook. I tested it on a synthetic task where the input sequence contains a variable number of subsequences. Some subsequences' values are all smaller 5 and some are all greater than 5. The task is to learn the bounaries of those subsequences that are greater than 5. 

The model works well on both learning the correct number of pointers(boundaries) and correct pointer positions. 

However, after talking to the original author (Dr. Vinyals), I realize that I can solve this task simply by following the original seq2seq practice, where I can add a end-of-sequence token at the end of the output sequence during training so that the model can learn to point to the EOS token when it's done outputting.