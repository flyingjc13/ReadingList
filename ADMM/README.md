## My Thoughts on ADMM for NN optimization

[Paper](https://arxiv.org/pdf/1605.02026.pdf)

The optimization of NNs is formulated as an ADMM optimization problem.
Note that the Lagrange multiplier is only added for the final output, unlike in the classical ADMM where it is added for each constraint. This is claimed to be more stable. 

The main advantage is that it supports parallelization and can be easily distributed on many nodes.

In terms of convergence performance, it does not seem to have a significant advantage over gradient-based methods according to the experiments in the paper. But I suspect that ADMM suffers less from local minima problems (but not supported by experimented results). (Although there is proof that GD can achieve global minima as long as the model is large enough. [ref1](https://arxiv.org/pdf/1811.03804.pdf) [ref2](http://proceedings.mlr.press/v80/laurent18a/laurent18a.pdf))

I am thinking of extending this method to RNNs. I suspect that such methods can alleviate the vanishing gradient problem. 