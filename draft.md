# Features in bilinear models

## Motivation
Sparse autoencoders (SAEs) have been successfully used to find interpretable features in LLMs ([Sharkey et al 2022](https://www.alignmentforum.org/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition); [Cunningham et al 2023](https://arxiv.org/abs/2309.08600); [Bricken et al 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-feature-ablations)) but it's unclear if the features SAEs find correspond to those used for computation in the models. For example it's possible that SAEs capture additional features based on the statistics of their training data. In the ideal case, we'd be able to derive features directly from the model's weights. 

Being able to "decompile" a model by rewriting its computation in terms of a sparse feature basis is an important step in the long term mechanistic interpretability agenda ([Sharkey 2024](https://www.alignmentforum.org/posts/64MizJXzyvrYpeKqm/sparsify-a-mechanistic-interpretability-research-agenda)). Currently the nonlinearities in models (and SAEs) make it difficult to decompile models because features can potentially interact in complicated ways. Recent work has focused on training "transcoders" (as opposed to autoencoders) to convert features in one layer to those in the next ([Riggs Smith et al 2024](https://www.alignmentforum.org/posts/7fxusXdkMNmAhkAfc/finding-sparse-linear-connections-between-features-in-llms)). 

An alternative to transcoders is to choose a nonlinearity that makes decompiling easier. [Sharkey 2023](https://arxiv.org/abs/2305.03452) suggested that bilinear layers of the form $g(x) = (W_1 x + b_1) \odot (W_2 x + b_2)$ have simple nonlinearities that may make their interpretations easier. Bilinear layers have several nice properties:
  - _Bilinear layers have comparable performance to other nonlinearities._ A comparison of activations functions found that bilinear activations outperformed ReLU and GELU in transformer models ([Shazeer 2022](https://arxiv.org/abs/2002.05202)) and similar but slightly worse performance to SwiGLU, a modern version of the Gated Linear Unit (GLU) that is used in LLama and PaLM models. The bilinear activation can be seen as the simplest type of GLU.
  - _Computations can be expressed in terms of linear operations with a third order tensor._ This means we can leverage tensor or matrix decomposition, such as singular value decomposition, to understand the weights. For other activations, we can still do matrix decompositions on the weights but its unclear how to understand the results after applying the nonlinearity. 
  - _Input pairs can act as a basis for bilinear outputs._ This is related to the previous point. Given a set of input features ${v_i}$ with activations ${a_i}$, the output can be expressed as a sum $\sum_{ij} a_i a_j u_{ij}$ where $u_{ij}$ is the output when the input is exactly $v_i + v_j$. The set of ${u_{ij}}$ are a basis for outputs and we can potentially analyze them instead of outputs over a dataset. For example we could train an SAE over $u_{ij}$ instead of a dataset. This may be a way to derive features purely from the model weights.

As a proof of concept we explored how to interpret bilinear layers trained for simple tasks, such as classifying MNIST digits. 

## The bilinear tensor
We can express the action of the bilinear layer, $g(x) = (W_1 x + b_1) \odot (W_2 x + b_2)$, fully through a third order tensor that we'll denoted $B$ for "bilinear". Using the usual trick, we can fold the bias into the weight matrices as an additional column, eg $[W_1; b_1]$. 


Two linear layers, less weights than a full three tensor
folding bias into the weights
can save memory by keeping 
B tensor, symmetrifying

## Transforming to a feature basis

Dotting B with a feature directions to get a feature’s quadratic “kernel” of interactions
Eigenvectors of the quadratic kernel to get linear kernels

Pseudoinverse trick to project onto a feature basis


### MNIST digit directions

### MNIST SAE features

## Deriving features from the model weights

### MNIST SVD

### MNIST SAE over input pair vectors

## Future Work

