# Bilinear Layers for Interp: MNIST case study
by Michael Pearce and Thomas Dooms

## Motivation
Sparse autoencoders (SAEs) have been successfully used to find interpretable features in LLMs ([Sharkey et al 2022](https://www.alignmentforum.org/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition); [Cunningham et al 2023](https://arxiv.org/abs/2309.08600); [Bricken et al 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-feature-ablations)) but it's unclear if the features SAEs find correspond to those used for computation in the models. For example it's possible that SAEs capture additional features based on the statistics of their training data. In the ideal case, we'd be able to derive features directly from the model's weights. 

Being able to "decompile" a model by rewriting its computation in terms of a sparse feature basis is an important step in the long term mechanistic interpretability agenda ([Sharkey 2024](https://www.alignmentforum.org/posts/64MizJXzyvrYpeKqm/sparsify-a-mechanistic-interpretability-research-agenda)). Currently the nonlinearities in models (and SAEs) make it difficult to decompile models because features can potentially interact in complicated ways. Recent work has focused on training "transcoders" (as opposed to autoencoders) to convert features in one layer to those in the next ([Riggs Smith et al 2024](https://www.alignmentforum.org/posts/7fxusXdkMNmAhkAfc/finding-sparse-linear-connections-between-features-in-llms)). 

An alternative to transcoders is to choose a nonlinearity that makes decompiling easier. [Sharkey 2023](https://arxiv.org/abs/2305.03452) suggested that bilinear layers of the form $g(x) = (W_1 x + b_1) \odot (W_2 x + b_2)$ have simple nonlinearities that may make their interpretations easier. Bilinear layers have several nice properties:
  - _Bilinear layers have comparable performance to other nonlinearities._ A comparison of activations functions found that bilinear activations outperformed ReLU and GELU in transformer models ([Shazeer 2022](https://arxiv.org/abs/2002.05202)) and similar but slightly worse performance to SwiGLU, a modern version of the Gated Linear Unit (GLU) that is used in LLama and PaLM models. The bilinear activation can be seen as the simplest type of GLU.
  - _Computations can be expressed in terms of linear operations with a third order tensor._ This means we can leverage tensor or matrix decompositions, such as singular value decomposition, to understand the weights. For other activation functions, we could still do matrix decompositions on the weights but it would be unclear how to understand the results after applying the nonlinearity. 
  - _Input pairs can act as a basis for bilinear outputs._ This is related to the previous point. Given a set of input features ${v_i}$ with activations ${a_i}$, the output can be expressed as a sum $\sum_{ij} a_i a_j u_{ij}$ where $u_{ij}$ is the output when the input is exactly $v_i + v_j$. The set of ${u_{ij}}$ are a basis for outputs and we can potentially analyze them instead of outputs over a dataset. For example we could train an SAE over $u_{ij}$ instead of a dataset. This may be a way to derive features purely from the model weights.

As a proof of concept we explored how to interpret bilinear layers trained for simple tasks, such as classifying MNIST digits. 

## The bilinear tensor
We can express the action of the bilinear layer, $g(x) = (W_1 x + b_1) \odot (W_2 x + b_2)$, fully through a third order tensor that we'll denoted $B$ for "bilinear":
  - _Including the bias_: Using the usual trick, we can fold the bias into the weight matrices as an additional column, $[W_1, b_1]\rightarrow W_1$, and add a constant value to the input $[x; 1]\rightarrow x$. We can then write the bilinear layer as
```math
g(x) = (W_1 x)\odot(W_2 x) = \sum_{ij} (W_1)_{ai}(W_2)_{aj} x_i x_j \equiv \sum_{ij} B_{aij} x_i x_j
```
  - _Keeping the symmetric part_: Since the same input is used for both the $i,j$ indices only the symmetric part of $B$ contributes to the sum, so we can instead define the bilinear tensor as 
```math
B_{aij} \equiv \frac{1}{2}(W_1)_{ai} (W_2)_{aj} + \frac{1}{2}(W_1)_{aj}(W_2)_{ai}
```
Making the symmetry explicit will simplify some of the decompositions we consider later. 

Since the bilinear tensor is constructed from two matrices, it has fewer parameters than a full rank 3-tensor and is faster to train. Constructing the full bilinear tensor can require a large amount of memory, so it's often easier to keep the $W_1$ and $W_2$ and construct $B$ on the fly when needed. 

## Transforming to a feature basis
Say we're given a set of embedding and unembedding weight matrices, $W_\text{in}$ and $W_\text{out}$, that transform from input features or into output features. We can then transform the bilinear tensor to be into both feature bases:
```math
\tilde{B}_{a'i'j'} = \sum_{aij} (W_\text{out})_{a'a} B_{aij} (W_\text{in})_{i i'}(W_\text{in})_{j j'}
```
**The pseudoinverse trick for feature dictionaries**: Instead of an unembedding matrix we might be given a dictionary $D$ of features as its columns. These features might come from a sparse autoencoder where the activations are given by a ReLU or some other nonlinear function. Nonlinear activations are difficult to incorporate into a model without altering the computations (due to reconstruction error) or keeping the interactions between features interpretable.

However, we can use the pseudoinverse [[wiki](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)], denoted $D^+$, in place of the unembedding matrix above. For the equation $D a = x$, the pseudo-inverse gives the least squares solution for the activations as $a = D^+ x$. When there's not a unique solution, the pseudoinverse gives the solution with the smallest L2 norm. 

A convenient property of $D^+$ is that $D D^+$ is a projection onto the subspace spanned by the features in $D$. This means that if the set of features is full rank (which is reasonable given the typically large number of features) then $D D^+$ equals the identity, so we can transform to the feature basis with no reconstruction loss, although the feature activations will likely not be as sparse. 

**Interaction matrices**: In the feature basis, the bilinear tensor plays the role of the "transcoder" and we can explicitly see how interactions between in-features produce an activation for an out-feature. For example, for out-feature $a$ we can get its symmetric interaction matrix as $Q_{ij} = B_{aij}$. 

The interaction matrix $Q$ can be analyzed in a variety of ways. If $Q$ happens to be sparse, then it's already easy to interpret. If $Q$ is somewhat sparse, we can cluster the interactions to find sets of inputs that interact strongly with each other. 

If the interactions in $Q$ are dense, we can look at its matrix decompositions and derive a set of linear "kernels". Since $Q$ is symmetric, the eigendecomposition takes a simple form: the eigenvalues are real and the eigenvectors are all orthogonal. The eigendecomposition is essentially the same as the SVD (the singular values are the absolute value of the eigenvalues. For an eigendecomposition, $Q = \sum_i \lambda_i q_i q_i^T$, we can express the activation output as
```math
x^T Q x = \sum_i \lambda_i (q_i^T x)^2
```
so the eigenvectors $q_i$ act as a set of linear "kernels". By keeping the large magnitude eigenvalues (typically a small percentage) we can get a low rank approximation of the interaction matrices. 

### MNIST digit basis
As an example of interpreting bilinear layers, we trained a bilinear model to classify handwritten digits in the MNIST dataset [training details in Appendix]. To have clear input and output features we used a single bilinear layer with a final linear readout, which we can use as $W_\text{out}$ to transform into the digit feature basis.

For each digit $d$, the interaction matrix $Q_{ij}^{(d)} = \tilde{B}_{dij}$ involves interactions between many pairs of input pixels. Given the 2d structure of the inputs, it's easier to interpret the eigenvectors of $Q$ which are themselves 2d images, instead of parsing the interactions between pixels directly. The figure below shows the top 3 positive and negative eigenvectors for each digit. It's clear that the positive eigenvectors (top rows) capture features we might expect: circular curves for 0, a straight line for 1, parallel vertical lines for 4, a horizontal line for 5, etc. 

Given an input $x$, the logit for digit $d$ is $x^T Q^{(d)} x = \sum_i  \lambda_i (q_i^T x)^2$, so it's the square of an eigenvector's overlap with the input that contributes. This means that both the blue (pos) and red (neg) parts contribute in the same way, ie the sign of $q^Tx$ does not matter. For example, the eigenvectors for 1 contain both red and blue lines which can have high overlap for 1's with different slants. 

The computation using $Q$ is still a superposition over different features for each digit, for example over different locations in the image, different slants, and different digit shapes. The eigenvectors (and their separation in blue/red components) help disentangle this superposition to an extent. The eigenvectors are all orthogonal so each captures a different aspect (eg the 3rd positive eig for 1 adds a bottom horizontal line, the 3rd positive eig for 7 adds a horizontal cross) but the orthogonality constraint can also their interpretation less clear. There is likely a better approach to deriving interpretable features from $Q$ which we haven't fully explored, for example looking at the set of local maxima/minima for $x^T Q x$ might find a set of clear digits shapes 

An important point is that the 10 different $Q$'s fully capture the model's computations. We could rewrite the model fully in terms of their eigenvectors if desired. This shows the advantages of the bilinear structure. For other nonlinearities it's difficult to escape the "neuron" basis in which the nonlinearity is applied when trying to connect the output features to input ones. 

![image](/images/MNIST_digit_basis_1K.png)

### MNIST SAE features using the pseudoinverse trick

As an additional example, we trained a sparse autoencoder on the outputs of the bilinear layer (before the linear readout) [details in Appendix]. The SAE has an average of L0 of 5.2 but only 87 features are active (out of 2000 starting features) and a MSE/$L2^2$ of 0.28. We didn't try too hard to optimize the SAE training to minimize dead features or prevent dense features because we're mainly interested in illustrating how a set of dictionary features $D$ can be incorporated into the bilinear model. 

The plots below show the top four features by mean activation. We see some similar eigenvectors to the digit basis results although noiser. Similar digits can be combined in the positive direction of an SAE feature (2 and 6, 4 and 9) while the negative direction can contains constrasting digits (3 vs 4 in feature 1).

![image](/images/MNIST_SAE_dataset_relu.png)

The interaction matrices $Q$ are derived using the pseudoinverse trick described above using $W_\text{out} = D^+$. The fact that the logit directions mostly make sense in terms of the eigenvectors is a sign that the pseudoinverse is a reasonable way to incorporate the dictionary features. In fact, when restricting to the 87 active features, the **pseudoinverse activations improve the loss recovered from 87.1% for the ReLU activations to 98.1%**, using a random guessing baseline for the loss. The plot below shows that the pseudoinverse activations are well correlated with the nonzero ReLU activations and are typically small values for the zero ReLU activations. 

![image](/images/MNIST_relu_vs_pinv_activations.png)

## Deriving features from the model weights

### MNIST SVD

![image](/images/MNIST_svd_features_1K.png)

![image](/images/MNIST_topk_bottleneck_accuracy_1K.png)

![image](/images/MNIST_best_match_similarity_1K.png)


### MNIST SAE over input pair vectors

![image](/images/MNIST_SAE_btensor_relu.png)

## Future Work

# Appendix 

## MNIST training
[[Colab](https://colab.research.google.com/drive/12sE0jLTgY4_77ia7gRdCOo8-e52mJXRx?usp=sharing)]
[add colabs for other hidden dims]
- single layer model validation accuracy of 97.8%.
- Similar performance for ReLU
- Hidden dim: 1000 (higher dim minimizes random interference between input pair features)

## SAE training
**SAE trained over input dataset** [[Colab](https://colab.research.google.com/drive/19H1a_qy_RkqWzwV8i4W2gnuV3HwB1fhR?usp=sharing)]
- Starting features: 2000
- $L0$: 5.2
- $MSE/L2^2$: 0.28
- Dead feature frac: 96%
- Active features: 87
- [architecture]
- penalty on feature similarity

**SAE trained over bilinear tensor** [[Colab](https://colab.research.google.com/drive/1UCrpT-zod4ylPMaaqmz2BYrvYZYA_ndP?usp=sharing)]
- Starting features: 5000
- $L0$: 30.4
- $MSE/L2^2$: 0.29
- Dead feature frac: 29%
- Active features: 3544


