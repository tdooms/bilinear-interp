# Bilinear Layers for Mech-Interp: an MNIST case study
by Michael Pearce and Thomas Dooms

## Motivation
Sparse autoencoders (SAEs) have been successfully used to find interpretable features in LLMs ([Sharkey et al 2022](https://www.alignmentforum.org/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition); [Cunningham et al 2023](https://arxiv.org/abs/2309.08600); [Bricken et al 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-feature-ablations)) but it's unclear if the features SAEs find correspond to those used for computation in the models. For example it's possible that SAEs capture additional features based on the statistics of their training data. In the ideal case, we'd be able to derive features directly from the model's weights. 

Being able to "decompile" a model by rewriting its computation in terms of a sparse feature basis is an important step in the long term mechanistic interpretability agenda ([Sharkey 2024](https://www.alignmentforum.org/posts/64MizJXzyvrYpeKqm/sparsify-a-mechanistic-interpretability-research-agenda)). Currently the nonlinearities in models (and SAEs) make it difficult to decompile models because features can potentially interact in complicated ways. Recent work has focused on training "transcoders" (as opposed to autoencoders) to convert features in one layer to those in the next ([Riggs Smith et al 2024](https://www.alignmentforum.org/posts/7fxusXdkMNmAhkAfc/finding-sparse-linear-connections-between-features-in-llms)). 

An alternative to transcoders is to choose a nonlinearity that makes decompiling easier. [Sharkey 2023](https://arxiv.org/abs/2305.03452) suggested that bilinear layers of the form $g(x) = (Wx + b_w) \odot (V x + b_v)$ have simple nonlinearities that may make their interpretations easier. Bilinear layers have several nice properties:
  - **Bilinear layers have comparable performance to other nonlinearities.** A comparison of activations functions found that bilinear activations outperformed than ReLU and GELU in transformer models ([Shazeer 2022](https://arxiv.org/abs/2002.05202)) and similar performance to SwiGLU, a modern version of the Gated Linear Unit (GLU) that is used in LLama and PaLM models. The bilinear activation can be seen as the simplest type of GLU.
  - **Computations can be expressed in terms of linear operations with a third order tensor.** This means we can leverage tensor or matrix decompositions, such as singular value decomposition, to understand the weights. For other activation functions, we could still do matrix decompositions on the weights but we cannot propagate the transformations through the privileged basis created by the elementwise nonlinearity. 
  - **Input pairs can act as a basis for bilinear outputs.** This is related to the previous point. Given a set of input features $\vec{v_i}$ with activations ${a_i}$, the output can be expressed as a sum $\sum_{ij} a_i a_j \vec{r}\_{ij}$ where $\vec{r}\_{ij}$ is the output when the input is exactly $\vec{v_i} + \vec{v_j}$. The set of $\vec{r}\_{ij}$ are a basis for outputs and we can potentially analyze them instead of outputs over a dataset. For example we could train an SAE over $\vec{r}\_{ij}$ instead of a dataset. This may be a way to derive features purely from the model weights.

As a proof of concept we explored how to interpret bilinear layers trained for simple tasks, such as classifying MNIST digits. 

## The bilinear tensor
We can express the action of the bilinear layer, $g(x) = (Wx + b_w) \odot (V x + b_v)$, fully through a third order tensor that we'll denote $B$ for "bilinear":
  - _Including the bias_: Using the usual trick, we can fold the bias into the weight matrices as an additional column, $[W, b_w]\rightarrow W$, and add a constant value to the input $[x; 1]\rightarrow x$. We can then write the bilinear layer as
```math
g(x) = (W x)\odot(V x) = \sum_{ij} W_{ai}V_{aj} x_i x_j \equiv \sum_{ij} B_{aij} x_i x_j
```
  - _Keeping the symmetric part_: Since the same input is used for both the $i,j$ indices only the symmetric part of $B$ contributes to the sum, so we can instead define the bilinear tensor as 
```math
B_{aij} \equiv \frac{1}{2} \left( W_{ai} V_{aj} + W_{aj}V_{ai}\right)
```

Since the bilinear tensor is constructed from two matrices, it has fewer parameters than a full rank 3-tensor and is faster to train. Constructing the full bilinear tensor can require a large amount of memory, so it's often easier to keep the $W$ and $V$ and construct $B$ on the fly when needed. 

## Transforming to a feature basis
Say we're given a set of embedding and unembedding weight matrices, $E$ and $U$, that transform from input features or into output features. We can then transform the bilinear tensor to be into both feature bases:
```math
\tilde{B}_{a'i'j'} = \sum_{aij} U_{a'a} B_{aij}E_{i i'}E_{j j'}
```
### The pseudoinverse trick for feature dictionaries
Instead of an unembedding matrix we might be given a dictionary $D$ of features as its columns. These features might come from a sparse autoencoder where the activations are given by a ReLU or some other nonlinear function. Nonlinear activations are difficult to incorporate into a model without altering the computations (due to reconstruction error) or keeping the interactions between features interpretable.

However, we can use the pseudoinverse [[wiki](https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse)], denoted $D^+$, in place of the unembedding matrix above. For the equation $D a = x$, the pseudo-inverse gives the least squares solution for the activations as $a = D^+ x$. When there's not a unique solution, the pseudoinverse gives the solution with the smallest L2 norm. 

A convenient property of $D^+$ is that $D D^+$ is a projection onto the subspace spanned by treating the features in $D$ as a vector basis. This means that if the set of features is a complete basis (full rank), which is reasonable given the typically large number of features, then $D D^+$ equals the identity, so we can transform to the feature basis with no reconstruction loss, although the feature activations will likely not be as sparse. 

### Interaction matrices
In the feature basis, the bilinear tensor plays the role of the "transcoder" and we can explicitly see how interactions between in-features produce an activation for an out-feature. For example, for out-feature $a$ we can get its symmetric interaction matrix as $Q_{ij} = B_{aij}$. 

The interaction matrix $Q$ can be analyzed in a variety of ways. If $Q$ happens to be sparse, then it's already easy to interpret. If $Q$ is somewhat sparse, we can perhaps cluster the interactions to find sets of inputs that interact strongly with each other. 

If the interactions in $Q$ are dense, we can look at its matrix decompositions and derive a set of linear "kernels". Since $Q$ is symmetric, the eigendecomposition takes a simple form: the eigenvalues are real and the eigenvectors are all orthogonal. The eigendecomposition is essentially the same as the SVD (the singular values are the absolute value of the eigenvalues). For an eigendecomposition, $Q = \sum_i \lambda_i q_i q_i^T$, we can express the activation output as
```math
x^T Q x = \sum_i \lambda_i (q_i^T x)^2
```
so the eigenvectors $q_i$ act as a set of linear "kernels". By keeping the large magnitude eigenvalues (typically a small percentage) we can get a low rank approximation of the interaction matrices. 

### The MNIST digit basis
As an example of interpreting bilinear layers, we trained a bilinear model to classify handwritten digits in the MNIST dataset [training details in Appendix]. To have clear input and output features we used a single bilinear layer with a final linear readout, which we can use as the unembedding $U$ to transform into the digit feature basis.

For each digit $d$, the interaction matrix $Q_{ij}^{(d)} = \tilde{B}_{dij}$ involves interactions between many pairs of input pixels. Given the 2d structure of the inputs, it's easier to interpret the eigenvectors of $Q$ which are themselves 2d images, instead of parsing the interactions between pixels directly. The figure below shows the top 3 positive and negative eigenvectors for each digit. It's clear that the positive eigenvectors (top rows) capture features we might expect: circular curves for 0, a straight line for 1, parallel vertical lines for 4, a horizontal line for 5, etc. 

Given an input $x$, the logit for digit $d$ is $x^T Q^{(d)} x = \sum_i  \lambda_i (q_i^T x)^2$, so it's the square of an eigenvector's overlap with the input that contributes. This means that both the blue (pos) and red (neg) parts contribute in the same way, ie the sign of $q^Tx$ does not matter. For example, the eigenvectors for 1 contain both red and blue lines which can have high overlap for 1's with different slants. 

The computation using $Q$ is still a superposition over different features for each digit, for example over different locations in the image, different slants, and different digit shapes. The eigenvectors (and their separation in blue/red components) help disentangle this superposition to an extent. The eigenvectors are all orthogonal so each captures a different aspect (eg the 3rd positive eig for 1 adds a bottom horizontal line, the 3rd positive eig for 7 adds a horizontal cross) but the orthogonality constraint can also make their interpretation less clear. There is likely a better approach to deriving interpretable features from $Q$ which we haven't fully explored, for example looking at the set of local maxima/minima for $x^T Q x$ might find a set of clear digits shapes. 

An important point is that the 10 different $Q$'s fully capture the model's computations. We could rewrite the model fully in terms of their eigenvectors if desired. This shows the advantages of the bilinear structure. For other nonlinearities it's difficult to escape the "neuron" basis in which the nonlinearity is applied when trying to connect the output features to input ones. 

![image](/images/MNIST_digit_basis_1K.png)

### SAE features using the pseudoinverse trick

As an additional example, we trained a sparse autoencoder on the outputs of the bilinear layer (before the linear readout) [details in Appendix]. The SAE has an average of L0 of 5.2 but only 87 features are active (out of 2000 starting features) and a MSE/$L2^2$ of 0.28. We didn't try too hard to optimize the SAE training to minimize dead features or prevent dense features because we're mainly interested in illustrating how a set of dictionary features $D$ can be incorporated into the bilinear model. 

The plots below show the top four features by mean activation. We see some similar eigenvectors to the digit basis results although noiser. Similar digits can be combined in the positive direction of an SAE feature (2 and 6, 4 and 9) while the negative direction can contains constrasting digits (3 vs 4 in feature 1).

![image](/images/MNIST_SAE_dataset_relu.png)

The interaction matrices $Q$ are derived using the pseudoinverse trick described above using $U = D^+$. The fact that the logit directions mostly make sense in terms of the eigenvectors is a sign that the pseudoinverse is a reasonable way to incorporate the dictionary features. In fact, when restricting to the 87 active features, the **pseudoinverse activations improve the loss recovered from 87.1% for the ReLU activations to 98.1%**, using a random guessing baseline for the loss. The plot below shows that the pseudoinverse activations are well correlated with the nonzero ReLU activations and are typically small values for the zero ReLU activations.

![image](/images/MNIST_relu_vs_pinv_activations.png)

## Deriving features from the model weights
An exciting aspect of bilinear layers is that they suggest a few approaches for deriving features directly from the model weights. These approaches rely on the fact that **pairs of input features can act as a basis for the layer's outputs**. Given a set of input features $\vec{v_i}$ with activations ${a_i}$, the output can be expressed as a sum $\sum_{ij} a_i a_j \vec{r}\_{ij}$ where $\vec{r}\_{ij}$ is the output when the input is exactly $\vec{v_i} + \vec{v_j}$. So finding a set of features to describe the outputs $\vec{r}\_{ij}$ also finds features for all outputs of the bilinear layer. 

We've explored two basic approaches:

**Singular Value Decomposition (SVD)**: We can treat the set of outputs $\vec{r}\_{ij}$ as a matrix, which ends up being equivalent to a vectorized version of the bilinear tensor $B_{a(ij)} = (\vec{r}\_{ij})\_a$ where $(ij)$ indexes the pair of inputs. The singular value decomposition (SVD) allows us to obtain the best low rank approximations for this matrix.

The SVD gives $B_{a(ij)} = \sum_s  R_{as} \sigma_s Q_{s(ij)}$ where $\sigma_s$ are the singular values and $R$ and $Q$ are orthonormal matrices. For each singular value component, $Q_{s(ij)}$ can be considered an interaction matrix, $Q^{(s)}\_{ij}$ which we can analyze through its eigenvectors as before. The component's output is simply $\vec{r}\_s$.

**Sparse autoencoders (SAEs)**: We can treat the set of input pair vectors $\vec{r}\_{ij}$ as a dataset for training an SAE. If we can decompose each input pair vector into a set of features, then we can use those same features to decompose any output of the bilinear layer. Each feature would correspond to a weighted sum over the set of input pairs with nonzero activations. 

### SVD on the bilinear tensor

The plots below show the $Q$ eigenvectors for the top SVD components. We see similar eigenvectors to the digit basis ones, particularly for 2, 5, 3, 6, and 7, so the features generally seem reasonable. Some SVD components combine different digits, for example component 1 has 4/9 (positive direction) vs 2 (neg) and component 2 has 3 vs 6, so generally each component has more superposition than the interaction matrices in the digit basis. 

![image](/images/MNIST_svd_features_1K.png)

The SVD is particularly useful for identifying the most important components of a matrix, which allows us to find its optimal low-rank approximation. The number of SVD components is equal to the hidden dimension (in this case $d=1000$), as long as the number of input pairs is larger. The model predictions, however, only depend on a small number of components. Keeping only the top 20 components results in only a 1% drop in the validation accuracy while the top 50 components results in no drop at all. The less important SVD components can be shown to have low overlap with the logit directions. These results demonstrate that the SVD correctly identifies the most important components, without even knowing the logit directions explicitly.  

Although the SVD gives a small set of the important components, it's unclear if they correspond to "true" features. For MNIST, it's even possible that the features are dense, instead of sparse, and form an embedding space that allows for continuous deformations between digits. The top 5 components are reproducible over different initializations of the model, in the exact same order [see Appendix]. The higher components drop off in reproducibility, possibly due to the fact that the SVD is less stable between components with similar singular values. 

![image](/images/MNIST_topk_bottleneck_accuracy_1K.png)

### SAE over input pair vectors

The plots below show the results of training an SAE over the set of input pair vectors, $\vec{r}\_{ij}$. We selected the SAE features out of the top 10 that showed activations with greater specificity for individual digits. 

Even though the training set consisted of outputs when only two pixels are active, the SAE features make sense as curves over hundreds of pixels. This demonstrates the utility of treating the input pair vectors as a basis for the outputs. Each pair of pixels contributes to a small set of features but many pixel pairs can contribute to the same feature, creating curves and shapes. For this SAE, L0 = 30.4 with 3500 active features. 

It's perhaps surprising to find features that seem to represent individual digits (eg, 0 in feature 3, 7 in feature 5, and 6 in feature 8) instead of simply curve segments (eg, upper curves vs lines for feature 2). This might be because there are only 10 relevant logit directions, one for each digit, so it's possible to recover these directions in the bilinear layer outputs. 

![image](/images/MNIST_SAE_btensor_relu.png)

## Takeaways

This exploratory analysis of MNIST demonstrates that bilinear layers show promise in a few areas:
  - Bilinear layers may help interpret output features in terms of input features in a way that is compatible with the model's computations. They might even make training separate "transcoders" for connecting input and output features unnecessary.
  - For a given output feature/direction, there is an interaction matrix $Q$ that determines the output feature activation as $x^T Q x$. The eigenvectors of $Q$ help disentangle the superposition of feature computations in $Q$. For example, for a digit's logit directions, different eigenvectors represent different shapes and locations of the digit and some even represent optional features like a "cross" for the digit 7.
  - It may be possible to derive output features directly from the bilinear model weights. Since the pairs of inputs can be used as a basis for bilinear outputs, we can analyze this basis through singular value decomposition (SVD) or sparse autoencoders to derive features. For a single layer MNIST model, SVD finds a set of 50 output features that fully recover the accuracy. 

We also introduced a "pseudoinverse trick" for incorporating a dictionary of features $D$ into a model with minimal reconstruction loss using the pseudinverse $D^+$. If the set of features in $D$ are a complete basis for the hidden dim (likely for a large set of features), the projection operator $D D^+$ would equal the identity so we can insert $DD^+$ into the model without changing the model. In our case, we found that the pseudoinverse activations are well-correlated with the ReLU-based activations and remain small when the ReLU activations are zero. 

## Future Work
There are several interesting directions to try next:
  - Training and analyzing bilinear models for language modeling. Compared to MNIST, language modeling might result in sparser interactions. It would be interesting to see how interpretable the model-derived features are, perhaps in terms grammatical and semantic concepts.
  - Using bilinear layers for modeling boolean functions. Because of their pair-wise interactions between inputs, bilinear models can likely represent boolean ANDs more easily than other nonlinearities.
  - Exploring alternatives to eigendecomposition for analyzing the interaction matrices $Q$. Clustering or finding all local extrema would be interesting approach to try.

# Appendix 

## MNIST training
**Single Layer Model** [[Colab](https://colab.research.google.com/drive/12sE0jLTgY4_77ia7gRdCOo8-e52mJXRx?usp=sharing)]
[add colabs for other hidden dims]
- Validation accuracy: 97.8%. We found similar performance for ReLU models matched for the same parameter count.
- Hidden dim: 1000.
  - Higher hidden dimensions have smaller interference between features in superposition, leading to cleaner representations.
  - We found similar results for models with hidden dims of 300 [[Colab](https://colab.research.google.com/drive/1XbkcFQC8oXxRdXDqrdMfNGmBEahz0ZyJ?usp=sharing)] and 3000 [[Colab](https://colab.research.google.com/drive/1TGPwh8EKW8SzNLC06LuEnXIN2QW7h-EP?usp=sharing)]
- Weight decay: 1. We found that depending on the initialization distribution, the bilinear layer had noise orthogonal to the logit directions that complicated interpretation. Weight decay was effective in removing this orthogonal noise. 

![image](/images/MNIST_best_match_similarity_1K.png)

## SAE training
**SAE trained over input dataset** [[Colab](https://colab.research.google.com/drive/19H1a_qy_RkqWzwV8i4W2gnuV3HwB1fhR?usp=sharing)]
- Starting features: 2000
- $L0$: 5.2
- $MSE/L2^2$: 0.28
- Dead feature frac: 96%
- Active features: 87
- Architecture: We used a standard SAE model with a ReLU nonlinearity. We included both encoder and decoder biases and substracted the decoder bias from the initial input. We normalized the decoder features to have unit norm. 
- Feature similarity penalty: We initially found that certain pairs of features were in opposite directions, suggesting that cancellations between those features is possible. To limit this we included a penalty on feature similarity using an $L_p$ norm with $p = 10$ so that the norm was weighted much more towards the highest (absolute) similarities. 

**SAE trained over bilinear tensor** [[Colab](https://colab.research.google.com/drive/1UCrpT-zod4ylPMaaqmz2BYrvYZYA_ndP?usp=sharing)]
- Starting features: 5000
- $L0$: 30.4
- $MSE/L2^2$: 0.29
- Dead feature frac: 29%
- Active features: 3544
- Same architecture and feature similarity penalty as above.



