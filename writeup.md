# Bilinear Models of Decomposition

**By Thomas Dooms and Michael Pearce**

This document is an intermediate write-up of our current theoretic efforts in studying bilinear layers. It serves as an introduction, a motivation and as an overview of our current ideas. We also provide an accompanying notebook showcasing more empirical findings through plots.

---

## Introduction
Activation functions are a vital component in deep networks to learn complex functions. The sole purpose of an activation function is to introduce non-linearity into the network to prevent matrix collapse. It is often argued that the simplest non-linearity is piecewise linearity, which is exactly what the most popular activation function, the ReLU, does.

However, in terms of interpretability, ReLUs are very difficult to study. Intuitively, the issue is that it is only possible to know the output of a ReLU by passing in the input, not solely from the weights. While it is possible to discern structure and circuits in these weights, this is often done by means of sampling the input or using gradient based techniques to generate visualizations which may not reflect the full complexity of the models. This makes it impossible to make strong guarantees on which outputs models will be able to produce. This has been observed in the form of adversarial examples, in which a slight perturbation is applied to the input to confuse the model into making bogus predictions.

This undesirable property of the ReLU has led to MLPs and similar structures being famously hard to interpret. As a consequence, several papers ([5], [6]) have used quadratic activation functions to perform theoretic analyses. Unfortunately, simple quadratic activation functions result in terrible accuracy due to as they do not lead to universal function approximators ([7]). However, as a substitute, it is possible to use bilinear activation functions. These functions posses appealing characteristic like the quadratic activations while being comparable to (and even surpassing) ReLUs in accuracy in large models as established in ([4]). Therefore, in this document, we make the design decision to replace ReLUs with the more interpretable bilinear layer. We provide an introduction to these layers and provide an overview of our current efforts in interpreting simple models using them.

> The mentioned paper ([4]) has my favourite conclusion of any paper I have ever read: "*We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence*". In a sense, in this project, we aim to provide this divine explanation.

### Bilinear Maps
Like most things in life, bilinear maps are a very natural idea once one is familiar with it, but it can be a bit tricky to wrap your head around at first. If you're already familiar with the concept, you're free to skip this part, the next section covers how we use them in neural networks. What follows is a very intuitive explanation of bilinear maps, followed by a more mathematical definition.

Let's start with a linear map, while the term may be new, everyone should know this concept. In essence, a matrix is the most general form of a linear map. One can think of a matrix as a function that takes a vector as input and returns a new vector as output. It does so while conserving linear properties that makes them so useful to work with.

A bilinear map is not so different, it takes in two vectors and returns a vector according to very similar linear properties. These properties boil down to: if one input vector is kept constant, a bilinear map operates exactly the same as a linear map. The following are three intuitive explanations.

- A bilinear map is a function that takes in a vector and returns a matrix, this matrix can then be used to compute the actual output vector.

- Bilinear maps perform two linear operations in sequence on different inputs.

- The mathematical properties of a bilinear map are comparable to ordinary (scalar) multiplication. Multiplication takes in two numbers and returns another and each of the inputs is linear if we freeze the other.

Importantly, while bilinear maps are linear to their inputs if the other is frozen, it is not linear in general. For instance, scalar multiplication is quadratic ($s(x, x) = x^2$). That said, let's get a bit into the weeds of the maths behind bilinear maps. Let's refresh the properties of a linear map $m$:
$$m(\vec{u} + \vec{v}) = m(\vec{u}) + m(\vec{v})$$
$$m(\lambda \vec{u}) = \lambda m(\vec{u})$$
Where $\vec{u}$ and $\vec{v}$ are arbitrary vectors and $\lambda$ a scalar. These properties should feel quite natural, intuitively, it is possible to "pull out" any term out of the map. These maps mostly operate like normal numbers that everyone is used to work with. In contrast a bilinear map has the slightly more complicated constraints.
$$b(\lambda \vec{u}, \vec{v}) = \lambda b(\vec{u}, \vec{v}) \text{ and } b(\vec{u}, \lambda \vec{v}) = \lambda b(\vec{u}, \vec{v})$$
$$b(\vec{u_1} + \vec{u_2}, \vec{v}) = b(\vec{u_1}, \vec{v}) + b(\vec{u_2}, \vec{v}) \text{ and } b(\vec{u}, \vec{v_1} + \vec{v_2}) = b(\vec{u}, \vec{v_1}) + b(\vec{u}, \vec{v_2})$$
This may seem very arbitrary at first but if you squint a bit, you can see that if you ignore the $\vec{v}$ in the left equations, this is identical to the linear case.

### Bilinear Layers
A normal layer in a neural network is structured as follows.
$$h_{i+1} = ReLU(W_i h_i + b_i)$$
We will study the following structure.
$$h_{i+1} = (W_i h_i + b_i) \odot (V_i h_i + c_i)$$
Here, $W_i$ and $V_i$ are matrices of the same dimensionality and $b_i$ and $c_i$ are biases. The $\odot$ denotes an element-wise product. The astute reader will probably notice that this is not at all a bilinear map; the biases make this non linear. However, without biases we cannot represent potentially essential operations such as the identity operation (seriously, try this if you don't believe me). In a further section we will solve this to have the best of both worlds. As lightly covered in the intro, there are a handful of reasons for studying this structure.
#### 1) Linearizability
Just as linear maps can be represented as a matrix, bilinear maps can be represented as a tensor. A matrix is simply a rank 2 tensor (sometimes called a 1-1 tensor, meaning 1 input dim and 1 output dim). 
To describe bilinear maps, we need a rank 3 tensor (2 input dim and 1 output dim). While this tensor can become quite large, we can still employ very simple techniques to analyze this.

> I can't stress enough how important this is! We have a single mathematical object that represents the full computation of a layer. The main issue with a ReLU is that it requires to know the input to determine the output, which makes it extremely hard to define guarantees, this is completely obviated with this approach. Furthermore, the normal weight matrix is impossible to study because the ReLU may just change things around. This single object makes it possible to definitively make guarantees about behaviour and extract algorithms without ever evaluating the (single-layer) network.

#### 2) Foldability
Its possible to integrate normal linear maps (such as an embedding/projection or unembedding/classifier) into the abovementioned tensor. In essence, this allows us to create a tensor that encapsulates the exact computation of 3 full layers at once. In transformers, this makes it incredibly easy to compute exact Q, K, and V compositions.

#### 3) Capability
Bilinear layers with biases can represent a large category of interactions between features, every quadratic function with 2 inputs to be precise. This allows them to exactly compute any kind of binary gate (yes, even XORs) in a single layer. They can even approximate arbitrary ternary operations to a high degree. The representational capacity of this operation is honestly breathtaking.

## Motivation & Goals
Hopefully, by now the reader is convinced that this research direction merits exploration. The main goal of this project is to fully understand small networks that only use bilinear layers. We do so by studying different settings in an attempt to find the best interpretability methods. Intuitively, we aim to solve the most fundamental challenge in mechanistic interpretability; fully understand a high-dimensional linear object. To achieve this, we are currently pursuing two threads of research.

### Low Dimensional
Studying low-dimension toy models allows us to design interpretability techniques in a controlled environment (where the ground truth is known) to evaluate which approaches work best. This provides us with the building blocks on how these novel layers operate. The following is a simplified TODO list.

- **Find the best layer composition to study.**
	- [x] How can weights and biases be integrated into one structure?
	- [x] How can we visualise the full low-dimensional tensor?
- **Fully explain arbitrary computation**
	- [x] What operations can the model solve?
	- [x] Can we extract the exact operations from the weights alone?
- **Recreate and understand superposition.**
	- [x] Which geometric structures do they form?
	- [x] When does the model exhibit this behaviour?
	- [x] Can we extract superposition from the weights?
- **Can the above tasks the performed in SVD?**
	- [ ] Can computation still be extracted?
	- [ ] Can superposition still be extracted?
	- [ ] Does this approach scale to compositions?

### High Dimensional
In high dimensional cases, the main goal is to decompose to more basic interpretable blocks. These "interpretable blocks" are often referred to as features.

The difficulty of this study mostly lies not in the extraction of features, as is the case with ReLUs but in how to decompose them. Intuitively, due to the linearity, the high dimensional space can be manipulated to our wishes, but we don't yet know what our wishes for interpretability are. The following is a tentative list on how to get there (but were not fully sure yet).

- **SVD**
	- [ ] Study the Q matrix
- **SAE**
  - [ ] Pseudo-inverse tick
  - [ ] SAE on the weights
  - [ ] SAE on the activations

## Setup

### Model Tensors
> As we are analysing tensors and performing operations on them, we will make use of tensor products. For each of these instances, the (in my mind easier to understand) Einsum notation will be provided as well. However, as always, the presented concepts are much more important than the exact math.

In the spirit of uniformity, we use a shared model across all experiments. This model consists of three layers: a linear embedding, a bilinear layer, and a linear unembedding. This model has 4 matrices, which we call $E$, $V$, $W$, and $U$. As described above, we can combine $W$ and $V$ into a single rank 3 tensor,  we do so using the the following product.

```python
B = einsum(V, W, "output input1, output input2 -> output input1 input2")
```

This may seem a bit unfamiliar at first, however, we can think about the last two dimensions as an "interaction" matrix for each possible output (this will be explained later). Given this tensor, we can compute $Wh \odot Vh$ as $h^TBh$.

### Tensor Folding
More important than having a single tensor to describe the bilinear map is that we can fold other linear maps into this tensor. First let's define some notation:
- Inputs (aka the features) are denoted as ``in``.
- Embeddings (after the embedding projection) are denoted as ``emb``.
- Unembeddings (after the bilinear map) are denoted as ``unemb``.
- Outputs (after the unembedding) are denoted as ``out``.

We can describe a bilinear map followed by an unembedding as $h^T(B \otimes U) h$, which converts to the following einsum.

```python
UB = einsum(B, U, "unemb emb1 emb2, out unemb -> out emb1 emb2")
```

In the same way, the embedding matrix can be folded too ($h^T (E \otimes B \otimes E) h$).

```python
BE = einsum(B, E, E, "unemb emb1 emb2, emb1 in1, emb2 in2 -> out in1 in2")
```

Lastly, these multiplications can be chained by substituting $B$ by $BE$ or $UB$ in the first and second formula respectively. We call the resulting tensor $UBE$.

Importantly, as each of these described tensors is rank 3, **every** analysis technique we describe can be performed on any of them.  

### Including Biases
The importance of biases lie in their independence of the input. In extreme cases, such as a completely zero input, a matrix is not able to produce anything else than this zero vector. However, this independence from the input is also what makes biases slightly annoying to deal with.

Fortunately, we can use a trick: make the bias actually depend on a part of the input and ensure that this part is constant. Practically speaking, if a constant one is appended to the input vector and an extra row is added to the weight matrix, the elements in this row act as biases. In essence, a bias is just an ordinary weight multiplied by a constant one. Formulaically, we get:

$$\begin{bmatrix} h_0 \\ h_1 \\ 1 \end{bmatrix}^T
\cdot
\begin{bmatrix} W_{00} & W_{01}\\ W_{10} & W_{11} \\ b_0 & b_1 \end{bmatrix}$$

If expanded, this results in the following, which is obviously identical to using normal biases.
$$h_0 \cdot W_{00} + h_0 \cdot W_{10} + 1 \cdot b_0 \text{ and } h_1 \cdot W_{01} + h_1 \cdot W_{11} + 1 \cdot b_1$$
Back to the bilinear setting, we apply this technique to both sides. Now, a network can learn identity functions using the following weights $(I \cdot h + \vec{0}) \odot (O \cdot h + \vec{1})$ where $I$ and $O$ are the identity and zero matrix respectively.

> This procedure can actually be done in sequence; by also adding an additional row with a constant on at the end, this constant can be effortlessly propagated through the network to calculate all biases. While the alternative solution of always re-concatenating this constant may seem appealing, it will cause trouble when folding matrices.

### Feature Interactions
Let's decompose what the network can learn a bit more formally. This can be done by distributing over the element-wise product. We can see 3 forms of interactions which determine the output of the layer.
$$(Wh + b) \odot (Vh + c) = Wh \odot Vh + Wh \odot c + b \odot Vh + b \odot c$$
**Constant.** Starting at the simple case of bias-bias ($b \odot c$), this is simply the product of both sides, it is completely independent of the input and therefore called the *constant* component.

**Linear.** The next case are the linear interactions ($Wh \odot c + b \odot Vh$). In the general case, this can happen for either side of the input but in our case this boils down to the same (as we will exploit a bit later).

**Quadratic.** Lastly, we have the quadratic interactions ($Wh \odot Vh$). These interactions happen between any of the input pairs, this becomes clear if we write out the formula. Given input $[x, y]$, this term becomes.

$$(W \cdot \begin{bmatrix} x \\ y \end{bmatrix}) \odot (V \cdot \begin{bmatrix} x \\ y \end{bmatrix}) 
= \begin{bmatrix} x \cdot W_{00} + y \cdot W_{10} \\ x \cdot W_{01} + y \cdot W_{11} \end{bmatrix} \odot \begin{bmatrix} x \cdot V_{00} + y \cdot V_{10} \\ x \cdot V_{01} + y \cdot V_{11} \end{bmatrix}$$

$$\begin{bmatrix} x^2 \cdot W_{00} \cdot V_{00} + xy \cdot W_{00} \cdot V_{10} + yx \cdot W_{10} \cdot V_{00} + y^2 \cdot W_{10} \cdot V_{10} \\ x^2 \cdot W_{01} \cdot V_{01} + xy \cdot W_{01} \cdot V_{11} + yx \cdot W_{11} \cdot V_{01} + y^2 \cdot W_{11} \cdot V_{11} \end{bmatrix}$$

### Input Symmetry
Hopefully, you now have a good feel on what our bilinear layer can represent. There is one property that we haven't yet exploited to simplify this representation. Specifically, given that both out inputs are always the same, we can simplify all feature pair interactions. In the above example (given that $xy = yx$), we can rewrite the following in a few ways.

$$\begin{align*}
&= xy \cdot W_{01} \cdot V_{11} + yx \cdot W_{11} \cdot V_{01} \\
&= xy \cdot (W_{01} \cdot V_{11} + W_{11} \cdot V_{01}) + yx \cdot 0 \\
&=  0.5 \cdot xy \cdot (W_{01} \cdot V_{11} + W_{11} \cdot V_{01}) + 0.5 \cdot yx \cdot(W_{01} \cdot V_{11} + W_{11} \cdot V_{01}) \\
\end{align*}$$

All computations are exactly equal, however, the interaction matrix first is not guaranteed to have any structure. The second is guaranteed to be a upper or lower triangular matrix. The third is guaranteed to be symmetric (equal to its transpose). In this project, we picked the symmetric matrix as this generally results in nicer plots and properties. This calculation is quite simple (*.mT* transposes the last two dimensions):

```python
B_symm = 0.5 * B + 0.5 * B.mT
```

### Interaction Formula
``TODO: this notation really sucks``

Given all the above knowledge, we write down the closed formula for determining an output given a pair of input features. Unfortunately, we are dealing with a considerable amount of variables, to slightly counteract this, we will very strongly abuse some notations. Specifically, if $x$ or $y$ are in subscript, they denote the index of where variables $x$ and $y$ occur.

$f^o_{xy} = (W_{x} \cdot x + W_{y} \cdot y + W_{-1}) \odot (V_{x} \cdot x + V_{y} \cdot y + V_{-1})$

$f^o_{xy} = x^2 \cdot W_{xo} \cdot V_{xo} + xy \cdot W_{xo} \cdot V_{yo} + yx \cdot W_{yo} \cdot V_{xo} + y^2 \cdot W_{yo} \cdot V_{yo}$

$f^o_{xy} = aa^o \cdot x^2 + bb^o \cdot y^2 + 2ab^o \cdot xy + a^o \cdot x + b^o \cdot y + \gamma^o$

Where $aa^o = W_{xo} \cdot V_{xo}$, ...

Generally, we will ignore the superscript that indicates the output feature for simplicity. Second, $aa$ refers to a fully distinct variable from $a$. This is slightly strange, but preferable over using arbitrary letters for each weight.

### Principal Components
Until now, most of the analysis has been very low-dimensional; focussing on feature interactions. In higher dimensions, it becomes intractable to study each interaction. For instance, on MNIST, which has 784 features, this already results in ~300.000 interactions.

``TODO``

## Results
Currently, our results mostly consist of three distinct categories.

### Compression 
...

### Computation
...

### Decomposition
...

## Attribution
Michael focussed on the high dimensional case of MNIST while Thomas focussed on low-dimensional toy models. In terms of teamwork, Michael took the role of supervisor, suggesting ideas and new experiments while Thomas' contributions were mostly code-based.

## References
- <a id="1">[1]</a> Decomposing Language Models With Dictionary Learning [link](https://transformer-circuits.pub/2023/monosemantic-features/index.html). Bricken et al. 10/2023.
- <a id="2">[2]</a> Toy Models of Superposition [link](https://transformer-circuits.pub/2022/toy_model/index.html). Elhange et al. 09/2022.
- <a id="3">[3]</a> A technical note on bilinear layers for interpretability [link](https://arxiv.org/abs/2305.03452). Lee Sharkey 05/2023
- <a id="4">[4]</a> GLU Variants Improve Transformer [link](https://arxiv.org/abs/2002.05202). Noam Shazeer 02/2022
- <a id="5">[5]</a> Polysemanticity and Capacity in Neural Networks [link](https://arxiv.org/abs/2210.01892). Scherlis et al. 07/2023
- <a id="6">[6]</a> Toward A Mathematical Framework for Computation in Superposition [link](https://www.lesswrong.com/posts/2roZtSr5TGmLjXMnT). 01/2024
- <a id="7">[7]</a> Multilayer Feedforward Networks With Non-Polynomial Activation Functions Can Approximate Any Function [link](https://archive.nyu.edu/bitstream/2451/14384/1/IS-91-26.pdf). Leshno et al. 09/1991
