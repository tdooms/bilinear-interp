# Tensor Math Primer

**By Thomas Dooms and Michael Pearce**

This document offers an intuitive overview into the world of tensors and their utility in interpretability. The contents are aimed at researchers that have solid foundations into transformer mechanistic interpretability but don't especially know how higher-order tensors work.

> **Prerequisites**
> 
> A Mathematical Framework for Transformer Circuits [[link](https://transformer-circuits.pub/2021/framework/index.html)]

## Bilinear Maps
Like most things in life, bilinear maps are a very natural concept once one is familiar with it, but it can be a bit tricky to wrap your head around at first. What follows is a very intuitive explanation of bilinear maps, followed by a more mathematical definition.

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

## Bilinear Layers

A normal layer in a neural network is structured as follows.

$$h^{out} = ReLU(W h + b)$$

In contrast, a bilinear layer takes on the following structure.

$$h^{out} = (W h + b) \odot (V h + c)$$

Here, $W$ and $V$ are weight matrices and $b$ and $c$ are biases of some layer. The $\odot$ denotes an element-wise product. This is not a bilinear map due to the biases. However, without biases we cannot represent potentially essential operations such as the identity operation (seriously, try this if you don't believe me).

### Including Biases

The importance of biases lie in their independence of the input. In extreme cases, such as a completely zero input, a linear operation is not able to produce anything else than this zero vector. However, this independence from the input is also what makes biases slightly annoying to deal with. Fortunately, we can use a trick: make the bias actually depend on a part of the input and ensure that this part is constant. Practically speaking, if a constant one is appended to the input vector and an extra row is added to the weight matrix, the elements in this row act as biases. In essence, a bias is just an ordinary weight multiplied by a constant one. Formulaically, for a 2x2 matrix, we get:

$$\begin{bmatrix} h^{out}_0 \\ h^{out}_1 \end{bmatrix} = \begin{bmatrix} W_{00} & W_{01} & b_0\\ W_{10} & W_{11} & b_1\\ \end{bmatrix} \cdot \begin{bmatrix} h_0 \\ h_1 \\ 1 \end{bmatrix}$$

If expanded, this results in the following, which is obviously identical to using normal biases.

$$h^{out}_0 = h_0 \cdot W_{00} + h_0 \cdot W_{10} + 1 \cdot b_0$$
$$h^{out}_1 = h_1 \cdot W_{01} + h_1 \cdot W_{11} + 1 \cdot b_1$$

Back to the bilinear setting, we apply this technique to both sides. Now, a network can learn identity functions using the following weights $h^{out} = (I \cdot h + \vec{0}) \odot (O \cdot h + \vec{1})$ where $I$ and $O$ are the identity and zero matrix respectively. In summary, including biases scales both $W$ and $V$ by one column but does not change any properties. Therefore, writing $h^{out} = (W h) \odot (V h)$ is essentially equivalent to $h^{out} = (W h + b) \odot (V h + c)$.

> By also concatenating an additional row of the form $[0, 0, ..., 1]$, the constant can be effortlessly propagated through the network to calculate all consequent biases.

### Linearizability

> The math of the following sections can be a bit tricky to understand on a first read. The aim should be to understand the ideas, not the exact computation.

Just as linear maps can be represented as a matrix, bilinear maps can be represented as a tensor. A matrix is simply a rank 2 tensor (sometimes called a 1-1 tensor, meaning 1 input dim and 1 output dim).
To describe bilinear maps, we need a rank 3 tensor (2 input dim and 1 output dim). While this tensor can become quite large, we can still employ very simple techniques to analyze this. It's difficult to stress enough how important this is. There is have a single mathematical object that represents the full computation of a layer. The main issue with a ReLU is that it requires to know the input to determine the output, which makes it extremely hard to define guarantees, this is completely obviated with this approach. Furthermore, the normal weight matrix is impossible to study because the ReLU may just change things around. This single object makes it possible to definitively make guarantees about behaviour and extract algorithms without ever evaluating the (single-layer) network.

Constructing the rank-3 tensor is actually computationally very straightforward but can be difficult to understand intuitively. Let's start from a matrix multiplication, as an example we'll use the embedding and unembedding matrices We have one matrix that maps the input tokens into a hidden/latent space and the other maps it back up to output tokens. The multiplication can be written as:

$$(\text{out}, \text{hid}) \times (\text{hid}, \text{in}) = (\text{out}, \text{in})$$

> If this seems weird, take a look at "einsums".

This should make sense, a matrix multiplication takes an inner product for each input and output, thereby it essentially removes 2 dimensions from the input matrices. Using the same notation, we can also "multiply" the $W$ and $V$ matrices. However, as an output of this operation we want a structure that accepts two inputs and provides one output.

$$ (\text{out}, \text{in1}) \times (\text{out}, \text{in2}) = (\text{out}, \text{in1}, \text{in2})$$

This looks a bit strange but in essence, it's doing an element-wise multiplication over the $\text{out}$ dimension. In essence, that's also what our original bilinear layer did, take both inputs and perform an element-wise multiplication over the output dimension. We generally call this tensor $B$ (as this describes the $B$ilinear operation).

### Foldability

Its possible to integrate normal linear maps (such as an embedding/projection or unembedding/classifier) into the above-mentioned tensor. Say we have the following network.

$$ h_0 = E x $$
$$ h_1 = (W h_0) \odot (V h_0) $$
$$ y = U h_1 $$

As both $E$ and $U$ are linear, we can actually "fold" these into the bilinear operation.

$$ y = U((W E h_0) \odot (W E x)) = (U W E x) \odot (U v E x)$$

We can simplify $U$, $W$/$V$, and $E$ into the matrices $W^*$ and $V^*$ respectively, yielding:

$$ y = (W^* x) \odot (V^* x) $$

Akin to including the biases, our $W$ and $V$ matrices have changed in dimensionality but not in properties. This again means that all techniques, such as the tensor construction, can be applied to this folded form. In essence, this allows us to create a tensor that encapsulates the exact computation of 3 full layers at once. For instance, in transformers, this allows us to fold the embedding and unembedding into an arbitrary (bilinear) MLP. This means we have a single object for describing this indirect path.

### Capability

Bilinear layers with biases can represent a large category of interactions between features, every quadratic function with 2 inputs to be precise. This allows them to exactly compute any kind of binary gate (yes, even XORs) in a single layer. They can even approximate arbitrary ternary operations to a high degree. The representational capacity of these layers is honestly breathtaking.

## Feature Interactions

We have a good understanding of how to manipulate and fold these bilinear tensors but not yet of what they inherently mean.

> In higher dimensions, this information is less useful. Only read if you're very interested in the exact workings of this bilinear layer.

### Decomposition

Let's decompose what the network can learn a bit more formally. This can be done by distributing over the element-wise product. We can see 3 forms of interactions which determine the output of the layer.
$$(Wh + b) \odot (Vh + c) = (Wh \odot Vh) + (Wh \odot c + b \odot Vh) + (b \odot c)$$
**Constant.** Starting at the simple case of bias-bias ($b \odot c$), this is simply the product of both sides, it is completely independent of the input and therefore called the *constant* component.

**Linear.** The next case are the linear interactions ($Wh \odot c + b \odot Vh$). In the general case, this can happen for either side of the input but in our case this boils down to the same (as we will exploit a bit later).

**Quadratic.** Lastly, we have the quadratic interactions ($Wh \odot Vh$). These interactions happen between any of the input pairs, this becomes clear if we write out the formula. Given input $[x, y]$, this term becomes.

$$ = (W \cdot \begin{bmatrix} x \\ y \end{bmatrix}) \odot (V \cdot \begin{bmatrix} x \\ y \end{bmatrix}) $$
$$ = \begin{bmatrix} x \cdot W_{00} + y \cdot W_{10} \\ x \cdot W_{01} + y \cdot W_{11} \end{bmatrix} \odot \begin{bmatrix} x \cdot V_{00} + y \cdot V_{10} \\ x \cdot V_{01} + y \cdot V_{11} \end{bmatrix}$$

$$=\begin{bmatrix} x^2 \cdot W_{00} \cdot V_{00} + xy \cdot W_{00} \cdot V_{10} + yx \cdot W_{10} \cdot V_{00} + y^2 \cdot W_{10} \cdot V_{10} \\ x^2 \cdot W_{01} \cdot V_{01} + xy \cdot W_{01} \cdot V_{11} + yx \cdot W_{11} \cdot V_{01} + y^2 \cdot W_{11} \cdot V_{11} \end{bmatrix}$$

### Input Symmetry

Hopefully, you now have a good feel on what our bilinear layer can represent. There is one property that we haven't yet exploited to simplify this representation. Specifically, given that both out inputs are always the same, we can simplify all feature pair interactions. In the above example, since we use real numbers $xy = yx$, therefore, we can rewrite the following in a few ways.

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

Given all the above knowledge, we write down the closed formula for determining an output given a pair of input features. We denote this pair of features with $x$ and $y$, we denote their indices with $i$ and $j$ respectively. Furthermore, we denote the output index as $o$ and the bias index as $-1$.

$f^o_{ij} = (W_{i,o} \cdot x + W_{j,o} \cdot y + W_{-1,o}) \odot (V_{i,o} \cdot x + V_{j,o} \cdot y + V_{-1,o})$

$f^o_{xy} = x^2 \cdot W_{xo} \cdot V_{xo} + xy \cdot W_{xo} \cdot V_{yo} + yx \cdot W_{yo} \cdot V_{xo} + y^2 \cdot W_{yo} \cdot V_{yo}$

$f^o_{xy} = aa^o \cdot x^2 + bb^o \cdot y^2 + 2ab^o \cdot xy + a^o \cdot x + b^o \cdot y + \gamma^o$

Where $aa^o = W_{xo} \cdot V_{xo}$, ...

Generally, we will ignore the superscript that indicates the output feature for simplicity. Second, $aa$ refers to a fully distinct variable from $a$. This is slightly strange, but preferable over using arbitrary letters for each weight.


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

> The mentioned paper ([4]) has my favourite conclusion of any paper I have ever read: "*We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence*". In a sense, in this project, we aim to provide this divine explanation.

## References
- <a id="1">[1]</a> Decomposing Language Models With Dictionary Learning [link](https://transformer-circuits.pub/2023/monosemantic-features/index.html). Bricken et al. 10/2023.
- <a id="2">[2]</a> Toy Models of Superposition [link](https://transformer-circuits.pub/2022/toy_model/index.html). Elhange et al. 09/2022.
- <a id="3">[3]</a> A technical note on bilinear layers for interpretability [link](https://arxiv.org/abs/2305.03452). Lee Sharkey 05/2023
- <a id="4">[4]</a> GLU Variants Improve Transformer [link](https://arxiv.org/abs/2002.05202). Noam Shazeer 02/2022
- <a id="5">[5]</a> Polysemanticity and Capacity in Neural Networks [link](https://arxiv.org/abs/2210.01892). Scherlis et al. 07/2023
- <a id="6">[6]</a> Toward A Mathematical Framework for Computation in Superposition [link](https://www.lesswrong.com/posts/2roZtSr5TGmLjXMnT). 01/2024
- <a id="7">[7]</a> Multilayer Feedforward Networks With Non-Polynomial Activation Functions Can Approximate Any Function [link](https://archive.nyu.edu/bitstream/2451/14384/1/IS-91-26.pdf). Leshno et al. 09/1991
