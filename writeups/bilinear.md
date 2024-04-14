# Bilinear Layers

This document offers an intuitive explanation the math behind bilinear layers and their decomposition. The contents are aimed at researchers that have solid foundations into transformer mechanistic interpretability but don't especially know how higher-order tensors work.

> **Prerequisites**
>
> A Mathematical Framework for Transformer Circuits [[link](https://transformer-circuits.pub/2021/framework/index.html)]
>
> A general sense for interpretability research and linear algebra

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

$$\vec{y} = ReLU(W \vec{x} + \vec{b})$$

In contrast, a bilinear layer takes on the following structure.

$$\vec{y} = (W \vec{x} + \vec{b}) \odot (V \vec{x} + \vec{c})$$

Here, $W$ and $V$ are weight matrices and $\vec{b}$ and $\vec{c}$ are biases of some layer. The $\odot$ denotes an element-wise product. This is not a bilinear map due to the biases. However, without biases we cannot represent potentially essential operations such as the identity operation (seriously, try this if you don't believe me).

### Including Biases

The importance of biases lie in their independence of the input. In extreme cases, such as a completely zero input, a linear operation is not able to produce anything else than this zero vector. However, this independence from the input is also what makes biases slightly annoying to deal with. Fortunately, we can use a trick: make the bias actually depend on a part of the input and ensure that this part is constant. Practically speaking, if a constant one is appended to the input vector and an extra row is added to the weight matrix, the elements in this row act as biases. In essence, a bias is just an ordinary weight multiplied by a constant one. Formulaically, for a 2x2 matrix, we get:

$$\begin{bmatrix} y_0 \\ y_1 \end{bmatrix} = \begin{bmatrix} W_{00} & W_{01} & b_0\\ W_{10} & W_{11} & b_1\\ \end{bmatrix} \cdot \begin{bmatrix} x_0 \\ x_1 \\ 1 \end{bmatrix}$$

If expanded, this results in the following, which is obviously identical to using normal biases.

$$y_0 = x_0 \cdot W_{00} + x_0 \cdot W_{10} + 1 \cdot b_0$$
$$y_1 = x_1 \cdot W_{01} + x_1 \cdot W_{11} + 1 \cdot b_1$$

Back to the bilinear setting, we apply this technique to both sides. Now, a network can learn identity functions using the following weights $\vec{y} = (I \cdot \vec{x} + \vec{0}) \odot (O \cdot \vec{x} + \vec{1})$ where $I$ and $O$ are the identity and zero matrix respectively. In summary, including biases scales both $W$ and $V$ by one column but does not change any properties. Therefore, writing $\vec{y} = (W \vec{x}) \odot (V \vec{x})$ is essentially equivalent to $\vec{y} = (W \vec{x} + \vec{b}) \odot (V \vec{x} + \vec{c})$.

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

$$ \vec{h_0} = E \vec{x} $$
$$ \vec{h_1} = (W \vec{h_0}) \odot (V \vec{h_0}) $$
$$ \vec{y} = U \vec{h_1} $$

As both $E$ and $U$ are linear, we can actually "fold" these into the bilinear operation.

$$ \vec{y} = U((W E \vec{x}) \odot (W E \vec{x})) = (U W E \vec{x}) \odot (U V E \vec{x})$$

We can simplify $U$, $W$/$V$, and $E$ into the matrices $W^*$ and $V^*$ respectively, yielding:

$$ \vec{y} = (W^* \vec{x}) \odot (V^* \vec{x}) $$

Akin to including the biases, our $W$ and $V$ matrices have changed in dimensionality but not in properties. This again means that all techniques, such as the tensor construction, can be applied to this folded form. In essence, this allows us to create a tensor that encapsulates the exact computation of 3 full layers at once. For instance, in transformers, this allows us to fold the embedding and unembedding into an arbitrary (bilinear) MLP. This means we have a single object for describing this indirect path.

### Capability

Bilinear layers with biases can represent a large category of interactions between features, every quadratic function with 2 inputs to be precise. This allows them to exactly compute any kind of binary gate (yes, even XORs) in a single layer. They can even approximate arbitrary ternary operations to a high degree. The representational capacity of these layers is honestly breathtaking.

## Interactions

We have a good understanding of how to manipulate and fold these bilinear tensors but not yet of what they inherently mean.

### Interaction Matrix

To get a sense on how input features interact, let's write out a toy example. We use an input vector with two elements $\vec{x} = [x_0, x_1]$. We will study output $a$.

$$y_a = (W_a \cdot \vec{x}) \odot (V_a \cdot \vec{x})$$

$$y_a = (W_{a0}\cdot x_0 + W_{a1}\cdot x_1) \odot (V_{a0} \cdot x_0 + V_{a1} \cdot x_1)$$

$$y_a = W_{a0}V_{a0} \cdot x_0^2 + W_{a0}V_{a1} \cdot x_0x_1 + W_{a1}V_{a0} \cdot x_1 x_0 + W_{a1} V_{a1} \cdot x_1^2$$

$$y_a = \sum_{i,j} \begin{bmatrix} W_{a0}V_{a0} & W_{a0}V_{a1} \\ W_{a1}V_{a0} & W_{a1}V_{a1} \end{bmatrix}_{ij} \cdot \begin{bmatrix} x_0^2 & x_0x_1 \\ x_1x_0 & x_1^2 \end{bmatrix}_{ij}$$

In the general case, we can rewrite this as the following, where $\otimes$ is an outer product. Here, we call $W_{a} \otimes V_{a}$ our interaction matrix $B_a$ and $\vec{x} \otimes \vec{x}$ the feature matrix $F$. The interaction matrix essentially determines the weight by which each feature is scaled.

$$y_a = \sum_{i,j} \left[ (W_{a} \otimes V_{a})_{ij} \cdot (\vec{x} \otimes \vec{x})_{ij} \right] = \sum_{i,j} B_{aij} F_{ij}$$

Note that the naming of $B$ is not a coincidence. This is actually the exact tensor that we use to describe our the full bilinear layer. Intuitively, $B$ is a stack of interaction matrices.

### Symmetrification

Given that $F$ is symmetric, we can perform the following.

$$ B_a F = B_a^T F = \dfrac{1}{2}(B_a + B_a^T)F = B'_aF$$

In essence, $B'_a$ is a symmetric version of $B^a$ that performs the same computation. Symmetric matrices are generally easier to work with, so we always perform this simplification trick.

 Therefore, performing this trick in code is exceedingly simple ($.mT$ does a transpose of the last two dimensions).

```python
B_symm = 0.5 * (B + B.mT)
```

### Bias Decomposition

> In higher dimensions, this information is less useful. Only read if you're very interested in the exact workings of biases in bilinear layers and it's expressive properties.

Given this interaction matrix, let's decompose what bias-infused networks can learn a bit more formally. This can be done by distributing over the element-wise product. We can see 3 forms of interactions which determine the output of the layer.

$$(W \vec{x} + \vec{b}) \odot (V \vec{x} + \vec{c}) = (W \vec{x} \odot V \vec{x}) + (W \vec{x} \odot \vec{c} + \vec{b} \odot V \vec{x}) + (\vec{b} \odot \vec{c})$$

**Quadratic.** First, we have the quadratic interactions ($W\vec{x} \odot V\vec{x}$). These interactions happen between any of the input pairs as seen above.

**Linear.** The next case are the linear interactions ($W\vec{x} \odot \vec{c} + \vec{b} \odot V\vec{x}$). Since $\vec{b}$ and $\vec{c}$ are simply a constant vectors, we can write $L \vec{x} = (W \vec{c} + V \vec{b}) \vec{x}$. Just as $B$ is a stack of interaction quadratic matrices, $L$ is a stack of linear interaction vectors.

**Constant.** The simple case of bias-bias ($\vec{b} \odot \vec{c}$), this is simply the product of both sides, it is completely independent of the input and therefore called the *constant* component. For clarity, this is a stack of single values for each output.

It may seem strange that we spent effort at first to incorporate biases into the network just to decompose them again. The main reason is that these bias-related objects are actually related to the interactions and are therefore more interpretable.

<!-- Given all the above knowledge, we write down the closed formula for determining an output given a pair of input features. We denote this pair of features with $x$ and $y$, we denote their indices with $i$ and $j$ respectively. Furthermore, we denote the output index as $o$ and the bias index as $-1$.

$f^o_{ij} = (W_{i,o} \cdot x + W_{j,o} \cdot y + W_{-1,o}) \odot (V_{i,o} \cdot x + V_{j,o} \cdot y + V_{-1,o})$

$f^o_{xy} = x^2 \cdot W_{xo} \cdot V_{xo} + xy \cdot W_{xo} \cdot V_{yo} + yx \cdot W_{yo} \cdot V_{xo} + y^2 \cdot W_{yo} \cdot V_{yo}$

$f^o_{xy} = aa^o \cdot x^2 + bb^o \cdot y^2 + 2ab^o \cdot xy + a^o \cdot x + b^o \cdot y + \gamma^o$

Where $aa^o = W_{xo} \cdot V_{xo}$, ...

Generally, we will ignore the superscript that indicates the output feature for simplicity. Second, $aa$ refers to a fully distinct variable from $a$. This is slightly strange, but preferable over using arbitrary letters for each weight. -->

## Tensor Networks

``todo``

## Transformer Circuits

Given all this background knowledge, it's time to turn towards transformers and how these can be studied. In short, the methodology is very similar to current mechanistic interpretability techniques. The only exception is that bilinear layers also allow us to study the MLP but it requires some smart approaches. In short, constructing the $B$ tensor in larger models is computationally infeasible, therefore we usually analyze sensible subsets.

### Objects

Objects represent a subset of model behavior we wish to study.

- $B^+ = PB + I$: The MLP + residual stream tensor.
- $B^+OV$: The right-attention interaction tensor.

### Projections

- $T(.) = U(E^T . E)$: The full token space projection.
- $D(.) =  U\Delta(E^T . E)$: The diagonal or direct input output token map.
- $A(.)$ a filter projection defined by the top-k entries of a QK circuit. In code: ``[:, T(QK).mean(0).topk, T(QK).mean(1).topk]``.
- $\mathcal{P}_{tok}$ indexes at a certain axis according to a token.

### Analysis

- SVD
- Max activation
- ...

### Examples

- $T_{game}(B^+)$ is the interaction tensor for the token ``game``.
- $D(B^+)$ defines the bi-grams learned in an MLP according to their strength.
- $D(B^+OV)$ defines a map for V-composition with an MLP.


It's generally not very difficult to find 