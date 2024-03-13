# Inspecting superposition by incorporating biases into the weights.
We wish to study the superposition phenomenon in bilinear layers.

To achieve this, we largely copy the setup from Anthropic. Namely, we use a (sparse) input feature vector, project this into a lower dimensional space and then project back up to reconstruct the original values. We are mostly interested in their study of ReLUs and comapare this to bilinear activations. Their approach is the following.
$$ h = Wx $$
$$\hat{y} = ReLU(h)W^T + b$$
In terms of a bilinear layer, this looks like the following:

$$ h = Px $$
$$ \hat{y} = (Wh + b^w) \odot (Vh + b^h) $$
> Using a bias on a single side should be sufficient to for this task, however, using them on both sides actually simplifies further analysis.

## Math

$$ (Wh + b^w) \odot (Vh + b_v) $$
Using the distributivity of elemetwise products we get.
$$ (Wh \odot Vh) + (b^w \odot Vh) + (Wh \odot b^v) + (b^w \odot b^v) $$

We will now look in-depth term by term for visual clarity. We can decompose $h$ into its original features through $P$.
$$ Wh \odot Vh = (\sum_i W_iP_ix) \odot (\sum_j V_jP_jx)$$
Again, using distributivity, we can decompose this into a single long sum.

$$ Wh \odot Vh = \sum_i \sum_j (W_iP_ix) \odot (V_jP_jx) $$
Looking at the second term, we can follow a similar procedure.

$$ b^w \odot Vh = (\sum_i b^w_i) \odot (\sum_j V_jP_jx) $$
We can write $b^w_i$ as the product of a bias matrix $B^w_i$ and the identity matrix $I$, which is multiplied by a value that is always 1.
$$ b^w \odot Vh = (\sum_i B^w_i I_i 1) \odot (\sum_j V_jP_jx) $$
$$ b^w \odot Vh = \sum_i \sum_j (B^w_i I_i 1) \odot (V_jP_jx) $$
TODO: I have to write some more stuff here probably.

This large sum can them be recomposed into.

$$ (Wh \odot Vh) + (b^w \odot Vh) = \sum_i \sum_j (W_iP_ix) \odot (V_jP_jx) + (B^w_i I_i 1) \odot (V_jP_jx)$$
$$ (Wh \odot Vh) + (b^w \odot Vh) = \sum_i \sum_j [(W_iP_ix) + (B^w_i I_i 1)] \odot (V_jP_jx)$$

$$ (Wh \odot Vh) + (b^w \odot Vh) = (\sum_i W_iP_ix + B^w_i I_i 1) \odot (\sum_j V_jP_jx)$$
## TL;DR
Instead of using biases, we can simply adapt the weight matrices as follows.
$P: n\_features \times n\_hidden$ -> $P^*: (n\_features + 1) \times (n\_hidden + 1)$

By constructing the block matrix with and extra row and columns full of zeros except a 1 at the edge $P^* = [[P, 0], [0, 1]]$. In a sense, we are pushing a constant value of 1 as an extra dimension through the projection.

Both $W$ and $V$ are updated by adding a row to each, which represent the learnt biases.

## Observations
- $n\_hidden = 1$ and $n\_features = 3$ results in the model learning only 2 features in superposition at each instance 
- The model often shifts towards these two features in superposition.
- The model is able to learn superposition of 4 and 5 values.

## Pictures

[This](./images/basis_features.png) shows the learnt features.
[This](./images/pairwise_2_6.png) shows the pairwise interactions between features.

The second clearly corresponds to how the model is represented internally and shows clear superposition.
