# Superposition with Bilinear layers

In this folder, we aim to study superposition in bilinear layers. Studying the low-dimensional behavior of these layers will hopefully reveal generally applicable knowledge to study these networks. The following is a list of items we wish to study with their respective files.

## Finding the best way to study low-dimensional models

- [x] [simple](./simple.py): Get a sense of the structure of superposition within bilinear layers
  - It generally seems like there are multiple (symmetric and anti-symmetric) optima to which these models optimize.
  - In low sparsities, capacity is distributed across all features, no matter their importance.
- [x] [biases](./biases.py): Incorporate the bias into the weight matrix by adding a constant dimension to W and V.
  - I have made a lengthy write-up [here](./biases.md)
  - Bilinear layers have lots of clear bipolar features.
  - The model seems tot struggle to represent more features into a single neuron.

## Model reactions to different scenarios

- [x] [correlation](./correlation.py): Correlation and anti-correlation according to [the original paper](https://transformer-circuits.pub/2022/toy_model/index.html).
  - Models that receive pairwise correlated features learn to sum them together in low sparsities.
  - Models that receive pairwise anti-correlated features can simply learn twice as many features.
- [x] [geometry](./geometry.py): What kinds of geometry can the bilinear model learn?
  - We test this by removing the projection matrix by a handcrafted one.
  - There is not much to say here, it can learn all kinds of geometries in the way you'd expect.
  - I should maybe check if a ReLU can also learn all of this.
- [ ] : Capacity and alternate setup according to [this Redwood paper](https://arxiv.org/abs/2210.01892).
- [ ] : Boolean composition and computation according to [this LessWrong post](https://www.lesswrong.com/posts/2roZtSr5TGmLjXMnT/toward-a-mathematical-framework-for-computation-in).
  - This will be discussed in the computation folder as there are way more experiments in this direction.

## Leveraging these findings

- [ ] Is there a way in which we can use the correlation and anti-correlation findings to improve high dimensional feature extraction?
