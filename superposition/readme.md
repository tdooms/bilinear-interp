# Superposition with Bilinear layers

In this folder, we aim to study superposition in bilinear layers. Studying the low-dimensional behavior of these layers will hopefully reveal generally applicable knowledge to study these networks. The following is a list of items we wish to study with their respective files.

#### Finding the best way to study low-dimensional models.
- [x] [simple](./simple.py): get a sense of the structure of superposition within bilinear layers
  - It generally seems like there are multiple (symmetric and anti-symmetric) optima to which these models optimize.
  - Capacity is distributed across all features, no matter their importance. There is never a 1-1 mapping.
- [x] [incorporate](./incorporate.py): We can maybe incorporate the bias into the weight matrix by adding a constant dimension to W and V.
  - I have made a lengthy write-up [here](./biases.md)
  - Bilinear layers have lots of clear bipolar features
  - The model seems tot struggle to represent more features into a single neuron.

#### Model reactions to different scenarios.
- [ ] : Correlation and anti-correlation according to [the original paper](https://transformer-circuits.pub/2022/toy_model/index.html).
- [ ] : Capacity and alternate setup according to [this Redwood paper](https://arxiv.org/abs/2210.01892).
- [ ] : Boolean composition and computation according to [this LessWrong post](https://www.lesswrong.com/posts/2roZtSr5TGmLjXMnT/toward-a-mathematical-framework-for-computation-in).

After this is done, it is probably possible to get a good view of how to continue the interpretation work.
