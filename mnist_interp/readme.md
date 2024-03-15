# MNIST features in bilinear models
## Motivation
- **What are features?**
  - "Features" are currently a somewhat fuzzy concept to describe separable aspects of a dataset. It's sometimes possible to "know it when you see it".
  - At different layers in a model, different features may be linearly accessible so that a simple linear layer can be used to readoff the feature values. This means that the next layer of a model can easily use the linearly accessible features output by a previous layer. 
  - _Sparse autoencoders (SAEs)_ have been successfully used to find interpretable features in LLMs [[1]](#1). But it's unclear if the features they find correspond to those used for computation in the models. 
    - For example, sparse autoencoders are trained over activations from a set of inputs. It's possible for SAEs to combine different linearly accessible features that are correlated into a single feature that might only be calculated by the model in a downstream layer. So SAEs might unintentionally find features from downstream layers.
  - _Computational features_: In some sense, the model's weights should already reflect the statistics of the inputs so it should be possible to derive features from the model's weights alone. This might help limit our understanding to the features that the model's computations rely on instead of those that can be inferred over the input statistics. 
- **Rewriting computations with features**
  - In the ideal case, we'd be able to rewrite the full model in terms of features. This would demonstrate how higher-order features are computed in terms of lower-order ones, for example. 
  - Nonlinearities such as ReLU make this challenging because features can interact in ways that are difficult to describe.
  - SAEs are often trained with nonlinear activations like ReLU. So even though SAEs might describe the features present in a layer, they do not allow those features to be easily understood in terms of earlier features.
- **Bilinear layers may allow us to trace features through a model**
  - Bilinear layers of the form $g(x) = (W_1 x + b_1) \odot (W_2 x + b_2)$ are perhaps the simplest nonlinear activation since they only have terms quadratic or linear in $x$.
  - Sharkey (2023) [[2]](#2) has suggested that bilinear layers may aid mechanistic interpretability because they are easier to analyze. They can be expressed using only linear operations and third order tensors, so techniques from linear algebra can be applied. It might be possible to "understand a smaller number of primitive features that bilinear
layers use to ‘construct’ their (potentially exponential) larger number of features."
  - _Bilinear layers may have little to no cost in terms of performance_. A comparison of activations functions found that bilinear activations outperformed ReLU in transformer models [[3]](#3). It had only slightly worse performance compared to SwiGLU, a modern version of the Gated Linear Unit (GLU) that is used in LLama and PaLM models. The bilinear activation can be seen as the simplest type of GLU.
- **Our goal**
  - Our goal is to demonstrate the utility of bilinear layers by extracting features for a bilinear model trained on a simple tasks, such as MNIST, and tracing their computation through the model. Ideally these features will be constructed solely from the model weights.
 
  
# References
- <a id="1">[1]</a> Towards Monosemanticity: Decomposing Language Models With Dictionary Learning [[link](https://transformer-circuits.pub/2023/monosemantic-features/index.html)] Bricken et al 10/2024. Anthropic. 
- <a id="2">[2]</a> A technical note on bilinear layers for interpretability [[link](https://arxiv.org/abs/2305.03452)]. Lee Sharkey 05/2023
- <a id="3">[3]</a> GLU Variants Improve Transformer [[link](https://arxiv.org/abs/2002.05202)] Noam Shazeer 02/2022
