# Text Dump

Don't look too deep here, this is mostly a file in which I dump snippets of written text that I didn't want to throw away but that dion't fit anywhere elese

---

Activation functions are a vital component in deep networks to learn complex functions. The sole purpose of an activation function is to introduce non-linearity into the network to prevent matrix collapse. It is often argued that the simplest non-linearity is piecewise linearity, which is exactly what the most popular activation function, the ReLU, does.

However, in terms of interpretability, ReLUs are very difficult to study. Intuitively, the issue is that it is only possible to know the output of a ReLU by passing in the input, not solely from the weights. While it is possible to discern structure and circuits in these weights, this is often done by means of sampling the input or using gradient based techniques to generate visualizations which may not reflect the full complexity of the models. This makes it impossible to make strong guarantees on which outputs models will be able to produce. This has been observed in the form of adversarial examples, in which a slight perturbation is applied to the input to confuse the model into making bogus predictions.

This undesirable property of the ReLU has led to MLPs and similar structures being famously hard to interpret. As a consequence, several papers ([5], [6]) have used quadratic activation functions to perform theoretic analyses. Unfortunately, simple quadratic activation functions result in terrible accuracy due to as they do not lead to universal function approximators ([7]). However, as a substitute, it is possible to use bilinear layers. This operation possesses appealing characteristic like the quadratic activations while being comparable to (and even surpassing) ReLUs in accuracy in large models as established in ([4]). Recently, most cutting-edge models such as Llama-2 have started using variants of it. Therefore, in this document, we make the design decision to replace ReLUs with the more interpretable bilinear layer. We provide an introduction to these layers and provide an overview of our current efforts in interpreting simple models using them.



$$ = (W \cdot \begin{bmatrix} x \\ y \end{bmatrix}) \odot (V \cdot \begin{bmatrix} x \\ y \end{bmatrix}) $$
$$ = \begin{bmatrix} x \cdot W_{00} + y \cdot W_{10} \\ x \cdot W_{01} + y \cdot W_{11} \end{bmatrix} \odot \begin{bmatrix} x \cdot V_{00} + y \cdot V_{10} \\ x \cdot V_{01} + y \cdot V_{11} \end{bmatrix}$$

$$=\begin{bmatrix} x^2 \cdot W_{00} \cdot V_{00} + xy \cdot W_{00} \cdot V_{10} + yx \cdot W_{10} \cdot V_{00} + y^2 \cdot W_{10} \cdot V_{10} \\ x^2 \cdot W_{01} \cdot V_{01} + xy \cdot W_{01} \cdot V_{11} + yx \cdot W_{11} \cdot V_{01} + y^2 \cdot W_{11} \cdot V_{11} \end{bmatrix}$$



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