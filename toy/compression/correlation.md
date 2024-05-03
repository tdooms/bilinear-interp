# Correlation and Superposition


### Observations (correlated features)

- In cases of little sparsity, it seems that the model strongly relies on the correlation.
- It learns to pair these features and sum them together in the output
  - Given correlated features a and b: then O(a) = O(b) = 0.5a + 0.5b
- In higher sparsity regimes, it fully disentangles the features, despite their correlation.

![Image](./images/correlation_4_12.png)


### Observations (anti-correlated features)

- The case of anti-correlation is quite straightforward, the model can simply learn twice the amount of features no matter the sparsity.
- In higher sparsities, it switches towards more complex composition to learn all features but the structure remains a bit.

![Image](./images/anticorrelation_4_12.png)
