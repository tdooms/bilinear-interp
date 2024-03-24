# 

### AND gate

|$a^2$|$b^2$|$2ab$|$a$|$b$|$\gamma$|out|
|:---:|:---:|:---:|:-:|:-:|:------:|:-:|
|0|0|0|0|0|1|0|
|1|0|0|1|0|1|0|
|0|1|0|0|1|1|0|
|1|1|1|1|1|1|1|

$\gamma = 0$

$a^2+a = 0$

$b^2+b = 0$

$2ab = 1$

### OR gate

|$a^2$|$b^2$|$2ab$|$a$|$b$|$\gamma$|out|
|:---:|:---:|:---:|:-:|:-:|:------:|:-:|
|0|0|0|0|0|1|0|
|1|0|0|1|0|1|1|
|0|1|0|0|1|1|1|
|1|1|1|1|1|1|1|

$\gamma = 0$

$a^2+a = 1$

$b^2+b = 1$

$2ab = -1$

### General Boolean Computation

In general, given a boolean truth table with entries $t_{00}$, $t_{01}$, $t_{10}$ and $t_{11}$, we can find a closed formula for the optimal weights. We achieve this by simply generalizing the above to this general case, we get:

$\gamma = t_{00}$

$a^2 + a = t_{10} - t_{00}$

$b^2 + b = t_{01} - t_{00}$

$2ab = t_{11} - t_{10} - t_{01} -t_{00}$

The following are a few examples

#### 3 AND gate + 2 OR gate

To get a more complex truth table, we can combine gates in arbitrary ways. Let's try $t_{xy} = 3 * \text{AND}(x, y) + 2 * \text{OR}(x, y)$

$\gamma = 0$

$a^2 + a = (2 + 0) - 0 = 2$

$b^2 + b = (2 + 0) - 0 = 2$

$2ab = (2 + 3) - 2 - 2 - 0 = 1$

When we train a model on this truth table, we get:

![image](../images/3and_2or_feat0.png)

While there are not exact numbers, the light blue parts are $0.5 \pm 0.01$ and the darker parts are $1.5 \pm 0.01$, which satisfies our constraints. So, this simple derivation allows us to "predict" the optimal solution of a bilinear layer.



### Superposition (a)

|$a^2$|$b^2$|$2ab$|$a$|$b$|$\gamma$|out1|out2|
|:---:|:---:|:---:|:-:|:-:|:------:|:--:|:--:|
|0|0|0|0|0|1|0|0|
|1|0|0|1|0|1|1|0|
|0|1|0|0|1|1|0|1|
|1|1|1|1|1|1|0|0|

$\gamma = 0$

$a^2+a = (1, 0)$

$b^2+b = (0, 1)$

$2ab = -1$

> For the other superposition side, just swap the location of the ones in the output.


### Superposition $\odot$ AND

|$a^2$|$b^2$|$2ab$|$a$|$b$|$\gamma$|out1|out2|
|:---:|:---:|:---:|:-:|:-:|:------:|:--:|:--:|
|0|0|0|0|0|1|0|
|1|0|0|1|0|1|0|
|0|1|0|0|1|1|0|
|1|1|1|1|1|1|1|