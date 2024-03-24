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

### 3 AND gate + 2 OR gate

$\gamma = 0$

$a^2+a = 0+2$

$b^2+b = 0+2$

$(a^2+a) + (b^2+b) + 2ab=2+3 \Rightarrow 2ab = 1$


### General Boolean Computation
In general, given a truth table with entries $t_{00}$, $t_{01}$, $t_{10}$ and $t_{11}$, we can find a closed formula for the optimal weights.
We achieve this by simply generalizing the above to this general case, we get:

$ \gamma = t_{00}$
$ a^2 + a = t_{10} - t{00} $
$ b^2 + b = t_{01} - t{00} $
$ 2ab = t_{11} - t_{10} - t_{01} -t{00} $

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