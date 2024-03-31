# Polar analysis Q & A

#### Q: Can we derive the "capacity" of a certain feature from the integral of the polar function?

Maybe, but this doesn't seem like the answer is very clear cut. It's not like a feature in
superposition has area 0.5 and a feature without superposition has 1. I currently see 2 ways forward
on this issue: take a smart norm on the b tensor or analyze the phase interference.

#### Q: How well can we detect superposition? Does this work for more complex settings?

The largest limitation of the current approach is that we can only study two hidden dimensions.
However, within these two dimensions, I haven't found any scaling issues to more complex settings.
For instance, in the case of a pentagon superposition, the input directions are clearly visible.
In less sparse cases, the model prefers to just encode two digons. This is clearly visible, where the 4 features are orthogonal to each other and 1 feature predicts a constant value (a perfect circle).

#### Q: What happens to the polar plots when we add unembeddings? Do they generalize nicely?

Nope! Adding the unembedding (even an orthogonal one) generally strongly disturbs the plots. Maybe, in the orthogonal case that there is a way to still extract some value regarding superposition but this probably doesn't generalize to any transformation.

