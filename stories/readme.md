# Analyzing Bilinear Language Models

We want to demonstrate that bilinear layers, when used in transformer models are able to scale to language tasks. In a sense, this has been shown in [the Noam paper](). However, this paper only demonstrated that the accuracy of bilinear transformers matched or even outperformed normal GeLU-based transformers. In this document, we want to tackle the ambitious task of using out interpretability techniques to thoroughly understand this class of models. We aim to achieve this on the simplest (real-world) language task; children's stories. [TinyStories]() is an awesome dataset due to its simplicity. In short, it contains 2 million stories using the vocabulary of a 3-year old. This makes it possible to use tiny models to fit this dataset. Fascinatingly, these tiny models (with even less than 1M parameters) can coherently construct simple sentences and relate certain words together to tell a story. 

## Setup

Usually, language models are trained with a single goal in mind, achieve the lowest loss (or complexity) possible. In this work, this isn't the case, we strive to optimize for interpretability. This means that as long as the model can generate coherent output, we don't really care about how good the generated text is. This allows us to make some strong design choices which significantly simplify the model. The following sections describe the setup and design process in no particular order.

### Architecture

HuggingFace contains a wide range of transformer models that can be tweaked without much effort. However, 