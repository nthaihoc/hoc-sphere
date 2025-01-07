# Self-Supervised Learning Model: A Simple Framework for Contrastive Learning of Visual Representations
---

## Outlines

[**I. Introduction**](#introduction)

[**II. Methods**](#ii-methods)

[**III. Implementation plan**](#iii-implementation)

---
## I. Introduction

==**(+) Key findings of SimCLR**==

- Composition of data augmentations plays a critical role in defining effective predictive tasks.
- Nonlinear transformation between the representations and the contrastive loss substantially improves the quality of the learned representations.
- Benefits from larger batch sizes and more training steps compared to supervised learning.

<figure markdown="span">
    ![](../../assets/performance_ssl.png){width=65%}
</figure>

==**(+) Main notes**==

- Most mainstream approaches fall into one of two group: generative and discriminative.
- Discriminative approaches use objective functions similar to those used for supervised learning, but train networks to perform pretext tasks where both the inputs and labels are created from an unlabeled dataset.
    - Based on contrastive learning in the latent space.
    - More better than heuristics to design pretext tasks.

---
## II. Methods

==**(+) Contrastive Learning Framework**==

<figure markdown="span">
    ![](../../assets/architecture_ssl.png){width=70%}
</figure>

- *Data augmentation*: Stochastic transforms any given data example randomly resulting in two correlated views of the same example ($x_i$ and $x_j$).
- *Base encoder*: $f(.)$ that extracts representations vectors from augmented data examples. Where $h_i \in \mathbb{R}^d$ is output of the average pooling layer.
- *Projection head*: $g(.)$ that maps representations to the space latent where contrastive loss is applied.
- *Contrastive loss function*: Given set $x_k$ including a positive pair of examples $x_i$ and $x_j$. Aims to identify $x_j \in x_{k}$ with $k \neq i$ for a given $x_i$.

$$
 L_(i,j) = -log \frac{exp(sim(z_i, z_j)/ \tau)}{\sum^{2N}_{k=1} \mathbb{1}_{k \neq i} \exp(sim(z_i, z_k)/ \tau)} 
$$

- Where $sim(z_i, z_j)$ is cosine similar between $z_i$ and $z_j$ vector, $\tau$ is temperature parameter.   


==**(+) SimCLR pseudo code**==

{++Important steps in SimCLR++} 

1. Data augmentation from an example data.
2. Feature extraction is performed on all examples using a pre-trained backbone, and then they are mapped to the latent space by the projection head.
3. In the latent space, compute cosine similarity for all examples (including positive and negative pair).
4. Compute the contrastive loss function based on cosine similarity of all examples computed in step 3.
5. Finally, update the networks $f(.)$ and $g(.)$ to minimize $\mathcal{L} \text{(loss function)}$. Retain the encoder network $f(.)$ and discard $g(.)$. 

<figure markdown="span">
    ![](../../assets/pseudo_code_ssl.png){width=70%}
</figure>

==**(+) Discussion**==

{++Data++}

- Data augmentation defines predictive tasks.
- Composition of data augmentation operations is crucial for learning good representations.
- Contrastive learning needs stronger data augmentation than supervised learning.

{++Architectures++}

- Unsupervised learning benefits more from bigger models than its supervised counterpart.

<figure markdown="span">
    ![](../../assets/performance_ssl_01.png){width=70%}
</figure>

- Nonlinear projection head improves the representation quality of the layer before it.

{++Loss Functions++}

- Logistic Loss
$$
\mathcal{L(y, \hat{y})} = -y \text{ } log(\hat{y}) - (1 - y) \text{ } log(1 - \hat{y})
$$

^^Disadvantages :^^ Cannot effectively leverage hard negatives.

- Margin Loss
$$
\mathcal{L}_\text{margin} = \max (0, m + d_p - d_n)
$$

^^Disadvanteges :^^ Choose the appropriate margin $m$ value. Focuses semi-hard negatives but identify semi-hard negatives sample is very hard. 

- NT-Xent Loss

$$
 L_(i,j) = -log \frac{exp(sim(z_i, z_j)/ \tau)}{\sum^{2N}_{k=1} \mathbb{1}_{k \neq i} \exp(sim(z_i, z_k)/ \tau)} 
$$

^^Advantages :^^ Leverages both positive and negative samples. Use cosine similarity. Automatically focuses on hard negatives using the temperature parameter $\tau$. 

---
## III. Implementation plan

==**(!!!) Some of the training strategies and approaches for the cervical cancer dataset**==

^^Stage 1: Choose the model backbone^^

Using some pre-trained models on the ImageNet dataset:

- Directly use pre-trained models as the backbone and then perform fine-tuning on a small amount of labeled data.
- Leverage pre-trained models and then continue training the model on an unlabeled dataset.
- Train the model from scratch with dataset consisting solely of cellular images.

^^Stage 2: Evaluation the model backbone^^

After the model has been trained on unlabeled data, where representations are learned in latent space using NT-Xent loss, followed by evaluation with some linear layers.

- Retain the weights of the model trained on unlabeled data, then add some linear layers and continue training the model on a small amount of labeled data.
- Evaluate the performance of the model on validation and test data.

^^Stage 3: Fine-tune the model for the downstream task^^

Perform fine-tuning on the model that has the best performance after evaluation in stage 2 on labeled dataset. 

- Fine-tune the model sequentially with different proportions of labeled data (10%, 20%, 30%, 40%, 50%) and evaluate its performance.


---
<br>