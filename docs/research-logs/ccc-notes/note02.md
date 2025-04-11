# :material-note-edit-outline: Implementation SimCLR Model
---

## Outlines

- [**I. End-to-End Model Training Process**](#1-end-to-end-model-training-process)
- [**II. NT-Xent Loss**](#ii-nt-xent-loss)
- [**III. LARS Optimizer**](#iii-lars-optimizer)

---
## I. End-to-End Model Training Process

``` mermaid
flowchart LR

    subgraph task01[Pretext Task]

        A(((x))) --> |AG'| x_1([x_1])
        A(((x))) --> |AG'| x_2([x_2])

        x_1 --> en01[base encoder]
        x_2 --> en01

        en01 --> |feature map| h_1{{h_1}}
        en01 --> |feature map| h_2{{h_2}}

    
        h_1 --> |MLP'| z_1[[z_1]]
        h_2 --> |MLP'| z_2[[z_2]]

        z_1 --> |LN'| linear{linear layers}
        z_2 --> |LN'| linear

    end

    subgraph task02[Downstream Task]
    inputs(((x))) ---> |inputs| en02[base encoder]

    en02 --> finetune{{fine_tune}}
    labels(((y))) ---> |labels| finetune

    finetune --> |evaluation| eval01{Precision/Recall/F1}
    end

    en01 --> en02
```

==**(+) Model training process consists of two main tasks**==

{++a/ Pretext task++}

* Stage 01: Using the unlabeled dataset ($x$), apply augmentation techniques (AG') as **random crop resize** and **color distortion**, **gaussian blue** to create two views, view1 ($x_1$) and view2 ($x_2$). Positive pair are $(x_1, x_2)$.
* Stage 02: Leverage the pre-trained model like **ResNet50**, **ResNet101**, **VGG19** and **InceptionResNetv2** on **IMAGENET** dataset as feature extractor for augmented dataset obtain the feature map $(h_1, h_2)$. 
* Stage 03: In this stage, a linear layer is applied (MLP') as a projection from the feature map to the embedding space. Use **32/64/128** as the output dimensions of the projection for experimentation.
* Stage 04: Training the model using the transformed dataset in the embedding space, leveraging NT-Xent loss to optimizer positive pairs and maximize negative pair. After the training process finished, retain the **base encoder** and throw away (MLP').
* Stage 05: Finally, evaluate the model's performance by using a linear classifier (linear evaluation protocol) and training it on labeled data.

{++b/ Downstream task++}

* Stage 01: Choose the models with the best performance after being evaluated in task 1, then fine-tuning these on labeled dataset ($x, y$) for classification task.
* Stage 02: Evaluate the model's performance using classification metrics such as **Precicion**, **Recall** and **F1-score**. 


## II. NT-Xent Loss

{++a/ Explanation and formula++}

The SimCLR model uses NT-Xent loss (normalized temperature scaled cross entropy loss). We have a data point set $x = [x_1, x_2,\dots, x_n]$. By default, the SimCLR model generates two versions of each input data point using augmentation methods. As results we will have 2N data points. Suppose a data point $x_1$ is augmented into two versions $x_i$ and $x_j$, which considered a positive pair. Then the loss function of examples $(x_i, x_j)$ is defined as:
$$
l(i, j) = l(j, i) = -\log \frac{exp(sim(z_i, z_j)) / \tau}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]}exp(sim(z_i, z_k)) / \tau}
$$

Then, the loss function for all data points is:
$$
\mathcal{L} = \frac{1}{2N} \sum_{k=1}^{N} \left[l(2k-1, 2k) + l(2k, 2k-1) \right]
$$

Where:

- $z_i, z_j$: outputs of $x_i, x_j$ after feature extracted by the projection head.
- $sim(z_i, z_j)$: represents the cosine similarity between these vectors.
- $\tau$: temperature parameter.

**(+) The goal of NT-Xent loss :** NT-Xent loss optimizers the unsupervised model by learning strong representations, where points from the same class are grouped closely together, while points from different classes are clearly separated. 

The $\tau$ parameter plays a crucial role in controlling how the model distinguishes between positive and negative pairs, and how it learns embeddings.

- When $\tau$ small, the model will put more focus on pushing negative pairs further apart, leading to clearer separation between positive and negative pairs.
- When $\tau$ larger, the model becomes less focused on the negative pairs, helping the model to learn more flexible representations, but may reduce its ability to precisely distinguish between negative pairs. 

{++b/ Manual calculation example++}

Suppose we have a data set $x = [x_1, x_2]$ as inputs. With batch size of 2, after applying augmentation techniques, two augmented views are generated: $v_1 = [x_{1i}, x_{2i}]$ and $v_2 = [x_{1j}, x_{2j}]$, where each view represents a different augmentation of the corresponding input data. 

Positive pairs is $(x_{1i}, x_{1j})$; $(x_{2i}, x_{2j})$. Below, cosine similarity matrix of all the examples.

|      | $x_{1i}$ | $x_{2i}$ | $x_{1j}$ | $x_{2j}$| 
| :--: | :------: | :------: | :------: | :-----: | 
| $x_{1i}$ | 1 | 0.63 | 0.77 | 0.70 |
| $x_{2i}$ | 0.63 | 1 | 0.67 | 0.84 | 
| $x_{1j}$ | 0.77 | 0.67 | 1 | 0.64 |
| $x_{2j}$ | 0.70 | 0.84 | 0.64 | 1 |

- Stage 01: Filter the value cosine similarity of positive pairs: $(x_{1i}, x_{1j}) = (x_{1j}, x_{1i}) =0.77$, $(x_{2i}, x_{2j}) = (x_{2j}, x_{2i}) = 0.84$ and throw away the diagonal of cosine similarity matrix are always 1. With $\tau = 1$, a new cosine similarity matrix is created below.

|      | $x_{1i}$ | $x_{2i}$ | $x_{1j}$ | $x_{2j}$| 
| :--: | :------: | :------: | :------: | :-----: | 
| $x_{1i}$ | - | 0.63 | - | 0.70 |
| $x_{2i}$ | 0.63 | - | 0.67 | - | 
| $x_{1j}$ | - | 0.67 | - | 0.64 |
| $x_{2j}$ | 0.70 | - | 0.64 | - |

- Stage 02: Combine the positive pairs and negative pairs. 

|   | $x_{1i}$ | $x_{2i}$ | $x_{1j}$ | $x_{2j}$| 
| :--: | :------: | :------: | :------: | :-----: | 
| $(x_{1i}, x_{1j})=0.77$ | - | 0.63 | - | 0.70 |
| $(x_{1j}, x_{1i})=0.77$ | 0.63 | - | 0.67 | - | 
| $(x_{2i}, x_{2j})=0.84$ | - | 0.67 | - | 0.64 |
| $(x_{2j}, x_{2i})=0.84$ | 0.70 | - | 0.64 | - |
 
- Stage 03: Applying the softmax function to transform the cosine similarity values into probabilities.Consider the positive pairs as label 0, and the rest as label 1. 

$$
S(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

Where:

- $e$: is represents the exponential of the logit
- $K$: is the number of classes

Compute the softmax values for each row in the matrix.

=> $S_{r1} = \left [\frac{e^{0.77}}{e^{0.63} + e^{0.77} + e^{0.70}}; \frac{e^{0.63}}{e^{0.63} + e^{0.77} + e^{0.70}}; \frac{e^{0.7}}{e^{0.63} + e^{0.77} + e^{0.70}} \right] = [0.3569; 0.3103; 03328]$

Similarly, compute the softmax values for the remaining rows in the matrix. The vector $[0.3569; 0.3103; 0.3328]$ consists of softmax values, with the first element being the probability of the positive pair. Therefore, by applying cross-entropy, we can calculate the loss for each positive pair.

=> $l(x_{1i}, x_{1j}) = -\log(0.3569) = -1.0303$

Conduct computation of the loss for the remaining positive pairs. Finally, the loss function on the batch is:

=> $\mathcal{L} = \frac{1}{4} \left[l(x_{1i}, x_{1j}) + l(x_{1j}, x_{1i}) + l(x_{2i}, x_{2j}) + l(x_{2j}, x_{2i})   \right]$

## III. LARS Optimizer

==**(+) What is LARS optimizer ?**==

LARS (Layer-wise Adaptive Rate Scaling) is an optimization algorithm, specifically designed for the training large-scale deep learning models, particularly in scenarios involving large-batch training. It addresses issues related to non-uniform learning rates across layers, ensuring faster convergence and improved stability during training.

When the training large-scale models with large-batch, the ratio between the norm of weights and gradients varies significantly across layers. Besides slow or unstable convergence. To address these problems, LARS introduces layer-wise learning rate, which are adapted for layer based on the relative magnitudes of the weights and gradients. 

==**(+) LARS optimizer works**==

At each layer $l$, the weights of the model updated by LARS as follows:
$$
\Delta w_{t}^{l} = \gamma \cdot \lambda^{l} \cdot \nabla L(w_{t}^{l})
$$

where $\gamma$ is a global learning rate and $\lambda^{l}$ is a local learning rate, defined for each layer through $\eta$ (trust coefficient $\eta$ < 1):
$$
\lambda^{l} = \eta \cdot \frac{||w^l||}{||\nabla L(w^l)||}
$$

LARS can be used to balance the local learning rate and the weight decay term $\beta$, and then applied to the update process, then $\lambda^{l}$ is defined:
$$
\lambda^{l} = \frac{||w^{l}||}{||\nabla L(w^l)|| + \beta \cdot ||w^{l}||}
$$

Momentum helps prevent the weights from updating too rapidly and reduces oscillations during optimization. The momentum for layer $l$ at step $t+1$ is updated as follows:
$$
v_{t+1}^{l} = \mu \cdot v_{t}^{l} + \gamma_{t+1} \cdot \lambda^{l} \cdot (||\nabla L(w_{t}^{l})|| + \beta w_{t}^{l})
$$

Where:

- $v_{t+1}^{l}$ is the momentum at step $t+1$ for layer $l$.
- $\mu$ is the momentum factor, typically between 0 and 1.
- $v_{t}^{l}$ is the momentum at the current step $t$ for layer $l$.
- $\gamma_{t+1}$ is the local learning rate at step $t+1$.
- $\nabla L(w_{t}^{l})$ is the gradient of the loss $L$ with respect to the weights $w_{t}^{l}$ at step $t$.
- $\beta w_{t}^{l}$ is the weight decay term (L2 regularization).



---
<br>