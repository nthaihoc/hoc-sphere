# :material-note-edit-outline: Implementation SimCLR Model
---

## Outlines

[**I. End-to-End Model Training Process**](#1-end-to-end-model-training-process)

[**II. NT-Xent Loss**](#ii-nt-xent-loss)

[**III. LARS Optimizer**](#iii-lars-optimizer)

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

Suppose we have a data set $x = [x_1, x_2]$ as inputs. After applying augmentation techniques, the resulting augmented set becomes $x = [x_{1i}, x_{1j}, x_{2i}, x_{2j}]$.

- Stage 01: Filter the positive pairs, the result includes 2 positive pairs $(x_{1i}, x_{1j}); (x_{2i}, x_{2j})$. 
- Stage 02. Suppose we have a matrix cosine similarity for all examples.

|      | $x_{1i}$ | $x_{1j}$ | $x_{2i}$ | $x_{2j}$|
| :--: | :------: | :------: | :------: | :-----: |
| $x_{1i}$ |   1  |   0.63   |    0.77  |    0.70 |
| $x_{1j}$ | 0.63 |     1    |    0.67  |    0.84 |
| $x_{2i}$ | 0.77 |    0.67  |     1    |    0.64 |
| $x_{2j}$ | 0.70 |    0.84  |    0.64  |    1    |





## III. LARS Optimizer

---
<br>