# The SimCLR Model

## I. The SimCLR framework

SimCLR is a method for learning generalized representations from unlabeled image data. It works by:
- Learning shared representations by maximizing similarity between different augmented views of the same image (attracting them).
- Reducing similarity between representations of different images (repelling them).
- The model can then be fine-tuned with a small amount of labeled data to effectively perform classification tasks.
- SimCLR randomly selects examples from the dataset and applies two simple augmentations (random cropping, color distortion, Gaussian blur) to create two distinct versions of the same image.
Reasons for using simple transformations:
  + To encourage consistent representations for different versions of the same image.
  + Labels are unavailable in pretraining, so object classes are unknown.
  + Simple transformations are sufficient for learning good representations, though more advanced transformations can also be used.

- SimCLR uses a CNN based on the ResNet architecture to compute image representations.
- An MLP amplifies invariant features and improves the network's ability to distinguish transformations of the same image.
- Stochastic Gradient Descent (SGD) is used to update CNN and MLP by minimizing the contrastive loss.
- After pretraining, CNN representations can be used directly or fine-tuned with labeled data for classification tasks.

## II. Understanding Contrastive Learning of Representations.

<figure markdown="span">
  ![Image title](../../assets/type_augumentation.png){ width="500" }
  <figcaption>Augumentation Techniques of SimCLR Framework</figcaption>
</figure>

- Finding 1: SimCLR's improvement over previous methods comes from the combination of design choices, not any single feature. Key findings include:
    - Importance of image transformation combinations: SimCLR maximizes agreement between different views of the same image. Combining transformations like random cropping and color distortion effectively prevents trivial solutions like matching color histograms.
    - Significance of combining cropping and color distortion: Cropping creates prediction tasks, such as global-to-local or neighboring views. However, similar color spaces across crops can lead the model to focus on matching colors. Independent color distortion removes this shortcut, forcing the model to learn meaningful, generalizable features.

- Finding 2: The nonlinear projection is important.
- Finding 3: Scaling up significantly improves performance.

## III. Experiment and Results

| STT | Model | Type Augumentation | F1-Score |
| :-: | :---: | :----------------: | :------: |
| 01 | SimCLR | Use all augumentatio techniques the framework provides | 73.17% |
| 02 | SimCLR | Cutout, Color distort, Gaussian Noise, Gaussian blur | 72.59% |
| 03 | SimCLR | Cutout, Color distort, Gaussian Noise, ROI | 65.23% |





