# :material-note-edit-outline: Vision Language Model - VLM
---

## Contents

- [**I. The problem**](#i-the-problem)
- [**II. Overview of the VLM**](#ii-overview-of-the-vlm)
- [**III. CLIP model (Contrastive Language-Image Pre-training)**](#iii-clip-model-contrastive-language-image-pretraining)
- [**IV. BLIP model (Bootstrapping Language-Image Pre-training)**](#iv-blip-model-bootstrapping-language-image-pre-training)

---

## I. The problem
## II. Overview of the VLM
## III. CLIP model (Contrastive Language-Image Pretraining0)
==**1. Overview**==

**Information:** The CLIP model was introduced in Jan, 2020 by OpenAI with paper title {++"Learning Transferable Visual Models from Natural Language Supervision"++}. This is a highlight architecture in combining language and image learning, opening up zero-shot learning for a wide range of computer vision tasks.

**Objectives of Model:** 

- Training the model to predict which caption matches which image on a dataset of 400 million pair (image, text), developed an efficient image representation learning model from scratch. 

- Then the training, can use natural language to refer to or describe visual concepts, allowing zero-shot transfer to a variety of tasks.

**Results achieved:** The model was evaluated on more than 30 datasets: character recognition, video action, geolocation and fine-grained object classification. Model performed competitively with supervised training methods, accuracy comparable to ResNet-50 on ImageNet without using its training set.

==**2. Architecture model**==

<figure markdown="span">
    ![](../../assets/architecture_clip.png){width=100%}
</figure>

**Image encoder:** Use ResNet50, ResNetD and ViT as the base architecture for the image encoder. Replace the GAP layer with an attention pooling mechanism - transformer style (multi-head QKV). 

**Text encoder:** 

- Transformer with the architecture modifications: 63M-parameter, 12-layer 512-wide model with 8 attention heads. 

- Text converted to token by BPE with a 49.152 vocab size and sequence length at 76.

**Multi-model embedding space:** Both image feature and text feature are layer normalized, then linearly projected into the multi-model embedding space to calculate the similarity between images and descriptions.

==**3. Zero-shot transfer**==

^^Progress:^^

1. Use image encoder to get image embedding, use text encoder to get embeddings for all class names.
2. CLIP computes cosine similarities between the image and text embeddings, scales them by a temperature parameter, then applies softmax to get probabilities. This works like a softmax classifier where both the inputs and class embeddings are L2-normalized, there's no bias term, and temperature controls the sharpness of the output. 

^^Multinomial logistic regression classifier:^^

$$\text{logit}_{i} = f_\text{img} \cdot f_\text{text, i}$$

where:

- $f_\text{img}$ is embedding image / inputs
- $f_\text{text, i}$ is embedding of text $i$ / weights
- $\text{logit}_{i}$ is cosine similarity. 

Devide the logits by a temperature ($\tau$) coefficient. Then pass it through softmax to get the classification probabilities.

$$P_{i} = \frac{e^{logit_i / \tau}}{\sum_{j}e^{logit_j / \tau}}$$

==**3. Evaluation zero-shot CLIP**==

**Objectives:** The main goal is to evaluate the quality of the representation learned by CLIP during its large-scale pre-training.

**More specifically:**

- {++Logistic Regression = Supervised baseline:++}
    - Logistic Regression is trained features extracted from a standard backbone (ResNet50, v.v), using labeled training.
    - It represents a simple and standard supervised learning baseline, commonly used to evaluate learned representations. 
- {++CLIP zero-shot = No training on the new datasets:++}
    - CLIP doesn't require fine-tuning or labels from the new dataset.
    - It simply matches image features with text embedding using cosine similarity.
    - Predictions is done directly using ís knowledge CLIP learned during pre-training.

==**4. Limitations**==

CLIP also struggles on some tasks, especially:

- Fine-grained classification, like telling apart car models, flower species, or airplane types.
- Abstract or systematic tasks, like counting objects in an image.
- New or uncommon tasks that probably weren't in CLIP's training set. 

## IV. BLIP model (Bootstrapping Language-Image Pre-training)

==**1. Overview**==

Vấn đề: 

- Hạn chế về mô hình: 

    - Encoder-only model (CLIP) chỉ tốt cho việc hiểu văn bản và ảnh, nhưng không phù hợp cho việc sinh văn bản. 
    - Encoder-Decoder model, tốt cho sinh văn bản nhưng khó áp dụng hiệu quả cho các tác vụ như tìm kiếm ảnh bằng văn bản. 

- Hạn chế về dữ liệu:

    - Các mô hình trước đó chủ yếu dựa vào dữ liệu hình ảnh và văn bản từ web.
    - Dữ liệu tuy lớn nhưng thường nhiễu và không lý tưởng cho việc huấn luyện.

Mục tiêu: Multimodal mixture of encoder-decoder (MED) được thiết kế linh hoạt có thể hoạt động dưới 3 chế độ. 

- Encoder đơn modal: chỉ xử lý văn bản hoặc ảnh.
- Image-grounded text encoder: Xử lý văn bản có tham chiếu ảnh.    
- Image-grounded text-decoder: Sinh văn bản từ ảnh.

Nhiệm vụ: BLIP huấn luyện MED với 3 nhiệm vụ chính:

- Contrastive learning: học cách phân biệt cặp ảnh-văn bản đúng sai
- Matching: xác định xem một ảnh và văn bản có khớp nhau không
- Language modeling: sinh văn bản dựa trên nội dung hình ảnh.

==**2. Kiến trúc mô hình**==

Bộ mã hóa ảnh

- Mô hình sử dụng ViT để trích xuất đặc trưng hình ảnh thành các vector embedding
- Thêm một token đặc biệt [CLS] để đại diện cho toàn bộ đặc trưng hình ảnh.

MED hoạt động ở 3 chế độ:

- Bộ mã hóa đơn modal: Xử lý ảnh và văn bản một cách riêng biệt. Với văn bản dùng mô hình giống như BERT, thêm CLS vào đầu câu để đại diện cho toàn câu.
- Bộ mã hóa văn bản dựa trên ảnh: Dùng để hiểu kết hợp cả hình và chữ: Thêm lớp Cross-Attetion giữa self-attention và Feed-forward tong mỗi block transformer của văn bản. Thêm token [encode] vào câu văn bản, output của token này sẽ là biển diễn đại diện của cặp ảnh và văn bản.
- Bộ sinh văn bản dựa trên ảnh: Dùng để tạo ra câu văn dựa trên ảnh, thay các lớp self-attention bằng causal-attention. Thêm [decode] để đánh dấu bắt đầu đoạn sinh và token kết thúc để chỉ điểm dừng.

==**3. Mục tiêu huấn luyện**==

Image-Text Contrastive Loss - ITC:

- Mục tiêu: căn chỉnh không gian đặc trưng giữa bộ biến đổi hỉnh ảnh và bộ biến đổi văn bản, làm cho các cặp ảnh và văn bản dương có biểu diễn giống nhau hơn so với các cặp âm.
- Dùng momentum encoder và nhãn mềm để xử lý khả năng có mẫu dương tiềm ẩn trong các mẫu âm

Image-Text Matching Loss - ITM:

- Mục tiêu: học biểu diễn đa mô hình thể hiện sự tương quan chi tiết giữa ảnh và văn bản.
- Sử dụng chiến lược hard negative mining để chọn những cặp âm có độ tương đồng cao để tạo ra mẫu âm khó.

Language Modeling Loss - LM:

- Mục tiêu: tạo văn bản mô tả ảnh bằng cách huấn luyện mô hình sinh văn bản dạng chuỗi.
- Khác với cách dùng MLM, LM giúp mô hình có khả năng tổng quát hơn trong việc tạo ra mô tả từ hình ảnh.






---


