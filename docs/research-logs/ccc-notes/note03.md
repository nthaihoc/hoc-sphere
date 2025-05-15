# Kế hoạch triển khai thực nghiệm và đánh giá kết quả mô hình SimCLR.
---

## Nội dung

**[I. Bộ dữ liệu](#i-bo-du-lieu)**

**[II. Thiết kế mô hình](#ii-thiet-ke-mo-hinh)**

**[III. Chiến lược thực nghiệm](#iii-chien-luoc-thuc-nghiem)**

**[IV. Đánh giá kết quả](#iv-danh-gia-ket-qua)**

**[V. Liên kết nhanh]()**

**[VI. Tài liệu tham khảo]()**

---
## I. Bộ dữ liệu

==**1. Mô tả bộ dữ liệu**==

Sử dụng bộ dữ liệu Ung thư cổ tử cung của bệnh viện A - Thái Nguyên, bao gồm 5 nhãn chính với số lượng là 22.434 ảnh. Đây là 5 lớp nhãn bệnh phổ biến của Ung thư cổ tử cung, được thống kê mô tả ở bảng dưới đây.

| **Label** | **Size** |
| :-------: | :------: |
|  ASC_H    |  5.669   |
|  ASC_US   |  3.869   |
|  HSIL     |  6.355   |
|  LSIL     |  4.780   |
|  SCC      |  1.761   |

==**2. Chiến lược chia dữ liệu**==

Bộ dữ liệu ban đầu được chia thành 3 tập riêng biệt là **train/dev/test** với tỷ lệ lần lượt là **70%/15%/15%**. Mỗi bộ dữ liệu này được tách biệt để đảm bảo tính công bằng trong huấn luyện và đánh giá. Các tập này sẽ được sử dụng một cách linh hoạt trong các giai đoạn, chiến lược thực nghiệm, được đề cập ở [**Phần III**](#iii-chien-luoc-thuc-nghiem). 

## II. Thiết kế mô hình

==**1. Mô hình SimCLR (pretraining).**==

- **Backbone :** Sử dụng các mô hình học sâu đa dạng, cân bằng giữa tốc độ và độ chính xác, phù hợp cho trích xuất đặc trưng từ ảnh y tế như: ResNet-50, ResNet-101, DenseNet-121, Inception-V3, EfficientNet-B0.

- **MLP layers :** Sử dụng làm đầu chiếu (projection head), sau khi các backbone tạo ra vector đặc trưng, chúng sẽ được đưa qua một MLP nhỏ (thường gồm 2 đến 3 lớp fully connected + hàm kích hoạt ReLU). Mục đích là biến đổi các embedding hình ảnh vào một không gian mới để tính toán loss.

- **Projection head:** Đây là kích thước đầu ra tùy chỉnh sau khi áp dụng MLP layers cho các vector đặc trưng được trích xuất từ backbone. Thông thường các vector đặc trưng sẽ được chuyển sang một không gian biểu diến mới có kích thước nhỏ hơn (128 hoặc 256).

| ^^Mô hình^^ | ^^Backbone^^ | ^^MLP Layers^^ | ^^Projection Head^^ |
| :-----: | :------- | :--------: | :-------------: |
|         | ResNet-50       |    |                  |
|         | DenseNet-121    |    |                  |
|  SimCLR | Inception-V3    | 2 (layer)  |         128 (output_dim)      |
|         | ResNet-101      |    |                  |
|         | EfficientNet-B0 |    |                  | 


==**2. Mô hình baseline.**==

Tiến hành thiết lập một số mô hình đơn giản ban đầu nhằm so sánh hiệu quả với các mô hình học representation (biểu diễn) như SimCLR. Lựa chọn 5 mô hình làm backbone cho SimCLR (như đã đề cập ở phía trên) làm các mô hình baseline.

## III. Chiến lược thực nghiệm

==**1. Huấn luyện SimCLR**==

**Giai đoạn 1: Pretraining SimCLR:**

- Thực hiện pretraining SimCLR cho bộ dữ liệu train + dev (đã được phân chia ở [**Phần I**](#i-bo-du-lieu)) và không sử dụng nhãn.
- Sử dụng hàm mất mát NT-Xent để học biểu diễn.
- Huấn luyện SimCLR lần lượt với 5 backbone đã được lựa chọn.

| ^^Model (backbone)^^ | ^^Augmentation^^ | ^^Loss^^ | ^^Optimizer^^ | ^^Batch Size^^ | ^^Epoch^^ |
| :---- |:---: | :--: | :------: | :--------: | :---: |
| SimCLR (ResNet-50) |          |      |      |       |      | 
| SimCLR (ResNet-101) |         |      |
| SimCLR (DenseNet-121) | (1) | NT-Xent | Adam(lr=0.3) | 128 | 100 |
| SimCLR (Inception-V3) |       |      |         |     |     |
| SimCLR (EfficientNet-B0) |    |      |         |     |     |

{++Ghi chú:++} (1) Các kỹ thuật augmentation được sử dụng bao gồm: RandomResizedCrop, RandomHorizontalFlip, RandomApply, RandomGrayScale, GaussianBlur, Normalize. 


**Giai đoạn 2: Fine-tuning (Linear protocol)**

- Để đánh giá khả năng tận dụng dữ liệu không gán nhãn của mô hình SimCLR, tiến hành fine-tune mô hình với các tỷ lệ khác nhau của tập train có nhãn: 10%, 30%, 50%, 70% và 100%. 

- Mục tiêu nhằm xác định tính hiệu quả của SimCLR khi số lượng nhãn giảm mạnh.

| ^^Model (backbone)^^ | ^^Augmentation^^ | ^^Loss^^ | ^^Optimizer^^ | ^^Batch Size^^ | ^^Epoch^^ |
| :---- |:---: | :--: | :------: | :--------: | :---: |
| SimCLR (ResNet-50) |          |      |      |       |      | 
| SimCLR (ResNet-101) |         |      |
| SimCLR (DenseNet-121) | (2) | Cross-Entropy | Adam(lr=1e-3) | 64 | 100 |
| SimCLR (Inception-V3) |       |      |         |     |     |
| SimCLR (EfficientNet-B0) |    |      |         |     |     |

{++Ghi chú:++} (2) Các kỹ thuật augmentation ở đây được sử dụng bao gồm: Resize, CenterCrop, RandomHorizontalFlip, Normalize.


## IV. Đánh giá kết quả


<br>

---