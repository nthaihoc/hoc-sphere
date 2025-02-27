# :material-note-edit-outline: Tổng hợp các kết quả thực nghiệm cho mô hình SimCLR
---

## I. Tập dữ liệu

- Tập A: Dữ liệu ban đầu có tổng là 49 nhãn

- Tập B: Dữ liệu gồm 5 nhãn chính cho bài toán

{++Tập dữ liệu được phân bổ thành những tập sau: ++}

- Tập dữ liệu huấn luyện ($A_1$): Tập này không sử dụng nhãn được dùng cho quá trình huấn luyện mô hình SimCLR.
    - Mẫu dữ liệu: Lấy tất cả các nhãn trừ **ASC_H, ASC_US, HSIL, LSIL, SCC** và những nhãn có số lượng ảnh quá ít.
    - Số lượng: Lấy 20 nhãn => ==Tổng số lượng nhãn là 30.000==.

- Tập dữ liệu kiểm định ($A_2$): Đây là tập dùng cho quá trình **Linear Protocol Evaluation**.
    - Lấy 30 nhãn, mỗi nhãn 100 mẫu => ==Tổng số lượng mẫu là 3000==.

- Tập dữ liệu huấn luyện ($B_1$): Tập này gồm 5 nhãn chính cho bài toán: **ASC_H, ASC_US, HSIL, LSIL, SCC**.
    - Dùng để fine-tune mô hình cho giai đoạn **downstream task** => ==Tổng là 15.000==.

- Tập dữ liệu kiểm tra ($B_2$): Tập này dùng cho quá trình đánh giá sau khi giai đoạn **downstream task** kết thúc.
    - Số lượng tập kiêm tra cho giai đoạn này là ==7.000== mẫu. 

## II. Kết quả thực nghiệm

==Giai đoạn 1: Huấn luyện mô hình SimCLR (pretext task)==:

- Trong giai đoạn này thực hiện huấn luyện lần lượt qua các backbone khác nhau như: **ResNet-50, ResNet-101, InceptionV3, InceptionResNetV2**.
- Huấn luyện trên các số lượng dữ liệu, batch-size và projection head khác nhau.
- Mặc định huấn luyện tất cả các backbone với 100 epochs, thuật toán tối ưu LARS và đánh giá qua NT-Xent Loss.

- Dùng giá trị hàm mất mát NT-Xent [(1)](../ccc-notes/note02.md) để theo dõi mức độ sai số của mô hình. Vì NT-Xent bản chất là một dạng của hàm cross-entropy, nên giá trị của nó luôn dương. Khi biểu diễn hai phép biến đổi $z_i$ và $z_j$ từ một mẫu mà giống hệt nhau thì cosine similarity đạt giá trị tối đa là 1, xác suất của cặp dương đạt 1, logarit của 1 là 0, nên hàm mất mát sẽ bằng 0 -> Nếu loss tiến tới gần 0, mô hình học tốt và có nhiều các cặp dương gần nhau, ngược lại loss cao, mô hình chưa tối ưu. 

- Dùng tập dữ liệu $A_1$ trong giai đoạn  này. Lấy tập dữ liệu ngẫu nhiên với 3 tỷ lệ 100%/70%/40%. 

^^a/ ResNet-50^^

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|           |  30.000  |    | 32/64/128 |  LARS   |   4.26/**4.17**/5.0  |
| ResNet-50 |  21.000  | 64 | 32/64/128 |  LARS   |   6.13/**5.92**/7.13 |
|           |  12.000  |    | 32/64/128 |  LARS   |   9.78/9.0/**8.85**|

-> Với batch-size=64, nhìn chung mô hình ResNet50 đa số đều đạt loss nhỏ với projection head=64. Mô hình có giá trị nhỏ nhất khi loss=4.17 và huấn luyện trên toàn bộ dữ liệu.

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|           |  30.000  |     | 32/64/128 |  LARS   | 3.87/**3.37**/3.96 |
| ResNet-50 |  21.000  | 128 | 32/64/128 |  LARS   | **4.36**/6.74/4.58 |
|           |  12.000  |     | 32/64/128 |  LARS   | **6.42**/8.18/7.05 |

-> Với batch-size=128, tất cả các giá trị loss đều có xu hướng giảm.Giá trị loss nhỏ nhất là 3.37 khi huấn luyện mô hình với toàn bộ dữ liệu và projection head=64. 

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|           |  30.000  |     | 32/64/128 |  LARS   | 3.18/**2.84**/2.88 |
| ResNet-50 |  21.000  | 256 | 32/64/128 |  LARS   | 4.12/**3.16**/6.08 |
|           |  12.000  |     | 32/64/128 |  LARS   | 6.10/5.89/**5.53** |

-> Với batch-size=256, các gía trị loss vẫn tiếp tục giảm, giá trị loss nhỏ nhất là 2.84 với projection head=64 khi huấn luyện trên toàn bộ dữ liệu.

^^b/ ResNet-101^^

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|           |   30.000 |    | 32/64/128 |  LARS   |  **4.57**/5.25/5.13 |
| ResNet-101|   21.000 | 64 | 32/64/128 |  LARS   |  5.11/**4.75**/**4.75** |
|           |   12.000 |    | 32/64/128 |  LARS   |  7.69/8.74/**6.25** |

-> Với batch-size=64, mô hình ResNet-101 đạt giá trị nhỏ nhất là 4.57 khi huấn luyện trên toàn bộ dữ liệu cùng với projection head=32.

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|           |   30.000 |     | 32/64/128 |  LARS   |  **3.58**/5.55/3.72 |
| ResNet-101|   21.000 | 128 | 32/64/128 |  LARS   |  4.10/3.77/**3.36** |
|           |   12.000 |     | 32/64/128 |  LARS   |  6.27/7.04/**5.19** |

-> Với batch-size=128, các giá trị loss đa số đều tiếp tục giảm, giá trị loss nhỏ nhất=3.36 khi huấn luyện mô hình với 70% dữ liệu và projection head=128.

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|           |   30.000 |    | 32/64/128 |  LARS   |  4.87/2.85/**2.91** |
| ResNet-101|   21.000 | 256 | 32/64/128 |  LARS   |  3.79/3.04/**2.78** |
|           |   12.000 |    | 32/64/128 |  LARS   |  **4.70**/5.30/4.78|

-> Với batch-size=256, mô hình đạt giá trị nhỏ nhất khi huấn luyện với 70% dữ liệu, projection head=128 với giá trị là 2.78.

^^c/ Inception-V3^^

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|              |  30.000 |    | 32/64/128 |  LARS | 3.57/4.27/**3.55** |
| Inception-V3 |  21.000 | 64 | 32/64/128 |  LARS | 4.36/5.77/**4.16**  |
|              |  12.000 |    | 32/64/128 |  LARS | **5.36**/7.49/5.53  |

-> Với batch-size=64, mô hình Inception-V3 có loss nhỏ nhất là 3.55 khi huấn luyện trên toàn bộ dữ liệu với projection-head=128.

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|              |  30.000 |     | 32/64/128 |  LARS | 3.28/3.06/**2.97** |
| Inception-V3 |  21.000 | 128 | 32/64/128 |  LARS | **3.22**/4.16/3.45 |
|              |  12.000 |     | 32/64/128 |  LARS | **3.97**/6.91/4.9  | 

-> Với giá trị projection-head=128, khi huấn luyện trên toàn bộ dữ liệu, mô hình tiếp tục đạt giá trị nhỏ nhất khi loss đạt 2.97. 

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|              |  30.000 |     | 32/64/128 |  LARS | 2.39/2.83/**2.7** |
| Inception-V3 |  21.000 | 256 | 32/64/128 |  LARS | **2.68**/3.65/2.92 |
|              |  12.000 |     | 32/64/128 |  LARS | **4.26**/7.21/4.96 |

-> Giá trị loss nhỏ nhất đạt 2.7 khi projection-head=128, batch-size=256 và huấn luyện trên toàn bộ dữ liệu. 

^^c/ InceptionResNet-V2^^

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|                    | 30.000 |    | 32/64/128 |  LARS  |   3.15/4.67/**3.11** |
| InceptionResNet-V2 | 21.000 | 64 | 32/64/128 |  LARS  |   3.13/3.0/**2.19** |
|                    | 12.000 |    | 32/64/128 |  LARS  |   3.62/**3.05**/3.77 |

-> Giá trị nhỏ nhất: 2.19 (ở InceptionResNet-V2, 21.000 mẫu).

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|                    | 30.000 |     | 32/64/128 |  LARS  | 2.95/4.42/**2.35** |
| InceptionResNet-V2 | 21.000 | 128 | 32/64/128 |  LARS  | 2.4/2.65/**1.97** |
|                    | 12.000 |     | 32/64/128 |  LARS  | 2.78/**2.71**/2.89 |

-> Giá trị nhỏ nhất: 1.97 (ở InceptionResNet-V2, 21.000 mẫu, batch-size 128).

| Backbone  | Total Data | Batch-size | Projection Output| Optimizer | NT-Xent Loss |
| :--:  | :-----: | :----: | :---:| :-------: | :----------: |
|                    | 30.000 |     | 32/64/128 |  LARS  | 2.43/3.36/**1.92** |
| InceptionResNet-V2 | 21.000 | 256 | 32/64/128 |  LARS  | **1.94**/2.23/5.62 |
|                    | 12.000 |     | 32/64/128 |  LARS  | **2.16**/2.24/2.72 |

-> Giá trị nhỏ nhất: 1.92 (tập 30.000 mẫu). 

==Kết luận giai đoạn 1==

--> {++Nhìn chung, batch-size lớn hơn và tập dữ liệu huấn luyện nhiều hơn có xu hướng cải thiện hiệu suất mô hình, giúp giảm NT-Xent Loss đáng kể. Khi dữ liệu huấn luyện tăng, loss giảm dần, cho thấy mô hình học biểu diễn tốt hơn.Bên cạnh đó, khi thử nghiệm với các projection-head khác nhau, kết quả loss có sự chênh lệch nhất định, nhưng không theo một quy luật đồng đều hay quá rõ ràng. Điều này gợi ý rằng, dù kiến trúc đóng vai trò quan trọng, nhưng batch-size và tỷ lệ dữ liệu huấn luyện vẫn là yếu tố có ảnh hưởng mạnh mẽ đến chất lượng học của mô hình++}

==> Từ kết quả giai đoạn này, lựa chọn các mô hình được huấn luyện với batch-size lớn nhất và có NT-Xent Loss thấp nhất theo từng projection-head:

+ ResNet-50: Mô hình **ResNet50_projection64**
+ ResNet-101: Mô hình **ResNet101_projection128**
+ Inception-V3: Mô hình **InceptionV3_projection128**
+ InceptionResNetV2: Mô hình **InceptionResNetV2_projection128**

==Giai đoạn 2: Đánh giá hiệu suất ban đầu (Linear Protocol)==

- Đây là giai đoạn với mục đích kiểm tra chất lượng biểu diễn của mô hình đã được học trong giai đoạn 1. Trong đó backbone của các mô hình trong giai đoạn 1 sẽ được giữ nguyên, chỉ thêm một lớp linear layer đơn giản để huấn luyện cho tập dữ liệu có nhãn.
- Việc sử dụng phương pháp này để đánh giá nhằm chứng tỏ rằng mô hình có biểu diễn mạnh mẽ mặc dù chỉ được huấn luyện ngay cả trên một lớp linear layer đơn giản. 
- Phương pháp này cũng đảm bảo công bằng giữa các mô hình khác nhau mà không ảnh hưởng bởi chiến lược tối ưu hóa hoặc fine-tuning đặc thù của từng mô hình. 
- Dùng tập dữ liệu $A_2$ trong giai đoạn này: Tập dữ liệu này được chia thành 3 tập nhỏ train/dev/test với tỷ lệ 70/15/15.
- Setup: Huấn luyện tất cả các mô hình với batch-size=256, learning-rate=0.2 (đạt cao hơn trong giai đoạn này giúp mô hình nhanh hội tụ, vì chỉ huấn luyện trên một lớp linear layer và các feature extractor đã được cố định, nên các tham số cần cập nhật là tương đối ít). Kết hợp early-stopping để kiểm soát quá trình huấn luyện.

| Model | Accuracy | Precision | Recall | F1-score |
| :---- | :------: | :-------: | :----: | :------: |
| ResNet50_projection64   | 0.568 | 0.58 | 0.553 | 0.561 |
| ResNet101_projection128 | 0.602 | 0.621 | 0.518 | 0.617 |
| InceptionV3_projection128 | **0.645** | 0.663 | 0.638 | **0.65** |
| InceptionResNetV2_projection128 | **0.699** | 0.719 | 0.687 | **0.698** |

==Kết luận giai đoạn 2==

--> {++Sau khi huấn luyện mô hình với một lớp linear layer đơn giản và đánh giá bởi các độ đo phân loai. Hai mô hình InceptionV3 và InceptionResNetV2 cho ra các giá trị tốt nhất++}.

==> Kết hợp với giai đoạn 1, nhận thấy rằng các mô hình có giá trị NT-Xent loss thấp thì sẽ có hiệu suất cao hơn trong giai đoạn này. Vì vậy việc lựa chọn các mô hình có giá trị loss thấp như ban đầu tương đối là hợp lý. Tiếp theo các mô hình chỉ được huấn luyện trên một lớp linear layer đơn giản nhưng cũng đã có được các hiệu suất trung bình, mô hình InceptionResNetV2 gần xấp xỉ 70% độ chính xác. 

==Giai đoạn 3: Gia đoạn downstream task==

- Giai đoạn này khác với giai đoạn 2, giai đoạn 2 chỉ huấn luyện classifier layer, ở giai đoạn này tiến hành fine-tune toàn bộ mô hình, giúp các feature extractor thích nghi với các nhiệm vụ cụ thể .

- Sử dụng tập dữ liệu $B_1$ và $B_2$ trong giai đoạn này.
- Setup: Huấn luyện tất cả các mô hình với batch-size=256, learning-rate=1e-4 (đạt nhỏ hơn, do fine-tune cả mô hình). Kết hợp early-stopping để kiểm soát quá trình huấn luyện.

| Model | Accuracy | Precision | Recall | F1-score |
| :---- | :------: | :-------: | :----: | :------: |
| ResNet50_projection64   | 0.671 | 0.629 | 0.652 | 0.6 |
| ResNet101_projection128 | **0.743** | 0.751 | 0.712 | **0.741** |
| InceptionV3_projection128 | **0.782** | 0.799 | 0.665 | **0.781** |
| InceptionResNetV2_projection128 | 0.735 | 0.742 | 0.711 | 0.728 |

==Kết luận giai đoạn 3==

--> {++So với giai đoạn chỉ huấn luyện classifier, tất cả các mô hình đều có độ chính xác cao hơn. Điều này chứng tỏ việc fine-tune toàn bộ mô hình giúp tận dụng tốt hơn các đặc trưng đã học được trong giai đoạn pretext task. Khifine-tune  toàn bộ mô hình, Precision, Recall và F1-score trở nên ổn định hơn, cho thấy mô hình có khả năng tổng quát tốt hơn trên tập dữ liệu.++}

==> InceptionV3_projection128, ResNet101_projection128 là hai mô hình tốt nhất trong số các mô hình được thử nghiệm với độ chính xác lần lượt là 75% và 78%.

^^!!! Trong các thử nghiệm này, xu hướng chung cho thấy rằng việc huấn luyện mô hình với tỷ lệ dữ liệu không nhãn càng lớn thường mang lại hiệu suất tốt hơn. Hầu hết các mô hình đều phản ánh điều này, tuy nhiên việc xác định lượng dữ liệu không nhãn "đủ" để tối ưu hóa mô hình vẫn là một thách thức.Ngoài ra, các mô hình có NT-Xent Loss nhỏ hơn thường học được biểu diễn tốt hơn, giúp chúng đạt hiệu suất cao hơn khi fine-tune trên dữ liệu có nhãn cho nhiệm vụ cụ thể. Điều này cho thấy giai đoạn pretext task đóng vai trò quan trọng trong việc cải thiện chất lượng của mô hình trong Self-Supervised Learning.^^

