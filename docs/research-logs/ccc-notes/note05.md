# Nghiên cứu kỹ thuật Ensemble Learning cho phân loại ung thư cổ tử cung tế bào học
---

## Nội dung

- [**I. Tổng quan về đề tài**](#i-mở-đầu)
- [**II. Cơ sở lý thuyết về Ensemble Learning**](#ii-cơ-sở-lý-thuyết-vể-ensemble-learning)
- [**III. Phương pháp thực nghiệm**](#iii-phương-pháp-thực-nghiệm)
- [**IV. Kết quả thực nghiệm và đánh giá**]()

---

## I. Mở đầu

**Giới thiệu đề tài:**

- Ung thư cổ tử cung là căn bệnh nguy hiểm gây tử vong cao, chúng được xem là nỗi sợ hãi của con người, đặc biệt là phụ nữ.
- Các phương pháp truyền thống sử dụng Pap smear, HPV và sinh thiết mô để chẩn đoán và phát hiện, nhưng vẫn tồn tại một số hạn chế về độ nhạy, thời gian chờ kết quả lâu, phụ thuộc vào kinh nghiệm của các bác sĩ.
- Chính vì những hạn chế trên, các phương pháp trí tuệ nhân tạo-AI đang được nghiên cứu và áp dụng để giảm thiểu đi những hạn chế này.

**Mục tiêu chính:** Tập trung đề xuất phát triển một mô hình AI dựa trên kỹ thuật Ensemble Learning nhằm hỗ trợ dự đoán và phân loại UTCTC thông qua hình ảnh tế bào học.

**Phạm vi nghiên cứu:**

- Tập trung vào ứng dụng kỹ thuật Ensemble Learning (EL) để chẩn đoán UTCTC thông qua phân tích hình ảnh tế bào học.
- Kết hợp giữa nghiên cứu lý thuyết và thực nghiệm nhằm đảm bảo tính toàn diện và khoa học cho đề tài. 

**Phương pháp nghiên cứu:**

{++*/ Nghiên cứu lý thuyết:++} Tìm hiểu các nghiên cứu trước đây sử dụng Deep Learning trong phân loại UTCTC. Phân tích hạn chế của mô hình đơn lẻ và lý do cần thiết phải áp dụng học tập kết hợp. Nghiên cứu các chiến lược kết hợp trong Ensemble Learning. Xây dựng và xác định hướng đi thực nghiệm phù hợp.

{++*/ Nghiên cứu thực nghiệm:++} Thực hiện quy trình tiền xử lý hình ảnh tế bào, huấn luyện các mô hình đơn lẻ và kết hợp. Đánh gía và so sánh hiệu suất bằng các chỉ số: Accuracy, Precision, Recall, F1-score. Đề xuất chiến lược tối ưu EL để áp dụng vào trong thực tế.

## II. Cơ sở lý thuyết vể Ensemble Learning

**Khái niệm:** Học tập kết hợp là kỹ thuật trong học máy mà kết hợp nhiều mô hình đơn lẻ, để tạo ra một mô hình tổng hợp mạnh mẽ hơn. Một số loại phổ biến là Bagging, Boosting, Stacking.

**Stacking:** Đây là kỹ thuật mà trong đó nhiều mô hình học (Level 0) sẽ "đua nhau dự đoán", rồi một mô hình khác (Level 1 - gọi là meta-model) sẽ học từ các kết quả đó để đưa ra kết quả cuối cùng.

## III. Phương pháp thực nghiệm

==**1. Dữ liệu nghiên cứu**==

{++*/ Tập dữ liệu++}

- Sử dụng bộ dữ liệu UTCTC của bệnh viện A-Thái Nguyên. Bộ dữ liệu này được xây dựng thông qua quá trình thu thập và gán nhãn thủ công bởi bác sĩ chuyên gia.
- Bộ dữ liệu có 22.434 hình ảnh, gồm 5 nhãn chính: ASC_H, ASC_US, HSIL, LSIL, SCC.

{++*/ Quy trình tiền xử lý dữ liệu++}

- Loại bỏ dữ liệu không phù hợp: loại bỏ ảnh mờ nhiễu, kích thước bất thường.
- Tăng cường dữ liệu: xoay ảnh, lật ảnh, phóng to/thu nhỏ, dịch ảnh.
- Chuẩn hóa giá trị điểm ảnh: chuẩn hóa cường độ pixel, đảm bảo đồng nhất về màu sắc và phân bổ giá trị.

{++*/ Chiến lược chia dữ liệu++}

- Áp dụng stratified split (chia phân tầng) duy trì tỷ lệ gốc, hạn chế mất cân bằng lớp.
- Dữ liệu chia thành 3 tập:
    - Train set (80%): dùng để huấn luyện mô hình, kết hợp với augumentation.
    - Dev set (10%): dùng để theo dõi hiệu suất trong quá trình điều chỉnh mô hình đơn / nhãn được dự đoán trên tập này được sử dụng làm đầu vào cho mô hình meta.
    - Test set (10%): đánh giá mô hình đơn lẻ và ensemble, kiểm tra khả năng tổng quát và ovefitting.

==**2. Thiết kế mô hình**==

{++*/ Lựa chọn mô hinh cơ sở++} Trong nghiên cứu, 6 mô hình CNN được sử dụng làm mô hình cơ sở cho kỹ thuật Ensemble Learning:

- MobileNetV2 - nhẹ, tối ưu cho thiết bị hạn chế tà nguyên.
- VGG16 - lâu đời nhưng vẫn hiệu quả, đặc biệt khi fine-tune.
- ResNet101, InceptionV3, InceptionResNetV2, Xception - mô hình sâu, mạnh tỏng trích xuất đặc trưng hình ảnh.

Tiêu chí lựa chọn mô hình:

- Hiệu quả cao trong bài toán phân loại ảnh y tế.
- Cân bằng giữa độ chính xác và tốc độ xử lý.
- Đa dạng kiến trúc, giúp tăng tính tổng quát khi kết hợp trong hệ EL.

{++*/ Lựa chọn mô hình meta++}

**Stacking:** Kết hợp đầu ra từ các mô hình cơ sở bằng mô hình học máy đơn giản. Các mô hình meta được lựa chọn bao gồm:

- Logistic regression: tổng hợp đầu ra tuyến tính, phù hợp cho bài toán phân loại nhị phân.
- SVM: tối ưu hóa ranh giới quyết định cho dữ liệu phân tách tốt.
- Random Forest: ổn định, giảm phương sai thông qua tổ hợp nhiều cây quyết định.
- Naive Bayes: đơn giản, hiệu quả với dữ liệu có đặc trưng độc lập.
- KNN: Phân loại dựa trên các mẫu láng giềng gần nhất, linh hoạt và thích ứng nhanh.

**Voting:** Kết hợp đầu ra từ nhiều mô hình cơ sở bằng hard voting - mô hình đưa ra quyết định theo đa số phiếu bầu.

==**3. Chiến lược huấn luyện**==

{++*/ Đào tạo mô hình cơ sở:++} Thí nghiệm huấn luyện mô hình trên các kích thước đầu vào khác nhau: 128/224/256.

{++*/ Đào tạo mô hình meta++} Sau khi quá trình huấn luyện các mô hình cơ sở kết thúc, sử dụng các kỹ thuật ensemble learning để học những dự đoán đầu ra từ các mô hình đơn lẻ.

Giả sử có mẫu dữ liệu dầu vào $x$, các mô hình đơn lần lượt đưa ra dự đoán:

- $\text{MobileNetV2}(x) = 0$
- $\text{InceptionV3}(x) = 1$
- $\text{InceptionResNetV2}(x) = 2$
- $\text{ResNet101}(x) = 2$
- $\text{VGG16}(x) = 0$
- $\text{Xception}(x) = 0$

**Voting:** 3/6 mô hình dự đoán $x$ thuộc về lớp 0 -> chọn lớp 0.

**Stacking** Coi vector dự đoán lần lượt cho mẫu $x$ là một điểm dữ liệu mới $\mathbf{x}=[0, 1, 2, 2, 0. 0]$, $y$ là nhãn thực tế -> đưa vào mô hình meta để huấn luyện.

## IV. Kết quả thực nghiệm và đánh giá

==**1. Đánh giá mô hình cơ sở**==

- Khi tăng kích thước ảnh từ 128, 224, 256 hiệu suất của hầu hết mô hình đều cải thiện, tuy nhiên không đồng nhất. Kích thước 224 mang lại hiệu suất tối ưu cho hầu hết các mô hình, cân bằng giữa độ chính xác và chi phí tính toán.
- InceptionResNetV2 và InceptionV3 đạt độ chính xác cao nhất trên mọi kích thước ảnh. 

==**2. Đánh giá mô hình Ensemble Learning**==

- Việc áp dụng EL giúp cải thiện độ chính xác, độ nhạy và F1-score so với mô hình đơn lẻ.
- Kích thước 224 cũng là tối ưu khi áp dụng EL, vì mọi mô hình đều có hiệu suất tốt.
- SVM và Logistic là hai mô hình đạt hiệu quả tốt nhất.

==**3. Nhận xét chung**==

- Kích thước ảnh và lựa chọn mô hình ảnh hưởng rõ rệt đến hiệu suất phân loại. Kích thuớc ảnh 224 cho kết quả tốt nhất trong các thực nghiệm.
- Ensemble Learning giúp cải thiện hiệu suất tổng thể, nhưng không phải lúc nào cũng vượt trội so với từng mô hình đơn lẻ.

==**4. Hạn chế và hướng nghiên cứu tương lai**==

- Giới hạn kỹ thuật EL. 
- Hạn chế về số lượng mô hình học sâu. 
- Vấn đề về mất cân bằng dữ liệu.
- Chia dữ liệu chưa tối ưu.
- Yêu cầu cao về tài nguyên tính toán.
- Mô hình hộp đen. 

---