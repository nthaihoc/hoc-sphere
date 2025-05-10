# Tổng hợp hiệu suất khi fine-tune BLIP/BLIP2 trên dữ liệu VQA
---

## I. Bộ dữ liệu

| STT | Bộ dữ liệu | Mô tả | Nguồn |
| :-: | :--------: | :---- | :--: |
| 01  | PathVQA | Bộ dữ liệu gồm 32.799 câu hỏi ề 4.998 hình ảnh bệnh lý. | [**[Link]**](https://huggingface.co/datasets/flaviagiammarino/path-vqa) |
| 02  | VQA-NLE-LlaVA | Bộ dữ liệu tổng hợp từ GQA với câu hỏi và câu trả lời kèm giải thích | [**[Link]**](https://huggingface.co/datasets/patrickamadeus/vqa-nle-llava) |
| 03  | BLIP_VQA_Vietnamese | Bộ dữ liệu tiếng việt với câu hỏi về phát hiện, đếm, nhận diện trong ảnh | [**[Link]**](https://huggingface.co/datasets/datnguyentien204/BLIP_VQA_Vietnamese) |
| 04  | Viet-Doc-VQA-flash2 | Bộ dữ liệu từ tài liệu giáo khoa với câu hỏi-trả lời | [**[Link]**](https://huggingface.co/datasets/5CD-AI/Viet-Doc-VQA-flash2) |

## II. Phương pháp thực nghiệm, đánh giá và kết quả

==**1. Phương pháp thực nghiệm**== 

Sử dụng mô hình BLIP/BLIP2 fine-tune cho nhiệm vụ vqa.

* Fine-tune cả hai mô hình trên 2 tập dữ liệu tiếng anh.

* Sử dụng mô hình đã được fine-tune trên hai bộ tiếng anh để fine-tune tiếp trên hai bộ dữ liệu tiếng việt.

==**2. Kết quả thực nghiệm**==

| **Bộ Dữ liệu** | **Phép Đo** | **BLIP** | **BLIP-2** |
|----------------|------------ |----------|------------|
| **PathVQA**    | Yes/No Accuracy | 81.60%    | 90.40%  |
|                           | EM       | 37.50%    | 39.00% |
| **VQA-NLE-LLaVA**         | EM        | 41.20%    | 43.80% |
|                           | BLEU-4                   | 25.0     | 27.3  |
|                           | ROUGE-L                  | 30.5     | 32.1  |
| **BLIP_VQA_Vietnamese**   | Accuracy                 | 74.80%    | 77.10% |
|                           | F1 Score                 | 78.00%     | 81.00% |
| **Viet-Doc-VQA-flash2**   | ANLS                     | 0.65     | 0.68  |

---