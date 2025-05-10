## 3.1 Mô hình gặp vấn đề trong xử lý các chuỗi dài.

Trước khi chúng ta đi sâu vào cơ chế tự chú ý (self-attention) - cốt lõi của các mô hình ngôn ngữ lớn, hãy cùng phân tích và thảo luận về các vấn đề mà các kiến trúc trước đây gặp phải, vốn không có cơ chế tự chú ý. Giả sử chúng ta muốn phát triển một mô hình dịch ngôn ngữ, mô hình này có nhiệm vụ dịch văn bản từ một ngôn ngữ nào đó sang một ngôn ngữ khác. [image], chúng ta không thể chỉ đơn giản dịch từng từ một do sự khác biệt về mặt cấu trúc ngữ pháp giữa ngôn ngữ nguồn và ngôn ngữ đích.

