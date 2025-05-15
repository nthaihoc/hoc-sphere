# Triển khai cơ chế Attention trong LLMs: Từ Self-Attention đến Multi-Head Attention

Trong bài giảng trước, chúng ta đã cùng thảo luận và các bạn đã biết cách chuẩn bị văn bản đầu vào để huấn luyện các mô hình ngôn ngữ lớn bằng cách chia nhỏ văn bản thành các token và subword, sau đó thực hiện mã hóa chúng thành các biểu diễn vector - hay còn gọi là embedding. Ngay bây giờ, chúng ta sẽ tìm hiểu một phần cốt lõi, trái tim của hầu hết các kiến trúc LLM - đó là cơ chế chú ý (attention mechanisms), như được minh họa trong **hình 3.1**. Chúng ta sẽ chủ yếu tìm hiểu các cơ chế attention một cách riêng biệt và tập trung vào cách thức hoạt động. Tiếp theo, tiến hành lập trình các phần còn lại của LLM xung quanh cơ chế self-attention (tự chú ý) để quan sát cách chúng hoạt động và xây dựng một mô hình có khả năng sinh văn bản.

<figure style="text-align: center;">
  <img src="" alt="Mô hình LLM" style="width: 50%; height: auto;">
  <figcaption><strong>Hình 3.1:</strong>Những giai đoạn quan trọng để xây dựng một mô hình LLM.</figcaption>
</figure>

Có rất nhiều biến thể khác nhau của cơ chế attention, trong đó điển hình là bốn (như được minh họa trong **Hình 3.2**). Những biến thể này được xây dựng kế tiếp nhau, và mục tiêu đạt được đó là hiện thực hóa cơ chế multi-head attention (chú ý đa đầu) một cách hiệu quả và dễ hiểu, đây là cơ chế nền tảng để tích hợp chúng vào kiến trúc LLM sẽ được lập trình trong chương tiếp theo.

<figure style="text-align: center;">
  <img src="" alt="Mô hình LLM" style="width: 50%; height: auto;">
  <figcaption><strong>Hình 3.2:</strong>Những biến thể điển hình của cơ chế attention sẽ được thảo luận trong chương này.</figcaption>
</figure>

==**3.1 Vấn đề khi xử lý các chuỗi dài của các mô hình xử lý ngôn ngữ truyền thống.**==

Trước khi đi lần lượt vào cơ chế self-attention, đầu tiên hãy cùng xem xét vấn đề mà các kiến trúc trước thời kỳ LLM gặp phải (những mô hình không sử dụng attention).

Giả sử chúng ta muốn xây dựng một mô hình dịch ngôn ngữ (language translation) nhằm chuyển đổi văn bản từ một ngôn ngữ này sang một ngôn ngữ khác. Nghĩ một cách nôm na, ta chỉ cần làm cho mô hình đơn giản nhất, là dịch từng từ một (minh họa trong **Hình 3.3**). Tuy nhiên điều này là không thể, bởi vì cấu trúc ngữ pháp giữa các ngôn ngữ gốc và ngôn ngữ đích thường rất khác nhau.

<figure style="text-align: center;">
  <img src="" alt="Mô hình LLM" style="width: 50%; height: auto;">
  <figcaption><strong>Hình 3.3:</strong>Mô phỏng mô hình dịch ngôn ngữ dịch từ tiếng việt sang tiếng anh.</figcaption>
</figure>

Để giải quyết vấn đề này, các nhà nghiên cứu đã sử dụng mô hình transformer - một mạng nơ-ron sâu bao gồm hai mô-đun con: encoder (bộ mã hóa) và decoder (bộ giải mã). Nhiệm vụ của encoder là đọc và xử lý toàn bộ văn bản đầu vào, sau đó decoder sẽ tạo ra văn bản đã được dịch.

Trước khi các mô hình transformer ra đời, các mạng nơ-ron hồi tiếp (RNN - Recurrent Neural Networks) là kiến trúc encoder-decoder phổ biến nhất trong các mô hình dịch ngôn ngữ. RNN là một loại mạng nơ-ron trong đó đầu ra của bước trước sẽ là đầu vào cho bước hiện tại, điều này làm chúng phù hợp để xử lý dữ liệu tuần tự như văn bản. 

Trong kiến trúc encoder-decoder của RNN, văn bản đầu vào được đưa vào encoder, nơi chúng được xử lý tuần tự từng bước. Encoder sẽ cập nhật trạng thái ẩn (hidden state) của dữ liệu đầu vào tại mỗi bước, mục tiêu chính là năm bắt toàn bộ ý nghĩa của câu đầu vào trong trạng thái ẩn cuối cùng như được minh họa trong **Hình 3.4**.

Decoder sau đó sẽ sử dụng các trạng thái ẩn cuối cùng mà encoder tạo ra, để bắt đầu tạo ra câu dịch từng từ một. Chúng cũng cập nhật luôn các trạng thái ẩn ở mỗi bước - trạng thái này có thể coi là những thông tin ngữ cảnh quan trọng, cần thiết để đưa ra dự đoán từ kế tiếp.

<figure style="text-align: center;">
  <img src="" alt="Mô hình LLM" style="width: 50%; height: auto;">
  <figcaption><strong>Hình 3.4:</strong>Bộ giải mã encoder-decoder của RNN khi xử lý dữ liệu đầu vào và sinh ra câu dịch.</figcaption>
</figure>

Ý tưởng cốt lõi của kiến trúc transformer ở đây là: phần encoder sẽ đảm nhiệm xử lý toàn bộ văn bản đầu vào thành một trạng thái ẩn - hay còn gọi là bộ nhớ tạm (memory cell). Decoder sau đó sẽ sử dụng trạng thái ẩn này để sinh đầu ra. Bạn có thể xem trạng thái ẩn như một vector embedding, khái niệm mà đã được nhắc tới trong chương 2.

Điểm hạn chế lớn của mô hình encoder-decoder RNN là: trong giai đoạn giải mã, chúng không thể truy cập trực tiếp vào các trạng thái ẩn trước đó từ encoder. Do vậy, chúng chỉ dựa vào trạng thái ẩn hiện tại, trạng thái này có thể coi là gói toàn bộ thông tin ngữ cảnh. Điều này có thể dẫn đến việc mất ngữ cảnh, đặc biệt là với những câu phức tạp có các mối liên hệ dài giữa các thành phần trong câu. Nhưng sự xuất hiện của những hạn chế này lại là động lực để các cơ chế attention ra đời.

==**3.2 Nắm bắt mối quan hệ phụ thuộc dữ liệu bằng cơ chế attention.**==

Mặc dù RNN hoạt động khá tốt khi dịch các câu ngắn, nhưng chúng không hoạt động hiệu quả với các văn bản dài, do không thể truy cập trực tiếp vào các từ trước đó trong đầu vào. Một hạn chế lớn của cách tiếp cận này là: RNN phải ghi nhớ toàn bộ câu đã được mã hóa trong một trạng thái ẩn duy nhất trước khi chuyển đến decoder (**Hình 3.4**).

Vào năm 2014, các nhà nghiên cứu đã phát triển cơ chế attention Bahdanau cho RNN - được đặt theo tên tác giả đầu tiên của bài báo. Cơ chế này chỉnh sửa kiến trúc encoder-decoder, cho phép decoder có thể chọn lọc truy cập vào các phần tử khác nhau của chuỗi đầu vào tại mỗi bước giải mã, minh hoạ trong **Hình 3.5**.