# Chẩn đoán quan hệ → bằng chứng thị giác → decoder

Chạy `RELATION_CONDITIONING_DIAGNOSTIC_20_2GPU.ipynb`, **Internet ON, T4 ×2, Save & Run All**. Hai GPU chia 20 ảnh, suy luận độc lập trên cùng checkpoint **F đã học**; không optimizer, backward hoặc cập nhật trọng số. Người dùng tự chạy.

## Mục đích và điều chỉnh giả định

F chưa vượt A về caption. Tuy nhiên “đưa quan hệ đúng vào decoder” cần được hiểu đúng: kiến trúc E/F dùng text làm query, còn token chuyển vào decoder là tổng có trọng số của đặc trưng ảnh. Nếu câu không đổi, lỗi có thể ở biểu diễn/khâu lấy bằng chứng, kết nối decoder, hoặc tính hợp lệ của cặp đối chiếu; không thể kết luận chỉ decoder có lỗi.

Một giả định trước đây quá mạnh: **nhãn V-COCO bằng 0 không đồng nghĩa câu ngôn ngữ tự nhiên đối lập chắc chắn sai**. Holding/carrying có thể đồng thời hợp lý; 30/80 cặp học quan hệ của F thuộc cặp này. Riding/holding cũng có thể chồng lấn tùy định nghĩa holding. Đây là nguy cơ âm không hợp lệ, chưa phải 30 cặp đã được xác minh sai. Không sửa điểm F; sửa cách diễn giải và yêu cầu xem ảnh.

Vì vậy notebook dùng tên **annotation_positive / annotation_negative**, không tuyên bố đây là oracle ngữ nghĩa thật. Chỉ các ca được người xem xác nhận phân biệt rõ mới có thể được diễn giải như đối chiếu đúng/sai, và vẫn chỉ là chẩn đoán hậu nghiệm.

## Chọn 20 ca

Lấy từ các ca wrong_predicate mà F boxed không xếp annotation dương cao hơn annotation âm. Có 22 cặp/21 ảnh đủ điều kiện; chọn bằng hash ID cố định, một cặp/ảnh, 20 ảnh. Không chọn theo kết quả phép thử mới. Tất cả thuộc caption train/restval, không dùng test. Đây là **mẫu có chủ đích sau khi thấy lỗi**, không đo hiệu năng đại diện hoặc dùng chọn checkpoint.

## Bốn điều kiện

| Điều kiện | Đầu vào module bằng chứng |
|---|---|
| off | Tắt toàn bộ token nhánh E/F; giữ base F đã học |
| upt | Prompt/box UPT như suy luận F thông thường |
| annotation_positive | Một triple S/O/R với box GT, text theo annotation dương |
| annotation_negative | Cùng box/slot/S/O, chỉ đổi text R sang annotation âm |

**So sánh có kiểm soát là positive với negative.** So annotation với UPT còn khác box và số slot, không thể quy lợi ích cho riêng quan hệ đúng.

Đo cả caption greedy, logP của hai câu mô tả (tổng và trung bình theo token, gồm EOS), độ đổi attention R và token bằng chứng R. Chênh logP là độ nhạy của mô hình, chưa phải đúng ngữ nghĩa hay truy nguồn từng từ. Không dùng khớp từ khóa để tự chấm caption.

Kiểm tra không đổi base memory và token S/O trong cặp có kiểm soát; replay điểm matcher F với sai số <1e-4; kiểm tra hash toàn model trước/sau và giữa hai GPU. Không có ngưỡng “đủ cải thiện” được chọn sau khi thấy kết quả. Đọc các đại lượng liên tục cùng ảnh/caption thực tế.

## Input

- **Checkpoint F cuối**, SHA `4938bfe875f05104cd0afca4918dad3761b181bece08d773ad2b71b6adb51043`. Không dùng E/bin cũ hoặc checkpoint nền.
- `visual-cache`, legacy `prompt-cache`, `upt-prompts-full-all-actions-v2`, MSCOCO2014 (JSON **và ảnh**).
- Không cần object detections, CHAIR, raw UPT hoặc nhãn V-COCO riêng.

Đường dẫn mặc định checkpoint là `/kaggle/working/semantic_evidence_verified_pairs_F/checkpoint.pth` nếu chạy cùng session F. Nếu mở notebook mới, thêm Saved Output/dataset chứa F rồi sửa **CHECKPOINT** bằng đường dẫn thực tế; không dò tìm hay tải lại version notebook. Các Input còn lại giữ như trước. COCO_IMAGES mặc định `/kaggle/input/datasets/vuthetam/mscoco-2014/images`.

Notebook kiểm tra đủ 20 ảnh trước suy luận. Nó chỉ đọc 20 ảnh cần cho review; việc Kaggle mount dataset vẫn do Kaggle quản lý. Notebook khoảng 50 KB, nhãn/mã được nén kèm hash để tránh lỗi nguồn >1 MB.

## Đọc kết quả

- Token R gần như không đổi khi đổi text: ưu tiên xem biểu diễn/khâu lấy bằng chứng; chưa quy lỗi cho decoder.
- Token R đổi nhưng likelihood/caption phản ứng yếu: phù hợp với giả thuyết decoder dùng tín hiệu này yếu, chưa chứng minh nguyên nhân duy nhất.
- Hai mô tả đều hợp lý theo ảnh: loại khỏi kết luận đúng/sai ngữ nghĩa, giữ riêng để báo cáo độ mơ hồ của annotation.
- Annotation dương làm câu có vẻ tốt hơn chỉ trong vài ảnh: chưa đủ chạy full hoặc tuyên bố đóng góp.

Gửi lại `relation_conditioning_diagnostic_20_review.zip`. ZIP có đủ **20 ảnh + HTML có box + caption bốn điều kiện + scores**. Mở `case_review_20.html`, có thể ghi nhận xét và tải JSON; không tự xem nhận xét của assistant là nhãn người.

## Kiểm tra local

`local_validation.json` ghi kiểm tra dữ liệu, kiểm soát can thiệp và payload notebook. Máy local không có PyTorch, chưa chạy model hoặc GPU; notebook có CPU synthetic preflight bắt buộc và kiểm tra trọng số bất biến lúc suy luận.

Chỉ có 7/20 ảnh sẵn trong output cũ ở máy local; đã tạo `case_review_available.html` với đúng 7 ảnh có sẵn, không để ảnh hỏng. Các ảnh/HTML local này không đưa lên GitHub. Notebook Kaggle xuất đủ 20 ảnh từ dataset người dùng gắn vào. Chưa hoàn thành xác minh trực quan cả 20 ca.
