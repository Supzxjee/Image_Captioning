# Kết quả F: chưa đạt điều kiện chạy full

F chạy đủ **1.250 cập nhật trên hai Tesla T4**, đúng bộ nhãn và giao thức đã khóa. Kiểm tra artifact PASS. Checkpoint nguồn, cohort, reward targets, caption nguồn và runtime khớp A/E. Đây không phải lỗi thiếu cặp loss hoặc gradient bằng không.

**Lưu ý diễn giải sau khi rà ca lỗi:** “annotation âm xác nhận” trong thiết kế F chỉ có nghĩa nhãn actor/action V-COCO bằng 0, không bảo đảm câu mô tả đối lập sai theo ngôn ngữ tự nhiên. Ví dụ holding và carrying có thể đồng thời hợp lý; 30/80 cặp sai quan hệ fit thuộc cặp hành động này. Điểm matcher dưới đây đo khớp annotation, không tự động là độ đúng ngữ nghĩa. Chẩn đoán tiếp theo cần kiểm tra trực quan; đây là điểm yếu trong giả định tạo âm của chúng ta, chưa phải lỗi nhãn đã được xác minh. Các số caption và quyết định chưa chạy full không thay đổi.

## Caption: chưa vượt A

Cùng 1.000 validation, raw greedy, seed42, không dùng test:

| Nhánh | CIDEr ↑ | Caption có ảo giác ↓ | Recall đối tượng ↑ |
|---|---:|---:|---:|
| A — huấn luyện tiếp nền | **1.205513** | 52/1000 | 45.804% |
| E — loss giả nhãn | 1.196816 | 56/1000 | 46.326% |
| F — loss cặp xác nhận | 1.198510 | 52/1000 | 46.296% |
| F tắt module bằng chứng | 1.197899 | **50/1000** | 46.296% |

F−A CIDEr = **−0.007003**, khoảng bootstrap ghép cặp 95% từ điểm CIDEr đã lưu **[−0.024844; +0.011496]**. F−E = +0.001694, khoảng **[−0.015219; +0.019013]**. Chưa có bằng chứng F tốt hơn A hoặc E đáng tin cậy; cũng không kết luận F kém A có ý nghĩa thống kê từ một seed.

Bật module so với tắt chỉ thêm **0.000611 CIDEr**, khoảng **[−0.005315; +0.006874]**, nhưng tăng 2 caption có ảo giác. Không dùng việc F hơn checkpoint nguồn (1.189066) làm bằng chứng đóng góp vì A cũng được huấn luyện tiếp và đạt cao hơn F.

Nhóm 269 ảnh có quan hệ: F hơn A +0.009460 CIDEr, khoảng **[−0.025534; +0.043107]**; không đủ bằng chứng lợi ích riêng cho quan hệ. Điểm CIDEr nhóm này cũng không trực tiếp đo đúng động tác.

## Matcher: cải thiện gán người, chưa học được quan hệ đủ tốt

Đúng cặp / tổng cặp, boxed; hòa tính 0.5. “Trước” là module F khởi tạo ngẫu nhiên, **không phải E đã huấn luyện**:

| Bộ | Bài toán | Trước F | Sau F | Chỉ dùng IoU box |
|---|---|---:|---:|---:|
| Calibration | Sai quan hệ | 11/24 | **9/24** | 12/24, toàn hòa |
| Calibration | Sai người | 56/116 | **90/116** | 104.5/116 |
| Probe cũ | Sai vật | 7/12 | **6/12** | 12/12 |
| Probe cũ | Sai quan hệ | 7/12 | **5/12** | 6/12, toàn hòa |
| Probe cũ | Sai người | 24/38 | **26/38** | 34/38 |

Calibration có nhiều cặp/ảnh: macro theo ảnh cho sai người tăng **50.25% → 71.14%**, nhưng IoU đạt **88.25%**. Vì vậy không được dùng riêng 90/116 để tuyên bố mô hình hiểu ngữ nghĩa thị giác hơn đối chứng hình học.

Trên cùng probe cũ, E đã huấn luyện từng đạt vật **8/12**, quan hệ **8/12**, người **27/38**; F không vượt các con số đó. Mẫu nhỏ và probe có thiên lệch hình học nên đây chỉ là chẩn đoán.

Loss căn chỉnh trung bình giảm **0.18277 → 0.02074**, tỷ lệ margin dương trên các batch fit tăng **62.63% → 96.75%** (100 bước đầu so với 250 bước cuối). Cả sáu nhóm tham số module có gradient khác 0 ở đủ 2.500 rank-step. Độ chính xác fit cao nhưng heldout quan hệ thấp gợi ý khả năng khái quát kém; chưa xác định được nguyên nhân duy nhất hoặc chứng minh chỉ cần học lâu hơn.

## Caption có dùng module không?

- Tắt module đổi **73/1000** caption, trong đó **15/269** ảnh có quan hệ.
- Tráo text đổi **3/1000** caption; NLL tham chiếu trung bình tăng khoảng **0.000017**.
- Tráo vùng đổi **55/1000** caption.

Có tác động đến đầu ra, nhưng chưa có lợi ích caption rõ và độ nhạy với text rất thấp trong phép thử này. Không thể suy ra decoder hoàn toàn bỏ prompt, hoặc gán nguồn từng từ từ các phép tráo này. Tráo text/vùng không phải nhãn âm quan hệ đã được xác nhận.

## Quyết định

**Giữ A làm đối chứng tốt nhất của loạt pilot; không chạy full F hoặc thêm seed F lúc này.** Nhãn xác nhận đã giúp một bài toán gán người, nhưng chưa đạt mục tiêu quan hệ/caption. Dừng việc tăng quy mô loss F hiện tại.

Bước thiết kế tiếp cần giải quyết hai yêu cầu: phân biệt động tác trên cùng người/vật bằng bằng chứng tương tác, và khiến decoder dùng sự khác biệt đó khi sinh cụm quan hệ. Trước khi đề xuất notebook mới, cần kiểm tra vài ca sai quan hệ thực tế và khả năng biểu diễn của CLIP patch/vùng hiện tại; không mặc định thêm loss/epoch sẽ sửa được.

## Phạm vi xác minh

Đã kiểm tra CRC ZIP, hash mã/nhãn/protocol, fit-only schedule, tách ảnh, 2 lịch sử 1.250 bước, loss/reward arithmetic, gradient norm giữa hai rank, các log preflight; đối chiếu ảnh/reference/caption nguồn A/E/F. Đã tổng hợp lại CHAIR từ parser output, CIDEr từ per-image score, toàn bộ matcher và độ nhạy từ prediction/score lưu trong ZIP. Auditor chung được kiểm tra hồi quy A/E: kết quả giống hệt báo cáo đã lưu.

Chưa chạy lại inference, tokenizer hoặc metric evaluator độc lập; ZIP không có checkpoint tensor/HDF5 để kiểm chứng lại trọng số cuối. Bootstrap dùng điểm CIDEr với IDF cố định đã lưu, không đo bất định do seed. F có thêm nhãn/ảnh phụ/tính toán; không cùng tổng ngân sách giám sát A/E. Validation và các bộ matcher từng được nghiên cứu trước, không phải test hoàn toàn mới. CHAIR đánh giá đối tượng, không đánh giá quan hệ.

Nguồn máy đọc: `F_result.json`, `F_training_diagnostics.json`; auditor: `audit_F_review.py`.
