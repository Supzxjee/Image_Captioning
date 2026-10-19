# UPT alignment pilot: một nhánh/notebook, hai GPU/nhánh

Có bốn notebook độc lập, đã nhúng đủ mã mới. **Không phải chia GPU 0 cho một nhánh và GPU 1 cho nhánh khác.** Hai GPU trong cùng notebook đồng bộ gradient bằng DDP để huấn luyện cùng một mô hình.

| Đợt | Notebook của bạn | Notebook của Ngân |
|---|---|---|
| 1 | `UPT_ALIGNMENT_A_2GPU.ipynb`: huấn luyện tiếp baseline | `UPT_ALIGNMENT_B_2GPU.ipynb`: thêm vùng ảnh, không thêm text UPT |
| 2 | `UPT_ALIGNMENT_C_2GPU.ipynb`: vùng ảnh + text, cộng trực tiếp | `UPT_ALIGNMENT_D_2GPU.ipynb`: text điều khiển cổng cho từng thành phần thị giác |

Chạy A/B trước và gửi hai ZIP review. Đọc kết quả rồi mới quyết định chạy C/D; chưa cần tiêu quota cho cả bốn ngay.

## Cách chạy

1. Import đúng một tệp `.ipynb` vào mỗi tài khoản Kaggle; chọn **GPU T4 ×2**, bật **Internet**.
2. Thêm sáu nguồn: checkpoint SCST object w05 legacy, prompt-cache `.pt` cũ, UPT prompt JSON mới, visual-cache HDF5, MSCOCO JSON, objectdetectioncache JSON. Tài khoản Ngân phải được quyền truy cập các dataset riêng tư. Sao chép đường dẫn mount thực tế vào cell cấu hình nếu khác.
3. Không cần raw UPT cache hoặc thư mục ảnh. Nếu thêm MSCOCO dataset thì Kaggle vẫn có thể chuẩn bị toàn bộ dataset trước khi Python bắt đầu; mã chỉ đọc `dataset_coco.json`.
4. Chọn **Save & Run All**. Không chạy lại các cell sau khi đã hoàn tất trong cùng session: mã chủ động từ chối ghi đè Output/stage cũ. Để chạy lại, bắt đầu session mới hoặc đổi cả OUTPUT và stage.
5. Khi hiện `PILOT_COMPLETE`, tải `upt_alignment_pilot_A_review.zip` / `..._B_review.zip` gửi lại. Giữ `checkpoint.pth` riêng trong Saved Output; ZIP review chỉ chứa báo cáo, dự đoán, lịch sử và mã nguồn.

Không cần tải thêm `.py` lên Kaggle. Repository và bộ CHAIR nằm trong `/tmp`; Output chỉ lưu thí nghiệm, tránh vấn đề hàng trăm tệp GitHub che mất checkpoint khi tạo dataset.

Checkpoint lưu kiểu `.pth` không phải ZIP để tránh Kaggle tự giải nén. Đây là định dạng mới `upt_alignment_pilot_v1`, chứa cả baseline và nhánh phụ trong `model_state_dict`; khi đánh giá tiếp cần dựng `PilotModel` tương ứng rồi load state, không đưa thẳng vào loader checkpoint baseline cũ.

## Module và loss thực sự làm gì?

Giữ nguyên Q-Former, decoder và **legacy prompt cache** đang dùng trong baseline. UPT JSON mới được thêm ở nhánh phụ, không thay thế cache cũ.

Mỗi quan hệ đã chọn có ba token: vùng chủ thể S, vùng đối tượng O, vùng tương tác R. R ghép đặc trưng S/O/vùng bao cả hai và hình học bbox. Mỗi vùng lấy trung bình các CLIP patch có tâm nằm trong bbox; đây là pooling thô trên lưới 14×14, không phải ROIAlign hay bộ xác minh quan hệ. Giả định visual cache được tạo bằng Resize(224,224) như pipeline hiện có. Tối đa ba quan hệ/ảnh; ảnh không có quan hệ dùng tối đa tám object, ảnh không có object được mask toàn bộ nhánh mới.

- A: giữ mô hình gốc, huấn luyện tiếp với ngân sách tương đương.
- B: chiếu các vùng ảnh thành memory token cho decoder; không thêm văn bản UPT. B vẫn dùng box được UPT lựa chọn, nên B không phải đối chứng độc lập hoàn toàn với UPT.
- C: cộng embedding CLIP text đóng băng với token vùng tương ứng trước khi thêm vào memory.
- D: `g = sigmoid(MLP([v,t,v*t]))`; thêm `g*LayerNorm(v)` vào memory. Cổng riêng theo S/O/R, khởi tạo 0,5. Đặt cổng sau LayerNorm để không bị chuẩn hóa triệt tiêu. Cổng này học mức hữu ích cho caption, chưa phải xác suất quan hệ đúng.

Loss chung sau warmup: **L = L_CE + 0,05 L_SCST**, với **R = CIDEr-D + 0,5 R_object** và **R_object = 0,5 recall đối tượng nổi bật − số loại đối tượng không được hỗ trợ**. SCST dùng chênh lệch thưởng câu lấy mẫu và câu greedy. 200 cập nhật đầu chỉ CE; không thêm loss quan hệ V-COCO hoặc contrastive loss trong pilot này. Chưa thể tuyên bố đã học được căn chỉnh đúng chỉ vì loss giảm.

## So sánh công bằng

Cùng checkpoint đầu vào, cohort, seed 42, LR nền 5e-6, LR nhánh mới 1e-4, một lượt 10.000 ảnh train, 1.250 cập nhật, batch toàn cục 8 = 4/GPU. Tham chiếu caption và nhãn thưởng chỉ lấy từ train. IDF thưởng CIDEr-D dùng toàn bộ train như pipeline baseline. Validation dùng 1.000 ảnh từ Karpathy val, không dùng test. Ảnh được lấy theo hash filename, không ưu tiên ảnh có quan hệ.

Cohort đã khóa: train 2.706 ảnh có quan hệ / 7.226 object-only / 68 no-objects; val 269 / 722 / 9. Không dùng nhóm 9 ảnh no-objects để kết luận thống kê mạnh.

Giải mã **greedy** cho cả checkpoint ban đầu và checkpoint cuối của mỗi nhánh. Điểm này không so trực tiếp với điểm test cũ dùng beam/rerank. Báo cáo CIDEr/BLEU/METEOR/ROUGE, CHAIR và CIDEr theo nhóm ảnh. CHAIR chỉ đánh giá đối tượng, không đánh giá đúng/sai của predicate. Bộ val đã được dùng trong lịch sử nghiên cứu, không phải test độc lập chưa từng xem.

Với B/C/D, thêm đánh giá tắt nhánh (có điểm chất lượng), tráo text, tráo vùng (độ đổi caption và NLL tham chiếu). Tráo theo ảnh khác cùng mode và giữ mask; không coi câu tráo là nhãn quan hệ sai chắc chắn. Với B, tráo text phải không tác động về mặt cấu trúc. Caption thay đổi nhiều chưa có nghĩa là tốt hơn.

Ưu tiên kết luận: B−A cho biết thêm vùng có hữu ích không; C−B cho biết text có ích không; D−C so sánh hai cách dùng text. C/D khác đường truyền và số tham số, nên chưa tách riêng được tác dụng của “gating” bằng phép so sánh này. Chỉ chuyển sang chạy lớn khi gain caption ổn định và CHAIR không xấu đi; pilot một seed chưa chứng minh được điều đó.

Hai tài khoản phải dùng dữ liệu trùng hash. So sánh thêm `cohort_sha256`, `initial_base_sha256`, dự đoán `origin` và runtime trong hai ZIP. Khác GPU/phần mềm có thể gây sai khác số học; nếu checkpoint ban đầu đã khác điểm đáng kể thì chưa quy gain cho module.

## Kiểm tra đã làm và giới hạn

Local đã kiểm tra cú pháp, ba unit test dữ liệu và box/instance link trên đúng 11.000 mục prompt thật. **Chưa chạy notebook, tải mô hình hoặc huấn luyện trên máy người dùng.** Unit test mô hình, smoke DDP hai GPU và smoke metric được nhúng để chạy trên Kaggle trước khi huấn luyện. Đây là mã pilot mới, chưa có số đo caption mới.
