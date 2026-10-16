# Kiểm tra tái lập SCST+CE: 32 batch, hai lần trên cùng GPU

Trạng thái: đã chuẩn bị mã; chưa chạy QFormer trên GPU. Đây là kiểm tra kỹ thuật trước khi sửa hàm thưởng, không phải kết quả cải thiện caption.

## Chạy thủ công

Trong notebook **QFormer UPT Val5000 20261015**, thay cell launcher cũ bằng nội dung `kaggle_scst_replay_32.py`, hoặc upload `SCST_CE_SAME_GPU_REPLAY_32.ipynb`. Bật Internet và GPU. T4 x2 cũng được, nhưng kiểm tra này chỉ dùng GPU 0; hai worker chạy **tuần tự trên cùng GPU**, mỗi worker là một tiến trình mới. Người dùng tự Run/Save & Run All.

Giữ các input checkpoint QFormer epoch10, prompt cache, visual cache và MSCOCO đang có. Không cần thêm input; không tải CLIP weights, không copy ba HDF5 sang `/tmp`. Các đường dẫn nằm đầu launcher để sửa trực tiếp. OUTPUT phải mới, tránh ghi đè kết quả trước.

## Thiết kế cố định

- Cùng checkpoint gốc đã khóa SHA, cùng 20.000 ảnh train/restval, seed42, batch8, AdamW5e-6, FP32, dropout off.
- Giữ objective `CE + 0.05 * SCST(CIDEr)`; không có object reward hay parser mới.
- Mỗi lần chỉ thực hiện **32 optimizer updates**, tối đa256 ảnh; IDF của reward vẫn từ đầy đủ20.000 ảnh train đã chọn. Không tính lại IDF trên256 ảnh.
- Strict deterministic algorithms, cuBLAS workspace cố định, math attention, TF32 off. Lỗi deterministic operation được báo ra, không tự chuyển sang chế độ lỏng hơn.
- So sánh runtime, trọng số khởi tạo; mỗi batch so sánh hash thứ tự ảnh, visual/prompt/caption/mask, RNG CUDA, token sample/greedy, loss và gradient norm, trọng số sau cập nhật. Kết thúc so sánh cả optimizer state.
- Bỏ qua duy nhất thời gian chạy. Không dùng sai số gần đúng để coi là PASS; chỉ lệch một giá trị loss cũng là FAIL.
- Không sinh caption validation/test, không đánh giá CIDEr/CHAIR, không lưu model checkpoint. Có danh sách val trong protocol để giữ cohort definition cũ; không dùng caption val để tính thưởng.

## Gửi lại

Tải `/kaggle/working/qformer_scst_ce_same_gpu_replay_32_review.zip` và gửi lại. ZIP chứa protocol, runtime, history, kết quả so sánh và log; không kèm weights nên nhỏ hơn các gói model trước.

`replay_comparison.json` có `PASS` nếu hai lần khớp hoàn toàn trong phạm vi kiểm tra. Nếu `FAIL`, xem bước/field lệch đầu tiên trong `differences`. Nếu worker lỗi, ZIP có `failure.json` và log, trạng thái INCOMPLETE; không được coi là kiểm tra tái lập thành công.

PASS chỉ xác nhận32 bước đầu trên cùngGPU/runtime; không đảm bảo2.500 bước, khácGPU, phiên bảnPyTorch khác, hoặc chất lượng caption. Sau khi xem kết quả mới quyết định sửa penalty theo số đối tượng và chạy đối chứng tiếp.
