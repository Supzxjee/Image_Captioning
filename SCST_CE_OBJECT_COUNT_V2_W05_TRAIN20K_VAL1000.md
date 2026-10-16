# Object Count v2: hệ số thưởng đối tượng0.5

Trạng thái: đã chuẩn bị notebook cho người dùng tự chạy, chưa chạy GPU. Đây là thử nghiệm thăm dò trên cùng tập validation đã xem, không phải đánh giá xác nhận độc lập.

## Thay đổi duy nhất trong huấn luyện

Giảm `--object-reward-weight` từ1.0 xuống0.5 so với R123. Giữ nguyên core commit `7a31327199bfd49684f5eb2b39bb2f7774795919`, parser `aliases_v2`, penalty `unique_count_v2` và toàn bộ cấu hình còn lại:

`R_object = 0.5 * salient_unique_recall - unsupported_unique_count`

`R_total = CIDEr_train + 0.5 * R_object`

`Loss = CE + 0.05 * SCST(R_total)`

Hệ số0.05 của SCST và hệ số1.0 của CE không đổi. Phần coverage lẫn penalty trong object reward đều giảm một nửa; đây không phải chỉ giảm riêng mức phạt. Hypothesis: bớt tác động của object reward có thể phục hồi CIDEr trong khi vẫn giữ một phần mức giảm ảo giác; chưa có bằng chứng điều này sẽ xảy ra.

## Đối chứng và ngân sách

- Control: CE+.05SCST(CIDEr), train lại từ QFormer epoch10 gốc.
- Treatment: CE+.05SCST(CIDEr+.5R_object), cùng checkpoint gốc.
- Giữ train20k/validation1k, batch8, một lượt2500 updates/nhánh, seed42, AdamW5e-6, FP32, dropout off và strict deterministic runtime.
- Hai nhánh train tuần tự trên GPU0. Không tăng ngân sách so với lần trước. Phần đánh giá dùng haiGPU, cùngbeam5 và fixed reranking .3decoder/.4CLIP/.3OCC. Không dùng test, không chỉnh reranker.
- Đối chiếu đồng thời control mới, original và R123 trọng số1.0. Nếu runtime/control không tái lập, phải ghi nhận trước khi quy khác biệt cho hệ số thưởng. Đọc cả số caption/mention ảo giác, recall và CIDEr; không kết luận thắng chỉ từ một độ đo.

R123 fixed: original CIDEr1.275819/CHAIRs3.8%; control1.260458/3.8%; weight1.0 đạt1.264003/3.2%. Các khoảng tin cậy CIDEr đều chứa0. Gate giữ nguyên theo runner, chưa có đủ cơ sở mở rộng train.

## Chạy thủ công

Trong **QFormer UPT Val5000 20261015**, upload `SCST_CE_OBJECT_COUNT_V2_W05_TRAIN20K_VAL1000.ipynb` hoặc thay cell cũ bằng `kaggle_scst_ce_object_count_v2_w05.py`. Giữ các input đang có, bật Internet và GPU T4x2. Các đường dẫn ở đầu launcher để sửa trực tiếp. Không thêm input hoặc copy toàn bộ HDF5 sang/tmp.

OUTPUT mới: `/kaggle/working/qformer_scst_ce_object_count_v2_w05_train20k_val1000`. Không dùng lại thư mục đã có. Người dùng tựRun/Save & Run All; assistant không submit notebook.

Gửi lại `qformer_scst_ce_object_count_v2_w05_train20k_val1000_review.zip`. Giữ riêng hai checkpoint trong OUTPUT ở `scst_ce/checkpoint.pth` và `scst_ce_object/checkpoint.pth`; ZIP review không chứaweights. Giữ output/checkpoint của R123 ở phiên bản trước để so sánh, tránh bị mất khi Kaggle cập nhật output.

## Kiểm tra chuẩn bị

Đã đối chiếu launcher với R123: chỉ khác hệ số thưởng, tênOUTPUT và banner. Notebook khớp chính xác launcher, không có cell đã chạy hay output. Cú phápPython và smoke test phép kết hợp reward0.5 được kiểm tra cục bộ; không chạy mô hình/huấn luyện hay notebookKaggle. Bộ audit được mở rộng để nhận đúng hai hệ số1.0/0.5 cho count_v2, giữ kiểm tra các biến thể cũ.
