# C0 — đối chứng sau pilot C/G

C0 khởi tạo từ A, không từ C endpoint. Chỉ đổi hệ số positive preservation của C từ .1 thành0; vẫn tính và ghi keep để theo dõi. SCST reward gốc A và rollout KL .1 giữ nguyên. Không phạt negative. Protocol template có keep_weight .1 chung, nhưng `objective_weights` ghi rõ keep0 cho C0 trong config, history, checkpoint và summary; effective weights mới là hệ số thực sự dùng.

- 4000 unique TRAIN images, cùng schedule/ref/seed42 và500updates/batch8 như C.
- Cùng hai decoder layers cuối, AdamW5e-6, native model/teacher/sampler/greedy và fixeddev1000.
- Không sửa model hoặc loss sau xem C0. Chỉ endpoint500. Đây là post-pilot ablation được chọn sau kết quả C/G.
- Các caption sampled có thể khác C sau update vì policy khác; không bắt buộc replay sampled captions C.
- Chuẩn bị mới từ5input gốc, không đòi dataset output C/G. Không dùng chế độ recovery cũ qua source version khác.

Mở `CONCEPT_MASS_C0_2GPU.ipynb` trên Kaggle, bật Internet và T4×2, Add Input giống C: pair-source-fullframe-ce, prompt-cache, visual-cache, objectdetectionecache, mscoco-2014(vuthetam). Kiểm5path ở cell đầu rồi Save & Run All.

Gửi `/kaggle/working/concept_mass_C0_v2_review.zip`; giữ `/kaggle/working/concept_mass_C0_v2/checkpoint.bin`.

So sánh C−C0 và C0−A: CIDEr/CHAIRs/CHAIRi/recall/length, paired image intervals và neutral-prefix diagnostic. Không tự quảng bá thắng trên dev thành test result, novelty proof hay giảm hallucination. 51vs52 caption hiện chưa đủ kết luận. Sau khi chốt model/protocol, tập đánh giá độc lập lớn hơn giúp xác nhận chất lượng; không dùng nó làm vòng tuning tiếp.

Local verification:13metadata/NumPy policy tests,4C0static/package checks,7existing notebook static/mock checks passed. No Torch/models/gradients locally; actual CUDA preflight and C0 metrics pending user-run. Caps prepare3600s(+120sclosegrace), GPU3600s, CPUmetrics900s are limits, not ETA. Quota24h is an old report, not current verified balance.
