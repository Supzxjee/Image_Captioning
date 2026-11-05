# Cache prompt CLIP V2 — hoàn tất ở Output 169

05/11/2026. Nguồn người dùng xác nhận: `ducanh2403/2-vi-cai-kia-qua-nhieu/169`. [Metadata kiểm tra](PROMPT_TEXT_CACHE_RESULT.json). ZIP SHA256 `e7415b3559fac5af37119e761dc8b243d3898ad70b8d332212d180011584c52c`.

## Đã có gì

- **318 prompt duy nhất → ma trận float32 318×512**, CLIP frozen trên CPU. Dùng `openai/clip-vit-base-patch16`, revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`; tám asset SHA khớp cấu hình CLIP đã khóa. Không có lỗi thiếu/thừa trọng số khi nạp. Dài nhất 13 token, không cắt prompt.
- Liên kết giữ nguyên từ mapping167: TRAIN 10.000 + VALID 5.000 ảnh, không TEST. TRAIN có 29.670 object slots và 4.839 quan hệ đã map; VALID có 14.690 object slots và 2.319 quan hệ. Đây là liên kết detector, chưa phải nhãn quan hệ do người kiểm chứng.
- Người dùng chạy đạt **7 kiểm tra mapping + 5 kiểm tra cache/transport + 3 neural fixture CPU**. Ba fixture kiểm đường đóng/rỗng giữ đầu ra gốc, dữ liệu sai bị từ chối, và đường text mở có tác động cùng gradient hữu hạn.
- Tác giả kiểm lại **57 mục metadata/log/source/hash** bằng `audit_prompt_text_result.py`: PASS; mã nguồn thực chạy `418a717`, 22 file Git pins. Không chạy lại mô hình hoặc notebook.

## Chưa thể kết luận gì

Ba neural fixture dùng CSA nhỏ khởi tạo ngẫu nhiên, **chưa nạp checkpoint161, chưa kiểm CUDA/DDP, chưa huấn luyện caption**. Gradient khác 0 không chứng minh mô hình đã dùng đúng quan hệ trong caption. Chưa có CIDEr/CHAIR mới; 117,06 vẫn thuộc pipeline cũ không có prompt UPT.

Small review ZIP không chứa NPY, toàn bộ links/texts hay trọng số CLIP. Hash và kiểm vector đầy đủ là bằng chứng từ lượt người dùng chạy, chưa được tác giả mở lại binary. Pha cache ghi 19,46 giây, có tải/nạp model và ghi payload, không bao gồm toàn bộ setup/API/cài thư viện/fixture; không dùng số này dự báo train hoặc tính quota. Các cảnh báo pip không làm lượt này thất bại.

## Giữ và dùng tiếp

Giữ toàn bộ `/kaggle/working/csa_prompt_text_cache_v2` trong Output169, đặc biệt `prompt_embeddings.npy`, `prompt_links.jsonl`, `prompt_texts.json`, `cohort.json`, `summary.json`, `cache_manifest.json`, `cache_receipt.json`. Không cần chạy lại UPT/mapping/CLIP.

Tiếp theo: chuẩn bị cùng đầu vào cho hai tài khoản, rồi hai nhánh **CONTROL / PROMPT, cùng checkpoint161, 500 update × global batch8 = 4.000 lượt ảnh**, giữ bank danh từ/động từ, cùng CE + 0,05 SCST. Kiểm checkpoint thật/CUDA/DDP trước train; VALID5000, chưa mở TEST. Người dùng báo còn 15 giờ GPU; chưa phân biệt mỗi tài khoản hay tổng, không tự cộng thành 30 giờ. Notebook lưu tài sản dùng CPU, không cần ngân sách GPU.

Lượt cache đầu thiếu `review_examples.json` được giữ trong lịch sử; V2 đã tải đủ và hoàn tất. Không cần phục hồi hoặc chạy lại lượt lỗi.
