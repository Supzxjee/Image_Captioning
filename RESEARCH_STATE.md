# Trạng thái nghiên cứu — đọc trước khi tiếp tục

Cập nhật: 29/10/2026, giờ Việt Nam. File này là bản tóm tắt để tiếp tục công việc; protocol và artifacts được dẫn bên dưới mới là nguồn chi tiết. Không cần đọc toàn bộ lịch sử chat.

## Mục tiêu và cách làm việc

- Tăng CIDEr và giảm hallucination so với baseline mạnh A; cần đóng góp căn chỉnh thị giác–ngữ nghĩa có bằng chứng cho bài báo đầu tiên tại tạp chí đại học trong nước.
- Cộng sự phải phản biện, không chiều ý. Astra định hướng, agent phản biện/triển khai, người dùng chốt hướng và tự chạy Kaggle.
- Không tự chạy mô hình/Torch/autograd local hoặc notebook Kaggle. Kiểm tra metadata/NumPy/static/mock được phép; ghi rõ giới hạn. Quy trình tại [AGENTS.md](AGENTS.md).

## Hiện đang ở đâu

**Đã chuẩn bị C0; chờ người dùng chạy và gửi kết quả. Chưa có kết quả C0, không biết job đã bắt đầu hay chưa.**

- Notebook: [CONCEPT_MASS_C0_2GPU.ipynb](experiments/concept_mass_grounding/CONCEPT_MASS_C0_2GPU.ipynb).
- Protocol: [C0_PROTOCOL.md](experiments/concept_mass_grounding/C0_PROTOCOL.md).
- Quyết định Astra: [ASTRA_CG_RESULT_DECISION_2026-10-29.md](experiments/concept_mass_grounding/ASTRA_CG_RESULT_DECISION_2026-10-29.md).
- Package commit `7c8b602`, runtime core `bb84be8`, đã push GitHub. Sheet R212 đã đọc lại và xác nhận.
- C0 khởi tạo A: SCST reward gốc + .1 rollout KL; keep=0, negative=0. C có thêm .1 keep. Chỉ tắt hệ số keep, vẫn tính/ghi keep để theo dõi. Same seed42,4000unique TRAIN/order/ref,500updates,batch8,optimizer5e-6,last2decoder layers,greedy,fixeddev1000.
- Đây là ablation được chọn **sau** pilot C/G. C0 không khởi tạo từ C và không cần chạy lại C/G. Notebook dùng stage mới từ5input gốc; xem protocol/cell đầu để biết path đầy đủ.
- A checkpoint: `/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin`, SHA256 `630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901`.
- Output chờ nhận: `/kaggle/working/concept_mass_C0_v2_review.zip`; giữ `/kaggle/working/concept_mass_C0_v2/checkpoint.bin`.
- Đã kiểm local:13metadata/NumPy +4C0package/static +7notebookstatic/mock PASS. Chưa kiểm CUDA hoặc metric C0. Quota24h là báo cáo cũ, không coi là số hiện tại.

## Bằng chứng mới nhất: C/G đã hoàn tất

[RESULT.md](experiments/concept_mass_grounding/RESULT.md), [RESULT.json](experiments/concept_mass_grounding/RESULT.json), [audit_results.py](experiments/concept_mass_grounding/audit_results.py). Commit `a33a537`; registry R211. Hai ZIP đã đọc một lần; text cache ignored tại `experiments/concept_mass_grounding/review/C_loaded.json` và `G_loaded.json` trên máy này.

| Cùng fixeddev1000 | A | C | G |
|---|---:|---:|---:|
| CIDEr | 1.205513 | 1.225678 | 1.229535 |
| Caption có ảo giác /1000 | 52 | 51 | 52 |
| Hallucinated mentions / object mentions | 53/1632 | 52/1632 | 53/1633 |
| Object recall | .458039 | .458654 | .457731 |

- C/G đủ500updates,4000uniqueTRAIN; cùng init/frozen/input/tensor identities; preflight cả2rankPASS; A historical caption mismatches0. Metadata audit185assertionsPASS; không load checkpoint/H5 binaries local.
- C−A CIDEr +.020166, paired95%CI[+.002931,+.038610]; G−A +.024023,[+.006684,+.042807]. G−C +.003857,[-.003236,+.011555]. Một seed/dev đã tái dùng, chưa dùng test.
- Không chứng minh giảm hallucination; G không có lợi ích riêng rõ so với C. G có211weak-negativeevents, không phải zero-event. Neutral-prefix separation G cũng không hơn C. Negative là nhãn yếu từ annotation/reference absence + detector veto, không phải verified visual absence.
- Quyết định: giữ C như ứng viên tăng chất lượng; dừng mở rộng G hiện tại. C0 kiểm xem preservation có đóng góp ngoài continuation+KL không. Chưa được gọi C là phương pháp grounding đã xác nhận.

## Bước tiếp theo và điều chưa chốt

1. Khi có C0 ZIP: kiểm complete/init/scope/source/weights/4000IDs/A parity; đọc ZIP một lần. So C−C0 và C0−A trên cùng1000IDs; CIDEr/CHAIRs/i/recall/length, pairedCI và conditioning. Lưu RESULT, báo Astra dữ kiện ngắn, push và ghi sheet.
2. Nếu C hơn C0 có bằng chứng, mới có cơ sở đầu tư xác nhận đóng góp preservation. Nếu bất định hoặc C0 tốt hơn, không ép claim hoặc tăng hệ số để cứu giả thuyết.
3. Người dùng hỏi tăng mẫu có giúp C ít halu hơn A không: **chưa duyệt mở rộng**. Tăng ảnh đánh giá độc lập giảm bất định; tăng ảnh huấn luyện đổi mô hình và cần C0 cùng ngân sách. Nếu khóa phương pháp rồi đánh giá lớn độc lập cho thấy giảm rõ, đó là bằng chứng có ích. Không dùng test làm vòng tuning tiếp.

## Các điểm phối hợp cần nhớ

- Repo GitHub: `https://github.com/Supzxjee/Image_Captioning`, branch main.
- Sheet `Experiment Registry`: `https://docs.google.com/spreadsheets/d/1XZ5iuyCs5JP8VE6dT7AYdVTiY1_nK4C_QdXs5ppw6FA/edit`, sheetId797499113. R212 ở hàng239 tại lần xác minh cuối; luôn đọc metadata/sentinel mới trước khi ghi.
- Chat Astra: `01a15544-6a7a-7903-a6e1-23e7b9f6caa6`, host local, tên `Đề xuất hướng căn chỉnh ngữ nghĩa–th`. Người dùng đã ủy quyền trao đổi nghiên cứu trong lịch sử phiên.
- A có prompt gốc; không gọi A là no-prompt. I/J và các nhánh trước chưa vượt A; chi tiết nằm trong thư mục thực nghiệm, không cần lặp mọi nhánh để tiếp tục C0.
