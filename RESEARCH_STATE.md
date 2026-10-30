# Trạng thái nghiên cứu — đọc trước khi tiếp tục

Cập nhật: 30/10/2026, giờ Việt Nam. File này là bản tóm tắt để tiếp tục công việc; protocol và artifacts được dẫn bên dưới mới là nguồn chi tiết. Không cần đọc toàn bộ lịch sử chat.

## Mục tiêu và cách làm việc

- Tăng CIDEr và giảm hallucination so với baseline mạnh A; cần đóng góp căn chỉnh thị giác–ngữ nghĩa có bằng chứng cho bài báo đầu tiên tại tạp chí đại học trong nước.
- Cộng sự phải phản biện, không chiều ý. Astra định hướng, agent phản biện/triển khai, người dùng chốt hướng và tự chạy Kaggle.
- Không tự chạy mô hình/Torch/autograd local hoặc notebook Kaggle. Kiểm tra metadata/NumPy/static/mock được phép; ghi rõ giới hạn. Quy trình tại [AGENTS.md](AGENTS.md).

## Hiện đang ở đâu

**User đã đồng ý chốt một evaluation A/C0/C, không training. Protocol khóa tại [EXPANDED_EVAL_PROTOCOL.md](experiments/concept_mass_grounding/EXPANDED_EVAL_PROTOCOL.md): đúng val_extra4000_cg = val5000 trừ currentdev1000, validation đã dùng trong lịch sử, loại dev1000/TRAIN; giữ3checkpoints/greedy/scorer. Chưa có notebook/run mới; manifest đã khóa, còn chờ lấy checkpoint version111(C)/112(C0) và quota hiện tại trước user-run. Evaluation không tự tạo contribution. Astra đã xác nhận sửa nhầm nhóm dev lịch sử; protocol/manifest đã push `bdeeab7`, sheet R214 ở hàng241 đã ghi và đọc lại xác nhận.**

- [C0_RESULT.md](experiments/concept_mass_grounding/C0_RESULT.md), [C0_RESULT.json](experiments/concept_mass_grounding/C0_RESULT.json), [audit_c0_results.py](experiments/concept_mass_grounding/audit_c0_results.py). Review text cached ignored tại `experiments/concept_mass_grounding/review/C0_loaded.json`, ZIP đã đọc1lần.
- C0 CIDEr1.230741, CHAIRs52/1000, hallucinated mentions53/1646,recall.460498,words10.044. A52caption/53mentions: chưa giảm hallucination; CHAIRi giảm do mẫu số tăng.
- C0−A CIDEr+.025229, paired95%CI[+.006253,+.043992]; C−C0−.005063,[-.014020,+.003438]. Gain vẫn có khi keep=0; chưa chứng minh lợi ích riêng của preservation. Không kết luận preservation gây hại hoặc hai nhánh tương đương.
- Same4000uniqueTRAIN/order/ref/seed42,500updates,batch8,init/frozen/input/tensor/payload hashes;2rankpreflightPASS,Aexact. Effective KL.1 keep0 negative0, zero weighted aux gradient ở46logged rank snapshots.274metadata assertionsPASS; không chạy/load model/H5 local.
- [C0_PROTOCOL.md](experiments/concept_mass_grounding/C0_PROTOCOL.md), [CONCEPT_MASS_C0_2GPU.ipynb](experiments/concept_mass_grounding/CONCEPT_MASS_C0_2GPU.ipynb). Package7c8b602/corebb84be8; ablation được chọn sau pilot C/G, test chưa dùng.
- Giữ `/kaggle/working/concept_mass_C0_v2/checkpoint.bin`, SHA256 `5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d`.
- Người dùng xác nhận nguồn checkpoint: C = notebook version111, C0 = version112 của notebook đang dùng `ducanh2403/2-vi-cai-kia-qua-nhieu`. Chưa có dataset mount; phải lấy đúngversion và xác minh SHA, không tự lấy latest. Chưa có quotaGPU cập nhật.
- A checkpoint `/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin`, SHA256 `630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901`.
- Kết quả C0 và audit đã push ở commit `5efa385`; sheet R213 đã xác minh. Đã gửi Astra số liệu và giới hạn, không yêu cầu đọc lạiZIP. Quota24h là báo cáo cũ, không coi là hiện tại.

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

1. C0 đã có kết quả: giữ nó làm ứng viên chất lượng/đối chứng, chưa chọn thành mô hình cuối. Hai objective keep/negative chưa có ích lợi riêng rõ trên endpoint; không gọi continuation+KL là phương pháp căn chỉnh mới.
2. Đọc phản hồi Astra mới nhất, trao đổi với người dùng và ghi quyết định trước nhánh tiếp theo. Không tự tăng steps/hệ số/seed hoặc mở một họ phương pháp mới để cứu claim. C hơn C0 chút ở neutral-prefix separation (.106577 so .106161), chưa thành lợi ích caption.
3. Người dùng đã đồng ý bàn và chốt một evaluation mở rộng, không training. Protocol/manifest đã khóa trước inference; chưa có run mới. Historical remainder4000 chứa797currentdevIDs nên đã sửa thành val_extra4000_cg theo exactgainmetadata; xem protocol. Mở rộng TRAIN vẫn chưa được duyệt. Tăng ảnh đánh giá độc lập giảm bất định; tăng ảnh huấn luyện đổi mô hình và cần đối chứng cùng ngân sách. Nếu khóa phương pháp rồi đánh giá lớn độc lập cho thấy giảm rõ, đó là bằng chứng có ích. Không dùng test làm vòng tuning tiếp.

## Các điểm phối hợp cần nhớ

- Repo GitHub: `https://github.com/Supzxjee/Image_Captioning`, branch main.
- Sheet `Experiment Registry`: `https://docs.google.com/spreadsheets/d/1XZ5iuyCs5JP8VE6dT7AYdVTiY1_nK4C_QdXs5ppw6FA/edit`, sheetId797499113. R213 kết quả C0 ở hàng240 đã ghi và đọc lại xác nhận; R212 ở hàng239; luôn đọc metadata/sentinel mới trước khi ghi.
- Chat Astra: `01a15544-6a7a-7903-a6e1-23e7b9f6caa6`, host local, tên `Đề xuất hướng căn chỉnh ngữ nghĩa–th`. Người dùng đã ủy quyền trao đổi nghiên cứu trong lịch sử phiên.
- A có prompt gốc; không gọi A là no-prompt. I/J và các nhánh trước chưa vượt A; chi tiết nằm trong thư mục thực nghiệm, không cần lặp mọi nhánh để tiếp tục C0.
