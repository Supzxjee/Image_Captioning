# CSA core: kiểm chứng kỹ thuật đạt, chưa đánh giá caption

Ngày audit:2026-10-22. ZIP người dùng chạy trên Kaggle: `csa_core_smoke_review.zip`,35905bytes, SHA256 `e458bd3f375b997cc628bf55fdd3608e1bc16cd0a957405312960d6cb65a2571`.

**Kết luận: TECHNICAL_SMOKE_REVIEW_AUDIT_PASS.** Lõi CSA có thể tích hợp và chạy backward/optimizer trên nền Q-Former hiện tại với T4×2 trong cấu hình thử. Chưa có metric caption mới, chưa chứng minh grounding hoặc lợi ích so với A.

| Kiểm tra | Kết quả |
|---|---|
| Source |9file embedded khớp hash protocol và notebook đã phát hành; reference core khớp commit0cc570b |
| CPU output |B2,N32 và150: Vo/Vr max absolute difference0 |
| CPU gradient |Input và93parameter tensors đối chiếu đạt rtol1e-4,atol1e-6 |
| Interface |Memory/mask synthetic và bypass memory/logits của checkpoint thật đạt |
| Bank |Runtime kiểm hash/shape/finite; frozen buffers, không đổi sau optimizer |
| Core |23.161.644 tham số mới;2object stages+1relation stage |
| DDP |Hai T4,4updates/rank; final core state hash giống nhau |
| Loss/gradient |Hữu hạn; gradient caption đi vào core, norm trước clip>0 cả4steps |
| Bộ nhớ mỗi GPU |Peak allocated tối đa1,3263GiB; reserved1,3574GiB |
| Tốc độ thử |CE step2:0,1746s; SCST steps3/4:0,8047s và0,4330s, lấy rank chậm hơn |
| Dữ liệu |8ảnh TRAIN/restval theo hash cố định, kiểm ID/split với JSONCOCO gốc; không dùng dev/test |

Objective:2CEsteps rồi2CE+.05SCSTsteps; rewardCIDErD+.5object_count_v2, fullTRAIN IDF như A; globalbatch8,LRbase5e-6/core1e-4 chỉ phục vụ probe. FP32, dropoutoff với gradients, AdamWweightdecay.01, clipglobalnorm1. Mỗi rank4ảnh, lặp lại cùng8ảnh qua4updates.

Core gradient norm trước clip lần lượt1,61438;0,60058;0,89301;1,42216. Globalnorm3,16113;2,26877;3,32774;5,50095 nên clip tác động8/8ranksteps. Không được xem norm trước clip là norm update thực tế. Loss âm ở một bước SCST không tự nó là lỗi: loss=CE+.05RL đã khớp số liệu.

CPU parity dùng **cùng author forward**, khác cách nạp/buffer: xác nhận thay đổi kỹ thuật không đổi phép tính, không phải kiểm hai triển khai độc lập hoặc chứng minh paper equations đúng. N32 là latent query, không phải32detector regions. Đầu ra Vo/Vr chưa tự xác nhận liên kết đối tượng–quan hệ đúng.

Peak/time chỉ của8ảnh này, loại trừ bootstrap/staging/IDF/evaluation; không dùng làm upper bound mọi batch hoặc dự báo tổng thời gian chắc chắn. CE giảm trên cùng8ảnh không phải bằng chứng tổng quát hóa. Không lưu checkpoint sau smoke để tránh dùng nhầm model chỉ chạy4updates.

ReviewZIP không chứa bank/checkpoint/staged tensors. Hash và tính bất biến của chúng được kiểm bởi source runtime đã đối chiếu, chưa có kiểm bytes độc lập trong audit này. ZIP có cảnh báo resolver liên quan gradio/diffusers và DDPfind_unused_parameters overhead; không ngăn kiểm chứng hoàn tất.

**Bước tiếp theo:** gửi summary cho Astra chốt một pilot có kiểm soát và lịch học phù hợp core khởi tạo mới; chưa tự động dùng1250updates hoặc triển khai thêm gate/loss/freeze. Không cần chạy lại smoke hiện tại.

Tái kiểm bằng stdlib, không chạy model:

```powershell
python experiments/csa_core_smoke/audit_review.py C:/Users/leduc/Downloads/csa_core_smoke_review.zip
```

Audit cohort dùng JSONCOCO đã có local ở path cố định trong script. Kết quả máy đọc: [SMOKE_RESULT.json](SMOKE_RESULT.json).
