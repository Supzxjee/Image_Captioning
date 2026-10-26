# Role-interval feasibility 128 — NO-GO cho cấu hình hiện tại

Lượt user-run hoàn tất. **Chưa chuyển sang train:** thiếu mật độ tín hiệu và phát
hiện lỗi ánh xạ ontology trong bộ chấm. Đây không phải kết quả mô hình mới thua A.

## Kết quả theo ngưỡng đã khóa trước

| Kiểm tra | Kết quả | Ngưỡng | Kết luận |
|---|---:|---:|---|
| FIT có cache đúng nguồn | 2.026 | ≥500 | Đạt |
| Replay tách biệt, có cache | 1.898 | ≥500 | Đạt; planned n1.896 sau làm tròn4 |
| Cặp sampled/greedy | 128 | 128 | Đủ |
| Cặp có role advantage khác0 | **6 (4,69%)** | ≥16 | **Không đạt** |
| Cặp interval/binding làm đổi quyết định so comparator | 8 | ≥8 | Đạt về số lượng; cả8 là interval, không có binding |
| Binding-operative | 0 | Báo riêng | Chưa có bằng chứng cho đóng góp này |
| Search-capped | 0 | Báo riêng | Không phải nguyên nhân thiếu tín hiệu |

128 cặp mất151,25 giây cho hash/preparation/generation/scoring theo protocol;
không gồm setup và xuất ảnh. A frozen, không optimizer, không training.

Trong256 caption, chỉ21 có graph rõ theo policy (8,20%),128 opaque (50%),107
ngoài scope (41,80%).91 opaque đến từ quy tắc person+on/with/spatial chưa được
đồng nhất với holding/riding. Đây là nguồn mất coverage lớn nhưng chưa chứng minh
chỉ cần nới parser là phương pháp sẽ tốt. Greedy có18 graph, sampled có3.

Cả21 graph chỉ có **một edge**. Dù67/128 ảnh có nhiều person annotated, câu sinh
ra không tạo được bài kiểm joint multi-action binding. Binding0 không chứng minh
binding vô ích; phép kiểm này chưa kích hoạt được cơ chế đó. Tám interval-operative
cases đều aG=0: chúng minh họa abstention so với unknown=0, không cung cấp thêm
tám cặp học có ordering xác định.

## Lỗi quan trọng: nhãn action0 chưa tương đương phủ định câu tiếng Anh

Toàn bộ8 unit bị scorer gọi `refuted` đều là `riding` với skateboard(4),
snowboard(3), skis(1). Scorer dùng V-COCO actor/action `ride=0` làm phủ định.
Tôi đã kiểm lại file official `vcoco_train.json`, khớp SHA trong provenance:
**đúng8 actor này có skateboard/snowboard/ski=1**, với role object ID khớp chính
box trong ảnh, đồng thời ride=0.

| COCO ID | Caption greedy | Positive action chính thức |
|---|---|---|
| 24446 | a man riding a skateboard down a street. | skateboard=1, ride=0 |
| 354145 | a man riding a snowboard down a snow covered slope. | snowboard=1, ride=0 |
| 577980 | a man riding skis down a snow covered slope. | ski=1, ride=0 |

`ride=0` trong cách phân loại action của V-COCO **không đủ để kết luận** các câu
trên sai về nội dung nhìn thấy. Không nhận tất cả câu này là đúng chỉ từ bảng
nhãn; kết luận ở đây là phép phủ định mà scorer dùng không được bảo đảm.
RESULT.json giữ8 counterexample đầy đủ (image/actor/object ID và nhãn gốc).

Đây là lỗi semantic mapping trong thiết kế/triển khai của chúng tôi, không phải
lỗi bạn chạy notebook. Kiểm thử logic bằng graph giả lập và tính đúng interval
không tự xác nhận taxonomy phù hợp nghĩa của caption thực tế.

**Cả4 aG dương** đều đến từ việc sampled bỏ các `riding` vừa bị gắn refuted theo
cách ánh xạ này. Hai aG âm đến từ việc sampled bỏ holding được xác nhận dương.
Vì vậy6 active pairs không chỉ ít: phần tín hiệu dương chưa đáng tin để dùng học.
Reward từng phần cũng không đánh giá được ngữ pháp/toàn-caption; có sampled kém
mạch lạc vẫn nhận aG dương. Chưa có lượt train để biết base SCST/CE có bù được không.

## Audit kỹ thuật đã hoàn thành

- 41 receipt payload hashes và9 runtime-source hashes khớp package gốc.
- Hai rank đúng protocol/stage/selection/source/records digest, IDs, seeds và
  frozen-A identity được ghi nhận; đủ128 cặp và đúng hash-first FIT selection.
- Replay lại dependency tokens→graph→interval bằng pure Python khớp128/128 cases;
  không gọi spaCy neural parser hoặc caption model local.
- 30 review cases đúng selection policy. Không có human audit đã hoàn thành.
- Raw stage HDF5/checkpoint không nằm trong ZIP, nên không nhận đã trực tiếp
  kiểm lại tensor bytes. Không chấm truth của ảnh, không tính CIDEr/CHAIR mới.

## Quyết định và bước tiếp

**Không train cấu hình này, không nới ngưỡng16 xuống6, không yêu cầu bạn chấm thêm30
chỉ để quyết định gate đã fail.** Giữ A làm baseline. Kết quả này cũng chưa phải
thất bại của mọi hướng role alignment hoặc mọi cách dùng V-COCO.

Đã gửi Astra báo cáo số liệu và8 counterexample chính xác để quyết định hướng
phát triển. Cần tách hai việc: sửa lỗi ánh xạ action/object/negation cho đúng nghĩa
và tìm một phương án có đủ tín hiệu thị giác-ngôn ngữ. Sửa cho qua128 ảnh đã xem
không phải bằng chứng phương pháp mới hoặc bảo đảm caption gain. Chưa launch
notebook/GPU/training tiếp theo.
