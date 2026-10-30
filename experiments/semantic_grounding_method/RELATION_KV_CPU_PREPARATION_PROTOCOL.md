# Bước CPU trước pilot K/V — nguồn nhãn và cache

30/10/2026, sau khi user đồng ý nearest-check rồi pilot nhỏ. [Nearest review](KV_NEAREST_METHOD_CLOSURE_2026-10-30.md) đã đóng ở mức đủ cơ sở kiểm nghiệm một cải tiến kế thừa hẹp, không chứng nhận novelty/hiệu quả.

## Mục đích và giới hạn

Một notebook **CPU, không train/model/encoder forward** chuẩn bị đúng nguồn và tensors có sẵn. Kết quả là input cho D/E sau khi kiểm tra coverage; chưa được dùng để nhận semantic-learning thành công. User tự chạy. Không tự chạy HDF5/Torch/model local.

### Ontology để chuẩn bị, trước khi đếm matched data

- `contains ↔ inside`: chỉ diễn giải **sự chứa đựng không gian vật lý** giữa hai đối tượng khác nhau; không đổi thành `has-part`, `holding`, `wearing` hoặc `on`.
- `touching ↔ touching`: chỉ diễn giải **tiếp xúc vật lý**, không hành động chủ ý của một vai.
- Không dùng left/right/front/behind vì VSR cho phép intrinsic OR relative frame. Không coi None/0/1/2 là camera frame. Không tự thêm above/below/near/next-to để tăng count.
- Đây là phạm vi semantics được khai báo cho pilot, không bảo đảm từng annotation nguồn hoàn hảo. Transport nhãn sang inverse view là hệ quả của giả định này, không nhãn người thứ hai. Các observed inverse views có truth mâu thuẫn bị quarantine toàn fact, không chọn một nhãn thuận lợi.

Tập ontology không được thay sau khi nhìn kết quả validation. Nếu source không phù hợp phạm vi literal hoặc quá ít dữ liệu, báo chưa ready thay vì mở rộng im lặng.

### Nguồn và split

1. VSR official random TRAIN, pin commit/hash hiện có. DEV/TEST chỉ đọc image IDs để loại; không đọc các nhãn ấy vào train.
2. Chỉ COCO IDs thuộc Karpathy TRAIN/restval. Loại128image held cũ, officialVSRdev/test images, Karpathyval/test. Không dùng old32panel/mixedAI labels.
3. Giữ observed truth với >=3 distinct agreeing validators/no dissent, parse đúng COCO category template, subject và object khác class. Cả hai phải có đúng một noncrowd GT instance; chưa nhận phương pháp xử lý multi-instance ambiguity.
4. Predicted bank top6 theo confidence/tie original index, không lọc nhãn detector cho matching. Ghép hai boxes phân biệt với GT bằng IoU>=.5; nếu không ghép được thì local supervision UNKNOWN/skip, không tạo negative hoặc dùng GT ROI lúc forward.
5. Dedup theo instance-grounded proposal fact orbit. Các nhãn mâu thuẫn bị quarantine. Một fact và hai equivalent views vẫn tính một event.
6. CaptionTRAIN4000 giữ cùng lịch source cohort của C/G/C0; dev1000 như cũ. Auxiliary held partition theo hash imageID mod5, chỉ ở các ảnh không thuộc captionTRAIN4000; ảnh thuộc captionTRAIN luôn fit. Đây là held khỏi lượt adaptation mới, không ảnh chưa từng xuất hiện trong pretrained C0.
7. MP chỉ tạo giữa observed true và observed false có cùng lexical fact-orbit/query trong cùng split; không dùng in-batch unknown, mọi khác-ảnh, mọi arg-swap hoặc absence làm negative. Nếu không có đủ group để tính loss, ghi chưa ready và bàn protocol trước model.

### Feasibility gates cho bước chuẩn bị

Operational minimum trước khi viết/khóa training cohort: fit>=128original facts và held>=32; cả containment/contact có >=8 true/8 false fit và >=4 true/4 false held; fit có >=8 lexical groups với observed true+false, held>=2. Đây là điều kiện vận hành để tránh pilot chỉ vài facts, **không power calculation/semantic significance gate**. Không tự nới khi fail; notebook vẫn lưu counts/rejections/receipts và ZIP nhỏ.

### Cache và artifacts

- Tái dùng native rawCLIP197×768, legacy prompt20×512/mask và detector cache; không new crops/GQA/GPU encoder trong bước này.
- Lấy official COCO2014 instance metadata qua helper HTTP range đã có, cả train/val members với SHA đã ghi trong TRAIN_PRESENCE; lọc theo wanted IDs, không tải pixels. Chỉ writer-owned temporary files được dọn.
- Copy tensor theo COCO IDs, kiểm metadata/dtype/shapes/finite/payload hash; không giả rowcount=đủID. Native C0 checkpoint chỉ hash, không load/execute.
- Output ít files: metadata/caption references, original/equivalent auxiliary facts và nguồn, raw tensor HDF5, detector bank, source/preparation receipt và review ZIP không chứa HDF5/checkpoint. Repo clone ở `/tmp`, không làm output có hàng trăm sourcefiles.
- `READY_FOR_MODEL_PREFLIGHT` chỉ khi data gates và cache integrity đều qua. Chưa CUDA/model PASS. Fail/incomplete vẫn giữ log/ZIP; không training tự động.

### Sau bước này

User gửi review ZIP để root khóa actual event counts/unknown policy/commonloss và chuẩn bị hai notebook D/E với cùng stage identity. Không tự dùng expanded4000 để tuning, không nhận một ZIP tên đẹp là complete. Quota báo khoảng20giờ/tài khoản; CPU preparation không cần accelerator và chưa xác định runtime. Không tăng steps/relations/beam để cứu kết quả.
