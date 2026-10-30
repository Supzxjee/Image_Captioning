# Chuẩn bị pilot D/E sau khi người dùng đồng ý

30/10/2026. Người dùng đồng ý giữ [hypothesis K/V](RELATION_KEY_FACT_VALUE_HYPOTHESIS_2026-10-30.md) làm ứng viên chính, hoàn tất kiểm tra nearest method rồi làm pilot nhỏ. Không có quyền tự chạy Kaggle hoặc neural local. Chưa được gọi đây là novelty đã xác nhận hoặc phương pháp đã có hiệu quả.

## Phạm vi đã đồng ý và công việc đang làm

- Astra phụ trách đóng nearest semantic/equivariant K/V literature comparison, tái dùng các primary notes đã đọc. Nếu gặp exact collision, báo người dùng trước khi chuẩn bị lượt train; không lấy search không thấy làm chứng minh mới.
- Root kiểm metadata/splits/nguồn nhãn, ghi thiết kế đối chứng và kiểm NumPy construction trước. Chưa stage một bộ training mới hoặc chạy extraction/model.
- Người dùng trả lời “20” cho câu hỏi giờ GPU **mỗi tài khoản** còn lại. Lập kế hoạch theo khoảng20giờ/tài khoản; không tự quy đổi thành wall time hoặc cho phép một lượt chạy20giờ. Thời hạn thực của notebook cần được đặt sau benchmark user-run và còn chừa ngân sách evaluation.
- Hai nhánh D/E phải cùng data, ontology, instances, equivalent views, F nonlinear, slots, decoder adaptation, losses, init/schedule và evaluation; chỉ khác hard value projection. Không thêm loss riêng E hoặc thay đối chứng sau xem kết quả.
- C0 đúng checkpoint là own reference. Không nhận mọi gain do decoder continuation/new memory là công của projection; cần E−D và E−C0.

## Dữ liệu đã kiểm chỉ bằng metadata

Nguồn VSR random TRAIN đã có local:7680records, pinned SHA `8b725a8621f2d0f94745f1fefdcf3ff8a7eb96fd358676296ff9c9a4ff463c6e`. Bộ256TRAIN/64held cũ được chọn sau pooled official splits, nên không dùng nguyên bộ để nhận officialTRAIN mới.

Karpathy JSON local tại `C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json`, SHA `2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d`, đã kiểm lại. Giữ TRAIN/restval, loại ảnh xuất hiện trong officialVSRdev/test và128ảnh held cũ. Chỉ đọc image IDs của các tập loại, không dùng held truth để chọn model/ontology.

- Trong captionTRAIN10k hiện có:415records/298images; sau unanimous>=3/no dissent còn318records/242images. Ví dụ P5 chưa khóa chỉ36records/35images, trước uniqueness/detector matching. Không suy nguồn fullTRAIN chỉ có36.
- Trong toàn KarpathyTRAIN/restval:4520records/3294images; unanimous>=3/no dissent còn3531records/2722images. Counts minh họa trước GT/detector matching:above113,below85,left of63,right of21,next to87,touching463,contains126,inside84. **Không chốt ontology theo số lượng này.**
- Source `reference_frame` phần lớn làNone. Theo primary source review của Astra, VSR left/right vàfront/behind chấp nhận intrinsic OR relative, nên không chuyển nhãn bằng inverse đơn giản cho nhóm ấy. None chưa được giải mã thành camera frame. Các inverse/symmetric maps khác cũng cần định nghĩa rõ; chưa khóa P5.
- GQA mới đọc schema, chưa tải/stage/map. VSR observed truth có thể là nguồn tối thiểu nếu đủ matching, ontology đúng và learning signal; chưa tự quyết bỏ/giữ GQA theo counts.
- **Sửa giả định sau khi đọc lại metadata gốc:** `C:/Users/leduc/Downloads/visual_cache_metadata.txt` báo ba shards41095/41096/41096rows, tổng123287ảnh. Số15k là cohort từng được stage, không phải giới hạn cache gốc. Candidate K/V dùng raw CLIP fractional pooling nên có thể tái dùng cache cho auxiliary images ngoàiTRAIN10k; stage vẫn phải `require_ids` và kiểm provenance/compatibility, không nhận chắc mọi ID có chỉ từ rowcount. Root chưa đọc HDF5 hay chạy encoder. Không dùng GT box thay predicted ROI lúc forward.

Đây là upper bounds của nguồn, không phải training cohort đã ready. Chưa kiểm GT uniqueness, predicted-box IoU, vocabulary/tokenization hoặc local semantic quality trên cohort mới.

## Checks không cần mô hình

Đã chạy `test_relation_kv_math.py`: **5tests PASS** bằng NumPy, gồm orbit projection/idempotence và key-swap/control distinction; joint permutation/null-only; value-energy reduction/constant-content invariance; additive-F opposite-orbit collapse; một interaction toy tránh collapse. Chúng chỉ kiểm công thức/giới hạn kiến trúc, không semantic proof hoặc CUDA/model PASS. Chưa import/load Torch/checkpoint hoặc chạy scorer. Source kiểm tra đã có trước khi ghi PASS; không gọi các assertions là các thực nghiệm mô hình độc lập.

## Những việc phải khóa trước notebook training

1. Nearest-method conclusion và claim scope; còn exact K/V collision thì báo lại.
2. Ontology inverse-closed có frame semantics, original-fact weighting và positive/negative/unknown policy; nguồn/mapping/ảnh/GT/predicted-box coverage cùng split IDs được hash.
3. Training-caption cohort, auxiliary cohort và image-held semantic checks; không re-use old32panel hoặc expanded4000 làm nguồn tuning. Official dev/test image IDs chỉ để loại.
4. Common caption objective/anchor, MP/matching losses và nguồn negatives được quan sát thật. Hệ số và500updates hiện chỉ là đề xuất, chưa config đã khóa.
5. Ngân sách stage/train/eval cụ thể sau speed/memory preflight user-run; fail/incomplete phải archive log, không nhận completed từ tên ZIP. Người dùng tự chạy.

Chưa có notebook training được bàn giao trong file này. Việc chuẩn bị được tiếp tục trong phạm vi đã đồng ý; không mở thêm họ phương pháp hoặc tự hứa đủ điều kiện nhận bài.
