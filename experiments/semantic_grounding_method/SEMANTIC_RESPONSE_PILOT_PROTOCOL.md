# Pilot: selective semantic response alignment

31/10/2026. Người dùng đồng ý phát triển hướng ba nhánh CE / CE+change / CE+change+stable. Bước hiện tại: khóa thiết kế và chuẩn bị dữ liệu; chưa chạy/ủy quyền tự chạy GPU. Novelty là ứng viên cải tiến có điều kiện, không claim nguyên lý phản thực mới.

## Phạm vi đầu tiên

Chỉ thay đổi **màu của cùng vật**; không trộn count, object insertion/removal hoặc quan hệ vào pilot đầu. Đây là thu hẹp sau feasibility, chưa phải lựa chọn trước khi xem20mẫu. Lý do: giảm thay đổi ngữ pháp, referent và negation, dễ xác minh changed/stable bằng ảnh. Không claim đã cải thiện object hallucination chỉ từ color accuracy; CHAIR không chấm lỗi màu.

Từ MED SFT revision38f969ac, tìm thêm tối đa30cặp VG có ít nhất2từ màu cơ bản trong mô tả nguồn. Đây chỉ là cách lấy ứng viên, không suy ra nhãn đúng. Chỉ nhận VG→COCO mapping có Karpathy train/restval; loại val/test, unknown và các ID dev đã biết. Không lấy DOCCI test/qual_test. Giữ toàn bộ lỗi tải/loại; không chọn theo kết quả mô hình. Có ngân sách stream archive tối đa128MiB cho việc tìm mẫu, không tải5GB.

Chấm ảnh gốc/sửa và224bilinear: màu có đổi rõ trên cùng vật, ảnh có sửa lan không, chọn ít nhất1Stable rõ độc lập với màu thay đổi. Assistant labels chưa phải human gold. Reviewer phải được ghi trong file; chưa xác nhận thì chưa coi là training-ready. Các mẫu20ảnh đã xem chỉ là development-of-data-policy, không dùng làmheldout đánh giá phương pháp.

## Thiết kế đã chọn

Backbone C0: CLIP → Q-Former → decoder. Chưa thêm reader/verifier. C0 frozen reference giữ lại nguyên bản để so sánh; adaptation mask phải dùng cùng nhau cho3nhánh và được ghi rõ trước khi viết trainer. Một ảnh inference cho ra caption thông thường; không cần ảnh phản thực lúc chạy thật.

Mỗi pair có cùng subject cụ thể; cụm màu cũ a, mới b; chọn cùng context c để so trên hai ảnh. d(I)=s(a|I,c)-s(b|I,c), với s là mean token logprob của chính decoder. Full sequences của từng candidate được teacher-force riêng, chỉ lấy score của span màu, không so logits ở các prefix khác nhau. Với candidate single-word color, c phải giống hệt giữa hai ảnh; identity của prompt-condition phải khóa trước run.

- Change loss: `[m-d(I)]+ + [m+d(I')]+` để ưu tiên đúng ở cả hai ảnh.
- Stable loss: với từng fact còn nguyên và distractor đã kiểm tra không đúng trên cả hai ảnh, giới hạn biến động relative score trong epsilon; CE fact thật trên cả hai ảnh được áp dụng cho **mọi nhánh**, không chỉ fullmethod. Không ép toàn bộ logits/hidden features giống nhau.
- CE nền: dùng cùng tập caption ảnh thật đã khóa; không bắt models mô tả mọi fact mỗi câu, không tự tạo full caption giả làmgroundtruth từ diffresponse.

Nhánh0: CE captions + CE đúng cho các candidate facts trên cặp ảnh.
Nhánh1: giống0 + Change.
Nhánh2: giống1 + Stable.

Giữ giống init, data, caption/fact exposure, batch order, optimizer, params và update count; chỉ bật/tắt loss. Nếu2dùng nhiềufactsover0/1 thì comparison không hợp lệ. Phải có paired-CE để phân biệt data augmentation khỏi loss mới. SCST chưa thêm vòng này.

## Điều kiện trước khi xuất notebook

1. Có nhãn Changed + Stable + false alternatives hợp lệ, source IDs/splits/hash và ảnh tải đủ; không làm tự động “đúng” từ diffcaption.
2. Chia theo source image/group, không tách các edit của cùng ảnh ra train/heldout; kiểm trùng ảnh/near-duplicate và mọi nguồn eval hiện tại có thể đối chiếu. Train/restval filtering là điều kiện cần, chưa chứng minh không overlap toàn lịch sử.
3. Chốt prompt-condition: cùng prompt trong pair giúp tách visual response nhưng có thể stale cho ảnh sửa; không tự bỏ native prompts rồi vẫn gọi cùng pipeline C0. Phải ghi treatment và baseline tương ứng trước code trainer.
4. C0 exact vocab và preprocessing kiểm bằng metadata. Không đơn giản thay color OOV bằng màu khác. Không import/load Torch local.
5. Chốt mask trainable và số giờ GPU hiện tại, schedule/weights/gates trước run. Số20h/account từ lượt D/E không dùng như quota hiện tại.

## Quyết định dựa trên đầu ra nào

Heldout pairs: đúng màu trước/sau, giữ đúngfact, đo cả teacher-forced score và caption tự sinh. Cải thiện score mà caption không đổi không đủ. COCOdev: CIDEr, CHAIRs/i, objectrecall và độ dài; không đổi mục tiêu giảmobjecthallucination thành chỉ sửa màu. Một seed và tập nhỏ chỉ dùng quyết định tiếp tục; không gọi kết quả bài báo cuối.

Nếu không thu đủ các cặp sạch với chi phí nhỏ, hoặc chỉ probe tốt hơn nhưng caption không cải thiện, dừng mở rộng. Không thêm nhiều module/epochs để cứu claim. Kiểm tra được dữ liệu không chứng minh novelty hoặc hiệu quả.
