# Kế hoạch tái hiện CSA Net nguyên bản trước khi cải tiến

Ngày 01/11/2026. **User chốt:** “tôi không cần đóng góp mới tôi cần xây dựng y chang csa từ đó mới phát triển thêm”. Đây là thay đổi mục tiêu trước mắt, không phải yêu cầu tiếp tục bảo vệ X018/X020. Ưu tiên tái hiện phương pháp tham chiếu, chưa cần đề xuất novelty. Người dùng tự chạy notebook; agent không chạy mô hình hoặc tự submit Kaggle.

## Chuẩn tái hiện

- Bài báo CSA-Net mà người dùng đã cung cấp, cùng code tác giả `CrossmodalGroup/CSA-Net` tại commit `0cc570ba7c50ef08c46a21439fe73ef0a5df97ab`.
- Đã xác minh lại bằng GitHub API ngày01/11: official HEAD vẫn là commit này; recursive tree239entries, không truncated. Chỉ đọc source/metadata, không import Torch hoặc chạy code tham chiếu.
- Code/paper/config có một số điểm khác nhau đã ghi trong [audit trước](CSA_INHERITANCE_FEASIBILITY.md). Phải ghi quyết định chuẩn code hay công thức ở từng khác biệt trước triển khai; không tự sửa rồi nhận tương đương.
- Phân biệt ba mốc: (1) kiến trúc/forward đúng chuẩn tham chiếu; (2) recipe/data/evaluation đúng bản được chọn; (3) kết quả benchmark được tái lập trong giới hạn seed/hardware. Hoàn thành một mốc không tự hoàn thành hai mốc còn lại; không cam kết bitwise hoặc đạt đúng điểm paper trước khi chạy.

## Pipeline cần giữ

Ảnh RGB → Swin và detector Deformable DETR theo source →150region features512D → hai stage căn chỉnh đối tượng từ noun semantic bank → một stage căn chỉnh quan hệ từ verb semantic bank → hai luồng object/relation → parallel adaptive decoder của tác giả →caption.

Semantic aggregation/prototype selection, self/cross attention, FFN/gate, hai output streams và decoder phải giữ theo chuẩn đã chọn. Kho danh từ4764×1024 và động từ975×1024 là RegionCLIP banks toàn cục của tác giả. Không tự thay bằng prompt UPT theo ảnh, raw CLIP caches hoặc Q-Former32queries.

| Thành phần | Tái hiện đầy đủ cần | Bản thử trước trong dự án |
|---|---|---|
| Đặc trưng ảnh | Detector/backbone gốc và trọng số phù hợp | CLIP→Q-Former32query |
| Semantic banks/core | Noun/verb banks và core gốc | Core/banks đã kế thừa, đã smoke |
| Decoder | Parallel adaptive decoder đọc object/relation streams | Nối memory vào decoder nền của mình |
| Dữ liệu/vocab | COCO/Karpathy và vocab/preprocessing bản tác giả | TRAIN10k/dev1000/native vocab/cache riêng |
| Huấn luyện/eval | XE rồi RL, scheduler/freezing/beam/scorer bản tham chiếu | Pilot1250+1250updates với recipe nền |

Kết quả [pilot cũ](../csa_pilot/PILOT_RESULT.md) được giữ làm lịch sử adaptation; không gọi đó là full reproduction hoặc bằng chứng CSA gốc thất bại.

## Tài nguyên đã kiểm và phần thiếu

1. Có code lõi, decoder, Swin source, train/eval entrypoints, config và noun/verb banks. Metadata/hashes của các assets đã kiểm nằm trong audit trước. Cùng repo có `coco_caption/data/vocab.json`.
2. **Thiếu source detector theo import:** `models/caption/detector.py` import `models.detection.det_module`, nhưng `models/detection/det_module.py` không có trong recursive tree chính thức đã kiểm. Cần xác định nguồn gốc và phiên bản đúng, không tự suy một module gần giống là bản tác giả.
3. **Thiếu checkpoint detector trong release đã kiểm:** config và README yêu cầu detector4ds, cấu hình trỏ `./coco_caption/detector_checkpoint_4ds.pth`; tệp không có trong tree và README đã đọc không cung cấp link tải. Điều này chưa chứng minh không tồn tại ở nơi khác; cần tìm nguồn chính thức/nguồn tác giả hoặc payload người dùng đã có.
4. Một số tệp khởi tạo package/import metrics cũng không có trong tree, ví dụ `datasets/caption/metrics/__init__.py`. Cần audit import graph toàn bộ. Chưa instantiate để kết luận số lỗi runtime hoặc yêu cầu vá nào đã được giải quyết.
5. Cần kiểm data IDs/splits/annotations, vocab và assets trích xuất detector đúng nguồn. Cache CLIP/Q-Former cũ không thay trực tiếp cache detector gốc.
6. Config khai báo finetune15XE+30SC; README ví dụ freezing10XE+10SC và zero finetune. Chưa tự chọn một recipe rồi gọi là đúng paper. Bảng đối chiếu paper/code/config/freezing, global-vs-per-rank batch, scheduler, beam baseline và metrics phải khóa trước notebook.

## Thứ tự thực hiện

1. Hoàn thiện inventory source/imports, dependency CUDA ops và nguồn checkpoint detector; tách rõ có/thiếu/chưa xác minh. Tìm thêm nguồn chính thức trước khi hỏi người dùng về payload họ có thể đã giữ.
2. Viết sơ đồ đúng CSA gốc, ghi tensor interfaces và mapping paper→source. Không thêm một module cải tiến, reward đối tượng hoặc KL của dự án hiện tại.
3. Chuẩn bị môi trường và smoke notebook trên Kaggle cho user, sau khi đủ tài nguyên; chỉ kiểm static/metadata tại local. Numerical/gradient/runtime parity do notebook user chạy xác nhận. Mọi compatibility patch phải có lý do và không âm thầm đổi thuật toán.
4. Kiểm data/vocab/preprocessing, checkpoint load coverage, decoder outputs, loss/sampler, scorer và frozen/training scopes trước run tốn GPU. Không chấp nhận silently dùng detector khởi tạo ngẫu nhiên do đường dẫn checkpoint không tồn tại.
5. Sau smoke và biết GPU quota hiện tại, lập lịch huấn luyện tham chiếu phù hợp. Nếu cần smoke/tập nhỏ vì tài nguyên, gọi đúng là smoke/tập nhỏ, không thay full-data recipe hoặc báo đã tái hiện benchmark. Chưa có ETA/epoch budget mới được xác nhận.
6. Lưu baseline tái hiện và kết quả kiểm chứng; sau đó mới phát triển phương pháp và so với chính baseline này. Bài toán novelty được để sau, không dùng nó chặn giai đoạn tái hiện.

Hiện tại mới hoàn tất việc khóa lại mục tiêu và audit một số assets. Chưa có trainer/notebook full CSA mới, tải checkpoint/model, feature extraction, training hoặc kết quả benchmark mới. Không tự sửa báo cáo Google Docs hoặc các số liệu lịch sử theo quyết định này.
