# Pipeline với icon tham số và query/residual tách riêng

07/11/2026, UTC+7. Bản vẽ bằng diagram-design v2.6.142 theo [brief mới đầy đủ](USER_BRIEF.txt). Đây là cập nhật hình minh họa của mô hình đang có, không phải thay đổi kiến trúc hoặc kết quả nghiên cứu. Bản trước được giữ nguyên.

## Tệp bàn giao

- [SVG chỉnh sửa được](captioning-pipeline.svg).
- [PDF vector](captioning-pipeline.pdf): một trang, chữ chọn được; một ảnh raster gốc được nhúng, các module, icon và đường nối là vector.
- [PNG 5120×2944](captioning-pipeline.png).
- [HTML độc lập](captioning-pipeline.html): sơ đồ và caption bên ngoài, không gọi font qua mạng.

## Caption tiếng Việt

**Kiến trúc mô hình chú thích ảnh với ngữ nghĩa toàn cục và prompt riêng ảnh.** CLIP và box YOLO tạo region features. Bank danh từ/động từ từ TRAIN được tổng hợp để hợp nhất tuần tự với các vùng, tạo hai memory riêng. Prompt từ UPT được nối với ROI hợp lệ và cập nhật memory qua adapter; mỗi adapter nhận cả visual query và đường residual. Relation slot dùng `Vo` gốc. Decoder đọc riêng `Vo′/Vr′`, sinh năm ứng viên và chọn caption bằng ba điểm reranking. Ảnh I là ảnh gốc không vẽ bbox; biểu diễn token chỉ minh họa.

**Icon tham số.** Bông tuyết xanh dương: frozen; ngọn lửa cam đỏ: trainable trong giai đoạn huấn luyện. Tất cả đều là icon vector có màu cố định, không dùng emoji. Khi suy luận, mọi trọng số giữ nguyên. Lửa tại box `Word embedding / Fixed position` chỉ áp dụng cho word embedding; positional encoding cố định. Lửa tại `Vocabulary projection / + softmax` chỉ áp dụng cho projection; softmax không có tham số học. Bông tuyết tại image–text alignment biểu thị CLIP đóng băng dùng để chấm điểm.

**Điều chỉnh theo phản hồi mới nhất:** bông tuyết xanh `#1677B8`; lửa viền `#D95F1A`, nền cam `#F6A23A`. Toàn bộ mũi tên/zoom guides là đường thẳng ngang/dọc, đổi hướng bằng góc vuông 90°. Yêu cầu mới này cập nhật quy ước icon đơn sắc và connector bo góc của brief/skill ban đầu.

## Chi tiết giữ đúng

- Dùng đúng ba fill pastel: neutral `#EDF3F8`, object/noun `#FCEBDD`, relation/verb `#E2F1F5`; nền trắng, chữ `#263238`, đường `#56616B`. `Vo/Vo′` cam, `Vr/Vr′` xanh, patch features và `V` trung tính.
- Ảnh chỉ đi vào một CLIP visual encoder ViT-B/16 lấy patches; không chạy CLIP riêng cho từng crop. YOLO không ghi phiên bản chưa xác minh. Nhánh box thể hiện tọa độ `B`, không dùng ảnh đã vẽ khung làm đầu vào ROI pooling.
- Có bước TRAIN caption → noun/verb lemmas → frozen CLIP text → fixed bank embeddings. Hai learned semantic aggregation tạo prototype riêng; không dùng caption VALID/TEST làm bank.
- Ngay sau object fusion 2, một nhánh đi ra `Vo`, một nhánh tiếp tục vào relation fusion. Không gộp `Vo/Vr`.
- Cả hai adapter có **hai đường riêng** từ memory: một vào masked attention làm visual query, một vào dấu cộng residual. Text embeddings và valid region links cùng đi vào linked slots. Relation slot dùng `Vo[s]/Vo[o]` gốc theo thứ tự, không dùng `Vo′`.
- Hai memory sau adapter đi riêng vào parallel cross-attention. Khung `×3` chỉ bao masked self-attention, parallel cross-attention, gated fusion + FFN; embedding, fixed positional encoding, vocabulary, beam và rerank ở ngoài.
- D, C, O có ba mũi tên riêng hội tụ vào weighted scoring and selection, trước final caption. Không gắn lửa cho ROI pooling, matching, softmax, beam, fixed scoring hoặc selection.

## Chi tiết giản lược

1. Residual/norm/FFN trong fusion, cosine temperature và selected-prototype alignment được gói trong module; `T1` vẫn truyền từ object fusion 1 sang object fusion 2.
2. Linked slots gói phép ghép text, original region features và geometry. Output projection được gói trong attention/residual update. Chi tiết mask, 8 heads, `Wout` zero-init và fallback không có slot → update bằng 0 chuyển ra caption kỹ thuật, không nhồi vào hình.
3. Công thức rerank cố định: `S(y)=0.3D(y)+0.4C(y)+0.3O(y)`. D và C min–max trên cùng 5 ứng viên; O theo normalization của bộ object consistency; hòa điểm ưu tiên beam đứng trước. Reference captions và CHAIR không tham gia chọn câu. CLIPScore dùng ảnh/caption; consistency dùng YOLO labels và objects được nhắc trong caption.
4. Sinh tự hồi quy từ BOS, không vẽ đường hồi tiếp dài. Các cổng `I`, `YOLO boxes`, memory lặp tên/màu là cùng dữ liệu, tránh dây xuyên các khung. Các nét đứt không đầu mũi tên chỉ giải thích block tổng quan, không phải luồng tensor bổ sung.
5. Training chỉ có một dòng dưới hình. Loss hiện tại là `LCE + 0.05 LSCST`, advantage CIDEr so sampling với greedy dùng reference TRAIN. Không thêm object reward; O thuộc reranking, CHAIR chỉ hậu kiểm.

Không dùng tên viết tắt phương pháp bị cấm trong hình; không thêm retrieval, BLIP hoặc Q-Former. Font khai báo Arial; môi trường render dùng Liberation Sans tương thích Arial và font ký hiệu hệ thống. Không có gradient, shadow hoặc hiệu ứng 3D. Các kích thước quan trọng giữ lại: patches 14×14×768, projection 768→512, V/memories N×512, prototypes 150×512, decoder ×3, beam K=5.

## QA và tái tạo

[Audit](figure_audit.json), [đo chữ](render_audit.json), [nguồn và hash](source_trace.json). Kiểm 67 tuyến và 58 đối tượng: không phát hiện giao cắt, chồng nét, transit qua module khác, node chồng nhau hoặc text overflow. Kiểm riêng màu icon/legend, connector không có lệnh curve và badge map, một ảnh không có box overlay, square data rectangles, query/residual ở cả hai adapter, hai zoom guides không arrowhead và các nhãn bị cấm. Đã xem PNG và PDF render. PDF có 311 vector drawings và một ảnh raster gốc.

`build_pipeline.py` dùng lại primitives của bản trước rồi dựng riêng hình này. `render_figure.py` gọi các helper SVG/PNG của skill qua Chromium hệ thống/local HTTP; không sửa skill. `audit_figure.py` kiểm geometry/PDF/content. Cần đường dẫn skill hiện có, Playwright, Chromium và PyMuPDF nếu tái tạo trên máy khác.

```bash
python build_pipeline.py
python render_figure.py
python audit_figure.py
```

Không chạy model/Torch/autograd/scorer/Kaggle, không thay metrics, Word hoặc Sheet. Các khối và chữ trong SVG chỉnh sửa riêng được; ảnh raster chỉ là asset minh họa đã có trong repo. Mũi tên không tự bám khi kéo module trong trình sửa SVG. Nên dàn toàn chiều rộng trang; bản vector cho phép chỉnh typography theo template tạp chí.
