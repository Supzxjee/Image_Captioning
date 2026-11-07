# Pipeline V4 — mở cơ chế xử lý

07/11/2026, Asia/Bangkok. Theo [brief người dùng](USER_BRIEF.txt), dùng diagram-design 2.6.142. [Phạm vi và lựa chọn bố cục](LAYOUT_BRIEF.md). Bản V3 giữ nguyên.

[Hình toàn bộ — PNG 7200×7552](captioning-mechanisms-v4.png) · [SVG chỉnh sửa được](captioning-mechanisms-v4.svg) · [PDF vector](captioning-mechanisms-v4.pdf) · [HTML tự chứa](captioning-mechanisms-v4.html).

[PDF 8 trang — từng panel, có mục lục](captioning-mechanisms-v4-panels.pdf) thuận tiện để đọc chi tiết.

Master là hình tham chiếu đầy đủ, canvas `fit` 3600×3776. Khi dàn trang báo, dùng từng panel ở dưới hoặc chia thành nhiều hình; thu cả master vào một cột sẽ làm chữ quá nhỏ. Chữ và đường nét SVG/PDF là vector; chỉ ảnh I là raster gốc, không thêm bbox. Ký hiệu tensor lặp cùng tên là cùng dữ liệu, giúp tránh các bus dài cắt nhau. Guide nét đứt A/B/C là liên kết phóng to, không phải luồng dữ liệu; D/E liên hệ bằng mã panel ở module tương ứng.

## Các panel riêng

| Nội dung | SVG chỉnh sửa | PDF vector | PNG |
|---|---|---|---|
| Luồng tổng thể, 5 giai đoạn | [SVG](panel-overview-v4.svg) | [PDF](panel-overview-v4.pdf) | [PNG](panel-overview-v4.png) |
| ROI pooling theo diện tích giao | [SVG](panel-roi-pooling-v4.svg) | [PDF](panel-roi-pooling-v4.pdf) | [PNG](panel-roi-pooling-v4.png) |
| A — bank TRAIN và aggregation | [SVG](panel-a-banks-v4.svg) | [PDF](panel-a-banks-v4.pdf) | [PNG](panel-a-banks-v4.png) |
| B — fusion chung, ba bộ tham số | [SVG](panel-b-fusion-v4.svg) | [PDF](panel-b-fusion-v4.pdf) | [PNG](panel-b-fusion-v4.png) |
| C — UPT/matching và hai adapter | [SVG](panel-c-adapters-v4.svg) | [PDF](panel-c-adapters-v4.pdf) | [PNG](panel-c-adapters-v4.png) |
| D — decoder song song và beam | [SVG](panel-d-decoder-v4.svg) | [PDF](panel-d-decoder-v4.pdf) | [PNG](panel-d-decoder-v4.png) |
| E — ba điểm reranking | [SVG](panel-e-reranking-v4.svg) | [PDF](panel-e-reranking-v4.pdf) | [PNG](panel-e-reranking-v4.png) |
| F — huấn luyện riêng | [SVG](panel-f-training-v4.svg) | [PDF](panel-f-training-v4.pdf) | [PNG](panel-f-training-v4.png) |

## Caption đề xuất

**Kiến trúc mô hình chú thích ảnh với bank ngữ nghĩa và prompt gắn vùng.** CLIP ViT-B/16 và box YOLO tạo region descriptors bằng ROI pooling có trọng số diện tích giao patch–box. Bank noun/verb được xây một lần từ reference TRAIN; hai aggregation riêng tạo các prototype mềm. Ba fusion lần lượt cập nhật đặc trưng object và relation, dùng cùng cấu trúc với tham số riêng. Prompt riêng ảnh từ UPT được mã hóa bằng frozen CLIP text và ghép tới ROI theo class/IoU. Mỗi Slot MLP nối embedding chữ, đặc trưng vùng từ **Vo gốc** và tọa độ box chuẩn hóa; relation giữ thứ tự hai endpoint. Hai adapter cập nhật residual qua attention có mask liên kết và output projection khởi tạo 0. Hai memory sau adapter đi riêng vào decoder song song. Reranking chọn trong năm ứng viên bằng `S = 0.3D + 0.4C + 0.3O`, D/C min–max trên cùng năm câu. Huấn luyện XE rồi XE + 0.05 SCST dùng CIDEr được đặt trong khung riêng. Tuyết xanh biểu thị tham số đóng băng; lửa cam biểu thị tham số học được khi huấn luyện.

**English.** Architecture of image captioning with semantic banks and region-linked prompts. Frozen visual features and detection boxes yield region descriptors through overlap-weighted patch pooling. Separate learned aggregators transform TRAIN-derived noun and verb banks into soft prototypes. Three semantic fusion stages produce object and relation memories. Image-specific UPT phrases and matched ROI links form slots by concatenating text embeddings, original object-region features and normalized box coordinates. Masked prompt adapters update the two memories through zero-initialized residual projections. A three-layer parallel decoder generates five beam candidates, followed by weighted likelihood, image–text alignment and object-consistency selection. Cross-entropy pretraining and subsequent SCST with cross-entropy are shown separately from inference. Snowflakes and flames denote frozen and trainable parameters during training.

## Chi tiết đã đối chiếu với mã

| Phần hình | Nguồn và cơ chế |
|---|---|
| Visual/ROI | `experiments/csa_paper/data.py::roi_pool`, `prepare_real_assets.py::make_regions`: bỏ CLS, grid14×14×768, diện tích giao từng patch, weighted mean; projection768→512 trong model. Không thay bằng crop encoder. |
| A | `model.py::SemanticAggregation`: Wa→MLP, softmax chiều bank, Wb và tích Aᵀ(EWb); noun/verb có tham số riêng. `prepare_real_assets.py`: POS/lemma TRAIN (train+restval). |
| B | `model.py::Alignment/Fusion/CSACaptioner.encode`: soft region attention τ=.1 → relevance → soft prototype weights; self/cross-attention và đủ residual/norm/FFN; gate trộn H,C; object2 nhận T1. |
| C | `prompt_adapter_data.py::join_entry/geometry/tensor_batch`, `prompt_adapter_model.py::RegionPromptAdapter/PromptCSA.encode`: class-consistent IoU≥.5; ROI i/cặp(s,o); concat text/Vo gốc/4 hoặc8 tọa độ; Linear→GELU→Linear; linked mask, zero-init output, empty-slot zero update; không gate adapter. |
| D | `model.py::DecoderLayer/CSACaptioner.decode/beam_search`: QKV self-attention là layer input; cross queries H, memories riêng; gate dùng e thô, input đầu là √512e+PE; residual FFN từ gated sum, không thêm H; heads8/layers3, beam5/max20, raw sum logp. |
| E | `prompt_scst_rerank.py::worker/fixed_rank`, `captioning/multiscore_reranking.py`: CLIPScore=2.5max(cos,0); D/C min–max trên5 (nếu bằng nhau trả1); O=(consistency+1)/2 với penalty1; weights .3/.4/.3, hòa ưu tiên beam trước. |
| F | `scst_pilot_model.py::rollout/policy_loss`, `run_prompt_scst.py`: sampling/greedy cùng mô hình, CIDEr TRAIN advantage detach, logp sampled; XE được tính lại trên batch hiện tại trong Stage2. Không object reward. |

[Source trace/SHA256](source_trace.json) khóa các file đã đọc. Sơ đồ nêu cấu trúc xử lý, không phải kết quả inference mới, phép đo mới hoặc đề xuất thay kiến trúc. O chỉ chọn caption; CHAIR chỉ đánh giá; reference không tham gia reranking. Dropout, padding safety và chi tiết khởi tạo ngoài zero-init được gói trong module.

## Kiểm tra và sửa hình

[Figure audit](figure_audit.json) · [Chromium text/icon/arrow audit](render_audit.json) · [Geometry](diagram_geometry.json).

Kiểm các route ngang/dọc, giao cắt/chồng nét/transit/node overlap; text trong node; icon với chữ; loose labels với connector; canvas bounds; nhãn Slot MLP/concat/Vo gốc/QKV, ba fusion, zero-init, loss và selector; đối chiếu hash HTML với lần render, kích thước PNG và PDF vector. Chạy self_check và verify-geometry của skill, rồi xem render. Các panel dịch trong master được kiểm độc lập; guide master được kiểm riêng. Chỉ static/graphics, không Torch/model/autograd/notebook/scorer/Kaggle, không metric/Word/Sheet.

Nguồn tọa độ/nhãn: [build_figure.py](build_figure.py). Từ thư mục này:

```bash
python build_figure.py
python render_figure.py
python assemble_panels.py
python audit_figure.py
```

Renderer dùng helper export SVG/PNG của skill và Chromium hệ thống qua local HTTP. Có thể truyền các stem cần xuất lại để giữ các bản không đổi. Sửa SVG trực tiếp trong Inkscape/Illustrator được; connector không tự bám khi kéo module. Font hệ thống Arial/Liberation Sans; không tải font mạng. Không có JavaScript hoặc nội dung ngoài mạng trong HTML/SVG bàn giao.
