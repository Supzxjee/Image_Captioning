# Pipeline CSA hiện tại — bàn giao X051-A

**Bản đề xuất mới nhất: [v3 — dựng lại dạng hình bài báo](v3/README.md).** User vẫn thấy v2 chưa đạt; v3 bỏ tiêu đề/chân trang lớn, dùng ảnh COCO thật, bank dạng cylinder, token chồng thẻ và ký hiệu frozen/trainable. [PNG v3](v3/01_overview.png), [PPTX v3](v3/CSA_pipeline_editable.pptx), [PDF v3](v3/CSA_pipeline_vector.pdf). Chưa có user chấp nhận thẩm mỹ. V1/v2 được giữ lịch sử; hình zoom encoder/decoder vẫn ở v2.

Ngày 02/11/2026. Vẽ theo X051 và hình mẫu user gửi trong chat: nền trắng, font serif, nét đen mảnh, khung offline nét đứt; các khối và đường nối giữ chỉnh sửa được. Nội dung là **CSA adaptation — cached CLIP ROI + semantic banks**, không phải pipeline retrieval/BLIP của hình mẫu.

## File sử dụng

- [PowerPoint chỉnh sửa được, 3 hình](CSA_pipeline_editable.pptx): native shapes/text/connectors, không nhúng cả sơ đồ thành ảnh.
- [PDF vector, 3 trang](CSA_pipeline_vector.pdf).
- Tổng quan: [PNG](01_overview.png), [SVG](01_overview.svg).
- Phóng to encoder: [PNG](02_encoder.png), [SVG](02_encoder.svg).
- Phóng to decoder và TRAIN/VALID: [PNG](03_decoder.png), [SVG](03_decoder.svg).
- Nguồn dựng hình: [build_pipeline.mjs](build_pipeline.mjs); bố cục máy đọc được: [diagram_layout.json](diagram_layout.json).

PPTX và SVG đều có text/khối/connector chỉnh sửa riêng. Trong PPTX, connector nhiều đoạn được lưu thành các đoạn riêng; khi di chuyển một module cần chỉnh các endpoint liên quan. F=frozen/cache; T=trainable. Batch dimension được bỏ khỏi nhãn tensor để hình gọn. N thay đổi theo ảnh; K=150 là số prototypes, không phải số vùng detection.

## Đối chiếu nguồn

Nguồn đọc ở commit `5a14d1b641732b42877ea8a458b8fc4e746a1643`:

| Đường trong hình | Nguồn xác định |
|---|---|
| Visual projection, hai aggregators, To→T1→object2, Vo→relation và nhánh Vo riêng | `experiments/csa_paper/model.py`: `CSACaptioner.encode`, `SemanticAggregation.forward` |
| Cosine, region temperature0.1, prototype weighting, scalar gate own/cross | `model.py`: `Alignment.forward`, `Fusion.forward` |
| Hai CA với cùng Q, raw previous-word embedding cho hai vector gates, không chia√2 | `model.py`: `DecoderLayer.forward`, `CSACaptioner.decode` |
| 3layers,8heads,FFN2048, vocabulary10000 và beam5/max20/raw score | `model.py`, `FULL_XE_10_PROTOCOL.md`, `full_xe_decode.py` |
| Bank5269×512/1594×512, raw ROI N×768, max87, fallback | X051; `FULL_ROI_RESULT.md`, `LEMMA_CORRECTION_RESULT.md` |
| TRAIN113287/566747events; VALID5000/25010refs; checkpoint selection | `FULL_XE_10_PROTOCOL.md`, `run_full_xe.py` |

`model.py` SHA256: `139edbb1b73caf3dadaf5c960b25542e282f88baf2ce9bdf141de922ceacbc09`. Không import hoặc chạy module model để dựng hình. Chi tiết paper/source khác nhau đã có trong `CSA_PAPER_IMPLEMENTATION_SPEC.md` và `CSA_PAPER_IMPLEMENTATION_REVIEW.md`; hình lấy **forward đang triển khai**, không thay nó bằng sơ đồ CSA gốc.

Các giới hạn cần giữ khi trình bày: producer boxes chưa được định danh thêm thành YOLO/DETR; pooled CLIP patches có ngữ cảnh toàn ảnh; global banks không lấy reference ảnh đang suy luận. TRAIN chỉ XE10, không SCST/RL. VALID beam generation không nhận reference prefix; teacher-forced NLL là nhánh eval riêng; CIDEr chỉ chọn checkpoint, không backward. TEST chưa dùng. Không nhận attention/prototype selection là chứng minh từng từ đã bám đúng box.

X052 thông báo attempt1 dừng trước TRAIN ở beam preflight và V2 dùng decoder per-image hiện có; forward không đổi. Hình mô tả kiến trúc/lịch đã chuẩn bị, không báo full10epochs đã thành công. Không tạo experiment hoặc Sheet row mới; chờ Codex1 review các đường nối trước khi dùng chính thức trong báo cáo.

## Kiểm tra xuất bản

Đã render và xem từng PNG, kiểm bản PDF vector và chữ tiếng Việt. PPTX package/font/layout và import lại bằng Artifact Tool đều pass; geometry findings0, warnings0 ở bản cuối. Kiểm XML cho thấy3slides đều có native shapes/connectors và0pictures, không phải screenshot trong slide. Chưa mở bằng Microsoft PowerPoint; các kiểm tra cấu trúc không thay kiểm tra phần mềm đó. Không có model/Torch/autograd/Kaggle hoặc trainer nào chạy/thay đổi trong lượt vẽ.

## Dựng lại

Dùng runtime Node được môi trường cung cấp, có `@oai/artifact-tool`, `skia-canvas`, cùng Python runtime có PyMuPDF. Font: Liberation Serif. Chạy từ root repo:

```bash
"$CODEX_PRIMARY_RUNTIME_NODE" docs/csa_pipeline_2026-11-02/build_pipeline.mjs
```

Trước khi dựng lại, chuyển bản PPTX cũ sang tên khác: finalizer từ chối ghi đè file đã tồn tại. PNG/SVG/layout/PDF được dựng lại từ cùng primitives; PDF dùng SVG→PyMuPDF vì exporter PDF tích hợp của runtime thiếu package standard-fonts. Dashes và arrowheads trong SVG được dựng thành vector segments/polygons để giữ đúng ở PDF. Các candidate/validation receipts nằm ngoài repo tại `/workspace/csa-pipeline-build/`; chỉ output cuối và source dựng hình được commit.
