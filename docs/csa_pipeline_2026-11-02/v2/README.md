# Pipeline CSA — bản sửa bố cục theo ảnh mẫu

Ngày 02/11/2026, bàn giao X051-B. User nhận xét bản đầu xấu và rối, yêu cầu làm như ảnh đã gửi. Bản này thay bố cục hình tổng quan: nhánh offline chạy ngang phía trên; ảnh và feature cache nằm bên trái; ba tầng CSA xếp dọc bên phải; decoder và caption đặt bên dưới. Dùng font serif, nét đen mảnh, module bo góc và khung nhóm nét đứt. Giảm nhãn phụ, bỏ badge F/T trên từng khối; frozen encoders được ghi trực tiếp và các module đang học có chú thích ở cuối hình. Chi tiết selection/gates để ở hai hình phóng to.

## File sử dụng

- [PNG tổng quan](01_overview.png) · [SVG tổng quan](01_overview.svg).
- [PPTX chỉnh sửa được — 3 hình](CSA_pipeline_editable.pptx).
- [PDF vector — 3 trang](CSA_pipeline_vector.pdf).
- Encoder: [PNG](02_encoder.png) · [SVG](02_encoder.svg).
- Decoder và TRAIN/VALID: [PNG](03_decoder.png) · [SVG](03_decoder.svg).
- Nguồn dựng: [build_pipeline.mjs](build_pipeline.mjs), [diagram_layout.json](diagram_layout.json); hash nguồn/output: [source_trace.json](source_trace.json).

PPTX chứa shape/text/connector riêng, không nhúng cả hình thành ảnh. Các đường gấp khúc gồm nhiều đoạn connector; di chuyển module cần sửa endpoint tương ứng. Font Liberation Serif, canvas1920×1180. Batch dimension được bỏ khỏi tensor labels.

## Nội dung và giới hạn

Giữ forward CSA adaptation đã đối chiếu ở [bản bàn giao đầu](../README.md#đối-chiếu-nguồn), model SHA256 `139edbb1b73caf3dadaf5c960b25542e282f88baf2ce9bdf141de922ceacbc09`. Nguồn code không đổi khi sửa hình: hai bank frozen, hai aggregators học, To→T1→object2, Vo→relation và nhánh Vo riêng vào parallel decoder. K150 là số prototypes; N thay đổi theo ảnh, max87 trong cache full. Main figure chỉ thể hiện decoder như một module; hai CA/vector gates độc lập và raw previous-word embedding nằm ở hình decoder.

TRAIN dùng reference prefix shift-right và XE; VALID sinh từ BOS/prefix tự sinh rồi CIDEr chọn checkpoint, không backward; teacher-forced NLL là eval riêng. Hình mô tả kiến trúc/lịch đã chuẩn bị; fullXE10 attempt1 dừng trước TRAIN theo X052, runtime V2 chưa có kết quả. Chỉ lấy bố cục của ảnh mẫu, không thêm retrieval/BLIP vào CSA. Không có thay đổi phương pháp, trainer hay kết quả thực nghiệm.

## Kiểm tra và dựng lại

Đã xem từng PNG và ba trang PDF; kiểm PPTX package/font/layout/import lại bằng Artifact Tool pass,0findings/0warnings. XML xác nhận3slides có native shapes/connectors và0pictures. Chưa mở Microsoft PowerPoint; không coi kiểm tra cấu trúc là kiểm tra phần mềm đó. Không chạy model/Torch/autograd/Kaggle.

Dùng runtime Node đã cung cấp, Artifact Tool, skia-canvas, Python có PyMuPDF:

```bash
"$CODEX_PRIMARY_RUNTIME_NODE" docs/csa_pipeline_2026-11-02/v2/build_pipeline.mjs
```

Chuyển PPTX cũ sang tên khác trước khi dựng lại vì finalizer từ chối ghi đè. PDF vector chuyển từ cùng SVG primitives qua PyMuPDF; arrowheads/dashes là vector segments/polygons. Candidate/validation receipts nằm ngoài repo tại `/workspace/csa-pipeline-build-v2/`. Bản đầu được giữ ở thư mục cha để lưu lịch sử; bản v2 là bản đề nghị dùng hiện tại. Review độc lập theo X051 vẫn pending.
