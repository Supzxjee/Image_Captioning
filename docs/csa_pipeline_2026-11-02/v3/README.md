# CSA figure — bản dựng lại sau phản hồi “vẫn chưa tới”

Thiết kế trước khi dựng, 02/11/2026: bỏ tiêu đề/chân trang dạng slide; dùng tỷ lệ hình bài báo, khối nhỏ và chữ serif; bank dạng cylinder, token features dạng chồng thẻ; ảnh COCO đầu vào thực đã có trong audit; đánh dấu frozen/trainable bằng ký hiệu nhỏ. Giữ nội dung CSA theo X051, không thay forward, trainer hoặc kết quả. Bản này là đề xuất tiếp theo, chưa được user chấp nhận về thẩm mỹ.

Ảnh minh họa COCO TRAIN263974 đã ghi tại `experiments/csa_paper/RID_IMAGE_SEMANTIC_CHECK.json`; chỉ minh họa đầu vào, không phải kết quả caption/detection của mô hình. Không vẽ thêm box detection hoặc câu caption giả trên ảnh. Hai hình zoom kỹ thuật vẫn có ở bản v2.

## File xem và chỉnh sửa

- [PNG tổng quan](01_overview.png), [SVG](01_overview.svg).
- [PowerPoint chỉnh sửa được, một hình](CSA_pipeline_editable.pptx).
- [PDF vector, một trang](CSA_pipeline_vector.pdf).
- [Nguồn dựng](build_pipeline.mjs), [bố cục](diagram_layout.json), [nguồn và hash](source_trace.json).
- Hình phóng to kỹ thuật: [encoder v2](../v2/02_encoder.png), [decoder v2](../v2/03_decoder.png).

Nội dung giữ theo X051: hai bank5269/1594 normalized512D, hai learned aggregators K150, `(V0,To)→(V1,T1)`, `(V1,T1)→Vo`, `(Vo,Tr)→Vr`, Vo có đường riêng vào decoder. Prefix TRAIN và VALID là hai chế độ thay thế; VALID beam5/CIDEr chỉ chọn checkpoint. Hai cross-attention/vector gates và thông số trainer nằm trong các hình zoom/đặc tả đã bàn giao. Hình chính không bổ sung giả thuyết, loss hoặc kết quả mới. N768 là raw ROI cache; projector học768→512; N biến thiên,max87 và genuine empty detections có full-frame fallback. CLIP patches vẫn chứa ngữ cảnh toàn ảnh, không phải object pixels cô lập.

Ảnh gốc: https://images.cocodataset.org/train2014/COCO_train2014_000000263974.jpg ; SHA256 `f208e31d4f689c88f1c0a6a1c583f1413c86424b10c52d62b8b6d3e2360e7406`, trùng audit đã ghi. Giữ nguyên tỷ lệ4:3, không sửa/cắt nội dung ảnh. Các ký hiệu bank/token là vector minh họa kiểu dữ liệu, không biểu diễn giá trị activation đo được.

Đã xem PNG và PDF vector. PPTX package/font/layout/import lại PASS,0findings/0warnings. Toàn bộ module/bank/token/text/đường nối là native editable shapes/connectors; chỉ bức ảnh COCO là picture. Chưa mở Microsoft PowerPoint. Không chạy model/Torch/autograd/Kaggle và không đổi trainer. Review kiến trúc độc lập và phản hồi thẩm mỹ của user vẫn pending.

Dựng lại bằng runtime Node/Python được môi trường cung cấp:

```bash
"$CODEX_PRIMARY_RUNTIME_NODE" docs/csa_pipeline_2026-11-02/v3/build_pipeline.mjs
```

Chuyển PPTX cũ sang tên khác trước khi dựng vì finalizer từ chối ghi đè. PNG/PPTX dựng bằng Artifact Tool; SVG dùng cùng tọa độ và text; PDF vector chuyển từ SVG qua PyMuPDF. Candidate/validation receipts ở `/workspace/csa-figure-v3/`, ngoài repo.
