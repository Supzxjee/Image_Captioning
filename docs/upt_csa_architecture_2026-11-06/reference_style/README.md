# Bản thử theo phong cách ảnh tham khảo

06/11/2026 (UTC+7). Người dùng yêu cầu thử bố cục giống ảnh CSA-Net vừa gửi trong chat. Vẫn dùng cấu hình của [brief gốc](../USER_BRIEF.txt); chỉ thay cách trình bày, không thay kiến trúc, mô hình, notebook hoặc kết quả.

[Tải SVG chỉnh sửa](upt-csa-reference-style.svg) · [PDF](upt-csa-reference-style.pdf) · [PNG 4400×2320](upt-csa-reference-style.png) · [HTML](upt-csa-reference-style.html).

## Chú thích ngắn

**Kiến trúc mô hình captioning kết hợp CSA và prompt gắn vùng.** Phía trên là hai tầng object fusion và một tầng relation fusion, nhận prototype từ bank TRAIN. Hai vùng nét đứt bên dưới mô tả prompt đối tượng và quan hệ riêng ảnh: CLIP mã hóa cụm từ, MLP tạo slot và masked attention cập nhật từng memory qua residual. Bộ sinh ngôn ngữ bên phải đọc riêng `Vo′/Vr′`, sinh năm ứng viên rồi reranking. Ảnh, các box màu và cụm từ minh họa chỉ giúp đọc hình, không phải kết quả detector hoặc caption mới được chạy.

## Những điểm lấy từ ảnh mẫu

- Cascade chạy ngang ở hàng trên; các thao tác attention/FFN được vẽ thành thanh đứng có mũi tên nội bộ.
- Khung semantic nét đứt cam/xanh, phrase pool dạng trụ, encoder chữ dạng nêm và memory dưới dạng token stack có màu.
- Decoder đứng bên phải, đọc từ dưới lên; bracket `×3` chỉ bao ba decoder layers, không bao embedding, vocabulary head hoặc beam search.
- Ảnh đầu vào và ví dụ liên kết ROI đặt bên trái; ma trận màu minh họa điều kiện mask trong adapter.

## Nội dung giữ đúng và chi tiết đã gói

- Giữ global noun/verb banks **tách biệt** với image-specific UPT phrases. `OSA/ORA` là tên ngắn trong hình cho prototype alignment/cross-attention của các Fusion block, không thêm module hoặc tuyên bố tái lập chính xác CSA-Net.
- Giữ `Vo` gốc trong cả hai slot MLP; relation slot giữ thứ tự `(s,o)` và hai box / geometry 8 chiều. Object geometry có 4 chiều.
- Cả hai adapter vẫn có Q/K/V, mask, output projection `Wout` và cộng residual. Text/prompt không đi thẳng vào decoder.
- Các token stack trùng tên `Vo′/Vr′` và cổng `Links ℓ` biểu thị cùng dữ liệu; dùng named ports để tránh dây cắt nhau. Memory đi vào hai nhánh parallel CA riêng biệt.
- Self-attention residual/norm và đường own-branch đi vào scalar fusion gate được gói trong Fusion block. Gate của decoder được gói trong `Gated fusion + FFN`; đầu vào gate có word embedding như mô hình gốc.
- Bản này tập trung hình overview theo ảnh mẫu. Mục tiêu huấn luyện chỉ còn một dòng ở dưới; công thức đầy đủ, quy tắc min–max, normalization và tie-break của reranking được giữ trong [bản ba panel](../README.md). Không bỏ hoặc đổi loss của mô hình.
- Ảnh minh họa tái sử dụng asset đã có trong repo `docs/csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg`. Các box được vẽ tay để minh họa; ma trận mask là ví dụ ký hiệu, không phải attention map đo được. Cụm `person / bicycle / person riding a bicycle` không được báo như output UPT.

## Tệp nguồn và kiểm tra

`build_reference.py` tạo HTML và geometry từ primitives của bản ba panel; `render_figure.py` dùng helper của diagram-design v2.6.142 để xuất SVG/PNG và Chromium để xuất PDF. `audit_figure.py` kiểm hình học, text bounds và PDF.

[QA](figure_audit.json): 54 tuyến, 35 đối tượng; không phát hiện giao cắt, chồng nét, transit qua node, node chồng nhau hoặc chữ tràn box. Skill self-check PASS; đã xem PNG và PDF render. PDF một trang có 225 vector drawings, chữ chọn được và **một ảnh raster minh họa** được tái sử dụng tại hai vị trí; các khối/đường/chữ vẫn vector. SVG giữ ảnh nhúng cùng các đối tượng vector chỉnh sửa được. Không chạy Torch/model/scorer/Kaggle.

```bash
python build_reference.py
python render_figure.py
python audit_figure.py
```
