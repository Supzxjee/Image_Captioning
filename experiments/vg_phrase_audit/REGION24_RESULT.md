# Phrase–region audit 24 ảnh — chưa chạy model

**Có nguồn mô tả–vùng hữu ích ngoài phạm vi cặp UPT. Nhưng inventory lọc tối thiểu chưa thể dùng nguyên trạng làm giám sát sạch.** Chưa triển khai G/T hoặc học region-captioning. Kết quả được gửi Astra để quyết định, không áp thêm ngưỡng cỡ mẫu bắt buộc.

## Inventory

Dùng hai annotation ZIP VG đã tải, đúng COCO split JSON và cache identity như audit trước. Chỉ train/restval; loại toàn bộ COCOval5000/test5000 và568known-exposed IDs. Không tải bộ ảnh VG, không mở rộng cache UPT. Sau giữ nguyên loại714ảnh mismatch tỉ lệ hình học, xét cả ảnh có và không có selected pair.

| Bước | Kết quả |
|---|---:|
| VG trong train10k cache sau known-ID exclusion | 4.093 ảnh |
| Qua kiểm tra tỉ lệ VG/COCO | 3.379 ảnh |
| Region gốc trước lọc tối thiểu | 168.646 |
| Phrase không rỗng, box gốc hợp lệ | **168.522 vùng / 3.361 ảnh** |
| Phrase khác nhau sau strip/lower để đếm | 131.719 |
| Ảnh không có selected UPT pair nhưng có ứng viên | **1.186 ảnh / 58.965 vùng** |

Loại123box không hợp lệ và1phrase rỗng. Không ép class COCO, một cạnh/hai object, ghép cặp, độ dài câu hoặc blacklist spatial. Mọi phrase/bbox gốc giữ nguyên. Median area region3.45%ảnh;3.431vùng chiếm>=75%ảnh. Đây là số lượng **ứng viên**, chưa phải số cặp giám sát đúng. Các câu một từ, mệnh đề chủ quan, vùng chỉ chứa một bộ phận và câu nhiều sự kiện vẫn có thể nằm trong inventory.

## Mẫu đã khóa

24ảnh khác nhau:8human–object,8spatial/other,8no-selected-UPT-pair. Loại30ảnh vừa xem khỏi mẫu mới. Dùng SHA256 ordering cố định, không chọn lại theo kết quả hình ảnh. Tags graph chỉ phân tầng, không là GT. Trước mở ảnh, nhận ra graph tag có thể gắn human–object cho phrase chỉ ghi “A gold necklace”, nên sửa tiêu chí chọn mẫu thêm human-word/relation-word cues cho hai nhóm liên quan; inventory vẫn lọc tối thiểu. Manifestv1 được giữ local, manifestv2 được khóa trước khi xem ảnh; mẫu cuối SHA256 nằm trong JSON kết quả. Không đổi mẫu sau khi xem ảnh. Kiểm tra byteSHA24/24COCOJPEG khớp cache.

## Đánh giá sơ bộ của trợ lý

| Câu hỏi | Phù hợp | Chưa rõ | Không phù hợp |
|---|---:|---:|---:|
| Toàn bộ phrase được ảnh hỗ trợ? | 16 | 6 | 2 |
| Vùng gốc chứa đủ bằng chứng? | 13 | 9 | 2 |
| Đúng instance được mô tả? | 20 | 2 | 2 |

**13/24mẫu đạt đồng thời cả ba tiêu chí theo xem ảnh sơ bộ:** human–object4/8, spatial/other3/8, no-pair6/8. Không tính nhãn unclear là false. Chưa có nhãn người xác nhận; đây là mẫu có chủ đích, không phải precision của VG và không so sánh thống kê với mẫu30phrase–pair khác ảnh trước đó. Chưa có kết quả sinh phrase/caption hoặc CIDEr mới.

Một số bằng chứng:

- **COCO295245 / VG2336827:** phrase nói hai surfboards dưới tay người đàn ông; ảnh lại là bé gái với bánh doughnut. Vùng gốc cực nhỏ, không chứa bằng chứng cho phrase. Tải riêng JPEG VG từ URL metadata để kiểm tra: vẫn cùng cảnh bé gái. Do đó mâu thuẫn không được giải thích chỉ bằng đổi JPEG COCO. Không suy rộng rằng toàn bộ mapping hoặc annotation đều sai.
- **COCO299346:** phrase nói mây trắng trên trời xanh; ảnh được xem là trời xanh có diều, còn crop gốc gần như trống xanh. Không tự sửa câu thành blue sky rồi tính annotation ban đầu là đạt.
- **COCO4424,431092:** câu có vẻ đúng trong toàn ảnh nhưng region chủ yếu chứa điện thoại+bàn tay hoặc quần shorts. Bằng chứng cho toàn bộ câu liên kết người–vật trong region riêng còn chưa rõ; không đòi bbox chứa mọi pixel cơ thể, cũng không tự mở rộng bbox.
- **COCO414112,340393:** dog/floor được khoanh đúng nhưng các mô tả sad-looking/dirty chưa đủ chắc. Đúng localization chưa bảo đảm mọi mệnh đề của phrase là đáng tin.
- **COCO535643,78426,231253:** các vùng urinal, dây điện thoại, kệ có đồ là ví dụ hữu ích trong ảnh có0selectedUPTpair. Không có cặp UPT không có nghĩa thiếu bằng chứng thị giác cho mọi mô tả.
- **COCO98497:** câu chỉ rõ hình vẽ người lướt sóng trên biển báo; không ép thành khẳng định có một người thật trong vùng đó.

## Diễn giải và bước tiếp

Bỏ yêu cầu ghép cặp giải phóng đáng kể **số lượng**, nhưng chưa tự giải quyết **tính đúng** và **độ đầy đủ của region**. Có lý do giữ hướng phrase–region để xem xét; chưa có cơ sở huấn luyện trực tiếp từ toàn bộ168.522 ứng viên hay tuyên bố căn chỉnh tốt hơn. Mốc1000ảnh/vài nghìn phrase của đề xuất trước chỉ là dự toán, không phải định luật hoặc lý do duy nhất để dừng một hướng.

Đề nghị Astra chọn bước tiếp dựa trên tóm tắt này: xử lý nguồn giám sát/tiêu chí evidence đáng tin hoặc đổi nguồn/mục tiêu, thay vì tự nối thêm heuristic để làm đẹp số lượng. Nếu về sau làm pilot, vẫn cần đối chứng cùng văn bản bổ sung nhưng không chỉ định vùng; inference/caption toàn ảnh không được dựa vào GT region. Region-captioning và CE đa nhiệm có tiền lệ, chưa có novelty được chứng minh. CLIP patch có global context nên ROI đọc patch cũng chưa bảo đảm bằng chứng thuần cục bộ.

## Artifact và tái lập

- [Inventory result](REGION_INVENTORY_RESULT.json), [locked24manifest](REGION24_MANIFEST.json).
- [Assistant review](REGION24_ASSISTANT_REVIEW.json): ba tiêu chí tách riêng, ghi chú từng mẫu; không humanGT.
- [COCO image byte provenance](REGION24_IMAGE_PROVENANCE.json), [VG source spot check](REGION24_SOURCE_CHECK.json).
- Local gallery: `review/region24/index.html`, có ảnh toàn cảnh, bbox gốc, crop gốc và ghi chú; JPEG không đẩy GitHub.
- `python experiments/vg_phrase_audit/audit_regions.py`; `python experiments/vg_phrase_audit/make_region_review.py`.
- Seven pure-data tests PASS, gồm ràng buộc split, box không clip/expand, lấy đủ24ảnh khác nhau và phân tầng ổn định. Numeric artifacts/mapping checked; visual labels do not auto-regenerate from code.

Không chạy GPU, encoder, notebook train hoặc model inference. Đây là audit nguồn giám sát trên giao dữ liệu đang có, không phải đánh giá toàn VG.
