# Chạy lại ngay bằng điểm chính thức của UPT

Notebook: `VCOCO_UPT_NATIVE_SCORE_AUDIT.ipynb`, hoặc dán toàn bộ
`kaggle_vcoco_upt_native_score_audit.py` vào một cell. Người dùng tự chạy,
Internet ON và GPU T4 x2. Không cần chờ hay attach full UPT cache.

Input duy nhất là MS COCO đang dùng:

```python
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
```

Giữ nguyên model HICO-DET pretrained, preprocessing, native proposal settings,
matching GT, 507 Calibration + 2.867 Development và policy >=90% known precision,
>=5 known accepts. Không fine-tune và không đánh giá caption.

Thay đổi duy nhất trong quyết định nhận claim: đọc trực tiếp `official_output`
của UPT, gồm sigmoid verb logit nhân prior người/vật; không tự tính công thức mới.
Native postprocessing chỉ xuất các tổ hợp có prior khác 0. Giữ raw claims riêng
để đối chiếu, không dùng raw score để chọn lại khi native score không đạt.
Nếu vẫn không đạt, tiếp tục abstain thay vì giảm mục tiêu precision.

Hai worker suy luận lại 3.374 ảnh trên hai GPU độc lập. Không chạy full 123.287 ảnh.
Tái dùng annotation và weights trong `/tmp/vcoco-upt-relation-audit` nếu session
cũ còn dữ liệu; session mới tự tải lại nguồn nhỏ và weights ~220 MB. Repository
được đặt trong `/tmp` để tránh hàng trăm file repo chiếm danh sách Output.

Core commit pin: `eb00d6c0c7d4ea6718de3cb55d1049a15ce9fbd5`.
Output mới `/kaggle/working/vcoco_upt_native_score_audit`; không ghi đè audit cũ.
Journals lần này giữ bbox, class, confidence, raw claims và native claim scores.
Calibration khóa ngưỡng trước khi tính Development. Đây là follow-up sau khi đã
xem kết quả raw-score thất bại, không phải một xác nhận độc lập mới.

Khi chạy xong gửi `vcoco_upt_native_score_audit_review.zip`.

Đã kiểm tra 32 CPU tests cho evaluator, nguồn nhãn, matching và split. Test mới
kiểm tra đúng cặp/verb, giữ nguyên official score, output rỗng và sai chiều dữ liệu.
Notebook khớp source và syntax PASS. Chưa chạy GPU; chưa biết native score có đạt
precision/coverage hữu ích hay không. Known-label precision không đánh giá đầy đủ
những target chưa có nhãn; unknown vẫn được báo riêng.
