# SPICE chi tiết — CONTROL188 / PROMPT64

Hai lượt CPU COMPLETE; caption nguồn CONTROL184/PROMPT61, trọng số CONTROL180/PROMPT60 giữ nguyên. Không huấn luyện, sinh câu hay chọn câu lại. 85 kiểm tra metadata PASS, SPICE All khớp kết quả chấm trước185/62.

| Cấu hình | Giải mã | SPICE tổng | Đối tượng | Quan hệ | Thuộc tính |
|---|---|---:|---:|---:|---:|
| CONTROL | normal | 19.2469 | 35.0505 | 5.3841 | 9.1328 |
| PROMPT | normal | 19.3663 | 35.3947 | 5.6055 | 8.8702 |
| CONTROL | fixed_rerank | 20.9465 | 37.9115 | 5.9781 | 10.4835 |
| PROMPT | fixed_rerank | 21.1609 | 38.2200 | 6.3467 | 10.4325 |

Điểm nhân100. normal là beam rank0, không phải greedy. Số ảnh có điểm hợp lệ: tổng/đối tượng5000; quan hệ4998; thuộc tính4989. Tập ảnh hợp lệ giữa hai cấu hình khớp theo từng nhóm; các điểm undefined bị loại khỏi trung bình, không điền0.

## Diễn giải

Sau reranking, PROMPT tăng0,3085đ đối tượng và0,3686đ quan hệ; thuộc tính giảm0,0510đ. Trước reranking, tương ứng+0,3441/+0,2214/−0,2626. Không thể nói mọi mặt ngữ nghĩa đều cải thiện. Kết hợp CHAIR trước đây, dữ liệu tiếp tục ủng hộ cấu hình PROMPT giảm ảo giác đối tượng, đồng thời có tín hiệu tốt hơn về mô tả quan hệ. Đây chưa cô lập đóng góp riêng của nhánh prompt quan hệ.

## Khoảng tin cậy chênh lệch PROMPT−CONTROL

| Giải mã | Nhóm | Chênh lệch | CI95 |
|---|---|---:|---|
| normal | All | +0.1193 | [-0.1004; +0.3331] |
| normal | Objects | +0.3441 | [+0.0086; +0.6619] |
| normal | Relations | +0.2214 | [-0.0587; +0.4877] |
| normal | Attributes | -0.2626 | [-0.6202; +0.0835] |
| fixed_rerank | All | +0.2144 | [-0.0118; +0.4369] |
| fixed_rerank | Objects | +0.3085 | [-0.0276; +0.6368] |
| fixed_rerank | Relations | +0.3686 | [+0.0903; +0.6432] |
| fixed_rerank | Attributes | -0.0510 | [-0.4106; +0.3234] |

Conditional paired bootstrap2000draws/seed42 trên điểm đã lưu, không chạy scorer. Sau rerank, CI quan hệ không chứa0 nhưng chưa hiệu chỉnh nhiều so sánh; không thay thế nhiều seed huấn luyện hoặc xác minh thị giác. CI tổng/đối tượng/thuộc tính còn chứa0.

- Conditional paired bootstrap 2000 draws seed42; one trained seed, not training variability; multiple subcategory comparisons unadjusted.
- Prior TEST exposure; C10/P9 start and adapter parameter count differ.
- SPICE reference scene-graph overlap is not region localization or visual relation verification.
- No model/Java/official scorer run by author. Source SaveVersions188/64 user-confirmed; not fetched again.

## Lưu trữ

Nguồn output người dùng xác nhận: `ducanh2403/2-vi-cai-kia-qua-nhieu/188` và `ducanh3/notebook7bc83c4c76/64`. ZIP chỉ đọc một lần, cache text tại `.review_cache/spice_subsets_v2`. Audit kiểm tra SHA payload, codeGit, PTB, cùng refs/IDs, raw→per-image, mẫu số và điểm tổng; không chạy Java/Torch/local model. CSV/JSON cùng thư mục.

Sheet Kết quả mới A52:L60 đã ghi/đọc lại72ô có dữ liệu khai báo (API content/format verification; không claim native visualQA). Không sửa Word trong lượt này.

Khuyến nghị: đưa bảngSPICE con và giới hạn vào bài; tiếp tục kiểm24mẫu ảnh. Chưa mở thêm training/RegionCLIP hay tự chạy object-only ablation.
