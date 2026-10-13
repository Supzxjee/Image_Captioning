# Kết quả diagnostic căn chỉnh theo instance: 512 train / 128 dev

Nguồn: instance_contrast_alignment_diagnostic_512_review.zip, code4efac29,
workflow3216790, seed42, CLIP frozen, GT boxes/classes, 20 epochs head.
Kaggle version chưa có trong manifest. Không phải caption training hoặc official HOI evaluation.

## Kết quả trên các nhãn đã biết

| Phương pháp | Macro AP | Holding | Riding | Sitting_on | Carrying | Eating | Talking_on |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP union cosine | 31.88 | 86.83 | 14.36 | 16.98 | 33.66 | 30.82 | 8.64 |
| Supervised, không contrast | 67.57 | 87.18 | 98.42 | 75.19 | 14.92 | 82.17 | 47.54 |
| Supervised + contrast | 69.53 | 88.67 | 99.24 | 78.69 | 16.81 | 90.09 | 43.65 |

AP trong bảng nhân100. Support dev positive theo thứ tự: 88,11,21,14,13,9;
known labels:114,124,97,123,118,124. Unknown không được coi là negative.
Contrast thêm1.96 điểm macro AP so với supervision đơn thuần, một lượt/seed nhỏ;
chưa có bằng chứng ý nghĩa thống kê hoặc tính ổn định. Talking_on giảm; carrying yếu.
Supervision là nguồn cải thiện lớn nhất trong lượt này.

Train loss0.7199→0.3481; ranking0.1990→0.0670. Không dùng loss giảm như chứng cứ đúng instance.

## Kiểm tra đúng instance từ predictions đã lưu

Recompute contrast_edges trên đúng128 dev rows; kiểm tra predictions cùng thứ tự IDs.
So sánh score positive lớn hơn score negative đã được xác nhận (không dùng threshold):

| Loại đối chứng | CLIP cosine | Không contrast | Có contrast |
|---|---:|---:|---:|
| Sai actor, cùng ảnh/B/action:10 edges,10 ảnh | 8/10 | 7/10 | 7/10 |
| Sai predicate, cùng A–B:566 edges,116 ảnh | 433/566 | 518/566 | 528/566 |

Mean actor gap:0.01119,0.00397,0.00495; không actor edge nào đạt margin0.2.
Train có64 actor edges và2183 predicate edges: actor chỉ2.85% tổng ranking edges.
Loss gộp trung bình edges chưa cân bằng hai loại. Các dev edges không độc lập;
không dùng566 như566 ảnh hoặc thử nghiệm thống kê độc lập.

Kết quả chưa chứng minh giải quyết sai actor. Chưa test wrong-object negatives.

## Threshold và coverage

Threshold fit chỉ trên train, mục tiêu90% empirical precision với ít nhất5 accepted.
Với contrast: holding precision91.67%/recall25%; riding91.67%/100%; sitting_on70.59%/57.14%;
eating80%/61.54%; carrying và talking_on đều abstain toàn bộ known dev labels.
Mục tiêu train90% không bảo đảm dev90%. GT class priors và thay đổi prevalence ảnh hưởng kết quả.

## Lỗi chọn mẫu được phát hiện

Train có461/512 positive-containing pairs; dev118/128. Hàm diagnostic_subset lấy nửa
positive trước rồi ưu tiên positive khi lấp chỗ trống, lệch so với ý định khoảng nửa positive.
Sửa fill ưu tiên negative trước và thêm regression test. Không sửa/relabel kết quả cũ;
lượt chạy này phải giữ distribution/provenance hiện tại. Có counterpart-positive thì tỷ lệ
sau sửa vẫn không nhất thiết chính xác50%, cần đọc pair_summary của lượt sau.

Train418 ảnh, dev116 ảnh, giao image IDs0; tất cả train/restval trong caption metadata.
SHA256 train/val JSON khớp pair_summary. Caption heldout không được dùng.

## Quyết định tiếp theo

Giữ scorer supervised làm nền phát triển. Chưa nối Decoder và chưa tạo full prompt cache.
Lượt tiếp cần mẫu sửa đúng, thêm số ảnh/cặp actor đã xác nhận và cân bằng loss theo loại
đối chứng; tách cải thiện do supervision khỏi cải thiện do instance contrast. Chưa triển khai
thay đổi loss trong bản này. Cần xem per-class support và lỗi carrying/talking_on.
Không chỉ tăng epoch hoặc dùng69.53% như precision/accuracy tổng thể.
