# Visual Genome phrase–pair feasibility audit

**Follow-up:** [Phrase–region audit24](REGION24_RESULT.md) đã hoàn tất theo đề nghị Astra:168.522region candidates/3.361images;13/24mẫu phù hợp đồng thời ba tiêu chí theo trợ lý chấm sơ bộ. Bỏ giới hạn ghép UPT mở được nguồn ứng viên, nhưng inventory chưa là nhãn sạch. Không chạy model/training. Phần dưới giữ kết quả lịch sử của audit phrase–pair.

**Decision: chưa đủ điều kiện chạy G/T với phần giao cache hiện tại.** Không viết hoặc chạy notebook huấn luyện mới. Không kết luận Visual Genome nói chung không phù hợp.

## Đã làm và kết quả

Tải đúng hai ZIP annotation từ [trang Visual Genome chính thức](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html): image metadata 1.78MB và region graphs 331.67MB nén. Region graphs giải nén logic 2.78GB được đọc tuần tự, không cần lưu JSON giải nén. Không tải bộ ảnh VG, không trích đặc trưng/mô hình; chỉ lấy 30 JPEG COCO train cho kiểm tra trực quan.

| Bước | Số lượng |
|---|---:|
| VG image metadata | 108.077 ảnh |
| Có COCO ID thuộc dataset đang dùng | 51.498 ảnh |
| Trong COCO train/restval | 47.210 ảnh |
| Trong COCO val/test — loại khỏi giám sát | 4.288 ảnh |
| Giao train10k cache hiện tại | 4.115 ảnh |
| Sau loại known-exposed IDs | 4.093 ảnh |
| Qua kiểm tra tỉ lệ ảnh và có selected pair | 2.188 ảnh |
| Regions được kiểm tra trên nhóm đó | 109.645 |
| Đúng một cạnh graph | 35.754 |
| Đúng hai object, human→COCO object, alias rõ | 2.223 |
| Cụm có nối từ vựng với hai đầu mút và predicate | 852 |
| Ghép duy nhất vào selected pair, cả hai IoU ≥.5, bỏ lặp | **468 ứng viên / 314 ảnh** |
| Cụm khác nhau sau chuẩn hóa văn bản | 440 |

Trong 4.093 ảnh, 714 bị loại vì tỉ lệ VG/cache không khớp đủ an toàn; 1.191 ảnh còn lại không có selected pair. Đây là các tầng loại khác nhau, không phải các lỗi cộng lên cùng đơn vị với region/phrase. Sau mức 852 cụm: 351 không ghép được cặp cùng lớp đủ IoU, 27 ghép được nhiều cặp, 6 trùng image-slot-phrase; còn468. Kết quả không đo khả năng của mọi native proposal UPT, chỉ tối đa8 cặp đã giữ trong cache.

Dùng đúng `dataset_coco.json` SHA256 `2fd999...b7600b8d`, phân biệt COCO ID với imgid. Cấm toàn bộ 5.000val +5.000test theo Karpathy JSON, không chỉ dev1000 hiện tại. Chỉ train/restval được vào tập ứng viên; loại thêm568COCO IDs từng dùng trong phrase audit (22 trong giao cache lần này). Tên thư mục train2014/val2014 không quyết định split học. Identity liên kết của30JPEG tải từ COCO khớp byteSHA trong cache30/30.

## Kiểm tra trực quan 30 mẫu

Chọn có chủ đích theo predicate, mỗi ảnh chỉ một mẫu, trước khi xem ảnh; **không đại diện phân phối và không dùng để suy ra precision toàn bộ**. Overlay: UPT S xanh lá/O cam; VG nét xanh dương mảnh. Bản [manifest](SAMPLE30_MANIFEST.json) giữ ID, phrase gốc, slot/native row, IoU và bbox; [ghi chú chấm](ASSISTANT_SAMPLE30_REVIEW.json) là **assistant_visual_provisional_not_human**, không phải nhãn người chấm.

- 23 mẫu có bằng chứng phù hợp theo xem ảnh sơ bộ.
- 4 mẫu chưa rõ: chi tiết quá nhỏ, chưa chắc liên kết con diều, hoặc cụm cụ thể hơn bằng chứng nhìn được.
- 3 mẫu loại/sửa: một ghép sai người, một biến thể spatial lọt bộ lọc event, một câu chứa thêm sự kiện ngoài cặp.

Ví dụ đáng chú ý:

- **COCO245806:** câu “man holding a white bowl” nhưng subject box chọn người phía sau đang bế em bé; bát thuộc người đang nấu phía trước. IoU VG↔UPT cao không bảo đảm VG đã gắn đúng quan hệ.
- **COCO349676:** graph chỉ có cạnh người nhìn hươu, nhưng phrase còn nói hai con hươu giao phối. Một cạnh trong graph không chứng minh phrase chỉ có một sự kiện hay mọi chi tiết đều đúng.
- **COCO187435:** nhiều diều, người cầm vật đỏ có thể là tay cuốn dây; không tự kết luận diều xanh được chọn là sai, giữ nhãn chưa rõ. Đây là sửa nhận định ban đầu sau khi xem ảnh gốc.
- **COCO445684:** “on a” là biến thể không bị blacklist exact-string “on” bắt. Hình có thể đúng, nhưng chưa đúng tiêu chí event-only của pilot. Các predicate thô trong report không phải các loại sự kiện độc lập; riding/rides/riding a không được tính như ba nghĩa khác nhau.

Gallery local được sinh ở `review/index.html`, cùng5contact sheets. Ảnh và annotation ZIP giữ local/ignored, không đẩy ảnh lên GitHub. Chưa yêu cầu user chấm thêm hoặc dùng các đánh giá của trợ lý làm GT huấn luyện.

## Quyết định và giới hạn

Mốc Astra đề nghị là khoảng1.000ảnh và vài nghìn cụm hợp lệ, không phải tính toán power thống kê. Ngay cả trước xác nhận ngữ nghĩa, giao hiện tại chỉ có314ảnh/468ứng viên. **Không đủ điều kiện chốt pilot G/T như đề xuất.** Chưa cần GPU. Không lấp thiếu bằng câu tổng hợp, giảm IoU, chấp nhận nhiều instance hoặc dùng val/test.

Đây là audit bảo thủ với alias exact/biến thể nhỏ, một cạnh/hai object, liên kết từ vựng và selected top8. Số bị loại không chứng minh chú thích sai. Ngược lại, qua bộ lọc cũng không chứng minh nhãn sạch: spatial variants, extra events và graph-instance errors đã thấy trong mẫu. Bộ lọc là công cụ đếm/triage, **chưa là bộ xây GT production**.

Tập VG train/restval rộng hơn có47.210ảnh, nên mở rộng targeted cache là một khả năng, nhưng chưa biết cần trích bao nhiêu mới có đủ link sạch. Cần Astra chọn giữa audit nguồn/ghép rộng hơn và đổi nguồn/mục tiêu giám sát, trước khi chi GPU. Cơ chế/loss G/T chưa được triển khai. Không có kết quả caption hay đóng góp mới được chứng minh từ audit này.

## Tái lập

Python stdlib, NumPy không cần cho audit chính; Pillow chỉ để dựng gallery. File tải có SHA256/CRC/provenance ghi trong [kết quả máy đọc](VG_FEASIBILITY_RESULT.json). Review cache phải đúng SHA receipt đã audit và JSONL payload hashes. Chưa kiểm tra HDF5 vector bytes vì ZIP review không chứa vectors; mapping dùng metadata selected-pair trong JSONL đã hash.

```powershell
python experiments/vg_phrase_audit/download_annotations.py
python -m unittest discover -s experiments/vg_phrase_audit -p 'test*.py' -v
python experiments/vg_phrase_audit/audit_data.py --dataset C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json --cache-review C:/Users/leduc/Downloads/upt_hidden_pairs_11k_review.zip
python experiments/vg_phrase_audit/make_review.py
```

Numeric audit ghi `review/RULE_AUDIT.json`, candidate list và sample manifest; kết quả tổng hợp/ghi chú xem ảnh đã lưu ở các JSON trên. Ghi chú xem ảnh không tự tái sinh bằng code. Four pure-data tests PASS: streaming/truncation, split leakage prevention, bbox+class instance matching/ambiguity, lexical/alias conservatism. Chưa chạy model hoặc training.
