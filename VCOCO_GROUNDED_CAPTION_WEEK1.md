# Tuần 18–24/10/2026: căn chỉnh quan hệ với vùng ảnh

## Trạng thái thực tế

Bắt đầu ngày 18/10, chốt báo cáo thứ Bảy 24/10. Đã chạy **kiểm kê annotation trên CPU**;
chưa huấn luyện mô hình, chưa chạy notebook Kaggle, chưa có chỉ số caption mới.
Notebook GPU sẽ do người dùng tự chạy. Không triển khai lại một bộ chấm điểm cũ dưới tên mới.

Mục tiêu: kiểm tra liệu quan hệ **dự đoán từ ảnh** và bằng chứng đúng cặp vùng có thể
cải thiện caption ngoài checkpoint SCST+CE+object reward đã có hay không.

Cập nhật cùng ngày: đã hoàn tất ghép box COCO và giữ mọi liên kết dương hợp lệ.
Xem `VCOCO_GROUNDED_CAPTION_BOX_JOIN_RESULT.md`: 3.469 liên kết, không cắt còn một
cặp mỗi ảnh; 16 tests PASS và export nguồn thật tái lập giống byte. Các hạn chế
“chưa kiểm tra box” ở phần inventory dưới đây mô tả bước kiểm kê trước cập nhật này;
box detector và hiệu quả caption vẫn chưa được đánh giá.

## Kết quả bước đầu trên nguồn thật

Nguồn [V-COCO chính thức](https://github.com/s-gupta/v-coco), revision
`489cc4db74f2f10ab4b134f67da3874afbf245ab`. Chỉ đọc annotation train/val,
không tải annotation V-COCO test. Metadata Karpathy lấy từ
https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip;
SHA256 `dataset_coco.json` khớp input SCST hiện tại.

| Nhóm đã khóa | Ảnh | Holding: liên kết dương / ảnh | Riding: liên kết dương / ảnh |
|---|---:|---:|---:|
| Relation fit, 80% official train | 2.026 | 1.141 / 981 | 189 / 91 |
| Relation calibration, 20% official train | 507 | 275 / 232 | 33 / 17 |
| Relation development, official val | 2.867 | 1.608 / 1.363 | 223 / 111 |

Official train tổng 2.533 ảnh, official val tổng 2.867 ảnh: **5.400 ảnh đều thuộc
caption train**, không phải caption validation. Split relation và split caption là
hai cách chia khác nhau. Calibration/dev không trùng fit; đều được loại khỏi
10.000 ảnh thích nghi caption được chuẩn bị cho pilot này.

Các số trên là **liên kết annotation**, chưa phải số cặp detector định vị được.
Ví dụ, relation fit có 1.473 nhãn hành động hold dương, nhưng 332 trường hợp không
có đối tượng vai trò được ghi rõ: chỉ còn 1.141 liên kết dương. Không bịa box hoặc
tự lấy một vật gần người nhất để điền vào chỗ thiếu. Riding fit chỉ có 91 ảnh:
không thể coi 189 liên kết là 189 ảnh độc lập.

Đã khóa thêm 10.000 ảnh thích nghi caption và chính tập tune1000 cũ của SCST
(shuffle validation theo seed 20261017, thứ tự hàng gốc). 4.000 validation còn lại
không dùng để chọn cấu hình trong pilot. Không chấm lại test.

Toàn bộ caption train/restval có 113.287 ảnh. Checkpoint gốc đã học trên tập này;
10k chỉ là **quy mô thích nghi**, không phải tổng dữ liệu mô hình từng thấy.
V-COCO development đã được xem ở các lần trước, không gọi là benchmark mới độc lập.

Aggregate: `audit/vcoco_grounded_caption_week1_data_audit.json`.
Danh sách ID khóa để tái lập: `audit/vcoco_grounded_caption_week1_groups.json`.
Không đưa caption tham chiếu, ảnh, annotation đầy đủ hoặc weights vào Git.

## Bài học phải giữ từ các lần trước

- `INSTANCE_ALIGNMENT_V2_DIAGNOSTIC_1024_RESULT.md`: học action semantics được
  nhưng head mới làm mất một số thứ tự instance tốt của CLIP.
- `PAIR_QUERY_ATTENTION_PILOT_RESULT.md`: attention theo query và query-blind
  cùng đúng 29/34 đối chứng; chưa chứng minh lợi ích của query-conditioned reader.
- `DECODER_RELATION_MEMORY_ADAPTATION_512_RESULT.md`: ghép memory quan hệ rồi học CE
  chưa thay đổi top caption trên 12 ảnh đánh giá; không có bằng chứng cải thiện caption.
- `SELECTIVE_EVIDENCE_ROUTING_V1_RESULT.md`: selection có cải thiện cục bộ nhưng
  coverage thấp; holding còn nhiều lỗi. Độ nhạy prompt hoặc loss giảm không đủ.

Vì vậy **cổng + attention + BCE quan hệ đơn thuần không được coi là phương pháp mới**.
Thiết kế C còn phải chỉ rõ giám sát nào buộc decoder liên kết cụm mô tả với đúng
bằng chứng. Không phát hành notebook train ba nhánh trước khi chốt được điểm này.

## Thiết kế A/B/C để kiểm tra, chưa phải kết quả

| Nhánh | Điều kiện |
|---|---|
| A | Tiếp tục checkpoint SCST+CE+object reward, không thêm quan hệ mới |
| B | Cùng checkpoint và objective, thêm prompt quan hệ dự đoán với fusion đơn giản |
| C | Cùng prompt/pair inventory của B; thêm mục tiêu căn chỉnh có tác động tới decoder |

Giữ cùng train10k, số bước, seed, batch, trainable caption backbone, reward,
beam và fixed reranking. A/B có cùng objective sinh caption; C bổ sung loss mới
nên phải báo rõ. Nếu C tăng số tham số, cần control tương ứng trước khi kết luận
lợi ích do loss/grounding thay vì capacity. Không đổi decoder checkpoint nền sang E2.

### Các quyết định cần hoàn thành trước GPU

1. **Box và nhãn:** đọc COCO instance annotations để kiểm tra role ID, loại crowd,
   box hỏng và đối tượng không tìm thấy. Đếm lại support sau lọc; inventory hiện tại
   chưa làm bước này. Không cần tải lại toàn bộ ảnh COCO.
2. **Nguồn dự đoán:** đối chiếu bộ HOI pretrained UPT hiện có với mô hình V-COCO nhỏ;
   không mặc định huấn luyện lại một MLP cũ. Chọn một nguồn cố định trước đánh giá
   caption, ghi nguồn pretraining và giới hạn overlap. GT box dùng cho giám sát;
   đầu vào triển khai phải là detector boxes. Không chọn cặp đầu vào bằng GT role.
3. **Mục tiêu decoder:** phân biệt rõ ghép cụm–vùng và dự đoán hành động. Caption
   không nhắc quan hệ không có nghĩa quan hệ sai; caption có cùng hai danh từ chưa
   xác định đúng instance. Chỉ sử dụng giám sát có ánh xạ đáng tin, không tự tráo
   predicate rồi gán âm. Nếu chưa đủ nhãn cho mục tiêu này, báo thiếu thay vì tự chấm.
4. **Kiểm tra phụ thuộc:** che/tắt bằng chứng hoặc thay đúng endpoint để kiểm tra
   ảnh hưởng có chọn lọc lên xác suất cụm tương ứng. Attention heatmap hay norm
   thay đổi không phải bằng chứng caption đúng. Chỉ tính accuracy đối chứng khi
   cả hai nhãn được xác nhận; các trường hợp unknown chỉ dùng đo độ nhạy.

Không có annotation quan hệ ở đầu vào đánh giá caption. Prompt và chọn cặp lúc
đánh giá phải đến từ ảnh/detector/predictor. Điểm predictor không phải ground truth
hay một bộ xác minh độc lập cho chính dự đoán đó.

## Mốc công việc và quyết định

- 18/10: kiểm kê nguồn thật, khóa ID, đọc lại thất bại cũ; hoàn thành bước inventory.
- 19/10: kiểm tra COCO role boxes và feature coverage; chốt objective C khác thử cũ.
- 20/10: đánh giá nguồn quan hệ trên development bằng detector boxes; báo cả coverage,
  từng predicate và lỗi ghép instance. Không dùng GT-matched subset thay cho end-to-end.
- 21/10: kiểm tra code/gradient/fallback và đối chứng trên mẫu nhỏ; chuẩn bị notebook GPU.
- 22/10: người dùng chạy pilot A/B/C nếu các điều kiện trên đạt, không tự submit Kaggle.
- 23/10: đánh giá cùng caption tune1000, phân tích CIDEr/BLEU4/METEOR/CHAIRs/i/Recall,
  báo cả toàn tập và tập con liên quan; không chọn tập con dựa theo kết quả model.
- 24/10: tổng hợp kết quả, hạn chế và quyết định mở rộng/dừng; không hứa trước mức tăng.

Nếu nguồn quan hệ chưa dùng được, không đốt GPU huấn luyện caption với prompt sai.
Nếu C chỉ tốt hơn A nhưng không tốt hơn B, chưa chứng minh lợi ích căn chỉnh.
Nếu C chỉ tăng điểm quan hệ mà caption không cải thiện, chưa đạt mục tiêu.
Chênh lệch nhỏ cần paired bootstrap theo ảnh; một seed chỉ là pilot, không đủ khẳng định
ưu thế ổn định. Tham số loss và checkpoint selection phải chốt trước khi chấm kết quả.

## Cơ sở tài liệu

- [Zhou et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhou_More_Grounded_Image_Captioning_by_Distilling_Image-Text_Matching_Model_CVPR_2020_paper.html):
  hướng dẫn attention của captioner bằng matching teacher và nghiên cứu matching reward.
- [Bica et al., SPARC, ICML 2024](https://proceedings.mlr.press/v235/bica24a.html):
  căn chỉnh token với nhóm patch, kết hợp loss cục bộ và toàn cục.
- [Singh et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.56/):
  dùng cấu trúc đồ thị từ văn bản để học tính tổ hợp và mẫu âm khó.

Các tài liệu là cơ sở thiết kế, không chứng minh rằng kết hợp các thành phần trên
là mới hoặc chắc chắn tăng CIDEr trong pipeline này.

## Tái lập bước inventory trên CPU

Chỉ cần Python stdlib; không torch, GPU hoặc ảnh. Cung cấp trực tiếp ba JSON:

```powershell
python audit_vcoco_alignment_data.py --caption-json "PATH/dataset_coco.json" --vcoco-train "PATH/vcoco_train.json" --vcoco-val "PATH/vcoco_val.json" --output "PATH/new_inventory"
python -m unittest tests.test_vcoco_alignment_data -v
```

Script kiểm hash cả ba input, giữ unknown, kiểm identity filename/COCO ID,
không ghi đè output đã khóa. Chưa có feature cache hoặc checkpoint mới.

Kiểm tra đã thực hiện: 7 unit tests PASS; chạy lại trên ba nguồn thật cho report
và danh sách ID giống hệt, SHA256 manifest
`6a95392e450a4b810f2937ea792b77abcc6dd4c467f22f140f2867f3b5630249`;
thử ghi đè output cũ bị từ chối. Syntax và `git diff --check` PASS.
