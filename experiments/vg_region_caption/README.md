# VG region-caption pilot G/T

**Kết quả đã nhận đủ A/G/T:** xem [báo cáo đối chứng](G_T_A_RESULT.md).
G tốt hơn T trên CIDEr của development nhưng chưa cải thiện đáng tin cậy so với A; chưa mở rộng cấu hình hiện tại.

**GV cũng đã hoàn tất:** [báo cáo GV/A/G/T](GV_RESULT.md). GV chưa vượt A/G; dừng chuỗi VG region-CE ở cache và quy mô hiện tại theo tiêu chí đã khóa.

Hai nhánh kiểm tra liệu chỉ định vùng trong nhiệm vụ mô tả phụ có giúp caption toàn ảnh hơn việc thêm đúng cùng lượng văn bản hay không. **Đã có kết quả pilot; chưa chứng minh đóng góp căn chỉnh nâng baseline.**

| | G | T (đối chứng thiết yếu) |
|---|---|---|
| Caption chính | Cùng baseline A, ảnh + legacy memory | Giống G |
| Phrase phụ | Patch trong bbox VG, mask ở mọi lớp Q-Former | Toàn bộ196patch |
| CLS / legacy trong nhiệm vụ phụ | Bỏ cả hai | Bỏ cả hai |
| Q-Former / decoder | Dùng chung với caption chính | Giống G |
| Tham số mới | 0 | 0 |
| Dữ liệu phụ, thứ tự, loss, optimizer | Giống nhau | Giống nhau |

`L = caption_CE + scst_weight*SCST + 0.1*phrase_CE`.
SCST chỉ cho caption chính: weight0 trong200updates đầu, sau đó.05, rewardCIDErD+.5object_count_v2.
PhraseCE tính mean trên token không PAD của mỗi rank, gồm EOS; DDP trung bình hai rank như captionCE hiện có. Tất cả chạy eval-mode có gradient như A, nên auxiliary không tiêu thụ dropout RNG.

## Dữ liệu khóa trước

- Cùng caption train10k/dev1000 và checkpoint ban đầu của A;1250updates, globalbatch8, seed42, LR5e-6, AdamW/gradientclip như A. CLIP frozen/cache, không encode ảnh mới.
- Inventory yếu168522 vùng trên3361ảnh. Loại ảnh ngoài train/restval,568ảnh đã biết và đồng loạt54ảnh đã xem hai audit. Không dùng nhãn assistant để lấy riêng mẫu đúng.
- Phrase gốc giữ nguyên; lowercase/split đúng tokenizer baseline, không sửa chính tả/dấu câu. Tối đa28từ để giữ BOS/EOS trong30token; **zero OOV**. Báo cáo OOV trước lọc2.149% số từ; không diễn giải thành chất lượng annotation.
- Bbox gốc hợp lệ và chứa>=4tâm patch; **không nới box/nearest fallback**. Cache bilinear224square, không center crop; kiểm tra metadata lúc chuẩn bị.
- Image-first hash cycles, cap5phrase/ảnh;10000phrase trên3307ảnh, gắn vào10000chỉ số captiontrain, cùng DistributedSampler seed42 cho G/T. Một batch phụ global8 mỗi update.
- Các phrase và lịch đã nhúng vào notebook, có SHA256 trong `AUXILIARY_MANIFEST.json`. **Không cần dataset VG riêng, không tải ảnh/annotation VG trên Kaggle.** File `.bin` trong repo là zlib JSON dữ liệu, không phải model checkpoint.
- Nguồn: Visual Genome region_graphs/image_data chính thức; annotation giữ quyền/điều khoản nguồn. Nguồn VG có lỗi đã quan sát; không coi payload là nhãn grounding chuẩn.

## Chạy trên Kaggle

Import **mỗi notebook vào một notebook riêng**, mỗi cái **GPU T4×2**, Internet ON, **Save & Run All**:

1. `VG_REGION_CAPTION_G_2GPU.ipynb`
2. `VG_REGION_CAPTION_T_2GPU.ipynb`

Hai tài khoản cần cùng quyền truy cập các dataset sau (giống A):

| Đầu vào | Đường dẫn mặc định |
|---|---|
| Baseline checkpoint | `/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth` |
| Legacy prompt | `/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt` |
| CLIP visual cache | `/kaggle/input/datasets/ducanh2403/visual-cache` |
| COCO JSON | `/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json` |
| Object detections | `/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json` |
| Hidden cache đã lưu | `/kaggle/input/datasets/ducanh2403/upt-hidden-pairs-caption11k` |

Sửa mount path ở cell đầu nếu khác; file bytes/SHA phải giữ nguyên. Hidden cache dùng để tái sử dụng kiểm tra/staging/cohort của A, **không đưa vector UPT vào G/T**. Đường dẫn cache phẳng, không thêm thư mục con `upt_hidden_pairs_11k`.

Notebook in log ngay từ cell đầu, dùng checkout tạm riêng và nguồn pin+overlayhash; cài dependencies và chạy synthetic CPU/DDP smoke trước khi train. Không tự tiếp tục nếu smoke thất bại. Stage khoảng9GB trong `/tmp`; output cuối chứa checkpoint `.bin` legacy serialization và reviewZIP, không chứa stage.

Gửi lại `vg_region_caption_G_review.zip` và `vg_region_caption_T_review.zip`; giữ riêng cả hai `checkpoint.bin`. Nếu lỗi, gửi phần lỗi đầu tiên hoặc `print(LOG.read_text(encoding='utf-8')[-12000:])`.

## Đánh giá và quyết định

- Rawgreedy original+end trên cùng dev1000 đã dùng nhiều lần. Không rerank, không chọn checkpoint, không dùng test.
- A lịch sử CIDEr1.205512599, CHAIRs.052, CHAIRi.032475490, recall.458038733. Trước so sánh cần khớp checkpoint, cohort, reward targets, original captions, tokenizer/evaluator và runtime. Nếu runtime/original khác, không tự quy chênh lệch cho phương pháp; cần A khớp lại.
- Gate G−A CIDEr>=.01, pairedbootstrap95%lower>0, G>original; CHAIRs/i không tăng và recall giảm tối đa.01 soA. Muốn quy lợi ích cho vùng còn cần G−T paired95%lower>0 và review mù trên cohort khóa trước.
- G và T cùng tăng ngang nhau: lợi ích từ thêm giám sát mô tả, **chưa chứng minh giá trị vùng**. Phrase loss giảm không thay thế chất lượng caption. Không tăng dữ liệu/dò trọng số tự động khi thất bại.
- Patch CLIP có ngữ cảnh toàn ảnh sẵn; mask chỉ giới hạn key Q-Former đọc, không cô lập bằng chứng hoàn toàn. Supervision yếu có thể truyền sai vào caption. CE đa nhiệm/region-captioning là phần kế thừa, chưa phải novelty tự thân.

## Kiểm tra đã làm / chưa làm

Local: pure-data tests, AST/hash/payload/notebook integrity, kiểm tra training-loop A chỉ thêm hook ghi diagnostic. Không có torch/h5py local; **không chạy model/training/notebook**. Synthetic gradient/mask/DDP tests được cung cấp để notebook chạy trên Kaggle trước train; chưa xác nhận GPU PASS ở thời điểm phát hành.
