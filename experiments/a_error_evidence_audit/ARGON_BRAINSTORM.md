# Brainstorm độc lập sau rà soát lỗi A — 27/10/2026

Bạn là cộng sự nghiên cứu phản biện. Tôi cần một phương pháp căn chỉnh ngữ nghĩa–
thị giác cho image captioning, có khả năng cải thiện baseline và đủ cơ sở viết
bài báo. Không cần đồng ý với hướng của Astra/root. Không hứa novelty hoặc gain.

## Bối cảnh tối thiểu

Baseline A có CLIP ViT-B/16, Q-Former32 queries/2layers, original semantic prompt
CLIP embeddings, attention/gating prompt–visual, Transformer caption decoder3layers.
Đã dùng SCST+CE và object reward rồi huấn luyện tiếp A. **A có prompt gốc**, không
phải baseline không prompt. Inference không có GT boxes, references hoặc V-COCO.
Train có thể dùng annotations/references đúng split. Caption inference hiện theo
greedy. Không có yêu cầu giữ nguyên mọi module nếu hướng khác có cơ sở tốt hơn.

A CIDEr1.205513 trên dev1000 và1.182248 trên val5000 vì các tập/IDF khác nhau;
không được lấy hai con số để suy weight drift. Các tập đã dùng nhiều lần, không
gọi là fresh unseen. Hiện chưa có nhánh alignment mới vượt A đáng tin.

Những gì đã thử, tránh lặp dưới tên mới:

- Noun evidence head: global G vs bbox-loss S. Cả hai giảm CIDEr và tăng CHAIR so
  A, dù object recall tăng. S−G chưa thấy lợi ích chắc.
- Native detail: cùng head nhận H(native) và L(degraded). H−A CIDEr−.010160,
  CI âm; H−L không rõ. Điều này không bác bỏ mọi cách dùng high-res.
- VG region phrase auxiliary: G−A +.002888, CI qua0; G−T(global phrase)+.022654,
  nhưng T thua A. Chưa chứng minh G tốt hơn baseline hay đúng grounding.
- Routing/location probes có tín hiệu attention hình học nhưng can thiệp mass
  chưa cho decoder utility/gain đáng tin. Attention đúng không tự là causal use.
- Stale-prompt matched C/T cả hai thua A. Chưa biết chính loss thêm gây hại vì
  T−C CI qua0. Không gọi negative result là architecture ceiling.
- Role-interval feasibility:128 FIT pairs, active6<16, binding0; scorer có lỗi
  ontology: V-COCO ride=0 không phủ định English riding+skateboard/snowboard/skis
  (đúng actor/object có action riêng=1). Đổi8 phủ định sai thành unknown trên cùng
  caption cũ chỉ còn2 active âm,0 dương. Dừng nhánh này; không cứu gate bằng nới
  parser trên128 ảnh đã xem. Graph rewards, semantic edits và RL tự chúng không mới.

## Bằng chứng mới: rà soát30 ảnh, giữ20 trường hợp trọng tâm

Chỉ đọc saved artifacts; không chạy model mới. Ghép9 review caches theo COCO ID,
kiểm A strings nhất quán. Chọn10 CHAIR+/6 holding-and/10 unflagged low-CIDEr dev/
4 unflagged low-CIDEr nondev, hash-first trong nhóm trước khi xem ảnh.

Trợ lý nhìn30 ảnh COCO, giữ20 focus gồm9 lỗi rõ,4 unsupported,6 under-specified,
1 uncertain;10 trường hợp khác gồm5 caption có cơ sở và5 chưa chắc. **Không phải
human gold hoặc20 lỗi chắc chắn; không là tỷ lệ lỗi trên tập dữ liệu.** Không dùng
reference omission, singular existential hoặc low CIDEr làm false labels.

Các case quan trọng (COCO IDs có thể tải ảnh theo URL chuẩn COCO val2014):

| ID | A | Quan sát và các nhánh cũ |
|---|---|---|
| 33924 | a person holding a toothbrush and a cat. | Tay cầm bàn chải; mèo nằm trên chăn. Sai phạm vi holding dù nouns đúng/CHAIR không flag. H/L,nounG/S giữ nguyên. |
| 167271 | a woman laying in bed while talking on a phone. | Thiết bị màn hình gập/cầm tay, refs chủ yếu DS, một câu tablet. H/VG/stale sửa talking→using nhưng vẫn phone. Không nhận DS là gold từ refs. |
| 196594 | a clock on a wall next to a sign |3clocks+3signs rõ;10 output giữ nguyên. Câu không sai tồn tại nhưng kém cụ thể. |
| 466945 | a man walking down a sidewalk next to a body of water. | Đồng hồ lớn tiền cảnh bị bỏ. NounG/S thêm đúng clock on a pole. A vẫn là một mô tả có cơ sở. |
| 261535 | a woman in a skirt is holding a tennis racket and tennis balls | Một yellow object ở tay, nhiều yellow objects trên sàn; không khẳng định subtype fruit/ball chắc chắn. NounG/S bỏballs; VGG bỏballs nhưng thêmheels dù chân trần. |
| 261068 | a woman is petting a sheep in a pen | Người chạm animal và cầm cào; petting có thể đúng nhưng thiếu hành động chính. NounG/S đổi dog on a leash, ảnh không hỗ trợ leash. |
| 185901 | a glass bowl filled with oranges on a white table. | Quả vàng dạng lemons; bốn refs lemons, một câu fruit. COCO không có lemon class riêng. |
| 304036 | a group of planes flying in the sky | Ảnh helicopters có rotor. |
| 514248 | a fire hydrant ... with a bird on the sidewalk. | Vật nhỏ là toy horse, không phải bird. |
| 393547 | a cat sitting on a table next to a mirror. | Mèo trên mặt gương nằm ngang, phản chiếu bên dưới. Không phủ định mọi phần đỡ gương là bàn. |

Counterexamples chống ép mọi caption thành lỗi: hải âu tiền cảnh+boat background
(100811) là mô tả hợp lý dù còn chim khác; người dùng phone dưới ô(371564) hợp lý;
pastry display(140391) có cơ sở dù CIDEr thấp; surfboard/storefront(32391) hoặc
board/canoe/paddle scene(447237) có thể under-specified nhưng không chốt sai
category vì refs bất đồng. Vật không nhắc tới không tự là lỗi cần sửa.

Trên20 focus: H và L mỗi nhánh giữ15/20 A; nounG giữ15,S giữ16. VG variants có14
case coverage, thay đổi7–9 câu; đổi câu không là sửa đúng. Không là10-way matched
ablation vì lịch training/objective khác nhau.

## Giới hạn bằng chứng — phải tính vào đề xuất

Có original pixels,5refs và các saved captions. CLIP visual-cache/original-prompt
embedding source/cohort có trong protocols, **nhưng chưa có raw per-case CLIP/Q
vectors hoặc exact original prompt text local**. Không được tái tạo prompt từ refs
hay lấy UPT prompt mới thay thế. UPT valid-pair count chỉ có ở14case overlap; chưa
có hidden vectors/native role scores. Count không chứng minh quan hệ đúng.

Do đó chưa xác định lỗi nằm ở encoder, alignment hay decoder. Người nhìn được
cue ở224×224 không chứng minh model phân biệt được. Native H/L captions hay cùng
câu không chứng minh mọi detail đều bị bỏ. VG region supervision không là oracle
region input khi caption inference. Không đổi luận điểm nhân quả chỉ từ bảng caption.

## Yêu cầu brainstorm

Astra vừa rút một đề nghị audit thêm4–6 cặp ảnh thật bằng frozen-A teacher forcing:
nó khác về kỹ thuật nhưng chưa có hai hành động nghiên cứu khác nhau tùy kết quả.
Dự án đã có CE/CROSS supplied-phrase preference tốt hơn free generation, pair-prefix
và F relation-token diagnostics. Vì vậy không chỉ đề nghị lại một phép đo likelihood
rồi kết luận sensitivity khác correctness. Nếu cần phép kiểm, phải gắn nó với một
quyết định phương pháp cụ thể mà kết quả có thể thay đổi.

1. Phản biện: ba cách giải thích nào còn phù hợp bằng chứng? Điều gì hiện chưa
   đủ căn cứ? Có nút thắt nào đáng theo nhất, hay cần một phép kiểm trước?
2. Tối đa hai cơ chế **khác thực chất** các heads/region losses/rewards đã thử.
   Chọn một nếu đủ cơ sở; nếu chưa đủ, chọn một phép kiểm nhỏ có khả năng đổi
   quyết định, không thêm một audit chung chung kéo dài.
3. Nếu đề xuất method: mô tả module/tensors/loss/supervision, đường inference,
   phần nào đóng băng/khởi tạo, tham số train, shortcut có thể phá method và đối
   chứng kiểm decoder sử dụng evidence. Không mặc định text→bbox là alignment.
4. Nêu phần kế thừa và giả thuyết mới; các prior gần nhất và overlap thực sự.
   Nếu chưa đọc/không tra được paper, ghi novelty chưa xác minh, không bịa citation.
5. Phép thử ít nhất để tiếp tục/dừng; phân biệt semantic correctness, causal use
   và caption gain. Không đổi gate sau khi xem kết quả hoặc gọi dev cũ là unseen.

Chưa được launch GPU/training. Tôi có thể chạy hai notebook T4×2 nhưng quota còn
lại chưa xác nhận. Hãy ưu tiên một quyết định nghiên cứu rõ ràng, không một danh
sách module nghe mới. Chúng tôi cần biết vì sao lần thử tới có căn cứ khác những
lần trước, đồng thời chấp nhận rằng hiện chưa có bằng chứng đủ nếu đúng là vậy.
