# Role-interval feasibility — user-run only

Đây là bước kiểm tra liệu tín hiệu ngữ nghĩa có đủ tin cậy và đủ dày để thử học.
**Không có training, optimizer, checkpoint mới hoặc kết quả caption mới trong gói này.**

## Chạy gì

Import `ROLE_INTERVAL_FEASIBILITY_2GPU.ipynb` vào notebook Kaggle. Internet ON,
T4 ×2, Save & Run All. Hai worker GPU độc lập, không DDP collectives.

Inputs, theo mặc định ở cell đầu:

| Input | Path |
|---|---|
| A đã huấn luyện tiếp | `/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin` |
| Prompt embeddings gốc | `/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt` |
| CLIP visual cache | `/kaggle/input/datasets/ducanh2403/visual-cache` |
| MSCOCO JSON + ảnh | `/kaggle/input/datasets/vuthetam/mscoco-2014` |
| Output preparation cũ | `/kaggle/input/notebooks/ducanh2403/stale` |

Nếu mount khác thì sửa các path ở cell đầu. Không thay checkpoint hoặc prompt
bằng phiên bản khác: notebook kiểm SHA. Không cần original prompt text JSON,
UPT cache, CHAIR hay API token. Annotation FIT được nhúng sẵn, có provenance.

Trước inference, notebook kiểm hash nguồn, split và coverage; chọn bằng hash,
không dùng điểm mô hình. Metadata local có2026 FIT và1898 replay tách biệt,
cap n1896 sau làm tròn4 **trước** khi kiểm actual cache coverage. Không lấy thêm
calibration/dev để lấp thiếu. Không dùng nhãn test hoặc caption reference để
tạo prompt hay chọn câu. Metadata replay cũ chỉ được đọc để lập kế hoạch;
HDF5 cũ chưa được xác minh lại và chưa dùng cho inference ở bước này.

## Nó làm gì

1. Cố định128 ảnh FIT đầu theo hash, một greedy và một multinomial sampled caption
   (temperature1, max_len30, seed42+COCO ID) từ A frozen; giữ nguyên input gốc.
2. spaCy3.8.7 + en_core_web_sm3.8.0 chuyển câu thành graph holding/riding.
   Wheel parser kiểm SHA; parser chỉ chạy trên Kaggle. Cùng một chủ thể dùng chung
   giữa các hành động. Gán GT nhất quán trên toàn connected component.
3. Đối chiếu official actor/action và role-instance IDs. Thiếu link, thiếu class,
   người chưa có nhãn hoặc parse chưa chắc → unknown. Chỉ refute nếu mọi cách gán
   tương thích đều mâu thuẫn explicit actor/action0, với full annotated-person inventory.
4. Hủy proposition chung, tính trực tiếp khoảng sampled−greedy. Khoảng chứa0
   thì aG=0. Báo riêng ảnh hưởng interval và consistent binding so với hai
   comparator đơn giản; đây không phải hai gold judges.
5. Xuất30 trường hợp để kiểm **phép suy diễn**, không yêu cầu chọn câu hay hơn.

Giới hạn bảo thủ của parser: tối đa3 edges/caption,6 entities/component;
whole uncertain parse giữ [-3,+3]. Không đồng nhất carrying/gripping/on với
holding/riding. Disjunction, modal, negation, passive, plural/quantifier chưa
hỗ trợ, ambiguous coreference và depiction context đều thành opaque.
Đây là policy hữu hạn, không chứng minh mọi câu ngoài whitelist đều được bắt.
Coarse person projection **không xác minh giới tính, thuộc tính hoặc selected bbox**.
Annotation completeness là giả định theo bộ dữ liệu, không chứng minh mọi
người/vật nhìn thấy trong ảnh đều đã được gán nhãn.

## Nhận kết quả

Gửi `role_interval_feasibility_128_review.zip` trong Output. ZIP giữ protocol,
nguồn, dependency tokens, graph, witness, khoảng và ảnh review; không chứa
checkpoint hoặc HDF5 đặc trưng.

`summary.json` có hai trạng thái:

- `FEASIBILITY_NO_GO`: thiếu16 active hoặc8 operative/128; không cần bắt người
  chấm thêm chỉ để cứu mật độ. Chưa train.
- `WAIT_HUMAN_REVIEW`: qua mật độ; mở `review/index.html` offline, kiểm30 phép
  suy diễn và xuất `role_interval_verified_review.json`. Cần≥27 correct,
  không có lỗi missing-link→false. Vẫn chưa tự train.
  Parser biến cấu trúc bất định thành khẳng định chắc chắn cũng là lỗi critical,
  không được có dù tổng correct đạt27/30.

Nếu không có binding-operative case, không coi binding đã được xác nhận dù
interval density đạt. Mẫu review chọn có chủ đích10 active,10 operative chưa
chọn,10 còn lại (thiếu thì bù bằng hash); tỷ lệ audit không là accuracy toàn tập.
Các trường hợp chạm `binding_cap` được báo riêng, không tính vào operative gate;
khác biệt vì cap tìm kiếm không phải bằng chứng identity có ích.

Budget30 phút bao gồm hash/prepare/generation/parser scoring; setup và xuất
hình nằm ngoài cap. Dừng khi vượt cap hoặc nguồn/split/frozen state sai;
không tự thay input, nới policy, chọn lại sample hoặc chạy training.
Nếu lỗi vẫn có review ZIP chứa log; dòng SEND không có nghĩa lượt chạy thành công.

## Kiểm tra local đã làm

Chỉ tạo metadata, synthetic dependency/graph fixtures, test audit/export và
AST/package checks. Không chạy spaCy neural model, caption inference hay training.
Kaggle runtime, parser trên câu thật, numerical density và human audit còn pending.

Parser primary sources:
[spaCy3.8.7](https://pypi.org/project/spacy/3.8.7/),
[official en_core_web_sm3.8.0 release](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0).
