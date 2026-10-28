# Frame pretraining — chốt implementation choices

28/10/2026. Review thiết kế root chuyển; chưa review code cụ thể, chưa neural/autograd/DDP validation hoặc launch GPU. Cohort,200updates và primary gates giữ nguyên.

## Chấp nhận auxiliary GRU head trainable

Chọn **GRU hidden256 + trainable Linear(256,10000)**, cùng initialization/parameter count giữa RelPos và RelContrast. Giữ A input word embeddings và token-ID vocabulary cố định. Không dùng frozen A fc_out + bridge trong pilot này.

Phải phân biệt ba thứ: fixed vocabulary IDs, frozen input embedding matrix và output-head weights. Hai thứ đầu cố định **không** có nghĩa auxiliary output head cố định. `Linear(256,10000,bias=True)` thêm2,570,000parameters; cộng vào actual trainable count và receipts. Original A fc_out của main captioner vẫn theo frozen policy cũ, không bị unfreeze/alias nhầm bởi auxiliary head.

Lý do: GRU có hidden space riêng; head tự học không bắt nó đi qua pretrained A fc_out geometry. Tuy nhiên đây là lựa chọn về mức tự do/optimization, không bảo đảm tốt hơn bridge. Với256pairs và200updates, head mới có thể overfit/underfit và language priors có thể lấn visual context. Không claim r được căn vào latent/vocabulary geometry của A chỉ vì dùng cùng token IDs/input embeddings. Representation transfer vẫn phải kiểm ở caption stage.

Primary pretrain probe vẫn là **correct-image ranking của cùng assertion trong từng arm**, không dùng train CE hoặc raw likelihood giữa hai arms làm bằng chứng semantic quality. Gate fail không cho phép đổi head/tie weights/chạy thêm để cứu trong cùng recipe; đó là kết quả âm hoặc unresolved optimization trong budget đã khóa, không bác mọi relational alignment.

## Chấp nhận dimensions và fractional ROIpool, giữ đúng evidence unit

- e256, union attention/value256, r512 và class embedding80x32 hợp lý;256 là internal dimension,512 là frame-memory interface. Kiểm actual shapes/counts ở user-run preflight.
- Fractional patch-area ROIpool: `e=sum_k w_k V_k / sum_k w_k`, weights/reductions/normalization tính FP32. Đối với valid nonzero box, sum weights phải dương/finite; không tạo zero feature cho malformed/empty box mà bỏ qua validation.
- Dùng196spatial tokens, tách CLS của source197tokens. Tọa độ phải qua đúng original A square-resize mapping; không tự chuyển sang center-crop.
- Fractional pooling tránh empty hard-cell selections; **không** tăng độ phân giải hoặc làm globally contextual CLIP features thành region-pure. Hai box trong cùng cell vẫn có thể đọc appearance giống nhau.
- `Union` cho interaction phải là **enclosing rectangle của hai boxes**, bao gồm vùng giữa chúng. `max(M_subject,M_object)` chỉ là OR của hai object masks, có thể bỏ mất phần giữa; không thay thế union-interaction footprint.
- Hai role-area masks giữ thứ tự subject/object, không cộng thành một mask duy nhất. Class IDs phải là canonical contiguousCOCO80 IDs đúng cached detector mapping, không raw category IDs1..90 có gaps.

## Kiểm tra cần có trong model preflight đã dự kiến

Cùng một initialization artifact cho toàn bộ encoder/GRU/head giữa arms; frozen embedding/source hashes bất biến. Tiny fixtures có thể kiểm constant-feature pooling invariance, valid thin-box normalization và role-mask/geometry ordering. Không assert mọi role swap phải đảo truth vì có symmetric relations.

User-run neural preflight cần xác nhận finite logits/losses, gradients từ L_pos đi qua GRU/head tới frame encoder, truth/pair losses không detach r ở RelContrast, đúng role-span masking và DDP normalization. Static metadata tests không thay thế các kiểm tra này. Không thêm arm hoặc thay cohort/hypothesis/gates từ review implementation này.
