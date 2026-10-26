# Kết luận trao đổi Astra–Codex sau C/T

26/10/2026. **Chọn ứng viên để kiểm tra feasibility, chưa duyệt train, chưa có gain
hoặc novelty được xác lập.** Bản phương pháp đầy đủ ở `METHOD_PROPOSAL.md`.

## Phương pháp đang chọn

Giữ kiến trúc A, chỉ mở decoder layer cuối (~4,204Mparams); không thêm verifier hoặc
memory khi inference. Đường ảnh mới vẫn dùng ảnh và prompt gốc, không GT/parser.
Tín hiệu học bổ sung đến từ việc đối chiếu mệnh đề caption với V-COCO actor–role
và COCO instance IDs, giữ entity binding nhất quán giữa các quan hệ trong cùng câu.

Ví dụ người1 cầm gậy,người2 cưỡi ngựa: không thưởng cả hai quan hệ cho câu nói
**một người** vừa cầm gậy vừa cưỡi ngựa bằng cách đổi actor giữa hai cạnh.
Nếu annotation không đủ phản bác mọi binding, giữ unknown; không suy từ thiếu link
thành false. Với caption không định danh actor, phạm vi là mệnh đề tồn tại có vai
nhất quán, chưa phải selected-instance bbox grounding.

Mỗi mệnh đề có utility interval: supported[w,w],refuted[−w,−w],unknown[−w,w].
So trực tiếp sampled và greedy caption. Sau hủy các mệnh đề chung, khoảng utility
của phần thêm/bỏ quyết định signed advantage a_G: dùng cận gần0 nếu khoảng hoàn toàn
dương/âm,chiaK=3 vàclip[−1,1]; nếu khoảng chứa0 thì a_G=0.

Control: L_C=CE−.05 stopgrad(R_base(sample)−R_base(greedy)) logp(sample).
Treatment: L_T=L_C−.01 stopgrad(a_G) logp(sample).
R_base giữ nguyên CIDEr-D +.5object_count_v2 đã có. Chỉ base term là SCST chuẩn;
role term là **masked pairwise policy-gradient surrogate**, không là unbiased
policy gradient của expected fixed scalar role reward. Đối chứng mới khớp regime
SCST và trainability, nên không cần C0 CE+KL như một arm bắt buộc trước đó.

## Phản biện đã dẫn tới sửa công thức

Bản Astra đầu chấm r_safe mỗi caption so với fixed A rồi trừ sample/greedy.
Codex chỉ ra unknown bonus0 vẫn nhận role advantage âm nếu greedy được thưởng.
Ví dụ unknown interval[−1,1] cóbonus0,known-positive[1,1] cóbonus1/3:
old additional reward advantage=.2(0−1/3)=−.066667.
Astra đồng ý sửa sang **direct pair interval**: ở ví dụ này[−2,0] chứa0 nên a_G=0;
chiều ngược[0,2] cũng bằng0. Unknown của một caption riêng lẻ không tự quyết định
advantage; phải xét khoảng của chính cặp sau triệt tiêu phần chung.

Không nhận conservatism của bound là bảo đảm chất lượng sau optimizer step. Bound
chỉ đáng tin trong ontology/parser/annotation đã chấp nhận. Base CE/SCST vẫn có thể
cập nhật mẫu có role advantage0.

## Bước được khuyến nghị trước train

Một packet feasibility: kiểm provenance/cache/split/role coverage;128FITimages cố
định lấy một frozen-A greedy và một sample; kiểm parser/binding/interval advantage;
audit30scoring decisions. Đây chưa phải train. Nếu chưa có saved caption đúngID,
người dùng cần chạy frozen model trên Kaggle; không chạy local.

Gate đề xuất trước dữ liệu: n>=500FITeligible;>=27/30derivations đúng và không có
unknown→false vì thiếu annotation;>=16/128pairs có a_G khác0;>=8cases cách xử lý
interval/identity thực sự khác bag-tuples hoặc unknown=0. Gate density không phải
power estimate. Nếu fail thì dừng ứng viên, không tự nới ontology/sample count.

Pool train dự kiến tối đa2.000FIT và replay factual disjoint tương đương. Không
nhập calibration/dev vào FIT. Inventory haiactions cũ chỉ85positiveFITimages giao
TRAIN10k; phải kiểm toàn FIT/cache,không giả định1.000positive ở main10k. Các ca nhiều
actor/refutation cần COCO instances đầy đủ; export linked endpoints không đủ.

Sau packet mới cân nhắc A + C_SCST + T_role,hai nhánh cùng data/seed/optimizer và chỉ
khác role term. Primary gate T−A>=.01CIDEr,CIpositive T−A/T−C,CHAIRs/i không xấu,
recall drop<=.005;secondaryrole-dev phải có tín hiệu và báo coverage/unknown. Một
seed/reuseddev chưa thay được final evaluation hoặc chứng minh alignment.

Encoderfrozen cho phép cache memory≤52×512(~0.53GB/5kimages),nhưng đọc visual/text
source và SCST rollouts/parser vẫn cần đo. Preparation cap30phút,train/end cap60phút
mỗiarm là đề xuất ngân sách,chưa phải ETA;quota còn lại chưa xác nhận.

## Khuyến nghị riêng của Codex

Tôi đồng ý **kiểm feasibility ứng viên này** trước khi viết notebook train. Nó xử
lý rõ hơn lỗi nhãn thiếu và tránh lặp lại các token/region adapters đã thử. Điểm
rủi ro quyết định là parser/entity consistency và mật độ cặp có ordering xác định;
không nên đặt tên module/loss rồi coi là đã có đóng góp. CEV-PG của Argon chưa có
label policy đủ chắc để chạy bốn nhánh như đề xuất.

SC-Captioner/FoCUS và các prior đã có semantic-component rewards. Phần đang khảo sát
là partial-annotation interval kết hợp binding nhất quán; novelty vẫn pending việc
đối chiếu conservative/partial-label RL. Mã/parser/loss/model mới chưa được viết,
không có GPU run mới. JSON prompt gốc local không có; audit Argon riêng vẫn pending,
nhưng phương án này không cần JSON đó để dựng prompt trung hòa.

## Cập nhật triển khai feasibility — 2026-10-26

Người dùng đã duyệt chuẩn bị packet. Có notebook unexecuted và mã scorer/parser
adapter/audit, chưa train hoặc chạy model. Xem README.md để chạy. Metadata builder
đã kiểm lại2026 FIT/1330 positive links/6284 actor-action labels; replay metadata
tách biệt1898 ảnh, cap dự kiến1896 trước actual cache coverage.

Theo review code của Astra, packet bổ sung: whole opaque giữ[-3,+3] với cap3edges
toàn projected caption; mixed unhandled clause phải opaque; worker provenance
gắn protocol/stage/selection/source/rank/records; hai critical flags cho audit;
binding search cap báo riêng và không tính là operative effect. Bag comparator
giữ cùng partial-label evaluator, chỉ bỏ joint identity/inequality để contrast
không bị lẫn khác biệt xử lý nhãn âm.

Đây là triển khai để falsify feasibility, không là xác nhận parser đúng trên câu
thật, hiệu quả caption, đóng góp mới hay chấp thuận training.
