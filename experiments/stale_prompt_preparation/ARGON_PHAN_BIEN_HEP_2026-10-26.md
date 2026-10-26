# Vòng phản biện hẹp cho Argon — stale semantic prompt, 26/10/2026

Bạn là người phản biện độc lập cho một đề tài căn chỉnh ngữ nghĩa–thị giác trong image
captioning. Astra chịu trách nhiệm chọn hướng nghiên cứu; cộng sự triển khai kiểm tra
giả định và code. Tôi chuyển tài liệu này thủ công từ tài khoản khác. Bạn không cần
truy cập chat cũ, raw ZIP hay toàn bộ repository để trả lời lượt này.

**Yêu cầu hiện tại:** đánh giá một thiết kế ứng viên trước khi khóa pilot. Không mở
brainstorm rộng hoặc đưa thêm năm kiến trúc. Bạn được phép **Reject** nếu nhãn/loss
sai hoặc ranh giới tính mới quá yếu; không cần tìm lý do đồng ý. Nếu Revise, đề xuất
sửa nhỏ nhất vẫn trả lời được giả thuyết chính. Chưa có thí nghiệm mới nào được chạy.

## 1. Mục tiêu và bằng chứng đang có

Mục tiêu cuối tuần là một phương pháp có đóng góp bảo vệ được và caption tốt hơn A.
Không có căn cứ hứa chắc đạt hạn này. Giải thích kết quả âm không thay cho phương pháp
mới mà đề tài đang cần. Hai tài khoản Kaggle, mỗi notebook T4x2; giờ còn lại thực tế
chưa xác nhận. Không chạy model local; người dùng tự chạy notebook Kaggle.

A: CLIP ViT-B/16 raw197x768 → projection512 → Q-Former32queries/2layers → prompt→Q
attention/gate/LN → memory [grounded prompt;Q] → Transformer decoder3layers. Prompt
CLIP text cache gốc20tokens, có nguồn text JSON đã tìm lại và xác minh với cached
embeddings. A đã được tối ưu SCST+CE+object reward và huấn luyện tiếp; đây không phải
baseline không prompt. A có thể chạy với cache visual/text, backbone frozen.

UPT/VG/CSA/routing/noun heads/native-resolution pilots trước chưa có gain đáng tin cậy
trước A. Không suy ra mọi alignment đều vô ích hay nguyên nhân do frozen decoder.

Frozen diagnostic gần nhất:

- 138/10000TRAIN (1,38%) đủ2nounclasses unique-instance và separated boxes,298phrases.
  Phrase→Q→patch maps thật tốt hơn image-mean và whole-map derangement ở box mass và
  axis order (74,82%); centroid inside GT chỉ28,26%. Weak automatic labels, không gold.
- GT-max-Q oracle có mass headroom, nhưng kiểm tra A/T/W bác bỏ utility của correction
  cụ thể: frozenA, convex mixing<=10% tại attended_prompt trước gate/LN, đúng-Q vs
  donor-Q matched pre-fusion norm.137interventions tăng raw target box mass trung bình
  +.02233 nhưng noun benefit T−A=−.002612nat/token,CI95[−.004846,−.000450]; selectivity
  −.003868,CI[−.007653,−.000152].
- Full138 free caption CIDEr A1.48534,T1.48735,W1.48318 (fixed TRAIN10kIDF). T−A+.00201,
  CI[−.00392,+.00889]; chỉ3captionđổi; CHAIR-flagged9→10,recall276→275/603. Technical
  source/hash/norm/parity checks PASS; post-fusion norms vẫn khác. Không train teacher
  distillation; không rút57ca benefit dương sau No-Go để cứu recipe.

Kết luận đúng phạm vi: attention mass oracle chưa là teacher hữu ích; không chứng minh
prompt overreliance là nguyên nhân. Hướng dưới là **giả thuyết mới**, chưa được xác nhận.
TRAIN/dev1000/val5000/test5000 đều đã được dùng/xem; không gọi bất kỳ split này là fresh
unseen benchmark chỉ bằng cách chia lại tên.

## 2. Giả thuyết ứng viên và bốn điều kiện

Hỏi: khi bằng chứng của target bị che, semantic prompt gốc có làm decoder tin target
quá mức không; hạn chế sự tăng ấy có cải thiện caption trên ảnh nguyên bản không?

Chọn <=2000ảnh từ locked TRAIN10k, metadata hash-first, một target class/ảnh. Target
phải có COCO instances, có trong prompt thực sự còn ở20token và first reference mention
là một token caption thuộc alias không nhập nhằng. Chỉ dùng single-token alias IDs
cho next-token class probability; loại compounds/ambiguous aliases không đo an toàn.

| | P1 prompt gốc | P0 prompt trung hòa |
|---|---|---|
| I1 ảnh gốc | factual | factual, bỏ cue trực tiếp |
| I0 ảnh che target | prompt cue đã lỗi thời | cue trực tiếp cũng bị trung hòa |

I0 che union segmentation của **mọi annotated instance cùng class**, bỏ target nếu
bất kỳ instance crowd/invalid; không bỏ qua instance lỗi rồi gọi đã che hết. Fill cố
địnhRGB(123,117,104), không inpaint. Diện tích mask khóa1%–40% để tránh pixel cực nhỏ
hoặc gần xóa cả ảnh. Annotation có thể thiếu instance, che context hoặc lộ shape của
target. **I0 không được gọi là chứng nhận absence**, không CE caption “xóa từ”.

Mã hóa lại pixels I0 bằng CLIP frozen, không zero tokens cũ vì features contextualized.
I1 dùng cache gốc với metadata square224bilinear/normalization đã xác minh; re-encode
I1 trên fixed16panel để kiểm preprocessing/cache parity.

P0 thay mọi mention alias của target bằng `object`, sửa immediate a→an, giữ phần còn
lại **trong cửa sổ text P1 thực sự giữ lại**. Cửa sổ phải replay đúng token IDs/mask
P1; loại P0 nếu vượt20tokens. Không đưa thêm đuôi vốn bị truncate vào P0 khi từ thay
thế ngắn hơn. Lưu offsets/tokenIDs cả hai. First noun phải khớp một token whitespace
nguyên vẹn, không suy vị trí từ số ký tự; loại compound dấu nối/phẩy nhập nhằng.
Encode lại cả chuỗi theo budget20. P1 giữ đúng cached vectors gốc và được kiểm
text→embedding binding. Không bơm reference vào prompt. Number agreement/predicate và
cue gián tiếp có thể còn; cần phản biện xem neutralization đủ ý nghĩa không.

Caption thật y dùng cho(I1,P1),(I1,P0). I0 chỉ dùng ràng buộc xác suất, không có caption
GT mới. Tất cả bốn điều kiện dùng cùng prefix trước first mention của target.

## 3. Loss và matched control đề xuất

p_c(Iv,Pp)=sum_{w in aliasIDs(c)} p(w|Iv,Pp,prefix); s_vp=log(clamp(p_c,1e-8,1−1e-8)).
Alias set khóa từ TRAIN vocabulary; punctuation variants hợp lệ là token riêng. Đây
là class token probability, không so caption nhiều từ hoặc suy quan hệ holding/riding.

L_base=CE(I1,P1;y)+.1 KL(p_A||p_theta) trên prefix/valid caption tokens.
L_neutral=CE(I1,P0;y).
L_vis=.5 sum_{p=0,1} ReLU(.2−s_1p+s_0p).
L_stale=ReLU(s_01−stopgrad(s_00)−.1).

C=L_base+.1L_neutral+.1L_vis.
T=C+.1L_stale.

Hai arm có cùng dữ liệu4conditions, trainability, data order, budget; khác duy nhất
L_stale. Gradient L_stale chỉ giảm s_01 nếu hinge active, không cho nâng s_00 để giảm
loss; không trực tiếp cấm prompt giúp ảnh factual. L_vis cập nhật cả visible/occluded.
CE factual có thể chống global noun suppression nhưng không bảo đảm loại shortcut.

Không thêm module inference. Frozen CLIP image/text/vis_projection; mở lớp QFormer
cuối, prompt_projection/MHA/gate/LN và cả decoder giống nhau ở C/T. LR5e-6 AdamW,
eval-mode có gradient. Đề xuất1250updates/globalbatch8 (one-pass TRAIN10k), auxiliary
batch8 mỗi5updates, <=2000aux images. Không SCST hay sweep ở lượt đầu. Hệ số/margins là
starting protocol chưa tối ưu, không nói có chứng cứ chúng là tốt nhất.

## 4. Bước đang chuẩn bị, không phải pilot đã được duyệt

Manifest metadata +fixed16panel ảnh/text; frozenA four-condition scores trên128ảnh
hash-first để biết stale boost/visual contrast có hiện hữu. Không lọc target sau khi
xem scores. Báo tỷ lệ boost>.1đãkhóa, quantiles, strata theo class/kíchthướcmask; không
đổi margin để tạo active cases. Cache lưu factual và occluded vision tokens+P0/P1 dùngchung.

Benchmark50batch forward–backward của đúng C/T objective trênT4x2, resetA/RNG giữaarm,
không optimizer.step, xác minh A weights giữ nguyên. Optimizer latency chưa đo, AdamW
state memory chỉ ước tính riêng. Trầnprep/cache30phút; projected train+dev <=2giờwallclock/arm.
Đây là budget ceiling, không phải ETA. Projection có safety1.5 và scoring reserve,
chưa chắc bao phủ mọi I/O. Không tự giảm steps/batch rồi gọi cùngprotocol nếu vượt.

Primary endpoint nếu pilot được chốt: rawgreedy trên ảnh/prompt gốc dev1000, sameIDF.
T>A ít nhất+.01CIDEr/lower95CI>0;T>C lower95CI>0;CHAIRs/i khôngxấu;recalldrop<=.005.
Length là diagnostic. Chỉ thắng ảnh che không đạt mục tiêu. Nếu thắng cần kiểm prompt
abandonment và mask-artifact control trước claim, nhưng không mở hàng loạt arms trước.

## 5. Phản biện cần bạn trả lời — tối đa khoảng1500từ

1. **Proceed/Revise/Reject?** Chỉ ra lỗi nền tảng nếu có; phân biệt nhãn sai, intervention
   không nhận diện được và giả thuyết có thể đúng nhưng chưa chứng minh.
2. L_stale có thật sự điều kiện hóa reliance vào visual evidence, hay dễ chỉ global
   noun suppression/prompt abandonment/nhận diện màu mask? Đưa phản ví dụ cụ thể.
3. Thiết kế2x2 và C/T đã tách contribution mới đủ chưa? Kiểm tra rẻ nhất nào nên có
   trước train; kiểm tra nào có thể để sau gain mà không làm pilot mất ý nghĩa?
4. Counterfactual mask, prefix và neutralization có làm nhãn/score class lệch không?
   Nếu sửa, đưa sửa nhỏ nhất, không tạo caption GT giả/đưa từ reference vào input.
5. Chọn tối đa2prior gần nhất để xét novelty, ưu tiên nguồn chính thức. Astra đã đọc
   *See or Guess: Counterfactually Regularized Image Captioning* (ACM MM2024), method
   4.1–4.4, và thừa nhận masking/likelihoodcontrast không mới. Hãy kiểm tra xem phần
   stale semantic cue conditioning có khác thực chất hay chỉ đổi tên objective.
6. Nêu **một điều kiện đủ mạnh để dừng** trước khi tốn GPUtrain và **một outcome** có
   thể khiến bạn đổi ý. Không hứa cải thiện hoặc paper novelty từ lý luận.

Nếu không đọc được full method của prior, nói rõ mới đọc abstract và cần phần nào.
Không bịa conference/quartile/công thức. Không mặc định phải chấp thuận kế hoạch của
Astra; cũng không bác bỏ cho có. Trả lời vào cơ chế đang xét, không quay lại brainstorm
mười hướng. Kết thúc bằng việc đầu tiên cụ thể nên làm trước khi khóa C/T pilot.


## Đồng bộ sau review triển khai của Astra

Main CE và output KL dùng đủ năm caption mỗi ảnh, chuẩn hóa theo tổng valid tokens
trên hai GPU (global-token mean). Auxiliary dùng một reference/prefix đã khóa.
Retained-window policy ở mục2 là bắt buộc: P0 không được lộ suffix ngoài cửa sổ P1.

Hai phản ví dụ cần xét: (1) silhouette sau fill vẫn cho biết đó là người/đối tượng,
nên COCO mask không cung cấp nhãn bắt buộc xác suất class phải giảm; (2) mô hình đổi
sang từ ngoài alias set, khiến điểm class giảm mà nội dung caption không đổi.
Không diễn giải giảm alias probability thành giảm hallucination nếu chưa kiểm caption.

Bộ chuẩn bị báo b1=s11-s10, b0=s01-s00, interaction=b1-b0 theo ảnh và exp(s)
cho cả bốn điều kiện. exp(s) kế thừa clipping [1e-8,1-1e-8], chỉ bao phủ alias set đã khóa.
Các phép này dùng lại bốn score có sẵn, không thêm forward hay chọn mẫu theo score.
Đây là diagnostic; interaction dương hoặc cost gate đạt không tự động duyệt train.
