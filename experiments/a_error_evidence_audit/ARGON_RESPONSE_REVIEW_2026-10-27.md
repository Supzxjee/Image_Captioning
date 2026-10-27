# Review Argon SD-RPR — 27/10/2026

**Có một ứng viên kiến trúc cụ thể đáng xem xét: thêm đường đọc raw CLIP patches
ở phía decoder, bỏ qua bước nén Q-Former. Chưa chấp thuận triển khai/train bản
SD-RPR nguyên văn.** Nhiều bảo đảm toán học, diễn giải nguyên nhân và novelty trong
đề xuất chưa đúng hoặc chưa có bằng chứng. File đính kèm là nội dung đề xuất, không
phải lệnh thực thi training.

Raw response được lưu ở ignored review/ARGON_RESPONSE_2026-10-27.md; SHA và nguồn
ở ARGON_RESPONSE_PROVENANCE.json. Không chạy caption model hoặc neural parser.

## 1. Giữ phần nào

Khác biệt có thể có giá trị so với các nhánh đã thử trong dự án: query phụ thuộc
trạng thái decoder đọc **raw197−CLS patches**, thay vì chỉ sửa Q/semantic memory
trước decoder hoặc cộng bias80 lớp giống nhau ở mọi bước. Đây là giả thuyết bổ
sung nguồn thông tin sau bottleneck, không phải bằng chứng bottleneck đã xóa cue.

Zero-initialized output residual có thể khôi phục A tại khởi tạo nếu mọi tensor
hữu hạn, đường A/masks/numerics không đổi. Cache hiện có phù hợp để xem xét một
pilot, nhưng chưa đo throughput/I/O và chưa có quyền tự launch.

## 2. Các sửa đổi bắt buộc trước khi thiết kế thực nghiệm

### Memory cố định không có nghĩa attention bất biến

`captioning/models.py:401–428` dùng3 lớp nn.TransformerDecoder. Query tại bước t
phụ thuộc prefix; attention lên cùng memory có thể đổi theo từng bước. Điều mới
tiềm năng là **nguồn raw patches/đường đọc thêm**, không phải lần đầu A có attention
động. u ở lớp3 cũng đã mang thông tin ảnh qua2 lớp cross-attention trước; không
phải trạng thái cú pháp/ngôn ngữ thuần túy.

ARGON_MATH_CHECK.json minh họa cùng hai memory vectors nhưng hai query tạo readout
[.982,.018] và[.018,.982]. Đây là ví dụ toán, không phải phép đo A.

### Trực giao một vector không loại bỏ thông tin đã có trong memory

e−proj_m(e) chỉ loại một hướng trong hệ tọa độ học được. Nó không là projection
ra ngoài span của toàn52 memory tokens, càng không là phép đo thông tin ngữ nghĩa
mới. Wm có thể đổi hướng/tiến về0; nhánh e⊙m vẫn nhận thành phần song song. Ví dụ
memory chứa hai trục: residual[0,3] trực giao m[2,0] nhưng vẫn hoàn toàn trong span
memory. Có thể thử operator này như heuristic sau, không gọi novelty/information
removal đã được bảo đảm.

### Chứng minh zero-evidence và chống shortcut chưa đúng

Đề xuất có bf trong GELU(Wf z+bf). Khi e=0 và z=0, bf=1 vẫn cho output khác0
(GELU(1)=.841345). Bias của value/output projections và LayerNorm affine cũng cần
được xác định. Xóa các bias ấy vẫn không chứng minh dùng đúng nội dung thị giác:
features có thành phần chung, query/h đã mang thông tin ảnh và token patch CLIP
đã contextualize toàn cảnh. TopK12 không bắt buộc12 patches liền nhau hay tương
đương một vùng vật thể6% diện tích với ngữ nghĩa biệt lập.

### Loss raw contribution không khớp contribution thật vào logits

Patch contrast đang dùng Wvocab·Δh, còn output dùng tanh(gamma)·g·Wvocab·Δh.
Gamma được phép âm và gates của ảnh thật/tráo có thể khác nhau. Raw margin đúng
hơn1 có thể thành contribution khác biệt−.7616 khi gamma=−1; ngay cả gamma dương,
gate .1/.9 và raw margins2/1 cho contribution difference−.7. Main loss có thể học
cách hòa giải, nhưng không có bảo đảm mà tài liệu tuyên bố. Cần xác định chính
đại lượng muốn giám sát và kiểm gradient/zero-init; chưa viết loss từ lời hứa đó.

### Top competitors không phải semantic negatives

Một token khác reference có thể là paraphrase hoặc một cách mô tả đúng khác.
Danh từ ở clause khác không tự sai ở vị trí đang xét; chính audit đã thấy thiếu
chi tiết không đồng nghĩa false. Nếu dùng restricted softmax như lexical CE thì
gọi đúng là reference-conditioned training, không biến nó thành negative gold.
Lọc đồng nghĩa “cùng ngữ cảnh” và source prompt text chưa có đặc tả thực thi.

Ảnh ngẫu nhiên có thể chứa cùng class/quan hệ. Contrast có thể học khớp identity
giữa h có nguồn từ ảnh thật và patches, không nhất thiết học phạm vi holding.
Không được gán nhãn “semantic false” cho random image bằng thiếu reference.

Ngoài ra, A chọn oranges trong câu greedy không xác nhận oranges là top1 dưới
prefix của reference khác. Các ví dụ hard-foil trong Argon là minh họa, chưa là
logits đã đo.

### Frozen weights không bảo đảm phân phối/fluency/CIDEr bất biến

Residual đổi logits, rồi đổi prefix ở bước sau dù toàn bộ A frozen. Sigmoid g
luôn dương với logits hữu hạn, không tự chỉ hoạt động ở bước bất định. Chính
noun G/S và native H/L đã giữ A frozen mà vẫn giảm CIDEr. Không có bằng chứng
full-sentence CE là nguyên nhân duy nhất mọi regression hoặc CE luôn làm phẳng
phân phối SCST.

Object reward chỉ chấm một taxonomy thô; điều đó không chứng minh nó gây các
subtype errors đã xem. Code object_count_v2 còn phạt recognized unsupported
classes, nên không thể nói mọi việc đổi sang COCO noun đều tối đa reward.

### Effective rank không chọn được nguồn semantic tốt hơn

Raw768 và projected512 có kích thước/scale khác nhau. Effective rank/singular
spectrum không kiểm khả năng phân biệt lemon/orange hay predicate scope. Không
dùng10phút rank check làm bằng chứng “projection xóa semantics”. Mặc định raw là
một thiết kế bypass có thể kiểm, không cần câu chuyện rank để cấp phép.

### Vocabulary và số tham số cần khai báo đúng

Saved exact A word2idx có10000 tokens. `lemon/lemons`, `helicopter`, `tablet` có;
`helicopters`, `DS`, `tablets` không có, kể cả khi kiểm variants dấu câu. Wvocab
đóng băng không mở thêm token: đây là existing-vocabulary readout rộng hơn80 lớp,
không phải open-vocabulary captioning. Không train một target đã thành UNK như thể
đang học sinh đúng từ bị thiếu.

Nếu fusion MLP có hidden128, sáu projection matrices đã425984weights, trước
bias/LN/gate; không phải395000. Nếu hidden width khác cần ghi rõ rồi đếm từ code.
Không tự sửa vocabulary trong pilot vì sẽ tạo thêm confound.

### Đối chứng và tiêu chí kết luận phải được sửa

T có patch-contrast, C_PermPatch không có: khác cả nguồn và objective, chưa isolate
được lợi ích patch đúng. Inference patch swap/static q1 là intervention có thể
OOD; giảm CIDEr dưới nó cho thấy dependence, không tự chứng minh grounding đúng.
CHAIR/recall cũng không đo predicate-scope correctness.

Chưa có đo đạc cho3–5phút/arm hoặc lập luận chắc chắn paired variance giảm. Gate
+.006 có thể được đề xuất cho method mới nhưng cần lý do prespecification, không
nhờ hạ từ+.01 và lời hứa frozen A giữ câu tốt. Val5000 cũng đã xem nhiều lần.
Một No-Go chỉ bác bỏ cấu hình/budget cụ thể; không chứng minh A đã bão hòa mọi
post-hoc alignment. Cả tuyên bố chắc chắn tránh distribution drift lẫn tuyên bố
chắc chắn chạm ceiling đều cần rút.

## 3. Đối chiếu prior bước đầu — không phải novelty clearance

Nguồn chính thức đã kiểm:

- [Visual Sentinel, CVPR2017](https://arxiv.org/pdf/1612.01887): đã đọc phần method
 2.1–2.3. Image context phụ thuộc hidden state ở từng từ; adaptive gate quyết định
 khi dùng visual signal. Do đó per-step readout/gating không tự mới.
- [Bottom-Up and Top-Down Attention, CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.html): metadata/abstract chính thức xác
 nhận object/salient-region attention. PDF lần này bị403/size limit; không nhận
 đã đọc lại toàn method.
- [RSTNet, CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RSTNet_Captioning_With_Adaptive_Attention_on_Visual_and_Non-Visual_Words_CVPR_2021_paper.html): metadata/abstract chính thức mô tả adaptive visual/non-visual word
 prediction trong Transformer. PDF lần này bị403; chưa loại trừ overlap chi tiết.

Academic-search MCP không được mount; dùng web để truy cập primary CVF/arXiv,
không dùng trích dẫn từ bài tổng hợp làm bằng chứng kỹ thuật. Đây là kiểm tra
prior giới hạn, chưa rà hết các raw-feature skips/adapters/contrastive decoding.
Việc ba nguyên liệu có tiền lệ không tự chứng minh mọi cách kết hợp thiếu mới;
ngược lại, kết hợp lại cũng không tự là một phương pháp mới.

## 4. Khuyến nghị root gửi Astra

Giữ **giả thuyết raw-patch bypass phía decoder** làm ứng viên để Astra cân nhắc,
không giữ nguyên lời giải thích static attention, guarantees trực giao, false
foil supervision hoặc ceiling. Trước notebook cần một module/loss contract ngắn,
nguồn target khả dụng, đối chứng chỉ đổi đúng yếu tố cần kiểm và outcome nào làm
chọn/bỏ cơ chế. Không ép cứu mọi chi tiết SD-RPR và không biến toy checks thành
feasibility/gain evidence.

## 5. Astra đã chốt ứng viên thiết kế

Giữ bypass Q-Former nhưng **đọc196 projected+ReLU patches512 chiều**, dùng đúng
frozen preprocessing A; không khẳng định bypass projection768→512. Đối chứng
đọc32 Q cùngfeature512, cùng4 linear128/no-bias/LN-no-affine:262144weights;
Wo=0 để khôi phục A tại khởi tạo. Hai nhánh cùng SCST trên phân phối sauresidual
và.1 KL neo A; bỏ toàn bộ topK/orthogonal/entropy/foil/random-negative loss.

Đề xuất500updates/arm trên2000 locked TRAIN images, end-only, dùng reward/IDF
đã pin. Gate P−A≥.01 cùng CI dương, P−Q CI dương và object safeguards; không hạ
gate hoặc dùng atlas làm validation. Cap thay ETA. Xem PATCH_READOUT_PILOT_CONTRACT.md
và phản hồi Astra đầy đủ. Đây là thiết kế để hoàn thiện pilot, **chưa có code ML,
notebook mới, run hoặc novelty được xác lập**.
