# Frame pretraining200 — quyết định sau kết quả

28/10/2026. Dựa trên RESULT.json/root source audit và đọc tĩnh frame_model.py/run_frames.py; không đọc lại ZIP, không load checkpoint/neural local hoặc launch GPU.

## 1. Chốt kết quả hiện tại

**Giữ `PRETRAIN_GATE_NO_GO`, không chuyển ba caption arms của recipe này.** Hai arm hoàn tất200updates, numerical/model preflight và source/metadata audit pass theo receipts. Không có bằng chứng cụ thể để quy kết quả cho lỗi implementation.

RelPos40/64=62.5%; RelContrast41/64=64.0625%; difference+1.5625pp, paired95%CI[-12.5,+15.625]pp. Không đạt accuracy70%, gain10pp hoặc CI lower>0. CI rộng nên chưa chứng minh equality/không có effect; đây là operational no-go của cấu hình/budget, không bác mọi relational representation.

13pairs chỉ Contrast đúng,12pairs chỉ Pos đúng; thay đổi gần cân bằng. Projective/containment tăng16->22/32, contact/proximity giảm24->19/32. Đây là descriptive heterogeneity, không cho phép bỏ contact/touching hoặc lấy subgroup tốt làm success. Chưa có caption nào được cải thiện trong thử nghiệm này.

TruthBCE.0244/pairloss.000663 ở update200 là **training minibatch/rank log**, không phải held-out truth accuracy hoặc full-TRAIN evaluation. PosCE2.7547 so với2.6987 cũng không cô lập được nguyên nhân. Không suy “classifier đã generalize”, “geometry được học còn contact thiếu resolution”, hoặc “encoder vô dụng” từ các số này.

## 2. Một điểm cấu trúc có thể giải thích thất bại của proxy

Code hiện tại:

`z=frame_encoder(I,subject_box,object_box)`,z512;

`d_r=truth_head_r(z)`;

`h0=tanh(initial(z))`,initial512->256;

`ell_G(q|z)=GRU_likelihood(q,h0)`.

Truth/pair losses cập nhật frame encoder/truth head; **không có direct gradient vào GRU/output/initial parameters** từ hai losses đó. GRU học CE trên positive frames, không học likelihood discrimination trên negative frames. Đây là intended multi-task design, không phải autograd bug.

Counterexample cấu trúc: nếu hai frame outputs khác nhau theo delta_z với `W_initial delta_z=0` nhưng `w_truth,r dot delta_z!=0`, truth head phân biệt được còn GRU có cùng h0, cùng teacher-forced probabilities. Rank512->256 cho phép các directions như vậy; chưa biết trained model có khai thác chúng không. Ngay cả khi thông tin đi qua h0, GRU vẫn có thể ưu tiên text prefix.

Do đó gate hiện tại đo **linguistic access qua một auxiliary GRU cụ thể**, không trực tiếp đo toàn bộ thông tin trong z hoặc khả năng một caption decoder khác đọc z. Đây là một hạn chế của gate tôi đề xuất. Nó không cho phép sửa hồi tố gate thành pass, nhưng chỉ ra phép chẩn đoán có decision value.

## 3. Bước tiếp theo tôi chọn: một frozen endpoint diagnostic

Chỉ inference trên checkpoints200 sẵn có, không optimizer/fitted probe, không đổi source labels/cohort hoặc threshold. Root có thể chuẩn bị user-run script sau khi user chốt; Astra không chạy neural local.

Xuất cùng256TRAIN và64HELD pair IDs:

1. RelContrast truth logits `d_plus,d_minus`, correct-image ranking và balanced binary accuracy với threshold0 cố định; confidence intervals trên heldout. Báo cả hai strata, không chỉ nhóm tốt. RelPos truth head chưa được supervised nên không dùng nó làm trained semantic baseline.
2. GRU likelihood gap của cả hai arms phân rã theo subject/predicate/object spans, với đúng weights cũ. Kiểm tổng ba thành phần khôi phục primary gap. Đây là component trace; không tự gọi subject-gap hoặc predicate-gap riêng là truth metric, và không chọn thành phần tốt nhất làm primary mới.
3. Report train/held gaps để phân biệt training fit với generalization; verify actual checkpoint/file hashes khi load. Không train một linear probe trên64heldout, không calibrate/rank-select head trên đó.

Cap một diagnostic run10phútT4x1 kể cảload/hash/eval (.167deviceh), không extrapolate11s warmtraining thành elapsed-time guarantee. Chỉ trả JSON nhỏ/source receipts. Original NO_GO không đổi dù diagnostic thuận lợi.

**Decision logic:** nếu truth head cũng không generalize đáng tin, chưa có bằng chứng một representation quan hệ tốt đang bị GRU che mất; dừng recipe cho deadline. Nếu truth head generalize rõ hơn GRU trên cùng held-out pairs, có lý do kiểm một thay đổi ở **đường học representation->ngôn ngữ**, thay vì thêm epochs hoặc đổi subgroup. Chỉ dùng kết quả như exploratory diagnosis trên reused data, không paper confirmation.

## 4. Nếu có bằng chứng cho readout gap: một thay đổi cơ chế đáng xem xét

Tôi chọn **bỏ auxiliary-GRU bridge khi học grounding, cho counterexample supervision đi qua actual caption decoder**. Không mở thêm một loạt tied-head/crop/weight variants.

Giữ ordered frame module làm representation candidate; nếu dùng encoder200 thì cả controls và proposed variant phải bắt đầu từ **cùng một encoder checkpoint**, không dùng initialization khác để quy gain cho loss. Reader/decoder nhận direct gradients từ normalized likelihood của **cả assertion tail predicate+object**, với cùng subject prefix trên I+/I-. Không phạt riêng relation token khi object đích chưa nằm trong prefix: relation đó có thể đúng với một object khác.

Với q=(s,r,o), định nghĩa:

`ell_plus = mean_logprob_theta(r,o | subject_prefix, I_plus)`;

`ell_minus = mean_logprob_theta(r,o | subject_prefix, I_minus)`;

`L_positive = -ell_plus`;

`L_language_pair = softplus(ell_minus-ell_plus)`.

Common caption SCST+KL giữ nguyên. Hai variants tối thiểu của một **protocol mới** sẽ dùng cùng positive loss/data/frame/decoder và chỉ khác language-pair term; original A là baseline. Positive term ngăn tạo ranking chỉ bằng làm cả hai likelihood tệ đi. Cần giữ entity/content guards, actual reader-use probe và free captions; không gọi score trên templates là caption gain.

Lợi ích cơ chế: counterexample gradient đi qua chính language readout cần sử dụng quan hệ, thay vì một classifier có thể đọc directions vô hình với GRU. Cache v105 đã có original A prompts/features nên điểm này nay có đầu vào phù hợp. Rủi ro vẫn lớn: suppress false tail không tự bảo đảm richer captions; strong A có thể bị lệch phân phối; reader có thể bị bỏ qua; latent CLIP regions không region-pure.

Đây là targeted redesign ứng viên, **không được tự chạy như continuation của failed200 protocol**. Trước run phải khóa objective/weights/endpoint/control và ghi đã dùng64pairs để chẩn đoán; dữ liệu này không còn untouched confirmation. Không thêm caption training chỉ vì auxiliary truth head cao mà gọi đã qua gate cũ.

**Novelty thẳng thắn:** direct contrastive caption likelihood đã có tiền lệ, ví dụ Dai & Lin2017 [primary](https://arxiv.org/pdf/1710.02534). Khác biệt có thể nghiên cứu là verified same-entity relational counterimages và direct regional-frame-to-caption transfer, không phải softplus/pair loss mới. Chưa có căn cứ novelty đủ mạnh cho paper chỉ từ sửa coupling này.

## 5. Khuyến nghị cho deadline

Làm tối đa frozen diagnostic ở mục3; không chạy caption arms hoặc tăng updates của recipe cũ. Nếu không phân biệt được representation failure với readout gap, giữ A và chốt negative pilot/data artifacts cho báo cáo. Nếu có readout-gap evidence, root có thể đưa **một** direct-decoder protocol cụ thể cho user chọn; không tự launch và không hứa sẽ có positive-method paper trước31/10.

Hiện tại: data/cache preparation và numerical implementation có kết quả tốt; hiệu quả semantic transfer qua GRU chưa được xác lập; free-caption improvement và originality đều chưa có. Chi phí GPU nhỏ không thay thế bằng chứng cho một lần train khác.
