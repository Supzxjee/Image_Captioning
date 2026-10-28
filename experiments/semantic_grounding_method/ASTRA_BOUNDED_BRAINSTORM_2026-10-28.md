# Một vòng chốt hướng sau originality review

> SUPERSEDED sau khi user làm rõ rằng yêu cầu là tìm một hướng mới về bản chất, không tiếp tục diagnose T. Xem [ASTRA_NEW_DIRECTION_RELATION_FRAMES_2026-10-28.md](ASTRA_NEW_DIRECTION_RELATION_FRAMES_2026-10-28.md). T/E+ applicability gate trong file này chỉ là lịch sử đề xuất, không phải bước tiếp theo được khuyến nghị. Không thay kết quả T500.

28/10/2026. Thiết kế nghiên cứu; chưa code/GPU. Không mở thêm vòng brainstorm tự động. Dựa vào protocol v0.3, audited C/H/T500 và literature review cùng ngày; không đọc lại ZIP.

## Quyết định

**Chọn hướng 1 bên dưới: giữ nguyên T như một ứng viên có điều kiện, kiểm tra net signed contribution so với positive attribution share. Chưa chọn T làm phương pháp đủ cho paper.** Trước một duration trial, dùng checkpoint sẵn có để kiểm tra xem khác biệt này có hiện diện đáng kể hay chỉ tồn tại trong ví dụ toán học. Nếu fail, đóng T trong đợt này, không sửa threshold/module để cứu.

Tôi chưa tìm ra trong ba hướng một phương pháp mới đủ vững để cam kết bài báo trước 31/10. Quyết định này chọn một câu hỏi nghiên cứu có thể bị bác với chi phí nhỏ, không lấy sunk cost làm lý do giữ T.

### Ba tầng và mức căn chỉnh được chọn

1. **Đúng vùng:** weak phrase–instance joins có thể sai; mask đúng tọa độ không chứng minh đúng referent.
2. **Đúng nghĩa:** labelbank hiện chỉ hỗ trợ head/name, không xác nhận modifier/quan hệ. Giới hạn ở tập tên gọi cùng thực thể, không ánh xạ mọi token thành box.
3. **Decoder sử dụng:** kiểm bằng free captions và output gain; attribution đúng accounting nhưng không đổi/không cải thiện câu vẫn là failure.

Quan hệ nên có thứ tự hai đối tượng và bằng chứng tương tác khi dữ liệu cho phép; union box không chứa thứ tự vai. Đúng về thiết kế ngữ nghĩa không đồng nghĩa khả thi với bank hiện có.

## 1. Net gain theo vùng thay cho positive attribution share — CHỌN CÓ ĐIỀU KIỆN

**Closest prior:** Energy/EPG, Rao et al., ICCV 2023, §3.3–3.4 Eq. (2),(6), đã đọc [primary full text](https://arxiv.org/html/2303.11932v2#S3). Prior tối ưu positive attribution share. **Chưa có bằng chứng Energy thất bại trên A.** Giả thuyết cụ thể: clipping phần âm có thể đánh giá cao một vùng có nhiều positive mass nhưng gần như không đóng góp vào net gain.

Ví dụ riêng trên ba patch, hai patch đầu trong vùng: `c=(1,-.99,.09)`. Khi đó `Delta=.10`, `CG=.01`, `CU=.09`, `rho_plus=1/1.09=.917`; net share chỉ `.10`. Hai đại lượng cho tín hiệu khác nhau, không chỉ khác ký hiệu.

**Module/loss:** giữ toàn bộ readout và loss T v0.3, không đổi weights, mask, margin hoặc parameter count:

`B = L_SCST + .1 KL + .1(Lgain + Lcancel)`;

`T = B + .1 relu(.8 Delta_plus - CG)`.

Đối chứng tối thiểu mới, gọi **E+**, dùng cùng F/path/c_j/mask và cùng B:

`rho_plus = sum_j M_j*relu(c_j)/(sum_j relu(c_j)+1e-8)`;

`E+ = B + .1 Delta_plus*relu(.8-rho_plus)`.

E+ là **Energy-derived positive-share control**, đã ghép cùng threshold và gain scaling của T để cô lập việc giữ dấu. Không gọi đây là reproduction nguyên bản Energy loss `-EPG`. Với contributions không âm và Delta dương, hai loss hướng vùng trùng nhau khi bỏ sai số denominator; khác biệt chỉ xuất hiện khi có signed contributions. Không cần sweep hoặc gradient-rescale để tạo kết quả mong muốn.

Giữ gradients qua Delta, rho_plus, probabilities và attribution integral; không detach chúng để tạo objective khác. Epsilon chỉ xử lý denominator bằng zero, không phải hyperparameter đem tune. Nhánh E+ cần numerical/autograd/DDP preflight riêng trước optimizer, không thừa hưởng kết quả pass T một cách mặc định.

**Dữ liệu:** dùng nguyên prepared TRAIN2k, reference groups/common-referent guards, weak box bank, frozen A, cache, original head initialization; không nhãn mới. Học ở mức lexical event của thực thể. Current C/H/T500 và 128 adapter-held-out groups dự kiến có thể dùng cho phép kiểm tra đầu tiên. Bank ngoài TRAIN2k không phải ảnh mới với A.

**Lý do có thể có ích:** nếu vùng chỉ có positive evidence bị counterevidence triệt tiêu, tăng positive share có thể không tăng net regional contribution. T tạo pressure khác tại đúng các ca ấy; lợi ích cho câu vẫn là giả thuyết phải đo. Budget cancellation dùng chung ở B giúp tránh quy một lợi ích của regularizer chung cho direction T.

**Phản ví dụ/shortcut:** Energy đạt EPG=1 cùng Delta>0 đã kéo theo CG>=Delta; T không sửa một lỗi ở optimum hoàn hảo ấy. Khoảng trống chỉ nằm ở finite optimization/trade-off. T cũng có lỗi ngược: trong vùng c=.08, ngoài vùng c=100 và -99.98 thì Delta=.10, net share=.8 nhưng positive share rất thấp; hai-group budget vẫn pass. Globally contextual CLIP tokens và decomposition gaming còn nguyên. Vì thế T không bảo đảm bằng chứng cục bộ, không nhận pixel causality/recognition certificate.

**Tính mới:** một adaptation cụ thể kiểm tác dụng giữ dấu trong gain localization. IG, box guidance, Energy concentration và cancellation đều có tiền lệ. B chung có cancellation nên T-E+ không chứng minh budget mới/hữu ích riêng. Nếu hai phương án tương đương trong thực tế, không có lý do paper để giữ T.

## 2. Ràng buộc output bằng can thiệp hữu hạn — KHÔNG CHỌN

**Closest prior:** *See or Guess: Counterfactually Regularized Image Captioning*, 2024. Đọc [primary §4.1–4.4, Eq. (4)–(6)](https://arxiv.org/html/2408.16809v1#S4): mask thực thể, dùng caption phản thực và regularize TE/NDE. Không nhận can thiệp vùng hoặc so likelihood trên cùng prefix là mới.

**Khoảng trống giả thuyết:** hiệu ứng mask có thể gồm damage do input lạ, không chỉ mất bằng chứng đích. Chưa kiểm prior này trên A; chưa chứng minh matched nuisance correction sửa được vấn đề.

**Cơ chế/loss đủ để đối chiếu:** giữ kiến trúc A, tune hai decoder layers cuối cùng với main SCST+KL. Với reference event S/prefix p, đặt `f_theta(I)=log P_theta(S|I,p)`. Xóa vùng G hoặc vùng đối chứng B có diện tích tương đương:

`d_theta=[f_theta(I)-f_theta(I_minusG)]-[f_theta(I)-f_theta(I_minusB)]`;

`L_cf=relu(m-(d_theta-d_A))`; thêm vào main loss.

Đây là kiểm mức sensitivity đặc hiệu vùng được học thêm, không dùng attention làm proxy. Có ích nếu subtraction thật sự loại được nuisance. Nhưng model có thể chỉ làm output trên ảnh xóa G tệ hơn, không làm caption factual tốt hơn. Vùng B cũng có thể mang ngữ nghĩa cần thiết; matched area không chứng minh matched semantic nuisance. Thêm baseline A chưa đủ tạo novelty so với NDE.

**Có/thiếu:** boxes/head names có; cache ảnh pixel sau xóa chưa có. Xóa cached CLIP tokens không tương đương xóa object, do tokens đã trộn toàn ảnh. Cần RGB verified + CLIP forward mới và định nghĩa vùng B hợp lệ; không có relation labels sạch.

**Test tối thiểu:** 64 ảnh cố định, factual/target-mask/control-mask, so A và một pilot với same-prefix scores lẫn free captions. Nếu target/control masking phá A tương tự nhau hoặc chỉ có masked output suy giảm, bác diễn giải learned grounding. Cap probe 20 phút T4x2 sau chuẩn bị; extraction/training chưa có throughput nên chưa xác lập tổng ngân sách. Không dựng hướng này trước 31/10: gần prior, tăng phụ thuộc dữ liệu và có shortcut mạnh. Các teacher-intervention cũ kém không trực tiếp bác loss mới, nhưng cũng không cung cấp bằng chứng để quay lại nó.

## 3. Biểu diễn cặp có thứ tự và giám sát vai — KHÔNG CHỌN TRONG ĐỢT NÀY

**Closest prior:** Object Relation Transformer, NeurIPS 2019. Đọc [primary §3.3 Eq. (6)–(9)](https://arxiv.org/pdf/1906.05963). Geometry offsets trong Eq. (6) dùng absolute distance; đổi trái/phải hoặc trên/dưới có thể mất dấu trong kênh geometry đó. Điều này không chứng minh toàn model không phân biệt được hướng, và signed geometry/scene-graph captioning đã có nhiều tiền lệ.

**Cơ chế:** pair token `r_so=MLP([z_s,z_o,z_union,g_signed])`, giữ thứ tự subject/object; decoder đọc cặp hoặc object tùy bước. Không gán predicate vào một box đơn. Loss dự kiến:

`L=L_main + lambda CE(predicate|r_so) + mu relu(m-s(q,r_so)+s(q,r_os))`.

Chỉ áp role-swap loss khi có verified asymmetric predicate và đảo vai thật sự sai. Đối chứng: cùng kiến trúc geometry/union-only và ordered-pair không có role loss.

**Lý do có thể có ích:** union pooling không phân biệt ai tác động lên ai; order + role supervision có thể dạy decoder phân biệt câu đổi vai. Bằng chứng ở A mới là một số lỗi định tính, không phải thống kê lỗi quan hệ đại diện. Không gọi hình học trái/phải là hiểu “holding/under/riding”.

**Có/thiếu và shortcut:** có object boxes/patch cache; chưa có clean predicate–argument bindings và nhãn negative cho role swaps. UPT/VCOCO missing label không phải phủ định; các lỗi này đã làm feasibility cũ fail. Cổng xác nhận cặp bằng object-name overlap không lấp được nhãn thiếu. Model có thể học person luôn là subject, thuộc predicate class, hoặc geometry template mà không dùng tương tác.

**Test tối thiểu:** cần trước tiên một split riêng có ít nhất 32 factual asymmetric tuples và 32 verified role counterexamples, tách ảnh và loại tổ hợp; rồi thử pair-order/swap ablation và free-generation. Đây là data prerequisite, không yêu cầu user đi gán nhãn ngay. Pilot 2 arms chỉ được cấp cap 4 device-GPU-hours sau khi bank tồn tại và nguồn verified; hiện chưa thể ước ETA hoặc cấp GO. Thêm order/gate vào BGRM hiện tại chưa có điểm mới đáng bảo vệ; không chọn chỉ vì các lỗi trông giống quan hệ.

## Phép quyết định nhỏ nhất cho hướng chọn

### Bước 0 — hiện tượng có tồn tại không, chưa training

- Khóa 128 hash-first eligible images ngoài adapter TRAIN2k từ sealed bank, một group/image theo policy cũ; không chọn theo model score/Atlas/dev. Dùng đúng sidecar/cache proposal trước, không neural extraction.
- Forward C500 và T500, không optimizer; tính từng c_j, Delta, CG/CU, rho_plus, within-group positive/negative mass và hai losses. Giữ completeness và 64/128-node check. Caption/logits không thay đổi do phép đo.
- Predeclare một gate chi phí, không power claim: trong **128 group C500**, có ít nhất **16** ca `Delta>=.001`, `rho_plus>=.8` và `CG/Delta<=.5`. Báo cả mẫu số đủ/thiếu và toàn phân bố, không chỉ examples được chọn. Đây kiểm applicability của counterexample, chưa kiểm Energy-trained model thất bại.
- Đồng thời báo lỗi ngược ở T500: net share pass nhưng rho_plus<.5; không giấu nó. Đặc biệt không tính Delta âm hoặc Delta gần zero thành mismatch ấn tượng.
- Nếu không có đủ eligible inputs, numerical checks fail, hoặc gate không đạt: **dừng hướng signed-share T trong đợt này**. Không hạ threshold, mở thêm ảnh cho đủ ca hoặc thay một architecture khác ngay.
- Thiếu nguồn hoặc lỗi số học là computational-incomplete, không bằng chứng cơ chế sai. Chỉ applicability gate fail mới cho thấy chưa đủ cơ sở theo đuổi khoảng trống này trên cohort cố định.
- Cap cả staging và hai forwards: 20 phút T4x2 sau khi payload có sẵn, tức tối đa .67 device-GPU-hours; estimated forward ít hơn training cũ nhưng chưa đo. Nếu nguồn cache cần dựng lại thì báo unavailable thay vì tải/extract để kéo dài vòng này.

### Bước 1 — chỉ nếu bước 0 đạt và user duyệt triển khai

So **T và E+** tới **2.500 total updates**, cùng initialization, optimizer, TRAIN2k/group schedule, B, decoding và receipts. E+ đi từ initialization chung 0->500; continuation phase 500->2500 dùng cùng RNG policy đã khóa như T. T resume verified checkpoint500 của chính nó. Chỉ final2500 là primary, không chọn best checkpoint. C/H500 lưu làm lịch sử, không giả làm matched2500 controls.

Main contrast chỉ thay positive clipping/normalization so với net signed share. Đây là lý do kiểm T lại có nội dung mới so với C/H/T500, không phải thử lại frozen readout bằng cùng giả thuyết.

Gate caption: T-A>=+.01 CIDEr; paired 95% CI lower(T-A)>0 và lower(T-E+)>0; CHAIRs/i<=A; recall>=A-.005. Report count/ratio riêng. Gate mechanism trên **cùng fixed128 groups**: T phải tăng joint pass(Delta>=m, net-share>=.8, group budget pass) so với E+ với paired CI lower>0, và mean Delta_T>=mean Delta_E+. Không coi giảm Delta để làm tỷ lệ đẹp là thành công. Kiểm rho_plus/lỗi ngược, absolute CG và trường hợp generated nouns có mapping hợp lệ, tách reference/generated prefixes.

128 groups đã dùng quyết định theo đuổi hypothesis nên endpoint này là exploratory mechanism evidence trên adapter-held-out data, không phải untouched confirmation. Cần một tập xác nhận độc lập nếu phát triển thành paper; không tự bổ sung nó trong vòng hiện tại.

Nếu không đạt một trong các điều kiện: dừng T, giữ A. Đây là exploratory gate, không corrected paper significance; reused dev và một seed vẫn hạn chế. Passing chưa đủ nói hơn nguyên bản Energy: cần đối chứng Energy nguyên bản và confirmation để làm claim rộng hơn. Không tự thêm các run ấy trong vòng này.

**Compute:** warm500+end trước khoảng230s/arm; ngoại suy 4.500 new updates cho hai arm khoảng35 phút tổng thời gian các arm dùng T4x2 trước cold-start/extra evaluation, chỉ là ước lượng thô, không elapsed-time promise khi chạy song song. Cap45phút/arm cho train+endpoint, tổng cả stage0/bootstrap/archive tối đa **4.5 device-GPU-hours**; kiểm balance từng account hiện tại trước user-run, không dùng số25h cũ. Nếu throughput/preflight không đáp ứng cap thì stop computational-incomplete, không đổi updates/batch/quadrature.

## Claim và lịch dừng

Nếu qua mọi gate, claim tối đa là lợi ích exploratory của net signed localization so với **matched Energy-derived positive-share control** trên frozen captioner; không first attribution grounding, không relation understanding hoặc local recognition. Cần nói rõ đây là adaptation của prior.

Nếu trước 29/10 chưa có bước0 và matched endpoint khả thi, hoặc tới30/10 chưa có evidence vượt prior/control và confirmation, **chưa có hướng đủ mạnh cho bài phương pháp mới trước31/10**. Đến31/10 chốt A/báo cáo evidence và limitations. Một kết quả âm không tạo positive-method paper; cũng không bác mọi semantic alignment.

Đây là quyết định thiết kế cuối của vòng brainstorm được yêu cầu. Không có code/launch authorization, không tự mở hướng4 hoặc thêm một vòng tìm phương pháp khác.
