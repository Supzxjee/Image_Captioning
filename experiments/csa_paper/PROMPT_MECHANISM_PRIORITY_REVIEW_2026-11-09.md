# Phản biện hướng phát triển — 09/11/2026

**Trạng thái: đề xuất để người dùng chốt, chưa phải protocol chạy đã duyệt.** Không code trainer/notebook, chạy model/scorer, đổi selector, TEST, Word hoặc Sheet. Đọc file context người dùng `ASTRA_CONTEXT_HUONG_PHAT_TRIEN_09_11_2026.md`, AGENTS/RESEARCH_STATE và các nguồn dưới đây tại Git `ae8ca55`. Checkout cũ đã fast-forward trước khi đọc báo cáo69/190.

## Kết luận ưu tiên

Ưu tiên **đối chứng huấn luyện để tách giá trị embedding chữ riêng ảnh khỏi phần adapter/ROI, sau đó kiểm riêng mask gắn vùng nếu bước đầu có tín hiệu**. Đây là sửa thiết kế thực nghiệm; chưa có bằng chứng đủ mạnh để ưu tiên module/loss mới. Thí nghiệm giúp quyết định thành phần cần phát triển, không tự tạo ra một pipeline có điểm cao hơn.

Nếu ưu tiên duy nhất là cải thiện điểm hệ thống ngay, matched continuation SCST là lựa chọn thứ hai có cơ sở từ69. Hai mục tiêu (cải thiện hệ thống, giải thích cơ chế) không đồng nhất. Không mặc định thêm XE, SCST, object reward hay đổi backbone.

## Bằng chứng đủ và chưa đủ

- XE10 đã hoàn tất. CONTROL10/PROMPT9: CIDEr99.4587/98.8453, CHAIRs6.60/4.40%, recall39.7664/40.5387%. Caption PROMPT ngắn hơn nhưng object mentions và recalled class–image pairs cao hơn; không thể giải thích toàn bộ giảm CHAIR bằng bớt nội dung. So endpoint10 có đánh đổi CIDEr rõ hơn. NLL chênh giữa chế độ train/valid không đủ chẩn đoán overfit.
- Cặp180/60 TEST: beam-rank0 CIDEr105.5606→107.3603, CHAIRs5.62→3.76%; rerank114.3671→115.3522, CHAIRs4.54→3.20%. CI có điều kiện của gain CIDEr sau rerank chứa0; không tuyên bố tăng chắc chắn hoặc tương đương thống kê chỉ từ đó. One seed, C10/P9 selected starts, +4.468736M adapter params và TEST đã từng được xem giới hạn suy luận.
- Inference zero/shuffle ở PROMPT60 làm CIDEr giảm mạnh. Đủ chứng minh nhạy và sử dụng text trong chế độ suy luận đã đo; không thay thế nhánh được huấn luyện với đầu vào không có text. Không đo lại ablation cũ để thay cho đối chứng còn thiếu.
- CHAIR là object hallucination. SPICE Relations có tín hiệu sau rerank (+.3686, CI conditional [.0903,.6432]) nhưng là reference-graph overlap; chưa chứng minh đúng tương tác trên pixel, nhánh relation có đóng góp riêng, hoặc ROI binding là nguyên nhân. Attributes không tăng.
- END69−60 fixed rerank: CIDEr+1.2623 (CI [.2358,2.2923]); CHAIRs−.06pp (37 repaired/34 introduced; CI chứa0), recall−.2825pp. Raw CHAIRi tăng2.8552→2.9123. Có tín hiệu continuation tăng CIDEr, chưa có tín hiệu giảm hallucination mạnh. CONTROL chưa có SCST epoch2 nên không gán chênh69/180 cho prompt.
- Quy tắc lịch sử69 chọn original60 do recall guard vẫn phải giữ trong hồ sơ. Người dùng đã nói recall là phụ: có thể xem END69 là ứng viên đánh đổi theo quyết định mới sau kết quả. Không biến guard cũ thành yêu cầu khoa học vĩnh viễn, không sửa nó hồi tố. `best.bin`69 là60; END là `checkpoint.bin`, SHA `c4acaf39d4444919b5c19ceb5ba3182251b752ea4d1392649c4919c048c352c2`.
- Oracle5≈139.75 không bảo đảm một selector không dùng refs lấy được khoảng26đ còn lại hoặc câu oracle đúng thị giác. CPU190 chỉ bộ cũ qua guard trong66 bộ; mức tăng meanCIDEr không guard tốt nhất chỉ≈.0313 và PROMPT giảm. Chưa có lý do ưu tiên dò lại trọng số.
- Grounding30k: loss TRAIN giảm nhưng gain heldout CIDEr≈.428đ/CHAIRs−.10pp, CI chứa0. Object reward full1 ở hệ bank cũ mất≈.704CIDEr so CE để giảm4caption bị đánh dấu. Không kết luận các họ phương pháp vô hiệu; cũng không mở lại cùng recipe chỉ vì có tên “grounding” hoặc “hallucination”.

**Hai suy luận cần bác bỏ:** XE là objective học caption có điều kiện, không tự là phép đo alignment. SCST tối ưu reward CIDEr cũng không tự xác minh alignment. Cả hai dùng được trong nghiên cứu cơ chế khi có can thiệp và đối chứng phù hợp; tăng CIDEr sau nhiều updates chỉ chứng minh hiệu quả của recipe continuation trong phạm vi đã đo.

## Hai phương án thực tế

| Phương án | Giá trị | Hạn chế / thứ tự |
|---|---|---|
| Paired training: đúng text vs zero text, rồi mask nếu cần | Lấp khoảng trống giá trị explicit text và kiểm một cơ chế cục bộ; dẫn đường phát triển | **Ưu tiên**; chưa bảo đảm cải thiện production score |
| CONTROL180 tiếp SCST epoch2 để so lịch học với PROMPT69, cùng recipe69 | Ước lượng gain do thêm tối ưu và đặt END69 trong so sánh công bằng hơn | Thứ hai nếu mục tiêu điểm số/ngân sách phù hợp; C10/P9 và khác capacity vẫn còn, không thành pure prompt ablation |

Chưa ưu tiên RegionCLIP hoặc objective/reranker mới vì thiếu bằng chứng rằng đó là nút thắt chính. Không đưa ra cam kết điểm hoặc thời gian.

## Thí nghiệm ưu tiên — thiết kế đề xuất

### Câu hỏi và nguồn khởi đầu

Ở một backbone captioning đã có semantic bank alignment, explicit image-specific text có mang lại cải thiện ngoài việc cung cấp các ROI/geometry/adapter không? Nếu có, hard linked mask có lợi thêm không?

Khởi đầu chung: **CONTROL178 epoch10**, trước SCST; receipt checkpoint SHA `3dbc24409259f2e1a062c674fd5cec495e12aa168bcc1b747c4c5fc1beec1c46`. Lấy cùng base weights, thêm hai adapter mới có cùng initialization; output projections zero-init. Không lấy trọng số adapter đã học của PROMPT58/60/69 cho nhánh đúng-text rồi ép nhánh khác học lại. Không dùng hệ CSA10/139 cũ thay CONTROL178.

Tất cả nhánh cập nhật cùng base caption model và hai adapter; encoders CLIP/UPT/detector vẫn frozen/cached. Cùng optimizer mới, không mang Adam khác lịch sử vào một nhánh. Bước0 phải cho cùng logits/caption với cùng base vì adapter delta=0; đây là invariant dự kiến cần user runtime xác minh, chưa được chạy.

### Các nhánh

| Nhánh | Text vào Slot MLP cả train/eval | Slot visual/geometry và attention mask |
|---|---|---|
| **A — linked text** | Đúng ảnh, đúng slot | Vo gốc + normalized boxes; linked mask hiện tại |
| **Z — zero text** | Zero cả TRAIN lẫn VALID | Giữ nguyên ROI index/pair, boxes, slot-valid và linked masks; adapters vẫn được học |
| **U — unrestricted mask** (chỉ sau khi A/Z có tín hiệu) | Đúng ảnh, đúng slot như A | Giữ nguyên slots/Vo/geometry; cho mỗi valid ROI đọc mọi valid slot của đúng loại, vẫn chặn padding và zero residual khi không có slot |

A−Z đo giá trị của **explicit text embeddings bổ sung trên cấu trúc ROI đã chọn**. Z vẫn giữ thông tin do UPT/class matching lựa ROI, không phải “xóa mọi ngữ nghĩa”. Zero khiến các cột text của Linear đầu không nhận tín hiệu đầu vào; cùng kích thước/tham số danh nghĩa không có nghĩa cùng capacity hiệu dụng. Không gọi phép này là kiểm soát hoàn toàn capacity/semantic meaning. Nó khắc phục lỗi suy luận từ inference-only OOD và mạnh hơn đối chứng bỏ toàn bộ adapter.

A−U kiểm **ràng buộc attention phải theo liên kết**, với cùng slot builders và số tham số hoạt động. U vẫn có Vo/boxes gắn vùng trong slot; không được gọi U là “không grounding”. Không tuyên bố thí nghiệm này tách riêng chất lượng box, class matching hoặc nhánh relation. Nếu cần claim như vậy phải có thiết kế tiếp theo, không ngầm suy ra từ CHAIR.

### Loss, lịch và phép thử nhỏ nhất

Chọn **XE cho pilot kiểm cơ chế**, không phải vì hệ hiện tại thiếu XE hoặc chỉ XE mới hợp lệ. Teacher forcing cho hai nhánh cùng target và tránh thêm biến thiên sample/reward; mã đã có recipe continuation nhỏ làm tham chiếu. Loss chỉ `L = LCE`, không thêm grounding/object reward/SCST.

Đề xuất số cụ thể để duyệt: TRAIN30k xác định trước, round-robin references, **2.000 optimizer updates/global32 = 64.000 caption events mỗi nhánh**, cùng order/seed/minibatch và dataset receipts; hai lượt30k cộng4k lượt ba. Chỉ dùng TRAIN/restval captions, bank/ROI134/prompt175/vocab của current pipeline. Phải kiểm coverage object-only/relation/empty trước khi niêm phong pool; nếu pool không đại diện thì sửa thiết kế trước chạy, không đổi sau xem điểm.

Fresh Adam lr1e−5, betas(.9,.99), eps1e−8, weight_decay0, FP32/clip-grad1, dropout train như XE base hiện có (0.2), evalOFF. Đây là lựa chọn pilot kế thừa recipe continuation cũ, **không phải hyperparameter đã xác nhận tối ưu cho adapter mới**. Không sweep lr/lambda hoặc tăng updates từng nhánh khi thấy kết quả.

Phép thử tối thiểu là A và Z; chỉ chạy U từ **cùng nguồn ban đầu, cùng schedule/2.000 updates** nếu qua bước này. Không khởi tạo U từ A endpoint. Reuse A endpoint cho đối chiếu U chỉ khi nguồn/config/schedule không đổi. Có thể xác nhận recipe bằng seed độc lập sau khi có tín hiệu; pilot một seed không đủ claim tổng quát.

Đo thực tế throughput, peak memory, staging, decode5000, rerank và scorer để đặt cap trước chạy; 13h trước69 không phải quota hiện còn. Không dùng worker101.9phút của69 dự báo cho pilot khác. Preflight nhỏ chỉ kiểm số học/mask/gradient/throughput trên TRAIN, không dùng VALID để lựa recipe. Nếu run không hoàn tất, đánh dấu incomplete; không so full A với partial Z.

### Đánh giá và checkpoint selection

- **Endpoint update2.000 đã định trước** là checkpoint chính; không chọn best theo CIDEr/CHAIR giữa nhiều mốc. Bảo toàn source bước0 để mô tả tiến triển; A−Z mới là đối chứng nhân quả có điều kiện của pilot. Thêm U chỉ sau trigger đã định.
- Primary: cùng VALID5000, **beam rank0** (beam5/max20, không gọi greedy), CIDEr×100 và CHAIRs. CHAIRi là kiểm tra hỗ trợ bắt buộc. Sau đó báo fixedrerank .3/.4/.3 như kết quả cấp hệ thống, không thay trọng số hoặc dùng nó để che beam-rank0 suy giảm.
- Secondary: recall/image–class pairs, object mentions, chiều dài, tỷ lệ caption≤2từ, SPICE Objects/Relations/Attributes, NLL. Recall giảm nhẹ được báo như tradeoff, không tự động veto. Kiểm nhóm giảm CHAIR có còn nêu đúng nội dung/quan hệ bằng review ảnh mù nhãn nhánh nếu có tín hiệu.
- Báo fullVALID và nhóm xác định từ metadata trước decode: object-only, có relation links, không có valid slots. Không lựa nhóm theo câu mô hình tạo ra hoặc metric mới. Empty-slot subgroup vẫn có thể khác giữa nhánh sau train vì base weights đã đổi; không dùng nó như invariant toàn mô hình.
- Paired image bootstrap trên per-image outputs, reaggregate ratio metrics, common reference/IDF convention; ghi rõ conditional một seed/reusedVALID. Không gọi một split mới của VALID là blind holdout mới. Không dùng TEST chọn loss/epoch/selector; TEST cũ đã exposure phải tiếp tục công khai.

### Ngưỡng ra quyết định — đề xuất, chưa khóa

Mục tiêu pilot là phát hiện lợi ích cơ chế đủ đáng xác nhận tiếp. Gợi ý A−Z: **CHAIRs giảm ít nhất0.3pp (15/5000 caption), CIDEr không giảm quá1.0đ**; báo CI paired để phân biệt tín hiệu rõ với chưa đủ lực. Nếu CI95 upper của ΔCHAIRs<0 và lower của ΔCIDEr>−1 thì có bằng chứng conditional mạnh hơn; các margin này là mức thực dụng đề xuất, không phải MCID đã được chứng minh. Phải chốt trước chạy.

Không yêu cầu mọi metric cùng tăng. Nếu chỉ CIDEr tăng mà CHAIR không cải thiện thì chỉ kết luận lợi ích cho chất lượng caption, chưa đủ claim giảm hallucination. Nếu CHAIR giảm cùng việc mất nhiều object mentions/coverage hoặc caption ngắn bất thường, chưa nhận là tốt hơn; phân tích loss of content trước. CHAIRi xấu rõ phải được giải thích, không bỏ qua vì CHAIRs đẹp.

- A>Z có tín hiệu: triển khai U để kiểm mask, rồi xác nhận một seed khác/lịch matched lớn hơn nếu ngân sách cho phép; không tuyên bố pilot đã giải thích toàn bộ điểm PROMPT60/69.
- A≈Z với CI rộng: **inconclusive**; không tự nhân epochs. Kiểm adapter đã rời zero-init và cohort có đủ slots trước khi bàn thêm kinh phí.
- A≈Z có CI đủ hẹp loại lợi ích thực dụng: hạ ưu tiên explicit text trong recipe này, ưu tiên kiểm ROI selection/adapter hoặc hướng tối ưu hệ thống; không phủ định mọi chế độ joint training từ đầu.
- U ngang hoặc hơn A: không bảo vệ hard mask vì tên “alignment”; cân nhắc U như biến thể đơn giản hơn và xác nhận quality/hallucination. A thắng U hỗ trợ mask hữu ích có điều kiện, chưa chứng minh quan hệ đúng trên pixel.

## Điều phải xác minh trước chốt run

Đã đọc `model.py`, `prompt_adapter_model.py`, `prompt_scratch_model.py`, `prompt_pilot_model.py`, các đoạn optimizer/sampling trong `run_prompt_scst.py`, và source/protocol continuation. Scratch wrapper hiện gọi prompt theo normal; không được coi tùy chọn zero ở inference là đã có nhánh train-zero. U đòi sửa `allowed` có kiểm padding/empty theo thiết kế, không bật toàn bộ padded slots.

Cần receipt/binary availability của CONTROL178 và exact state-dict prefix/base load; hash initial full adapter state giữa nhánh; mapping dữ liệu/current vocabulary; lịch caption-event và số liệu coverage; mask/runtime gradients/DDP/empty-slot checks do người dùng chạy; quota hiện tại trước cam kết. Chưa load binary hoặc kiểm neural locally. Bản continue-result cũ nói chưa đổi recall guard được đọc cùng quyết định mới trong RESEARCH_STATE/context09, không xem nó là yêu cầu hiện tại bất biến.

## Nguồn đã đối chiếu

- [XE quality](PROMPT_SCRATCH_QUALITY_RESULT.md), [XE checkpoint/history](PROMPT_SCRATCH_XE10_RESULT.md) — trạng thái CHAIR cũ đã được file quality bổ sung.
- [SCST180/60](PROMPT_SCST_RESULT.md), [TEST và inference controls](PROMPT_TEST_RESULT.md), [SPICE subsets](SPICE_SUBSETS_RESULT.md).
- [Continuation69](PROMPT60_CONTINUE_RESULT.md), [locked historical protocol](PROMPT_SCST_CONTINUE_PROTOCOL.md), [state](../../RESEARCH_STATE.md).
- [Calibration190](valid_rerank_diagnostic/CALIBRATION_RESULT.md), [oracle189](valid_rerank_diagnostic/ORACLE_RESULT.md).
- [Grounding30k](GROUNDING_30K_PILOT_RESULT.md), [recipe30k](GROUNDING_30K_PROTOCOL.md), [object reward full1](SCST_FULL1_RESULT.md).

Đây là đề xuất thực nghiệm dựa trên code/kết quả dự án; không viện dẫn một phương pháp mới từ literature hoặc khẳng định có hiệu quả mới được kiểm chứng. Không đọc lại ZIP hoặc chạy lại các phép chấm đã hoàn tất.
