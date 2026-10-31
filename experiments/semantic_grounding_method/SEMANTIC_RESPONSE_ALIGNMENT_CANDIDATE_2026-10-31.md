# Tìm hướng khác: căn chỉnh phản ứng ngữ nghĩa của bộ sinh caption

Ngày: 31/10/2026. Trạng thái: đề xuất để thảo luận; chưa chốt phương pháp, chưa novelty clearance, chưa triển khai/run. Không tiếp tục cứu editor C bằng cách đổi tên.

## 1. Câu hỏi và giả thuyết

Các nhánh trước có thể học được điểm quan hệ phụ nhưng caption ít thay đổi. Đề xuất chuyển đơn vị giám sát sang **phản ứng của chính decoder khi một sự thật thị giác thay đổi**.

Ví dụ giả định (chưa phải cặp ảnh đã kiểm tra): ảnh I có người cầm ván trượt, ảnh I' thay ván bằng vợt; người và áo xanh được giữ nguyên. Bộ sinh phải chuyển ưu tiên ván→vợt, đồng thời giữ nhận biết người/áo xanh. Không yêu cầu hai caption giống từng chữ hoặc bắt caption luôn đề cập tất cả sự thật.

Giả thuyết cần thử: giám sát có chọn lọc sự thay đổi và sự ổn định của các cụm mô tả trong decoder có ích hơn chỉ cho xem thêm các cặp ảnh/caption bằng CE. Đây là giả thuyết về hiệu quả, không phải kết quả đã có.

## 2. Những gì đã có — không nhận là mới

| Nguồn gốc | Phần đã xác minh | Giới hạn của lần đọc |
|---|---|---|
| [See or Guess, ACM MM2024](https://arxiv.org/html/2408.16809) | §4: dữ liệu che vùng ảnh, caption phản thực và TE/NDE regularization tác động tới xác suất sinh; NDE giữ văn cảnh để tách ảnh hưởng trực tiếp. | Đã đọc method; không gọi việc dùng decoder log-prob/counterfactual là mới. |
| [COMO, ECCV2024](https://njuhugn.github.io/pdf/COMO-eccv24.pdf) | Multimodal counterfactuals và hard-negative contrastive alignment. | PDF đọc phần mở đầu/method overview; chưa audit toàn bộ mọi objective/phụ lục. |
| [HALVA, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b73228d026ef49bfa49f95b7e330513e-Abstract-Conference.html) | Phrase-level alignment từ response pairs là prior trực tiếp. | Abstract/intro đã đọc; chưa full-method audit trong vòng này. |
| [CHiP, ICLR2025](https://arxiv.org/html/2501.16629) | §4, Eq11–12: visual preference cùng hierarchical text preference; rejected images có thể từ crop/rotation/noise. | Đọc source method; dùng visual preference hay token weighting không đủ tạo novelty. |
| [MED, NeurIPS2025](https://arxiv.org/html/2506.07227) | §3–4: minimal image edits và mô tả khác biệt; Eq7 huấn luyện mô tả khác biệt từ visual feature difference. | Abstract còn nêu feature consistency; cần đối chiếu code trước khi tái hiện đúng từng term. Không giản lược MED thành chỉ data augmentation. |
| [Grounding on Request, preprint 24/10/2026](https://arxiv.org/abs/2610.33841) | PDF §3 và AppendixC: CTM tăng log-likelihood đáp án mới, giảm đáp án cũ qua intervention. AppendixC còn có Stay objective và paired-SFT control. | Đã tải PDF và đọc công thức; chưa có xác nhận hội nghị. Đây là collision mới rất gần, phải tính vào novelty dù là preprint. |
| [EqSim, ICCV2023](https://openaccess.thecvf.com/content/ICCV2023/html/Wang_Equivariant_Similarity_for_Vision-Language_Foundation_Models_ICCV_2023_paper.html) | Abstract chính thức: similarity cần thay đổi tương ứng thay đổi ngữ nghĩa. | PDF web trả403; chưa full-method audit. Không nhận semantic equivariance là ý tưởng đầu tiên. |

Tìm thêm với các nhóm từ: counterfactual captioning + unchanged/token/preservation; semantic edit + selective invariance/equivariance; intervention response matrix. Không thấy khớp exact từ truy vấn không có nghĩa chứng minh chưa ai làm.

## 3. Cấu hình nghiên cứu cụ thể còn đáng đối chiếu

**Ma trận đáp ứng sự thật–can thiệp trên decoder caption**, thay cho một điểm toàn ảnh hoặc một câu trả lời đơn.

Với một ảnh và các sự thật đã kiểm tra f1...fk, một can thiệp e thay đúng một sự thật. Ghi nhãn hai tập:

- Changed: cụm đúng trước và sau can thiệp, đã kiểm chứng loại trừ nhau cho cùng đối tượng.
- Stable: sự thật thực sự còn đúng trên cả hai ảnh. Unknown bỏ qua; không suy ra từ việc caption tham chiếu không nhắc tới.

Cho cùng một ngữ cảnh văn bản c_j, định nghĩa d_j(I) là chênh lệch log-likelihood chuẩn hóa độ dài giữa hai cụm đối lập f_j và g_j trong **decoder đang sinh caption**. Hai cụm có thể nhiều từ; mỗi cụm được teacher-force đúng thứ tự riêng. So giữa I/I' luôn giữ nguyên cùng candidate, cùng c_j, cùng preprocessing và cùng prompt-condition. Không so logits tại cùng chỉ số sau hai prefix đã phân kỳ.

Ma trận R[e,j] = d_j(I'_e) - d_j(I). Mỗi hàng mô tả ảnh hưởng của một can thiệp lên nhiều quyết định ngữ nghĩa:

1. Với mục bị đổi từ f_j sang g_j, học hai điều kiện d_j(I) >= m và d_j(I') <= -m. Chỉ tăng hiệu giữa hai ảnh không bảo đảm chọn đúng cả hai.
2. Với mục Stable, phạt phần vượt ngưỡng |R[e,j]| > epsilon, kèm CE cho cụm thật trên cả hai ảnh. Không ép toàn bộ vocabulary/hidden states giống nhau.
3. Vẫn CE caption trên ảnh thật để giữ mục tiêu sinh mô tả thông thường. Không tự thêm SCST trước khi thấy tác dụng riêng của cơ chế.

Loss ứng viên:

`L = L_caption + lambda_change * mean([m-d_j(I)]+ + [m+d_j(I')]+) + lambda_stable * mean([abs(R[e,j])-epsilon]+)`

Chưa chốt m/epsilon/weights. Công thức hinge, contrastive và invariance đều là thành phần đã biết; không gọi chúng là toán tử mới. Ma trận là cách tổ chức giám sát/kiểm tra; đổi tên ma trận không tạo đóng góp.

## 4. Khác biệt dự kiến và điều chưa đứng vững

So CTM, candidate muốn học **phân bố ảnh hưởng trên nhiều sự thật cùng ảnh**: can thiệp một sự thật chỉ nên đổi các mô tả chịu ảnh hưởng, không lan sang những sự thật được xác minh còn nguyên. CTM/Stay đã làm change/stability cho lựa chọn nguồn ảnh–text; chuyển sang nhiều sự thật của caption và chống ảnh hưởng chéo là delta dự kiến, chưa đủ tự kết luận novelty.

So MED, mục tiêu là caption một ảnh bằng decoder gốc; cặp ảnh chỉ dùng trong huấn luyện, không yêu cầu đầu vào đôi khi inference. So EqSim/COMO, score cần gắn với quyết định sinh cụm từ, không chỉ dual-encoder similarity. So See or Guess, cần đối chiếu thêm xem selective off-target constraint và verified replacement pairs có tạo cải tiến thực ngoài TE/NDE không.

**Kết luận hiện tại: có thiết kế để kiểm tra, chưa có phương pháp mới được xác lập.** Muốn đưa vào bài báo phải xác minh exact delta trước và chứng minh lợi ích ngoài paired-CE/CTM. Không khẳng định “đầu tiên”, “causal alignment” hay “đủ novelty cho tạp chí” bằng tìm kiếm này.

## 5. Feasibility và phản ví dụ

- Có thể giữ CLIP/Q-Former/decoder của C0, chỉ thích nghi phần đã được phép train; inference một ảnh → caption, không cần editor/diffusion/ảnh thứ hai.
- Caches UPT hiện tại không chứa counterfactual image twins hay nhãn Stable, nên KHÔNG đủ dùng ngay. MED là nguồn cần kiểm tra release/license/schema/overlap; chưa xác nhận tải được toàn bộ dữ liệu hay đủ nhãn.
- Caption references không tự cung cấp nhãn sai/đúng đầy đủ. Cần kiểm tra một tập nhỏ image twins; không yêu cầu người dùng chấm hàng trăm mẫu trước khi thấy feasibility.
- Thay vật có thể đồng thời đổi quan hệ, occlusion, tư thế, shadows. Các sự thật chịu ảnh hưởng thật không được gán Stable. Cặp không kiểm soát được thì bỏ.
- Một từ có nhiều referent không thể giám sát chính xác bằng câu chung chung. Chỉ nhận cặp xác định được instance trong pilot; không claim đã giải quyết group/instance binding.
- Text probes có thể cải thiện mà free captions không đổi: phải đo caption tự sinh, không chỉ margin/R.
- Nếu legacy prompt đổi theo từng ảnh thì không còn tách hiệu ứng thị giác. Nếu giữ prompt cũ thì là điều kiện stale-prompt có chủ đích, phải ghi rõ và áp dụng giống mọi control. Cần chốt cách conditioning trước code; chưa tự chọn một cách.
- Không gọi sửa ảnh bằng diffusion là can thiệp nhân quả đã được đảm bảo. Đây là controlled behavioral training dưới giả định chất lượng edit.

## 6. Phép thử nhỏ nhất nếu user chốt hướng

Trước GPU: xác minh quyền truy cập/schema một nguồn có sẵn, kiểm tra khoảng20–30 cặp đã gồm Changed và ít nhất một Stable, khả năng phân biệt ở preprocessing thật, từ nằm trong vocab native và chia split theo ảnh nguồn. Không hợp lệ thì dừng trước huấn luyện.

Pilot tối thiểu cùng C0/init/trainable params/ảnh/labels/updates:

1. Paired CE: cùng toàn bộ ảnh và cụm đúng nhưng không coupling giữa hai ảnh.
2. Paired CE + change term: đối chứng gần CTM/TE, không gọi là tái hiện nguyên bản.
3. Paired CE + change + selective stability: candidate đầy đủ.

So trên held-out source images: đúng trước/sau đổi, giữ đúng sự thật Stable, và **caption tự sinh**. Muốn tiếp tục phải có tín hiệu tốt hơn paired CE/change-only, đồng thời CIDEr/CHAIR/recall/độ dài trên ảnh thật không bị đánh đổi tùy tiện. Chưa đặt gate số/ngân sách vì dữ liệu và user chưa chốt.

Khuyến nghị: đây là hướng dự phòng đáng thảo luận hơn thêm một grounding head độc lập. Chưa mở notebook lớn hoặc hứa có kết quả vượt baseline; nếu full prior check cho thấy exact mechanism đã có, chỉ có thể kế thừa có ghi nguồn rồi tìm cải tiến thực khác, không thu hẹp câu chữ để giữ claim mới.

## 7. Đã làm trong vòng này

- Web search đa nhóm từ; đọc các đoạn method được chỉ rõ trong bảng. GRD/editor review của vòng trước vẫn có hiệu lực, không xóa lịch sử.
- CTM PDF/text cached ignored tại `review/grounding_on_request.pdf` và `.txt`; không đưa full paper vào repo.
- Chỉ lưu thiết kế và trạng thái; chưa code neural/model/GPU/Kaggle, chưa dataset mới, chưa ghi thêm sheet cho brainstorm.

## 8. Kiểm tra tiếp theo yêu cầu người dùng — 31/10/2026

**Quyết định: SỬA và chỉ giữ như giả thuyết có điều kiện; chưa chốt đóng góp bài báo hoặc mở huấn luyện.**

### Prior sát và đánh giá lại

- Đã đọc HALVA §2 Eq1–3 đầy đủ hơn: likelihood đúng/sai ở phrase và reference-model token regularization là prior. HALVA không tự chứng minh trùng cross-image Stable của candidate, nhưng giảm mạnh phần có thể nhận mới.
- Đã đọc thêm [Debias Your VLM with Counterfactuals: A Unified Approach](https://openreview.net/pdf/17de59356537f7511f6f4179afd4be4327961002.pdf), bản anonymous under-review ICLR2024, §4.3–4.4 Eq7–9. Caption branch dùng ảnh/caption phản thực và language-model loss; retrieval branch phân biệt neutral query với query chứa thuộc tính bị đổi. Chưa xác minh final publication; không ghi đã accepted ICLR. Không thấy exact selective cross-fact caption loss trong các đoạn đã đọc.
- CTM+Stay, HALVA, See or Guess, MED, EqSim và COMO vẫn là prior gần. Không tìm thấy exact cấu hình trong lượt tìm có giới hạn; điều này không phải chứng nhận mới.
- Điểm yếu của bản thiết kế: loss đang cộng các contrast độc lập. Gọi chúng là ma trận R không tạo coupling mới, cũng không tự thành cơ chế căn chỉnh mới. Phần có thể bảo vệ chỉ là **ràng buộc giữ các sự thật không bị tác động, có nhãn hợp lệ, trên actual caption decoder** và giá trị thực nghiệm của nó ngoài change-only/paired-CE. Nguyên lý invariance đã biết; khả năng đây là adaptation/incremental contribution vẫn cao.
- Stable relative odds có thể đứng yên dù xác suất cả cụm đúng và cụm sai đều giảm. Vì vậy CE cho sự thật đúng trên từng ảnh là điều kiện cần của thiết kế, cùng đánh giá free captions; không được chỉ báo R đẹp là thành công.

### Nguồn dữ liệu đã xác minh thêm

1. GitHub MED redirect tới [Relaxed-System-Lab/hallu_med](https://github.com/Relaxed-System-Lab/hallu_med). GitHub API recursive tree hiện chỉ có README và scripts/utils, không có payload ảnh/nhãn. README nói dataset có sẵn nhưng không đưa link HF trực tiếp.
2. Tìm được [EasonFan/hallu_med](https://huggingface.co/datasets/EasonFan/hallu_med), public, ungated, metadata license apache-2.0; revision `38f969ac98d7db856ee8d57b1a23c4032cfff09e`. Tên tài khoản phù hợp dấu vết tác giả nhưng chưa có link từ repo xác nhận đây là release chính thức, nên provenance vẫn cần nối lại.
3. Đã đọc API file metadata: `edit_image.tar` 5,138,954,240 bytes; `bench_images.tar` 66,538,496 bytes; `med_sft_data.json` 26,939,203 bytes; còn `med_bench.json`, `med_real_set.json`. Chưa tải tar ảnh.
4. Đã đọc JSON train: **50,753 records**. Bản ghi có `messages` (câu hỏi và mô tả khác biệt) cùng `images` (ảnh gốc như `./VG_100k/2336913.jpg`, ảnh sửa `./edit_image/2336913_edit.jpg`). Mẫu đã đọc không có original full caption, Changed span, Stable facts, boxes hay matched-prefix annotations. Không coi tất cả record đều thiếu một trường chỉ từ một mẫu; chưa kiểm tra thống kê toàn schema.
5. Đã đọc hai benchmark records để xác minh định dạng MCQ; **không chọn chúng để fit/tune hoặc làm tập chấm đầu tiên rồi dùng lại cho đánh giá độc lập**. Các option có cả dạng “not mentioned”, cho thấy không thể tự chuyển câu khác biệt thành truth label mà không kiểm tra ảnh.
6. Nguồn dự phòng [Intel/COCO-Counterfactuals](https://huggingface.co/datasets/Intel/COCO-Counterfactuals/blob/main/README.md): card CC-BY-4.0, cặp caption thay noun và synthetic image pairs, có examples.jsonl/images.zip. Chưa tải/kiểm tra chất lượng/overlap; không tương đương ảnh thật chỉ thay đúng một vật đã được xác minh.

### Kết luận thực hiện

- Feasibility **một phần**: có candidate payload để kiểm tra, chưa đủ nhãn Stable, chưa kiểm ảnh, license/provenance/overlap/vocab native chưa khóa. Không phải đã có dataset dùng ngay.
- Khuyến nghị kiểm tra một mẫu nhỏ từ TRAIN có source IDs tách khỏi mọi eval, xác minh ảnh gốc/sửa và tối thiểu một Changed + một Stable thật. Chỉ sau khi nhãn đủ và người dùng chốt mới viết notebook/pilot.
- Trước khi trình đây là phương pháp mới, phải viết delta hẹp so CTM/See or Guess và chấp nhận khả năng chỉ là cải tiến có điều kiện. Nếu người dùng yêu cầu một nguyên lý căn chỉnh hoàn toàn mới đã được xác lập ngay bây giờ, **kết quả kiểm tra này chưa đáp ứng**.
- Không neural/model/GPU, không tải ảnh hàng GB, không sửa notebook, không phát sinh kết quả thực nghiệm hoặc sheet row.
