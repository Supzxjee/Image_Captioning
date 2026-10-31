# Kiểm tra tính mới hướng C: sửa caption có bằng chứng vùng ảnh

Ngày 31/10/2026. Codex1 tiếp quản vì Codex2 hết quota; user yêu cầu thực hiện kiểm tra X012. Đây là literature review và đề xuất, không phải protocol đã duyệt hoặc kết quả mô hình.

## Quyết định

**SỬA C ở mức đề xuất; chưa chốt là đóng góp mới, chưa mở run.** Độ tin cậy cao rằng sửa cục bộ, giữ cấu trúc, dùng region features để sinh từ sửa và dự đoán edit/content đều có prior. Chưa xác nhận exact collision của joint span–action–witness với selected-witness-only emissions. Không tìm thấy exact paper trong lượt này không phải bằng chứng phần còn lại mới.

Phát hiện quan trọng ngoài X011: phải đối chiếu **TIger ECCV2022** và **DECap ECCV2024**, không chỉ LURE/Woodpecker. So với một rewriter toàn câu không đủ chứng minh phần cải tiến vì explicit visual editors đã tồn tại.

## Nguồn sơ cấp đã đọc

Chỉ tóm tắt các phần thực sự truy cập, không nhận đã đọc mọi supplementary/code. Không dùng số benchmark editing để so ngang CIDEr trên cohort COCO captioning của ta.

| Nguồn và độ sâu | Cơ chế đã xác minh | Hệ quả đối với C |
|---|---|---|
| [TIger: Explicit Image Caption Editing, ECCV2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960111.pdf), §3–4.2, đặc biệt §4.1–4.2/Fig.4 | Tagger xóa/thêm, Inserter điền MASK; multimodal BERT đọc proposal features, tọa độ và token ảnh chung. Hidden của MASK đi qua vocab head để sinh từ. | Sửa từng từ, copy cấu trúc và đưa vùng ảnh trực tiếp vào đường sinh từ đều đã có. Phần method đọc chưa thấy một region-ID witness bắt buộc riêng cho mỗi replacement; không suy từ đó rằng toàn bộ prior khác cũng thiếu. |
| [DECap: Towards Generalized Explicit Caption Editing via Diffusion Mechanism, ECCV2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06001.pdf), §3.2–3.4, Eq.5–7 | Denoising dự đoán edit và content; bốn thao tác KEEP/DELETE/INSERT/REPLACE; Transformer multimodal, hai heads edit/vocab; loss edit và content, content chỉ cho INSERT/REPLACE. | Joint edit/content và REPLACE không mới. Chưa thấy selected-instance witness bottleneck trong các phần đã đọc. DECap khác DeCap ICLR2023 giải mã CLIP; không nhầm hai bài. |
| [LURE, ICLR2024](https://arxiv.org/html/2310.00754), §3.1–3.2/Algorithms1–2, §4 data | Revisor học image + corrupted description → corrected description; nhiễu co-occurrence, uncertainty, vị trí; IDK masking để xem lại nội dung đáng nghi. | Phát hiện từ nghi ngờ, mask và sửa theo ảnh đã có. Method đã đọc không đưa positive witness-ID supervision thành ràng buộc replacement bắt buộc. Không dùng điều này làm exact-collision clearance. |
| [Woodpecker](https://arxiv.org/html/2310.16045), §3.1–3.5 và implementation details | Detector/VQA trả lời câu hỏi; tạo visual claims; LLM sửa qua knowledge prompt và gắn boxes sau mô tả. | Bằng chứng vùng, xác minh trước sửa và đầu ra có box đều đã có. Khác selected-region neural emission, nhưng không được gọi evidence-guided revision là mới. |
| [Kim et al., Context-Aware Image Caption Editing via Hallucination-Resistant Visual Instruction Tuning, ICCV Workshops2025](https://openaccess.thecvf.com/content/ICCV2025W/CLVL/papers/Kim_Context-Aware_Image_Caption_Editing_via_Hallucination-Resistant_Visual_Instruction_Tuning_ICCVW_2025_paper.pdf), §2.1, §3.3–4.1/Fig.1–2 | Instruction tuning: chọn object và sửa caption giữ context; GPT tạo lỗi existence/attribute/interaction/count; có clean training examples. | Clean/no-edit và sửa tối thiểu không mới. Đây là workshop, không gán Q1/Q2 hoặc main-conference status. Vẫn là prior liên quan cần tính đến. |

## Phần nào phải bỏ khỏi claim

- “Có module chỉnh caption thay vì sinh lại” — prior explicit editing.
- “Có KEEP/REPLACE/DELETE và giữ từ đúng” — prior edit actions; giới hạn một edit là scope/chi phí, không tự là novelty.
- “Đặc trưng vùng tham gia xác suất từ thay thế” — TIger đã dùng region features trong Inserter. Phải nói rõ khác biệt giữa đọc cả bank và một selected-instance bottleneck; không nhận mọi prior chỉ có grounding head ngoài đường sinh.
- “Có boxes hoặc visual claims để giải thích” — prior evidence-guided revision.
- “Giữ clean captions/abstain khi thiếu bằng chứng” — cần cho chất lượng và kiểm soát lỗi, không tự là đóng góp mới.

## Phiên bản hẹp còn có thể đặc tả — giả thuyết, không novelty clearance

Một candidate để thầy góp ý: **sửa một cụm đối tượng với liên kết vùng–cụm được giám sát trực tiếp, dùng cùng vùng đã chọn để sinh replacement**.

Đường tính dự kiến:

1. C0 sinh draft. Bộ chọn dự đoán span/action/witness từ draft và proposals; có lựa chọn giữ nguyên khi không đủ bằng chứng.
2. Che span cần sửa. Replacement head chỉ đọc masked text state và feature của witness đã chọn; không nhận global visual hidden của C0 dưới tên text state. Không truyền classifier label để làm giả việc học từ pixels mà không ghi rõ.
3. Renderer copy tokens ngoài edit window đã định nghĩa; agreement/determiner cần scope trước. Một edit ban đầu; không nhận xử lý được relations/plurals/coreference.
4. Học joint edit–witness cùng replacement NLL từ nhãn được xác minh. Giám sát liên kết phải tới token generation, không chỉ auxiliary head. Prior pointer/grounded generation như NBT đã có; chuyển sang editor không tự đủ mới.

**Delta dự kiến so với TIger/DECap đã đọc:** thay multimodal bank fusion cho từ sửa bằng explicit selected-instance bottleneck, gắn nhãn witness với quyết định sửa. Đây là khác biệt cấu hình có thể đo, chưa phải kết luận chưa ai làm. Pointer, ROI restriction và supervised grounding đều kế thừa; cần lợi ích thực nghiệm của đúng ràng buộc này. Một câu sai có thể được sửa thành câu khác vẫn sai dù chọn đúng vùng; constraint không bảo đảm factuality.

Không thêm công thức contrastive/counterfactual rồi coi loss mới. Thử đổi witness chỉ là phép kiểm dependence; region khác không mặc nhiên là negative nếu cùng nghĩa/cùng class. Attention hoặc độ nhạy riêng không chứng minh grounding đúng.

## Feasibility và phép thử quyết định

- Chưa có xác nhận một tập native TRAIN drafts với spans lỗi, minimal corrections và positive witnesses đủ dùng. Các nhãn pair/VSR cũ không thay cho edit labels. Cache UPT có detections/HOI không cung cấp human truth cho mọi câu sửa.
- Không lấy khác reference, thiếu object trong reference hay detector miss làm nhãn sai/absent. Caption sửa có thể đúng ảnh nhưng không trùng reference; các trường hợp ấy cần annotation phù hợp.
- Nếu dùng benchmark editing public, kiểm tra split/image overlap trước: không đưa ảnh evaluation hiện có vào adaptation mới. Dữ liệu lỗi tổng hợp cần đánh giá riêng native C0 errors; clean drafts phải được giữ trong toàn cohort.
- Controls để bảo vệ đúng delta: cùng text/local head, nhãn, budget; so fusion cả bank với explicit witness, không chỉ C0 vs proposed. Oracle witness chỉ đo headroom, không làm kết quả inference. Giám sát thêm/capacity không được lẫn vào claim ràng buộc.
- Endpoint: CIDEr, CHAIR counts/ratios, recall/length; repaired/introduced errors và edit coverage; instance–replacement correctness trên panel có nhãn. Giảm CHAIR bằng xóa hoặc ít sửa không đủ đạt mục tiêu. Nếu witness không giúp so matched control, bỏ added-utility claim.
- Chưa stage dữ liệu, chưa đo throughput/cost; không ước lượng giờ GPU từ quota cũ. Không chạy model/notebook/Torch.

## Phạm vi tìm kiếm và khoảng trống

Workflow multi-source-search của nature-academic-search; academic MCP không có trong tool inventory, dùng web search tới nguồn arXiv/CVF/ECVA/ACL. Tìm theo các nhóm: image caption hallucination correction; region-grounded caption editing/replacement; TIger/DECap; edit+contrastive/counterfactual; region-conditioned/witness. Tìm cả prior cũ, không hạn chế Q-ranking cho novelty search. Các query có từ caption editing đôi lúc trả nhầm sửa ảnh/subtitles; đã loại khỏi bằng chứng.

CVPR2020 Show Edit and Tell PDF và Patch Matters2025 PDF trả403; chỉ thấy official landing/abstract, không dùng chúng để kết luận về bottleneck. Một số leads mới2026 (graph evidence repair/claim admission) không mở được full text/landing qua tool; không dùng snippet làm evidence về cơ chế hay ngày công bố. Chưa có systematic forward-citation review hoặc đảm bảo phủ tới2026. Không có exact-collision clearance.

Nếu tiếp tục, ưu tiên full texts sát phần còn lại: Patch Matters2025 và các region-grounded caption repair methods; đọc Show Edit and Tell2020 để khóa baseline lineage. Không cần user gửi lại TIger/DECap/LURE/Woodpecker vì method đã truy cập được trong lượt này.

## Khuyến nghị trình thầy

“Em đề xuất nghiên cứu sửa cục bộ lỗi đối tượng trong caption bằng liên kết trực tiếp giữa cụm thay thế và một vùng ảnh được chọn. Hướng này kế thừa explicit caption editing; phần em muốn kiểm là thay việc trộn toàn bộ vùng ảnh bằng một ràng buộc vùng–cụm có giám sát trong đường sinh từ. Hiện em đã đối chiếu TIger, DECap và các revisor, nhưng chưa xác nhận tính mới của ràng buộc này; em xin góp ý về phạm vi cải tiến trước khi triển khai.”

Đây là đề xuất có cơ chế để thảo luận, **không phải phương pháp mới đã được xác lập**. Khuyến nghị riêng của Codex1: giữ câu hỏi liên kết vùng–cụm, không duyệt một đợt huấn luyện editor lớn từ C hiện tại; còn phải chốt delta và nguồn edit labels. Deadline không làm thay đổi mức bằng chứng.
