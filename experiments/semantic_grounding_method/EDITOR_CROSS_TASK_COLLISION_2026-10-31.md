# Đối chiếu xuyên bài toán: collision của hướng local editor

31/10/2026. User yêu cầu tìm theo cơ chế thay vì chỉ caption editing. Codex1 dùng workflow tìm tài liệu sơ cấp; không model/code training. Đây là cập nhật bằng chứng, không quyết định triển khai thay user.

## Kết luận mới

**Không dùng selected-region generation/refinement làm claim mới độc lập.** Đã tìm thấy prior gần hơn ngoài caption editing: masked word learning có grounding, region-guided generation và region-conditioned report refinement. Không chứng minh exact pipeline KEEP/REPLACE–span–witness của ta trùng hoàn toàn; cũng không có căn cứ rằng thêm local-copy biến các cơ chế đã biết thành đóng góp đủ vững.

Đây là thay đổi khuyến nghị do đọc thêm nguồn sát hơn, không phải do có kết quả GPU mới. §8 của pipeline draft chỉ xác định khác biệt với TIger; chưa đủ định vị với các prior dưới đây. Tạm rút việc đề cử riêng bottleneck làm phần mới, giữ bản thiết kế để tham khảo/kế thừa.

## Bằng chứng bổ sung, mức đọc và liên kết

### 1. GRD-DCA — prior sát nhất về region-conditioned refinement

[Trang công bố MICCAI2026 Satellite/MI4MedFM](https://papers.miccai.org/miccai-2026-sat/MI4MedFM_002.html), [PDF](https://papers.miccai.org/miccai-2026-sat/paper/MI4MedFM_002.pdf). Đây là satellite workshop, không gọi main MICCAI. Đã tải PDF chính thức, đọc method §2.1–2.2/p3–5 và xem render Fig.2/p4.

Bài dự đoán box ở VLT-1, mask visual features rồi dùng vùng ấy cùng generated text/region representation trong VLT-2 để refinement. Trùng đáng kể lõi “predicted where → localized features → refined what”. Method đọc chưa nêu bounded noun-span KEEP/REPLACE và exact copy ngoài edit window. Khác domain và local-copy chưa đủ để coi cơ chế của ta mới; chưa kết luận exact collision toàn bộ.

PDF SHA256 `d375e49d90028eea5d577a8b816edffb3f0f6cd9d40cc3d4eaec52dbb70189db`,12pages. Bản PDF/text/render ở `review/grd_dca.*` bị git-ignore, không commit paper. Web direct-open lỗi cache miss; HTTP GET máy chủ MICCAI trả200, application/pdf. Đọc bằng pypdf; render bằng pypdfium2 (fitz không cài). Không có full code review hoặc independent result verification; không suy bypass đã được loại hết từ sơ đồ.

### 2. World-to-Words — ACL2023

[Official paper](https://aclanthology.org/2023.acl-long.31.pdf). Đã đọc task §2.1 và model §3.1/Fig.4. Input incomplete caption, dự đoán từ bị che cùng object localization; object representations đi vào text decoder; word–region alignment có positional/semantic supervision. Vì vậy “điền từ có giám sát vùng” đã có. Decoder đọc object representations, chưa thấy bắt buộc một selected witness duy nhất cho mỗi replacement trong phần đọc. Không có native draft edit detection/KEEP ở task đã kiểm. Không gọi grounding của họ chỉ là auxiliary ngoài language path.

### 3. RGRG — CVPR2023

[Official PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Tanida_Interactive_and_Explainable_Region-Guided_Radiology_Report_Generation_CVPR_2023_paper.pdf). Search tool trả trích đoạn primary §3.3–3.4/Eq.2–3 và Fig.2; direct PDF open403, chưa tải độc lập toàn văn. Có chọn anatomical regions và sinh câu từ associated region feature bằng pseudo self-attention. Chặn claim mới của “chọn vùng rồi sinh câu bằng feature vùng”. Không phải full caption correction pipeline; không lấy kết quả medical so trực tiếp CIDEr/CHAIR COCO của ta.

### 4. LaViDa / LaViDa-FIM — visually grounded text infilling

[Primary OpenReview PDF](https://openreview.net/pdf/75e11f13125217accdbe2ac0c664854d3040f492.pdf). Đã đọc §4.4, Fig.6 qualitative editing và architecture overview. Denoise phần text bị MASK với image/context; giới hạn text edit tự nó không mới. Chưa thấy edit–witness instance supervision/selected-region-only access trong các phần đọc. Không gán accepted venue theo URL hoặc demo; không dùng illustration làm bằng chứng factual editing endpoint.

GROUNDHOG CVPR2024 cũng là component prior: [author page](https://groundhog-mllm.github.io/) mô tả masked entity features, pointer input và region-conditioned response. Mới đọc project page, không dùng nó thay full-method collision proof.

## Bảng so cơ chế, không gộp các phần thành “một paper trùng toàn bộ”

| Tiêu chí | GRD-DCA | W2W | RGRG | LaViDa-FIM | Thiết kế của ta |
|---|---|---|---|---|---|
| Vai trò | Region-conditioned report refinement | Grounded masked-word learning | Region-guided report generation | Visual text infilling | Native-caption local correction |
| Vùng trước sinh/sửa | Predicted box/mask | Object embeddings vào text decoder | Selected anatomical feature | Chưa xác minh instance selector | Predicted witness |
| Grounding supervision | Box losses và alignment | OL và word–region alignment | Anatomical regions/sentences | Chưa xác minh per-edit witness labels | Gold edit–witness khi train |
| Local visual feature vào language computation | Có theo method | Có, không single-witness claim | Có theo primary excerpt | Có image tokens; không selected-only claim | Chỉ selected ROI, text-only context encoder |
| Fixed noun edit window/copy | Chưa thấy ở method | Không trong task đã đọc | Không trong task đã đọc | Masked text phần cần điền | Có theo draft spec |

“Chưa thấy” chỉ trong phần đã đọc, không phải xác nhận chắc chắn không có trong supplementary/code. Giới hạn một ROI thay nhiều ROI có thể là khác cấu hình; chưa có lý do/utility riêng đủ bảo vệ thành innovation. Không bỏ prior vì khác medical/natural images, khác tên hoặc chỉ là workshop.

## Phần còn lại và quyết định

- “Joint span–action–witness + giữ ngoài span + selected-region generation” vẫn chưa tìm được exact paper trong focused search này. Không dùng việc đó làm novelty clearance hoặc đếm ba khác biệt như ba đóng góp.
- Novelty risk hiện cao hơn trước. Hướng tổng quát có căn chỉnh thật, nhưng không đủ gọi mới riêng vì có đường dữ liệu vùng→từ.
- Nếu user/thầy muốn kế thừa, phải chọn một prior gần nhất và sửa một hạn chế kỹ thuật có căn cứ, dùng đối chứng cùng labels/budget; không chỉ chuyển report→caption hoặc thêm KEEP. Chưa phát hiện một hạn chế mới đã đo để biện hộ sửa đổi tiếp theo; không tự đẻ thêm loss/benchmark để cứu claim.
- Khuyến nghị Codex1: **không chốt C hiện tại là phương pháp mới và không viết notebook lớn từ claim bottleneck**. Có thể trình thầy đây là hướng kế thừa đang tìm cải tiến, kèm prior sát nhất và kết quả negative novelty review. Không kết luận mọi cải tiến local editor đều không thể có ích.

## Phạm vi tìm và điều chưa làm

Queries gồm caption editing+grounding/pointer/region selection/bounding box; visually grounded text editing; region-conditioned infilling; masked noun generation; report region-conditioned refinement. Có backward reference từ GRD-DCA đến RGRG. Tìm được primary ACL, CVF, MICCAI, OpenReview; loại image pixel editing, subtitles và sites tóm tắt khỏi method evidence. Academic MCP không có; dùng web + direct primary PDF retrieval khi web lỗi. Chưa systematic citation-forward/search mọi bài đến2026, không chứng minh absence.

Các file model/checkpoint/dataset không tải/chạy; không stage dữ liệu, không GPU, không mở lại ZIP. Chỉ metadata/document extraction/render và cập nhật context. Không cần sheet cho vòng literature review.
