# Chốt nghiên cứu sau expanded4000

**Amendment cùng ngày, sau quyết định này:** người dùng không chấp nhận thay mục tiêu bằng bài so sánh và yêu cầu tiếp tục tìm đóng góp căn chỉnh. Phần khuyến nghị scope empirical bên dưới là lịch sử đề xuất, chưa được duyệt. Giữ nguyên kết quả/gates và việc đóng nhánh C/G/C0; vòng brainstorm alignment mới được ghi tại `ALIGNMENT_RESTART_NOTES_2026-10-30.md`.

30/10/2026. Đọc AGENTS.md, RESEARCH_STATE và EXPANDED_RESULT.md; không đọc ZIP, không neural/scorer local. Audit, timing và binary identity theo nguồn root đã ghi; không lặp audit2261813bytes ZIP.

**Quyết định: đóng vòng C/G/C0 và evaluation này. Không training, tuning, seed sweep hoặc tăngn tiếp để cứu gate. Giữ C như ứng viên thực dụng có kết quả chất lượng tốt; giữ C0 làm đối chứng bắt buộc. Chưa đạt mục tiêu kép hoặc xác nhận lợi ích riêng của preservation.**

## 1. Kết luận khoa học

| 4000reused-val images | A | C0 | C |
|---|---:|---:|---:|
| CIDEr | 1.189771 | 1.204560 | 1.207042 |
| False captions | 239 | 235 | 230 |
| False mentions | 254 | 248 | 243 |
| Recall | .468087 | .471085 | .470296 |

- C−A CIDEr+.017271,95%pairedCI[+.008819,+.025853]: gain có bằng chứng trong evaluation này. C0−A cũng dương với CI không chứa0.
- C−A có ít hơn9false captions và11false mentions; đây là giảm numerator thực trên mẫu, không chỉ đổi mẫu số. Nhưng CHAIRs difference−.225pp,CI[−.600,+.150]pp; exactMcNemar p=.305677. Chưa xác nhận hallucination reduction.
- C−C0 có mean CIDEr+.002481 nhưng CI[−.001357,+.006263]; false captions giảm5/4000,CI vẫn chứa0,p=.1796875. Lợi ích riêng của preservation chưa thiết lập.
- Recall/CIDEr tradeoff guards qua, cả hai hallucination gates không qua. Theo protocol: preservation_relative=false,dualgoal=false,joint=false. Không đổi gate sau số liệu.

Không nói “không cải thiện gì”: quality gain có thật trong mẫu này; số lỗi thấp hơn cũng phải giữ trong bảng. Không nói “không có tác dụng” hay “tương đương”: uncertainty vẫn cho nhiều hiệu ứng có thể có. Không dùng CIDEr gain continuation để xác nhận semantic-alignment objective.

Chọn C làm working candidate là quyết định thực dụng theo các point estimates/chi phí inference bằng native A, không chứng nhận nó thắng C0 về thống kê. Giữ cả checkpoints, không xóa A/C0 hoặc đổi tên C0 thành novel method.

## 2. Vì sao dừng thay vì thử thêm?

C0 đã cho thấy gain vẫn có khi bỏ keep; expanded4000 không xác nhận added factuality của keep. G cũng chưa cho lợi ích riêng trước đó. Vì vậy chưa có một sửa tham số cụ thể nào được dữ liệu ưu tiên đủ để tiếp tục trước hạn.

Phân tích supervision noise/representation/optimization vẫn chưa xác định một nguyên nhân duy nhất. Không chọn “tănglambda”, “thêmnegatives” hoặc “thêmregion” như thể nguyên nhân đã biết. Thêm run chỉ để p nhỏ hơn sẽ đổi mục đích của evaluation đã khóa.

Các bước kỹ thuật còn lại: root hoàn tất artifact/registry/push, bảo toàn source/predictions/checkpoints. Không cần thêm neural diagnostic hoặc human audit để ra quyết định này. Không cộng rawCIDEr1000/4000; chưa cần pooled5000 để thay kết luận primary4000.

## 3. Phạm vi bài cụ thể để user duyệt

**Khuyến nghị riêng:** nếu ưu tiên có bản thảo31/10, trình user duyệt một bài thực nghiệm so sánh, thay vì hứa positive-method paper đã đạt mục tiêu gốc. Chưa viết toàn bài theo scope mới khi user chưa chốt.

Tên đề nghị trung tính:
**“Đánh giá bảo toàn xác suất khái niệm trong tinh chỉnh mô hình sinh mô tả ảnh”.**

Câu hỏi nghiên cứu: các mục tiêu semantic-class preservation/suppression có bổ sung ích lợi ngoài native SCST continuation với rolloutKL không?

Đóng góp có thể mô tả đúng:
1. Một cách hiện thực/kiểm nghiệm objective ở mức semantic-class event mass, kế thừa UL/KD và dùng nguồn weak image support; không nhận primitive mới hay đầu tiên.
2. Thiết kế ablation C0/C/G tách ảnh hưởng thêm concept objectives khỏi common continuation recipe, cùng cohort/parameters/budget.
3. Bằng chứng empirical: native continuation tăng CIDEr; concept constraints có point estimates tốt ở một số chỉ số nhưng chưa có added factuality được xác nhận theo gate.

Đây là **đề xuất/đánh giá phương pháp**, chưa là đóng góp mới đã chứng minh vượt baseline về cả hai mục tiêu. Không coi public code/metadata checks hoặc evaluation lớn tự là novelty.

Các bảng tối thiểu cho bản thảo:
- A/C0/C/G dev1000, đầy đủ cả negative correction G không đạt.
- A/C0/C remainder4000, per-cohort CIDEr IDF riêng.
- Paired effect/CI, discordants, exacttests và jointgateFAIL; không chỉ bảng point scores.
- Phạm vi training4000images/500updates, inference cost scope, weak-label/alias scope, one seed và lịch sử reusedvalidation.

Mở đầu kết quả có thể viết:
“Trên4000ảnh validation đã từng sử dụng, cấu hình C đạt CIDEr1.2070 so với1.1898 của A. Số caption có ảo giác theo CHAIR giảm239xuống230, nhưng khoảng tin cậy còn chứa0 và chưa xác nhận được hiệu quả riêng của positive preservation so với đối chứng không có thành phần này.”

Không viết “đã giảm hallucination đáng kể”, “chứng minh căn chỉnh đúng vùng”, “C0 là phương pháp semantic alignment mới” hoặc “không tốn chi phí toàn pipeline”. Không giấu C0/post-pilot selection/failedgates.

## 4. Nếu user giữ nguyên yêu cầu ban đầu

Nếu bắt buộc contribution căn chỉnh mới có added benefit và hallucination reduction đã chứng minh, phải báo rõ **chưa hoàn thành mục tiêu** và trao đổi lại kế hoạch/thời hạn. Tôi không đề nghị đổi deadline thành một cuộc đua chọn phương pháp mới chưa kiểm.

Đề tài/nội dung mới chỉ được mở sau khi user chốt; cần một hypothesis có lý do mới và đối chứng, không nối tiếp mọi failure bằng thêm một layer. Chưa chọn hoặc xin GPU cho một hướng mới trong quyết định này.

Nếu user đồng ý scope empirical, triển khai viết dựa trên kết quả đã khóa, relatedwork đã có và giới hạn trung thực; chưa có venue cụ thể nên không bảo đảm đủ tiêu chí hay accepted tại tạp chí trường.

**Đề nghị root nói với user:** “C có gain CIDEr và ít lỗi hơn trong mẫu, nhưng hai gate hallucination không qua nên chưa đạt mục tiêu kép. Tôi đề nghị dừng chạy và dùng kết quả này cho một bài so sánh objective có đối chứng, nếu bạn chấp nhận phạm vi ấy; nếu giữ yêu cầu phương pháp mới có hiệu quả đã chứng minh thì cần thêm một giai đoạn nghiên cứu, không hứa xong trước31/10.”
