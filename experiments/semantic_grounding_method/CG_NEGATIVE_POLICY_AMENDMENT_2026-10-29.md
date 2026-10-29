# C/G: sửa chính sách weak-negative trước training

29/10/2026. Research decision trước khi ship notebook hoặc chạy GPU. User đã xác nhận không còn notebook/code tạo detector cache. Root có pinned COCO TRAIN10k annotations, refs và detector JSON; image coverage/max87objects không xác định được upstream filtering.

**Đồng ý dùng protocol sửa đổi: COCO GT+refs định nghĩa weak-negative; detector là positive-hint veto một chiều. Không cần chạy extractor mới chỉ để tiếp tục dưới định nghĩa này.** Không gọi policy này là giữ nguyên strict detector-completeness policy trước đây.

## Vì sao đổi quyết định block trước đó?

Guard cũ yêu cầu biết detector không bỏ detections trên ngưỡng vì đã dùng detector absence như một phần negative evidence. Với nguồn detector không khôi phục được, guard đó đúng là chặn G.

Protocol mới thay định nghĩa đích học: “không được nguồn annotation/reference ghi nhận” là nhãn yếu từ COCO, chưa là “chắc chắn không có trong ảnh”. Detector absence không tham gia kết luận ấy. Một hint được ghi nhận có thể phủ quyết negative; hint không được ghi nhận không cung cấp thêm bằng chứng âm.

Đây là đổi supervision assumption minh bạch trước kết quả, không suy ngược unknown completeness thành complete và không giảm threshold để đủ mẫu. Kế hoạch implementation C/G có thể tiếp tục; không yêu cầu user xác nhận lại nguồn họ đã nói không còn.

## Chính sách chính thức

Với ảnh TRAIN i:

- \(P_i\): union của mọi COCO annotated class trên ảnh và mọi class-hint từ toàn bộ5refs. “Đủ annotation” nghĩa là giữ đủ bản ghi của nguồn COCO đã pin, không phải annotator nhìn thấy mọi vật thật.
- \(V_i\): mọi class-hint parseable/đúng mapping được ghi trong detector record; không suy diễn các class còn lại. Không dùng scores để tuyên bố negative hoặc bảo đảm completeness.
- \(E_i^+\): reference noun anchors có GT support, theo chính sách positive preservation đã chốt.

Class c được chọn cho negative loss khi caption sample thật sự nhắc c, ánh xạ noun/class không mơ hồ, dữ liệu GT/ref của ảnh hợp lệ, và:

\[
c\notin P_i,\qquad c\notin V_i.
\]

Ý nghĩa của hai điều kiện khác nhau: \(c\notin P_i\) tạo **annotation-conditioned weak-negative**; \(c\in V_i\) là lý do veto. \(c\notin V_i\) chỉ là “không thấy veto được ghi”, không là bằng chứng detector đã kiểm và bác bỏ c.

- Có hint detector nhưng không có GT/ref support: unknown cho negative loss; không tự tạo positive anchor.
- Record detector hiện diện nhưng thiếu class c hoặc là output rỗng: không cung cấp negative evidence; weak label vẫn phải đến từ GT/ref. Đây là điểm khác rõ với policy completeness cũ.
- Missing/corrupt/unmappable detector record: giữ unknown cho negative selection theo conservative data-validity guard; không fill thành một danh sách negative. Với full image-key coverage đã có, guard này chủ yếu xử lý bản ghi lỗi.
- Missing GT hoặc refs không đủ/không xác định: unknown, không coi empty set là nhãn vắng.
- Objects ngoài COCO80 và cách dùng lexical mơ hồ: unknown.

**Veto dùng bất kỳ hint hợp lệ đang có**, không còn cần chứng nhận “đã xuất đủ score≥.5”. Không tự gán confidence1 cho labels thiếu score. Nếu giữ filter score cho veto vì lý do khác thì đó phải là quyết định riêng được khóa; phương án hiện tại không cần filter ấy.

## Những điểm bắt buộc khi package

1. Tính GT presence từ toàn bộ annotation subset TRAIN, gồm các class của crowd/small boxes và những annotations bị loại khỏi routing-span bank. Không dùng17,904retained spans như danh sách đầy đủ vật có mặt.
2. Dùng đủ5refs để veto âm; mapping category IDs theo COCO80, không giả định IDs1..80 liên tục. Positive anchors và ambiguous spans tiếp tục theo masks riêng.
3. Pin SHA của inputs và version/hash policy mới. Ghi provenance detector unknown, vai trò positive-veto-only, negative source GT+refs, và rằng completeness không được xác minh. Không để một flag “full_detector_verified=true” chỉ vì JSON có đủ image IDs.
4. C/G cùng input membership, event selection policy, positive anchors và train schedule; chỉ G dùng L_neg như thiết kế. Không đổi optimizer, coefficients, alias probability hoặc sampler để cùng lúc thích nghi policy.
5. Xuất counts: GT/ref weak-negative candidates, số bị detector veto, số unknown/missing/ambiguous, usable events thực tế và class coverage. Counts là provenance/coverage, không ước lượng precision nhãn.
6. Nếu static eligibility rỗng thì chặn G trước train. Nếu G không nhận negative events thực tế thì không báo phép thử UL đã hoàn tất có hiệu lực; giữ zero-event handling số học đúng, không tự nới policy để tạo gradient.

## Giới hạn và diễn giải

GT/ref có thể bỏ sót vật thật; cache detector bị lọc có thể bỏ lỡ một veto lẽ ra hữu ích. Vậy false-negative supervision có thể nhiều hơn policy cũ giả định. Không biết hướng/độ lớn sai số từ provenance hiện có. Không nhận detector là một nguồn xác nhận độc lập.

Các bảo vệ vẫn là positive preservation, SCST CIDEr-object reward, unknown masks và đánh giá recall/CHAIR/CIDEr. Chúng không biến weak labels thành factual gold. Kết quả C/G chỉ kiểm objective dùng annotation-conditioned concept correction, chưa chứng minh phát hiện vật vắng về mặt thị giác.

Trong báo cáo, so sánh C/G và A dưới policy đã sửa; không trộn hai phiên bản rồi giữ nhãn “strict verified negatives”. Đây vẫn là cùng ứng viên semantic-class probability correction, nhưng target-label construction đã thay đổi và phải nêu rõ.

## Thông báo ngắn cho user trước run

“Do không còn nguồn tạo detector cache, chúng tôi giữ provenance của nó là chưa xác minh. Nhãn âm yếu nay dựa trên COCO TRAIN annotations và references; detector chỉ veto khi có hint đối tượng, không dùng việc detector không nhắc tới làm bằng chứng vật vắng. Quy tắc này được khóa trước training và còn rủi ro annotation bỏ sót.”

**Chốt: tiếp tục package native A/alias union và policy sửa đổi này. Không cần blocker provenance detector dưới định nghĩa mới; các checks dữ liệu/coverage và giới hạn nhãn yếu vẫn giữ. Chưa chạy GPU tại Astra.**
