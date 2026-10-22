# COCO caption–instance grounding: audit khả thi, 22/10/2026

**PARTIAL_BOX_SOURCE_ONLY_INTERFACE_UNVERIFIED_NO_PILOT.** Audit dữ liệu/code, chưa triển khai loss, module, notebook hoặc chạy mô hình. Các số dưới đây là độ phủ dữ liệu cục bộ, không phải kết quả grounding hay caption.

## Giả thuyết đang xét

Khi caption nhắc danh từ thuộc lớp COCO, có thể dùng vùng annotation để giám sát đường đọc ảnh đang phục vụ sinh caption. Nếu caption không chỉ rõ instance, dùng tập/hợp các vùng cùng lớp; không tự chọn một box. Đây là căn chỉnh đối tượng, chưa có giám sát quan hệ. Nguồn giám sát grounding/attention loss là phần kế thừa; chưa có novelty hoặc gain được xác nhận.

## 1. Nguồn box

Tại các vị trí project, `C:/KhoaLuanTotNghiep` và Downloads đã rà, chỉ tìm được bản trích `instances_vcoco_trainval2014.json`: **5.400 ảnh, 47.423 annotation, 80 lớp**. Có `image_id`, category, bbox xywh, iscrowd; bản trích không giữ segmentation. Provenance ghi nguồn official COCO `annotations/instances_train2014.json`, subset ảnh V-COCO; hash file trích khớp. Tên file có trainval không có nghĩa bao phủ full train+val COCO.

Chưa tìm thấy `instances_train2014.json`, `instances_val2014.json` đầy đủ hoặc ZIP annotations ở các vị trí đã tìm. Không coi thiếu file là bằng chứng phương pháp không khả thi. Chưa tải annotation mới, chưa thay box GT bằng YOLO/UPT.

Source CHAIR cho thấy cache lớp đối tượng `imid_to_objects` dùng đánh giá, không cung cấp nguồn bbox đầy đủ cho audit này. Không unpickle cache để đoán box.

## 2. Độ phủ chính xác trên train10k

Giữ nguyên train10k từ artifact A. Join qua Karpathy imgid → COCO ID → filename, kiểm train/restval và dataset SHA. Các caption dùng đúng năm câu đầu mỗi ảnh như recipe A.

| Hạng mục | Số |
|---|---:|
| Ảnh train | 10.000 |
| Caption | 50.000 |
| Ảnh có trong nguồn annotation cục bộ | 508 (5,08%) |
| Ảnh chưa có nguồn annotation | 9.492 |
| Span ứng viên từ bảng alias có sẵn | 65.236 |
| Span ở ảnh chưa có nguồn annotation | 61.165 |
| Span nối được với box cùng lớp, non-crowd | 3.932 |
| Span bị loại do alias dễ mơ hồ | 249 |
| Span bị loại do UNK | 1 |
| Span sau bộ lọc quy tắc | 3.682 |
| Ảnh có ít nhất một span sau bộ lọc | 501 |
| Span sau lọc có nhiều instance cùng lớp | 1.783 |
| Span sau lọc không chứa tâm patch nào trong union | 134 |
| Span sau lọc có union nhỏ hơn diện tích một patch | 190 |

Hai nhóm vùng nhỏ ở cuối **có thể chồng nhau**, không cộng lại. Diện tích union tính chính xác trên các bbox chữ nhật đã clip theo kích thước ảnh; không gọi là segmentation mask. Không dùng nearest-patch fallback để che mất vùng quá nhỏ.

Các span không nối được box là thiếu/không phù hợp supervision, **không phải negative label**. Nguồn V-COCO thiên về người: person chiếm **2.089/3.682** span sau lọc. Không ngoại suy coverage hay độ chính xác này sang full COCO.

## 3. Token span và 12 kiểm tra thủ công

Baseline tokenizer dùng `lower().split()`, giữ dấu câu trong token, có BOS/EOS, max length 30. Audit chỉ dùng normalization bỏ dấu câu biên để match alias nhưng giữ nguyên raw whitespace positions và token IDs. Không dùng chỉ số sau khi regex tách lại câu. Caption `phone.` có token riêng với dấu chấm; target position = word index + 1, decoder logit position = word index.

Vocabulary được tái dựng tĩnh bằng cùng Counter/training texts, chưa chạy tokenizer torch hoặc kiểm vocabulary bên trong checkpoint. Không tuyên bố runtime parity. Bộ lọc loại các surface mơ hồ như train/orange/remote/table; chưa có POS/context disambiguation. Các trường hợp như pizza box, nghĩa bóng hoặc danh từ trong hình quảng cáo cần chính sách rõ hơn trước khi gọi là gold noun grounding.

Chọn trước 12 span đầu theo hash cố định, rồi đọc kiểm tra raw caption/span, ID, kích thước ảnh, box và vị trí dự đoán. **Đây là kiểm tra cơ học bản ghi, chưa xem ảnh để xác minh toàn bộ nghĩa caption/box.**

| COCO ID | Span | Số box cùng lớp | Tâm patch trong union / 196 | Nhận xét |
|---|---|---:|---:|---|
| 64627 | man | 3 | 118 | Đúng lớp person; không chứng nhận một man là cả ba instance |
| 529017 | people | 13 | 6 | Nhiều người nhỏ; area-overlap khác center mask |
| 468966 | man | 1 | 45 | Chỉ số whitespace và BOS offset khớp |
| 580507 | surfboard | 1 | 15 | Span đa loại ngoài person; chỉ số khớp |
| 395304 | man | 1 | 6 | Region tương đối nhỏ |
| 230315 | man | 1 | 18 | Caption có typo ridding; không tự sửa token |
| 54928 | man | 1 | 49 | Box chạm gần biên ảnh; clip/normalize hợp lệ |
| 455667 | people | 13 | 79 | Câu tách policemen và people watching; union person còn gồm riders, không phải target referent chính xác |
| 179031 | people | 13 | 15 | Caption a couple không làm annotation giảm tự động còn hai box |
| 317349 | phone. | 2 | 2 | Dấu chấm giữ nguyên token; tổng union ~0,918 diện tích patch; câu là advertisement |
| 126255 | kids | 11 | 153 | Union rất rộng, tín hiệu vị trí yếu; không gọi 11 box đều là kids |
| 568155 | Person | 1 | 154 | Box lớn, dù một instance vẫn ràng buộc yếu |

Không thấy lỗi join/offset cơ học trong 12 mẫu đã đọc. **Không đủ kết luận độ chính xác ngữ nghĩa của parser hoặc grounding**, nhất là sample person chiếm đa số. Các assertions riêng kiểm longest multiword hot dog không bị dog đè, dấu câu, union overlap không đếm đôi và vùng nhỏ không có tâm patch.

## 4. Interface attention

Xem `ATTENTION_INTERFACE.md`. Source hiện tại đọc 197 CLIP tokens gồm CLS → 32 Q ở hai tầng Q-Former; decoder đọc cả **`[legacy memory; Q]`** qua ba tầng. Attention weights chưa được trả về và chưa kiểm runtime gradient/parity.

Một tích hai ma trận cuối tầng bỏ qua legacy branch, CLS, residual và trộn query. Chuẩn hóa lại chỉ trên Q hoặc chỉ spatial patches còn có thể che khối lượng ở các đường bị bỏ qua. Không gọi nó là truy nguyên nguồn thị giác đầy đủ. Có thể chỉ là routing regularizer nếu định nghĩa/đối chứng rõ.

## Kết luận và giới hạn thực hiện

**Chưa đủ điều kiện chạy pilot.** Cần giải quyết nguồn annotation train+val đầy đủ, chính sách span/class union và định nghĩa routing/normalization. Chưa thấy nhu cầu đổi kiến trúc lớn chỉ để lấy attention, nhưng chưa chứng minh instrumentation giữ baseline hoặc có gradient đúng. Không thêm grounding head độc lập để thay giả thuyết.

Đã báo summary cho Astra; không yêu cầu đọc lại artifacts. Thiếu nguồn box là thiếu tài nguyên có thể bổ sung, không phải bằng chứng route bị bác bỏ. Không chạy pilot chỉ với 501 ảnh thiên lệch rồi gọi là toàn train10k.

## Quyết định Astra sau audit

Astra chọn **kiểm chứng interface attention/gradient trước**, chưa tải thêm annotation hoặc chạy pilot. Định nghĩa phiên bản đầu:

- D là attention decoder tầng cuối, trung bình theo head, trên toàn memory `[L;Q]`; lấy Q slice **không chuẩn hóa lại**.
- C là attention Q-Former cross-attention tầng cuối, trung bình theo head, trên cả 197 CLS+patch tokens; lấy 196 patch slice **không chuẩn hóa lại**.
- `a[t,p] = sum_q D[t,q] * C[q,p]`; support là tổng `a[t,p] * bbox_patch_area_fraction[p]`; `L_route = -log(epsilon + support)`.

Định nghĩa này phạt cả việc đọc legacy/CLS thay cho nhánh patch được chọn; không chỉ dồn attention vào box. Đây là can thiệp routing có thể làm hại baseline, không phải phép đo đầy đủ hoặc nhân quả về grounding. Root đồng ý dùng định nghĩa tường minh này cho kiểm tra số học, **chưa đồng ý coi nó có lợi cho caption**.

Bước kế tiếp được đề xuất là một smoke notebook do user chạy Kaggle, **không optimizer update**, kiểm logits/greedy parity với tiêu chí khóa trước, attention gradients đến decoder và Q-Former, CLIP frozen, target-token offset, padding/causal mask, empty/full patch mask và ba loại mass riêng (decoder→Q, Q→patch, bbox). Không thêm head/tham số, không tự nới tolerance khi fail. Audit này chưa tạo hoặc chạy notebook đó.

Nếu smoke đạt mới bổ sung full annotation rồi làm sạch span có mục tiêu; nếu fail phải sửa implementation trước. Các vấn đề pizza box và union person quá rộng vẫn còn mở, không được smoke số học hợp thức hóa thành nhãn grounding sạch.

## Tái tạo

`audit.py` đọc JSON và source có sẵn; dùng NumPy, không import torch. Đường input và hashes được ghi trong `DATA_AUDIT.json`. Script tạo số liệu cùng 12 bản ghi có đủ raw token, annotation IDs, bbox và geometry. Không tải ảnh/annotation, không đọc trọng số mô hình.
