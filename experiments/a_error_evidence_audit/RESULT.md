# A: rà soát lỗi và bằng chứng đã có — 2026-10-27

**Đã ghép caption theo COCO ID, xem30 ảnh và lập20 trường hợp trọng tâm. Chưa tìm
được bằng chứng đủ để quy nguyên nhân cho riêng encoder, alignment hay decoder.**
Không có inference, training, metric run hoặc notebook mới.

## Tránh gọi tất cả ứng viên là lỗi

Mẫu30 được chọn có chủ đích, trước khi xem ảnh:10 CHAIR-positive có coverage
nhiều nhánh,6 câu có holding-and không bị CHAIR flag,10 câu unflagged có CIDEr
thấp/thiếu object mentions trong dev1000,4 trường hợp tương tự ngoài dev1000.
Trong từng nhóm, lấy theo hash COCO ID; không chọn theo nhánh nào thắng A.
Đây không là mẫu đại diện hoặc một benchmark mới.

Sau khi xem30 ảnh,20 focus gồm:

| Nhận định của trợ lý | Số trường hợp |
|---|---:|
| Lỗi khá rõ theo ảnh | 9 |
| Claim chưa có bằng chứng ảnh đủ chắc | 4 |
| Caption có cơ sở nhưng ít cụ thể | 6 |
| Chưa phân biệt chắc chi tiết | 1 |

10 trường hợp còn lại gồm5 caption có cơ sở và5 trường hợp chưa chắc. Toàn bộ
30 nhận định được giữ trong visual_notes.json, không giấu đối chứng bị loại khỏi
focus. **Chưa được người dùng xác nhận; không dùng làm nhãn train.** Những đánh giá
fine-grained, nhận định traffic density và mức thiếu chi tiết còn chủ quan.

## Các trường hợp có ích nhất

| COCO ID | A và quan sát | Nhánh cũ cho biết gì |
|---|---|---|
| 33924 | `holding a toothbrush and a cat`; tay cầm bàn chải, mèo nằm trên chăn. Danh từ có mặt nhưng phạm vi holding sai. CHAIR không flag. | H/L và noun G/S giữ nguyên. Không có trace dev1000 cho các nhánh khác ở ảnh này. |
| 167271 | `talking on a phone`; ảnh có thiết bị gập/màn hình cầm tay, references đa số Nintendo DS, một câu tablet. | H/VG G/GV/stale đổi talking→using nhưng vẫn gọi phone. Chưa sửa được việc nhận dạng vật. |
| 196594 | Ba đồng hồ và ba biển; A nói `a clock ... a sign`. Không sai về tồn tại nhưng mất cấu trúc nổi bật. | Cả10 output đã lưu giữ nguyên A. Không suy từ đó rằng chi tiết vắng trong CLIP hay decoder hoàn toàn không dùng ảnh. |
| 466945 | A chọn người đi cạnh biển, bỏ đồng hồ lớn ở tiền cảnh. A vẫn có thể đúng. | Noun G/S thêm đúng `clock on a pole`. Một ví dụ output có thể khai thác danh từ đúng, không là chứng minh head tốt hơn toàn tập. |
| 261535 | Racket ở một tay, một vật vàng ở tay kia; nhiều vật vàng trên sàn. A dùng holding với balls số nhiều. | Noun G/S bỏ balls. VG G cũng bỏ nhưng thêm heels dù chân trần. Không gọi mọi thay đổi là cải thiện. |
| 261068 | Tay chạm con vật, đồng thời cầm cào: petting có thể đúng, nhưng bỏ phần hành động với cào. | Noun G/S đổi sang dog on a leash; ảnh không hỗ trợ leash. Nhận đúng danh từ không đảm bảo quan hệ đúng. |
| 185901 | Lemons được gọi oranges; bốn reference nói lemons, một câu nói fruit. | COCO không có lớp lemon riêng. Một head theo lớp COCO không tự giải quyết đầy đủ loại lỗi này. |
| 304036 | Helicopters được gọi planes. | Cần phân biệt cách gọi cụ thể với việc nhận ra cảnh có phương tiện bay; không chỉ object presence. |

Các case32391/447237 được giữ là **under-specified**, không chốt caption A sai:
ván/bảng dáng surfboard có trong cảnh cửa tiệm; references của ảnh chèo nước bất
đồng boat/canoe/board. Case371564/140391/100811/230982/220732 là đối chứng cho
việc CIDEr thấp hoặc thiếu chi tiết không tự là lỗi semantic.

## Ghép nguồn đã kiểm

Đã dùng9 cached review, không mở lại ZIP lớn. A predictions và mapping image_id↔
COCO ID khớp khi nối các nguồn; references từ dataset có SHA đã khóa. Native H/L,
noun G/S có5000 ảnh; VG G/T/GV và stale C/T có1000 ảnh overlap. Không so giá trị
CIDEr/image giữa hai IDF dev1000 và val5000.

Trong20 focus, native H/L mỗi nhánh giữ nguyên15 câu; noun G giữ15 và S giữ16.
VG G/T/GV có14 case coverage, thay đổi lần lượt9/8/7 câu. Đây là **số câu đổi**,
không phải số sửa đúng. Không phải controlled10-way experiment: các nhánh khác
nhau về lịch huấn luyện, nguồn phụ và mục tiêu.

## Bằng chứng nào thực sự có, bằng chứng nào còn thiếu

| Nguồn | Có thể kiểm tra hiện tại | Không được suy ra |
|---|---|---|
| Ảnh gốc |30 ảnh; SHA và URL COCO được lưu; nhìn contact sheets và4 ảnh riêng | Người nhìn thấy cue không chứng minh model encode được cue |
| Preview224 | Bản PIL bilinear để người xem | Không phải tensor đầu vào/cache đã kiểm parity, không phải phép thử năng lực model |
| Caption/references | Cùng COCO ID, saved outputs của các nhánh,5 refs | References không đầy đủ; không nhắc không có nghĩa là false |
| CLIP/Q-Former | Saved protocols xác nhận cache source/cohort | Không có vectors/Q memory của từng case trong ZIP để kiểm thông tin bị mất ở đâu |
| Prompt gốc | Saved protocols xác nhận embedding source/cohort | Exact original text không được khôi phục; không dựng lại từ refs hay dùng prompt UPT thay thế |
| UPT | Valid-pair count trên14 case overlap có trace | Count không là đúng/sai quan hệ; không có hidden vectors/native role scores per case |
| Các nhánh vùng | Saved caption của VG/native detail | VG auxiliary regions không là oracle region input khi caption inference |

## Nút thắt đang được nghi ngờ — chưa được chứng minh

1. **Phạm vi và vai trò trong câu:** có object đúng nhưng tổ hợp sai. Ví dụ mèo+
   bàn chải chỉ ra hạn chế của object-presence reward. Không đủ để quay lại thiết
   kế ride=0→English-negative đã sai.
2. **Mức cụ thể và lựa chọn nội dung:** chi tiết lớn như ba đồng hồ hoặc đồng hồ
   tiền cảnh chưa được nhắc; noun-head đôi khi thêm được vật đúng nhưng cũng thêm
   quan hệ không có bằng chứng. Chưa phân biệt mất thông tin với chọn nội dung.
3. **Định danh ngoài taxonomy thô:** lemons/handheld console/helicopters không
   được giải quyết chỉ bằng tăng object recall trong80 lớp COCO. Chưa biết lỗi
   bắt đầu ở resolution/visual representation/prompt hay language prior.

Không tuyên bố đã tìm ra một causal bottleneck. Cần một phép kiểm có can thiệp
vào bằng chứng với các phần còn lại cố định để tách các giả thuyết; các saved
captions hiện tại chưa thay thế được phép kiểm đó. Chưa chọn kiến trúc/loss mới.

## Bàn giao

`review/index.html` hiển thị20 focus, ảnh gốc + preview224, notes, tất cả caption
nhánh có sẵn và references. `review/controls.html` giữ10 đối chứng/chưa chắc.
Không yêu cầu người dùng chấm thêm để cứu gate hoặc lấy điểm tốt hơn.

Đã gửi Astra tóm tắt bằng chứng, các phản ví dụ và khoảng trống dữ liệu, yêu cầu
quyết định nút thắt đáng theo/phép kiểm tối thiểu; không buộc một hướng module.
ARGON_BRAINSTORM.md là tài liệu độc lập để người dùng gửi Gemini Argon trên tài
khoản của mình. Không tự nhắn dịch vụ bên ngoài. Kết quả brainstorm không được
coi là novelty đã xác nhận hoặc quyền tự launch GPU.

## Chốt với Astra sau phản biện

Astra ban đầu ưu tiên phạm vi khẳng định như case33924 và đề nghị4–6 cặp ảnh thật,
giữ cùng caption/prompt rồi so chênh likelihood frozen A. Tôi đối chiếu với phrase
CE/CROSS, pair-prefix và F diagnostic đã có: sensitivity/teacher-forced preference
không bảo đảm free generation và vẫn chưa chọn được can thiệp vào encoder/memory/
decoder. Do chưa có hai hành động nghiên cứu khác nhau phụ thuộc kết quả, Astra
**rút đề nghị chuẩn bị/chạy bộ cặp mới ngay**.

Chốt: đóng atlas, gửi tổng hợp cho Argon; chưa có method mới được xác lập. Vòng đề
xuất tiếp phải chỉ ra cơ chế cụ thể, khác biệt với các nhánh cũ và kết quả nào sẽ
làm chọn/bỏ cơ chế đó. Không thêm manual-label/GPU audit chỉ vì phép đo khác về kỹ
thuật. Nếu một ứng viên sau này cần image×prompt probe, phải báo riêng từng prompt
gốc; không trung bình che các dấu đối nghịch hoặc mặc định neutral là cùng chế độ A.

Phản hồi của Astra dựa trên summary tôi gửi, không là visual review độc lập của
30 ảnh hoặc xác nhận gold cho9 nhận định lỗi rõ.
