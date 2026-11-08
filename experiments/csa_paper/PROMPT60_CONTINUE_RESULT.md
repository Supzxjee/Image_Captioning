# PROMPT học thêm epoch2 — nguồn Ngân69

Kiểm tra hoàn tất09/11/2026. User xác nhận SaveVersion69 của `ducanh3/notebook7bc83c4c76`. Đã hoàn tất14161update thêm /113287ảnh TRAIN, ORIGINAL/MID/END VALID5000, fixedrerank .3D+.4CLIP+.3O. Lỗi CPU-only trước đó không xuất hiện trong lượt này: runtime cu128/2GPU/NCCL và preflight đều PASS.

**Có tăng CIDEr và giảm nhẹ CHAIR sau rerank, nhưng recall giảm nhẹ.** Theo quy tắc đã chốt trước chạy, ORIGINAL là checkpoint duy nhất đủ điều kiện. Giữ source60 làm bản chính; bảo toàn END69 như ứng viên có đánh đổi, không gọi toàn bộ lượt này là không có cải thiện.

## Sau fixed rerank, cùng VALID5000

| Chỉ số | ORIGINAL60 | MID thêm7081update | END thêm14161update |
|---|---:|---:|---:|
| CIDEr ×100 |113,8004|114,2778|115,0628|
| BLEU4 ×100 |36,2831|36,4403|36,9083|
| ROUGE-L ×100 |56,5679|56,6843|56,8714|
| CHAIRs % |3,02|2,70|2,96|
| CHAIRi % |1,9274|1,7686|1,9197|
| Object recall % |44,5944|44,4061|44,3119|
| Caption bị CHAIR đánh dấu |151|135|148|
| Hallucinated mentions |154|140|151|
| Object mentions |7990|7916|7866|
| Recalled image–class pairs |7103|7073|7058|

END−ORIGINAL: CIDEr+1,26235điểm, CHAIRs−0,06điểm phần trăm (3caption), CHAIRi−0,00776điểm phần trăm, recall−0,28252điểm phần trăm (45cặp ảnh–lớp trên cùng15928cặp tham chiếu). MID giảm16caption bị đánh dấu nhưng mất30cặp đúng được nhắc; không đủ recall guard. Không tự chọn MID vì CHAIR thấp nhất hoặc END vì CIDEr cao nhất sau khi biết kết quả.

## Trước rerank: decoder rank0, không phải greedy

| Chỉ số | ORIGINAL60 | MID | END |
|---|---:|---:|---:|
| CIDEr ×100 |105,8365|105,3803|106,4984|
| CHAIRs % |4,06|3,80|4,04|
| CHAIRi % |2,8552|2,7557|2,9123|
| Recall % |41,4427|41,0660|41,0849|
| VALID teacher-forced NLL |2,423685|2,421547|2,419434|

Lưu ý END rank0 có CHAIRi cao hơn bản gốc; lợi ích CHAIRi nhỏ nêu trên là **sau reranking**, không áp dụng cho mọi cách giải mã. NLL giảm nhẹ không bảo đảm mọi chỉ số sinh câu cùng cải thiện. SCST hiện thưởng CIDEr; không có điều kiện bắt buộc recall không giảm trong loss. Đây là cách giải thích phù hợp với tradeoff, chưa xác định nguyên nhân từng lỗi.

## Kiểm tra, bất định và quyết định

Xem [audit](audit_prompt60_continue.py) và [JSON](PROMPT60_CONTINUE_RESULT.json) cho source pins, checkpoint/evaluation seals, toàn bộ15000lựa chọn rerank, ORIGINAL source60 replay và conditional paired bootstrap. Không chạy model hoặc official scorer local. Binary checkpoint không nằm trong reviewZIP nên SHA/restoreAdamRNG là runtime receipts và mã đã khóa, không phải tác giả tự load trọng số kiểm lại.

**15117 checks PASS.** END−ORIGINAL sau rerank,2000pairedbootstrap seed20261108: CIDEr+1,2623điểm, CI95[+0,2358;+2,2923]; CHAIRs−0,06điểm phần trăm, CI[−0,4005;+0,2600]; CHAIRi−0,00776điểm phần trăm, CI[−0,2338;+0,2048]; recall−0,2825điểm phần trăm, CI[−0,5576;0]. Tín hiệu tăngCIDEr rõ hơn tín hiệu giảmảo giác ở so sánh có điều kiện này. Net3caption ít bị đánh dấu là37cờ được sửa và34cờ phát sinh, không phải chỉ sửa3ảnh. MID net16 là49sửa/33phát sinh. Nhiều so sánh chưa điều chỉnh multiplicity; không diễn giải recall CI chạm0 như chứng minh không mất coverage.

Bootstrap chỉ resample ảnh trên một cặp checkpoint với corpusIDF cố định; không gồm biến thiên training seed/checkpoint selection và VALID đã tái dùng. Điều kiện operational recall>=original là quy tắc chọn đã khóa, khác với kiểm định ý nghĩa thống kê. Không dùng khoảng tin cậy sau chạy để tự nới điều kiện.

- Giữ bản chính theo `locked_selection.json`: ORIGINAL60, SHA `be2759b4a6f57af9bf8baeb13ee097bae2d8703d265b0791d9d39bb9c3dcbc63`.
- Giữ MID69 `mid.bin`: SHA `fd0708b7389034e24f3926d68fd7b7f715a5cc470c81d2f8af8edbfb7b504abc`.
- Giữ END69 `checkpoint.bin`: SHA `c4acaf39d4444919b5c19ceb5ba3182251b752ea4d1392649c4919c048c352c2`.
- `best.bin` của69 là bản gốc60, **không phải END mới**. Muốn nghiên cứu END phải lấy `checkpoint.bin` đúng SHA.
- Worker6111,73giây (~101,9phút), không gồm toàn bộ staging/CPUscoring/rerank/notebook setup, không phải quota tính phí.

Khuyến nghị không tự chạy thêm epoch hoặc TEST để cứu kết quả. END đáng giữ nếu người dùng muốn bàn về chấp nhận đánh đổi recall0,28điểm phần trăm, nhưng hiện chưa có quyết định đổi tiêu chí. Thí nghiệm này là continuation trong PROMPT, không tách riêng đóng góp nội dung prompt, không có kết quả TEST mới. Không sửa báo cáo Word.
