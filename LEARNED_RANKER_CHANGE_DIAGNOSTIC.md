# Vì sao regional mất CIDEr: phân rã165 caption thay đổi

**Ba caption được sửa hallucination không gây ra mức giảm CIDEr tổng.** Chúng đóng góp dương0.000652283CIDEr trên512 ảnh. Phần mất điểm nằm ở các thay đổi khác, nên chưa có cơ sở xem đây là đánh đổi bắt buộc giữa CIDEr và giảm hallucination.

Phân tích cùng predictions/reference của pilot, không train hoặc chọn lại checkpoint/ngưỡng:

| Nhóm đổi caption theo CHAIR | Ảnh | Đóng góp vào chênh CIDEr trên toàn512 ảnh |
|---|---:|---:|
| Cả fixed và regional không có object hallucination | 159 | −0.013518629 |
| Fixed hallucinated → regional không hallucinated | 3 | +0.000652283 |
| Cả hai còn hallucinated | 3 | +0.000271482 |
| Tổng | 165 | **−0.012594864** |

Trong165 câu đổi: **76 tăng CIDEr,77 giảm,12 bằng điểm**. Tổng đóng góp dương+0.053637038 bị phần âm−0.066231902 lấn át.12 câu bằng điểm có cùng văn bản sau PTB tokenization, nên thay đổi raw string không phải cải thiện nội dung.

Các mô tả khác, có giao nhau và không được cộng chéo với bảng trên:

- 131 trường hợp giữ cùng tập lớp đối tượng được parser nhận ra;34 trường hợp đổi tập đối tượng.
- 150 giữ cùng tập claim của parser6 hành động;15 khác.15 khác đóng góp−0.013260648, nhưng parser nhạy với trật tự câu và bỏ sót cấu trúc: **không kết luận15 trường hợp thực sự đổi/sai quan hệ**.
- Câu ngắn hơn74 trường hợp đóng góp+0.000467951;44 câu dài hơn đóng góp−0.002364723;47 câu cùng độ dài đóng góp−0.010698091. Chưa có cơ sở quy mức giảm chủ yếu cho việc rút ngắn caption.

Ví dụ một cặp cùng nhắc người/lướt sóng/surfboard: `a man riding a wave on top of a surfboard.` → `a man on a surfboard riding a wave.` vẫn giảm1.496267CIDEr tính riêng ảnh. Đây là nhạy với cách khớp reference; không phải bằng chứng câu sau sai ảnh. Không dùng ảnh này hoặc các case đã xem làm training labels mới.

## Hướng sửa có cơ sở hơn

Thử nghiệm kế tiếp nên học **lợi ích dự kiến so với lựa chọn fixed** và khả năng abstain/giữ nguyên khi bằng chứng đổi câu chưa đủ, thay vì chỉ tối ưu mọi so sánh cặp trong beam. Mục tiêu là giảm các thay đổi có hại, không ép câu phải thay đổi. Cách chọn ngưỡng/epoch phải dùng các nhóm train/tune/calibration tách riêng; không dò ngưỡng theo512 ảnh vừa phân tích. Đây là hướng thiết kế, **chưa phải thuật toán mới đã được train/đánh giá**.

CHAIR clean chỉ có nghĩa không phát hiện object hallucination theo evaluator; không bảo đảm đúng hành động, quan hệ, thuộc tính hay cả câu. Phân tích referenceCIDEr không thay thế đánh giá bằng chứng thị giác.512 ảnh là development validation đã xem lặp lại; không phải test độc lập.

## Notebook và dữ liệu

Người dùng xác nhận tên notebook Kaggle hiện dùng: **QFormer UPT Val5000 20261015**. Tên giữ từ lượt trước; phương pháp/split của lượt mới phải xác định từ `protocol.json` và `run_summary.json`, không suy từ tiêu đề. Thư mục pilot: `/kaggle/working/learned_caption_ranker_1024_512`. Không suy đoán mount path hay version notebook từ tên hiển thị. Người dùng tự chạy notebook; trợ lý chỉ phân tích local trong bước này.

[Script](analyze_ranker_caption_changes.py), [summary](audit/learned_ranker_change_diagnostic.json). Source SHA256 `b26b36118a1e9a2b57ceb99f6f07ab7e4f1d96ac50151d97903c5edbb80af162`. CIDEr tính lại cùng corpus512 khớp cả hai nhánh trong1e-10; tổng chênh từng ảnh khớp chênh corpus. Case details giữ local tại `C:/KhoaLuanTotNghiep/Code/output/ranker_caption_change_diagnostic/changed_cases.json`.
