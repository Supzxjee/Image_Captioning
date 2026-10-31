# MED Changed/Stable — kết quả kiểm tra dữ liệu nhỏ

31/10/2026. [Protocol](SEMANTIC_RESPONSE_FEASIBILITY_PROTOCOL.md), [20 annotations](SEMANTIC_RESPONSE_FEASIBILITY_REVIEW.json). Đây là assistant visual review, **0 nhãn được người dùng xác nhận**, chưa huấn luyện hoặc đánh giá mô hình.

## Kết quả

Lấy20entry đầu tiên của archive khớp metadata MED SFT, không chọn theo chất lượng ảnh. Tải đủ20ảnh sửa và17ảnh gốc VG;3ảnh gốc DOCCI chưa lấy được trong lượt này. Không thay mẫu tải lỗi bằng mẫu đẹp hơn. Đây là convenience feasibility sample, không phải random/representative sample hoặc ước lượng độ sạch toàn MED.

| Đánh giá sơ bộ | Số cặp | ID |
|---|---:|---|
| Có thể gắn Changed/Stable tương đối rõ | 3 | 7,12,17 |
| Có điều kiện: nhiều thuộc tính cùng vật bị sửa | 2 | 4,20 |
| Loại theo tiêu chí hiện tại | 12 | 1,3,5,6,8,10,11,13,14,15,18,19 |
| Thiếu ảnh gốc, không chấm | 3 | 2,9,16 |

Các ví dụ giữ tạm:

- #7: cà vạt xanh→đỏ; giữ sự thật có nhóm người đứng trước tòa nhà. Không gán identity/hành động cho từng người; còn artifact nhỏ.
- #12: bồn cầu xuất hiện ở vị trí trước đó không thấy bồn cầu; bồn rửa cạnh cửa sổ còn. Không gán Stable cho vật nhỏ sát vùng chèn.
- #17: lòng bồn tắm trắng→teal; bồn cầu cạnh bồn tắm còn rõ.
- #4 và20: màu sofa/thân máy bay đổi rõ, nhưng hình dáng/livery hoặc các phần khác cùng vật cũng bị sửa. Không dùng như edit một thuộc tính sạch mà bỏ qua các thay đổi liên quan.

Ví dụ lỗi nguồn quan trọng: #6 nói thêm tàu nhưng ảnh gốc đã có nhiều tàu; #11 nói thêm hươu nhưng ảnh gốc đã có hai hươu; #13 nói thêm đồ uống nhưng ảnh gốc đã có cốc; #19 nói thay mặt nước/cây bằng marina trong khi marina đã hiện diện ở ảnh gốc. Đây là nhận xét trên những cặp đã xem, không suy rộng tỷ lệ lỗi ra toàn dữ liệu.

## Kiểm tra đã làm và giới hạn

- Nguồn HF `EasonFan/hallu_med`, revision `38f969ac98d7db856ee8d57b1a23c4032cfff09e`; MED SFT JSON có50753rows, SHA256 `95fc6968cb6c3425c809beab085ebae6c3afea40ed017318edad3336e8641442`. Provenance official link từ repo MED vẫn chưa xác nhận.
- Chỉ stream archive tới20mục khớp, không tải cả5.139GB. Không extract đường dẫn tùy ý; chỉ ghi basename do script chỉ định.
- Xem17cặp tại display tối đa550px mỗi ảnh và17cặp224×224 bilinear. Resize theo `captioning/data.py:build_image_transform` (đọc source, không import Torch). RGB thumbnail chỉ kiểm tra khả năng nhìn bằng mắt, không chứng minh CLIP hoặc decoder phân biệt được.
- Giải mã34ảnh, đối chiếu34SHA trước/sau tạo trang review;20ID annotations khớp manifest. Không đo tensor/CUDA/model.
- Toàn bộ các từ trong5bộ candidate statements có mặt trong saved exact-A/C0 vocab10000. Đọc từ cached `C0_loaded.json`, member `metadata.json`, không load checkpoint. Điều này không chứng minh model hiểu từ hay tạo câu đúng.
- Mapping VG→COCO từ metadata đã lưu:6cặp xem được map vào Karpathy train/restval;11cặp chưa cóCOCOid. Chưa so tất cả cohort development từng dùng hoặc ảnh trùng gần. **Không được fit hoặc nhận đây là độc lập với eval hiện tại.** Ba DOCCI source names còn cótest/qual_test; không dùng làmtrain chỉ vì chúng nằm trongMED SFT.
- 5nhãn candidate không gồm hard-negative matching chi tiết hay canonical decoder prefix; chưa phải training-ready manifest.
- Không đổi nhãn theo mong muốn giữ candidate: giữ rõ12reject,3unavailable và ảnh/source descriptions để kiểm tra lại.

## Quyết định

**Có thể tạo nhãn Changed/Stable từ một số cặp, nhưng không thể dùng nguyên MED SFT JSON như supervision sạch cho phương pháp này.** Kiểm tra đã bác bỏ giả định “đã có nhãn đổi/giữ dùng ngay”.

Không mở training/full-cache MED ở thời điểm này. Khuyến nghị xem lại3cặp giữ tạm và2cặp có điều kiện trước khi quyết định đầu tư tạo/lọc nhãn. Nếu muốn đi tiếp, cần một tập nhỏ có source-split/overlap rõ và nhãn được xác nhận; phép thử paired-CE vs change-only vs selective-stability mới kiểm tra được giá trị cơ chế. Việc có3cặp thuận lợi không phải bằng chứng novelty, giảmCHAIR hoặc tăngCIDEr; user chưa duyệtpilotGPU.

## Artifact local

Trang xem ảnh: `review/med_feasibility/index.html` (ignored, không public ảnh nguồn trênGitHub). Trong trang có ảnh gốc/sửa, thumbnail224 và lý do giữ/loại; cả mô tả nguồn bị nghi sai vẫn được giữ riêng để đối chiếu. Manifest giữ index gốc, source names, ảnh hashes và mapping; raw metadata/ảnh không commit. Nhãn sơ bộ và báo cáo được commit để giữcontext. Không tạo thêm sheet row về hiệu quả mô hình từ kiểm tra dữ liệu này.
