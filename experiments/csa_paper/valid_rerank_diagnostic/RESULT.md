# Chẩn đoán bộ chọn caption trên VALID

Dùng END sau SCST/CE của CONTROL180 và PROMPT60, đúng5000VALID và5caption/ảnh. 20.045 kiểm tra dữ liệu/selector PASS; chỉ tính lại điểm D/C/O theo mã cố định và tổng hợp CIDEr/CHAIR đã lưu, không chạy model hoặc metric chính thức. Không sửa báo cáo của người dùng.

| Chỉ tiêu | CONTROL | PROMPT |
|---|---:|---:|
| CIDEr trước rerank | 105.1821 | 105.8365 |
| CIDEr sau rerank | 114.2522 | 113.8004 |
| Ảnh được đổi câu | 3061 | 3035 |
| Ảnh tăng CIDEr | 1834 | 1756 |
| Ảnh giảm CIDEr | 1225 | 1278 |
| Ảnh hòa CIDEr | 1941 | 1966 |
| Cờ CHAIR được bỏ | 93 | 85 |
| Cờ CHAIR phát sinh | 41 | 33 |
| Câu được chọn có lặp trigram | 209 | 227 |
| Câu rank0 có lặp trigram | 124 | 119 |
| Chọn câu lặp dù có câu không lặp | 151 | 165 |
| Cả5cùng điểm đối tượng | 2553 | 2682 |

## Kết luận tạm thời

Reranking có ích tổng thể: PROMPT+7,9639CIDEr và giảm52cờCHAIR (bỏ85, thêm33). Tuy vậy, 1278ảnh giảmCIDEr so rank0. Tổng phần tăng15,7541đ bị bù bởi phần giảm7,7902đ. Nếu biết tham chiếu để chọn giữa CHỈ rank0 và câu hiện chọn thì đạt121,5906; đây là upper bound có tham chiếu, không phải điểm của phương pháp mới và không bảo đảm cải thiện triển khai được.

Điểm object bằng nhau trên2682/5000ảnhPROMPT, nên ở hơn nửa cohort nó không tác động phân biệt5câu. Nhận diện được label không đồng nghĩa kiểm chứng quan hệ hoặc số lượng. CLIP score có thể ưu tiên câu lặp; không kết luận nguồn lỗi chỉ từ trọng số.

## Kiểm tra ảnh VALID

- COCO 373857: Ảnh có một TV. Cả năm câu đều lặp TV/television như hai vật riêng; chưa thấy ứng viên tốt cho cảnh này. Đây là lỗi bộ sinh ứng viên, không chỉ bộ chọn.
- COCO 139917: Ảnh có bé trai cầm bộ điều khiển, màn hình trò chơi. Rank0 mô tả chơi game phù hợp hơn; selected rank3 lặp in front of a tv. CLIP đóng góp0,4 so rank0 0,046, thắng phần giảm điểm decoder. Bộ chọn bỏ qua ứng viên tốt hơn về nội dung/ngôn ngữ trong ví dụ này.
- COCO 107257: Ảnh có bếp nấu bị bỏ bên túi rác, không phải refrigerator. Không ứng viên nào gọi đúng stove/oven; rank0 pile of trash chung chung ít sai hơn. Vừa thiếu ứng viên cụ thể đúng, vừa chọn thêm refrigerator sai.
- COCO 487421: Ảnh crop sát mì và rau; bàn không hiện rõ. Chọn thêm on a table làm CHAIR đánh dấu nhưng không thể xác minh chắc bàn không tồn tại. Object score cả5bằng0 nên không giúp phân biệt ứng viên; không coi CHAIR là nhãn sự thật thị giác.
- COCO 511674: Ảnh có hai chiếc kéo ở hai tay. Câu selected nhắc a pair of scissors and a pair of scissors bị heuristic repeated-trigram bắt, nhưng có cơ sở về số vật. Không được tự động coi mọi repeated-trigram là lỗi hoặc lọc cứng.
- COCO 85338: Ảnh có món ăn với thịt và rau trên nền bánh/đĩa; loại món khó chốt từ ảnh. Các câu đầy đủ đều lặp nặng; hai câu không lặp chỉ a table/a plate. Không có câu vừa đầy đủ vừa trôi chảy trong bộ5; đổi bộ chọn đơn thuần khó giải quyết.

Ảnh đều khớp SHA của input suy luận VALID đã lưu. Mẫu có chủ đích từ các nhóm lỗi, không ước lượng tỷ lệ lỗi chung; nhận xét của trợ lý chưa phải người dùng xác nhận.

## Bước kế tiếp đã chuẩn bị

Một notebook CPU duy nhất chấm CIDEr cho cả5rank ở cả2nhánh từ gói VALID6MB. Cùng5000refs/IDF cho từngrank; bắt buộc replay CIDEr từngảnh rank0 và câu đã chọn trước khi tính oracle5câu. Không cần inputDataset/checkpoint/APIsecret/GPU. Oracle dùng tham chiếu chỉ để chẩn đoán dư địa bộ chọn, không áp dụng lúc suy luận. Chưa có điểm oracle5câu; chưa đổi bất cứ quy tắc/trọng số nào.

Pure-data3tests PASS; mã scorer/notebook chỉ static/fixture tại máy tác giả. Java/CIDEr runtime do người dùng chạy. Source cachedZIP180/60, bundleSHA tại RESULT.json. RegionCLIP và training tiếp vẫn tạm hoãn.
