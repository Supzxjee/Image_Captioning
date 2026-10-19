# Kết quả chẩn đoán F — 20 ảnh, 20/10/2026

**Đổi text quan hệ làm đổi attention và token bằng chứng, nhưng không đổi caption greedy ở bất kỳ ảnh nào trong 20 ảnh khi so hai điều kiện annotation.** Phản ứng likelihood có thật nhưng nhỏ. Đây là bằng chứng đường quan hệ → sinh câu đang yếu trên mẫu này, chưa chứng minh lỗi nằm riêng ở decoder và chưa phải đánh giá chất lượng caption.

## Kiểm tra tính hợp lệ lần chạy

`relation_conditioning_diagnostic_20_review.zip` qua CRC; khớp 11 source nhúng và manifest cố định. Hai worker T4 hoàn thành 10 ảnh/worker, cùng full-model hash, báo trọng số không đổi; không huấn luyện, không test. Checkpoint F SHA `4938bfe875f05104cd0afca4918dad3761b181bece08d773ad2b71b6adb51043` đúng tệp đã chọn. Replay matcher sai số tối đa `3.13e-7`, dưới giới hạn `1e-4`. Log Kaggle ghi 3 data tests và 1 synthetic model test PASS.

Audit local tái tính mọi chỉ số từ JSON từng rank, kiểm tra mô tả/độ dài câu/thứ tự annotation và phép tính likelihood. Không chạy lại model, không đọc tensor checkpoint trong ZIP (không có tensor). Cảnh báo pip về gradio/diffusers không làm các kiểm tra/suy luận này thất bại; không nên coi môi trường đó đã được xác nhận cho các ứng dụng khác.

## Số liệu

| Điều kiện | Caption đổi so với tắt nhánh | Ưu tiên mô tả annotation dương theo logP tổng |
|---|---:|---:|
| Tắt nhánh E/F | 0/20 | 11/20 |
| Prompt/box UPT | 3/20 | 11/20 |
| Annotation dương, box GT | 2/20 | 11/20 |
| Annotation âm, cùng box GT | 2/20 | 11/20 |

Đối chiếu có kiểm soát **annotation dương với annotation âm** giữ nguyên S/O/box/số slot, chỉ đổi text R:

- Caption khác nhau: **0/20**.
- Cả bốn điều kiện giữ nguyên phía được ưu tiên ở từng ảnh: **20/20**; con số 11/20 không phải do các ảnh khác nhau bù trừ.
- Khoảng cách L1 attention R trung bình: **0.435869** (không phải độ chính xác).
- Khoảng cách tương đối L2 token R trung bình: **0.230696**; min **0.029037**, max **0.941822**. Không phải “cải thiện 23%”.
- Dịch chuyển trung bình của chênh logP(dương−âm): **−0.000885**; trị tuyệt đối trung bình **0.003725**, lớn nhất **0.014862**; 8 ảnh dịch về phía dương, 12 ảnh về phía âm.

Không suy diễn “hoàn toàn bỏ qua prompt”: UPT khác off ở 3 câu, annotation khác off ở 2 câu, và likelihood cũng đổi. Kết luận hẹp hơn là **thay động từ tại cùng cặp box chưa điều khiển được lựa chọn câu greedy**.

## Các ca cụ thể

`CASE_CAPTIONS.json` lưu cả 20 ca và bốn caption. Đã xem hai bảng ảnh có S/O box của 20 ảnh ở local; đây là quan sát của assistant, không phải nhãn human hoặc kiểm định đầy đủ các vùng rất nhỏ.

- **251696**: mô tả đối chiếu là *carrying a cell phone* / *talking on a cell phone*. Token R đổi tương đối L2 **0.9418**, lớn nhất mẫu; cả hai vẫn sinh `a man and woman on a subway train talking on their cell phones.` Box S chỉ người phụ nữ bên trái đang nhìn điện thoại, trong khi người bên cạnh áp điện thoại gần tai. Đây là ca nên kiểm tra sự khác nhau giữa mô tả cả ảnh và quan hệ của instance cụ thể; chưa tự gán verdict đúng/sai.
- **340406**: cả hai mô tả *holding a bicycle* / *riding a bicycle* đều sinh `a group of people waiting for a train to come in.` So off câu có đổi, nhưng đổi động từ R không đổi câu. Caption tổng thể có thể chọn cảnh tàu và nhóm người thay vì đối tượng nhỏ trong box.
- **495476**: cả hai mô tả *holding a backpack* / *carrying a backpack* đều sinh `a group of people riding on the back of an elephant.` S/O là người và ba lô nhỏ phía trái; caption chọn nhóm cưỡi voi. Việc bỏ qua một quan hệ phụ không tự động là caption sai.

## Diễn giải và khuyến nghị

**Bằng chứng:** truy vấn text ảnh hưởng cơ chế lấy bằng chứng và token thị giác; sự thay đổi đó chỉ tạo đáp ứng likelihood nhỏ và không thay đổi caption trong đối chiếu R. F trước đó cũng chưa vượt A về CIDEr.

**Giả thuyết, chưa được chứng minh:** chỉ dùng text để đổi trọng số trộn các patch trong cùng union box có thể chưa giữ đủ nội dung hành động; hoặc decoder ưu tiên memory nền và mô tả toàn ảnh. Hai nguyên nhân có thể cùng xảy ra. Không thể phân biệt chúng chỉ từ caption bất biến, và chuẩn L2 lớn không chứng minh token biểu diễn đúng ngữ nghĩa.

**Không khuyến nghị chạy full F, thêm seed F hoặc chỉ tăng trọng số loss matcher.** Bước tiếp theo nên chốt thiết kế một nhánh giữ được ngữ nghĩa quan hệ trong biểu diễn đưa vào decoder và một mục tiêu học nối nhánh đó với caption, rồi mới làm pilot nhỏ so với A cùng budget. Có thể thử một thành phần text residual được gate bằng bằng chứng thị giác, nhưng đó là phương án cần kiểm chứng: đưa text trực tiếp cũng có thể làm mô hình lặp lại quan hệ UPT sai. Cần kiểm tra on/off và tráo động từ, không chỉ giảm loss matcher.

Nếu dùng loss buộc decoder phân biệt hai câu, chỉ áp dụng với các cặp đã kiểm tra không chồng lấn và ngữ cảnh thật sự mô tả instance đó. Không ép mọi quan hệ V-COCO xuất hiện trong caption toàn ảnh, không tự biến nhãn action=0 thành phủ định ngôn ngữ. Đây là hướng thiết kế đề xuất, chưa triển khai hoặc có kết quả mới.

Mẫu gồm 20 ảnh lỗi được chọn hậu nghiệm, thuộc caption train/restval và từng được dùng chẩn đoán; không có tính đại diện hoặc đánh giá độc lập. Không có CIDEr/CHAIR mới, không thay đổi kết luận benchmark và không dùng các ảnh này để lựa chọn checkpoint.
