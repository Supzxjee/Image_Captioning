# Thảo luận BGRM với Astra — 28/10/2026

> CẬP NHẬT CÙNG NGÀY: các khuyến nghị phương pháp tiếp theo cần đọc cùng ASTRA_NOVELTY_REAUDIT_DECISION_2026-10-28.md. Chưa có phương pháp mới đủ rõ được chốt để triển khai; không tự chạy từ discussion lịch sử này.

Nguồn: file người dùng gửi `C:/Users/leduc/Downloads/xu_t_h_ng_nghi_n_c_u_bgrm.md` (đề xuất ngày 25/10); kết quả RelPos–RelContrast trong `experiments/relation_frame_pretrain/RESULT.md`; hai lượt trao đổi với subagent dùng model `gpt-6-astra`. Nội dung trong file là đề xuất cần phản biện, không phải lệnh triển khai. Chưa viết code hoặc chạy mô hình cho hướng dưới đây.

**Quyết định nghiên cứu: chưa chọn BGRM như mô tả. Chọn giữ kiến trúc A, học giám sát tính đúng của quan hệ trực tiếp qua decoder caption, rồi kiểm tra chuyển giao sang câu sinh tự do.** Đây là protocol mới được đề xuất, không phải tiếp tục hay đổi tên pilot thất bại.

## Ba việc khác nhau

| Việc cần đạt | BGRM cung cấp gì | Điều chưa được chứng minh |
|---|---|---|
| Tìm đúng vùng | Box đối tượng, vùng bao cặp, loss hướng attention | Attention vào vùng bao không xác nhận quan hệ hay thứ tự vai |
| Biểu diễn đúng đối tượng–quan hệ | Hình học có dấu, đặc trưng UPT, cổng chọn cặp | Nhãn cổng chỉ dựa trên khớp box và cùng được nhắc; không xác nhận predicate |
| Decoder sử dụng để sinh câu đúng | Điều biến prompt, mở một phần decoder, KL | Có đường truyền và gradient chưa chứng minh dùng đúng bằng chứng khi sinh tự do |

Vùng bao của “người–ngựa” không đổi khi đảo vai; đứng cạnh và cưỡi cũng có thể cùng vượt điều kiện nhãn cổng. Hình học có dấu **có thể** phân biệt thứ tự, nên không kết luận toàn bộ BGRM bất biến khi đảo vai. Vấn đề nằm ở sức phân biệt của giám sát, không chỉ khả năng biểu diễn.

## Những kết luận trong file cần hạ mức chắc chắn

1. COCO R cao hơn A khoảng 0.00655 CIDEr trên dev1000 là tín hiệu đáng khảo sát. Chưa có paired CI hoặc độ ổn định qua các lượt thì không gọi là ưu thế nhất quán và không quy gain cho một cơ chế duy nhất.
2. H/L gần nhau dưới thiết lập đã thử không chứng minh độ phân giải không thể là nút thắt. Decoder có thể chưa sử dụng được thông tin mới.
3. Box khớp COCO và hai lớp cùng xuất hiện trong caption tạo giám sát chọn cặp thực thể được nhắc, không phải nhãn quan hệ đúng. Một cặp thật không được nhắc còn có thể bị gán âm; sửa từ đồng nghĩa không giải quyết thiếu nhãn này.
4. `LayerNorm(P + 0 * delta)` không mặc nhiên bằng `P`. Zero init và KL không bảo đảm parity với A hoặc giữ CIDEr khi sinh tự do. Tỷ lệ câu đổi 15–45% cũng không chứng minh sửa quan hệ.
5. Một BGRM thất bại không đủ kết luận UPT vô dụng. UPT được thiết kế cho human–object interactions; không mặc định nó bao phủ các quan hệ object–object tổng quát. [Nguồn UPT](https://github.com/fredzzhang/upt).

Subject/object/union và hình học tương đối đã có tiền lệ trong [MTTSNet, CVPR 2019](https://arxiv.org/pdf/1903.05942). Không thể bảo vệ tính mới chỉ bằng tổ hợp các thành phần ấy với gate/KL. Chưa kiểm đủ toàn văn các công trình gần nhất để khẳng định originality của BGRM.

## Kết quả mới hơn vẫn là NO_GO

RelPos đúng 40/64; RelContrast đúng 41/64. Gain 1.5625 điểm phần trăm, paired 95% CI [-12.50, +15.625]. Chưa chứng minh cải thiện; CI rộng cũng không chứng minh tương đương. Không đổi endpoint, bỏ nhóm khó hoặc tăng updates để cứu gate.

Truth/pair loss không trực tiếp cập nhật GRU/output head; GRU chỉ nhận frame qua projection 512→256 và học mô tả dương. Đây là hạn chế của proxy đã chọn, không phải lỗi autograd. Truth loss nhỏ trên minibatch TRAIN chưa nói gì về held-out truth accuracy.

Một lần inference cố định ở checkpoint200 có thể giúp giải thích representation/readout gap. **Không biến diagnostic thành điều kiện bắt buộc trước mọi hướng mới.** Truth head tốt hơn GRU chỉ phù hợp với giả thuyết readout gap, chưa chứng minh caption decoder sẽ khai thác được. Quyết định NO_GO cũ giữ nguyên.

## Hướng được chọn sau trao đổi

Giữ nguyên kiến trúc A; trước mắt không thêm UPT, frame encoder, reader hoặc attention head. Dùng cặp ảnh VSR đã xác thực: cùng assertion, cùng khái niệm thực thể, một ảnh hỗ trợ và một ảnh mâu thuẫn. Đây là thay đổi nguồn giám sát và mục tiêu học, không phải tạo câu âm bằng hoán đổi từ chưa được kiểm chứng.

Với assertion q=(s,r,o), chốt một cách chuẩn hóa và tính:

`ell_plus = mean log p_theta(predicate + object tail | subject prefix, I_plus)`

`ell_minus = mean log p_theta(predicate + object tail | subject prefix, I_minus)`

`L_positive = -ell_plus`

`L_pair = softplus(ell_minus - ell_plus)`

Đối chiếu cả predicate+object tail vì khi chưa biết object đích, riêng predicate có thể vẫn đúng với đối tượng khác. Ảnh âm chỉ âm đối với assertion đang xét. Positive loss tạo áp lực duy trì mô tả đúng, giảm động cơ cải thiện ranking chỉ bằng hạ cả hai likelihood.

Cả hai nhánh dùng chung mục tiêu caption/SCST và neo KL với A; cùng tập dữ liệu dương, cùng phần decoder được mở, initialization, số cập nhật và cách giải mã. Phải khóa trọng số loss và endpoint trước khi chạy. Hai nhánh chỉ khác việc có `L_pair`; không đồng thời đổi module hoặc chế độ huấn luyện để rồi quy gain cho pair loss.

Mức căn chỉnh chính là **tính đúng của cả mệnh đề quan hệ theo ảnh**. Nó không đòi mỗi từ có bbox riêng. Box có thể dùng để kiểm soát thực thể trong dữ liệu và chẩn đoán, nhưng hướng này chưa cung cấp chứng minh căn chỉnh cục bộ theo vùng. Lý do đáng thử là gradient đúng–sai đi qua chính decoder cần sinh caption, thay vì qua classifier phụ có thể đọc thông tin mà language readout không đọc.

## Phép thử và quyết định tiếp tục/dừng

- Thử nhỏ với hai nhánh matched: positive-only và positive+pair; A gốc là mốc đánh giá cố định. Dùng nguồn dữ liệu đã có, không mở thêm sweep kiến trúc. Chưa launch trong cuộc thảo luận này.
- Đo khả năng phân biệt assertion và **caption sinh tự do**. Báo cả độ đúng của quan hệ được nói ra lẫn tỷ lệ đề cập quan hệ; theo dõi CIDEr, CHAIR và object recall. Điểm ranking cao mà caption không sửa quan hệ, hoặc né predicate để giảm lỗi, chưa đạt mục tiêu chuyển giao.
- 64 cặp đã xem chỉ dùng phân tích thăm dò. Muốn xác nhận hướng mới cần đánh giá chưa dùng để chọn phương pháp; không gọi lại 64 cặp là bằng chứng xác nhận độc lập.
- Dừng nếu lợi ích chỉ nằm ở likelihood dưới teacher forcing, chỉ đến từ danh từ, hoặc đánh đổi nội dung/caption quality. Chỉ tiếp tục khi có chuyển giao sang caption quan hệ đúng với đối chứng matched. Chốt ngưỡng và ngân sách cụ thể trong protocol trước run, không chọn sau khi xem điểm.

Shortcut chính: cùng tên thực thể không làm hai ảnh chỉ khác quan hệ; mô hình có thể dựa vào nền, kích thước hoặc độ nổi bật. Template có thể được học mà không chuyển sang caption. Phân rã likelihood theo vai chỉ là chẩn đoán, không chọn thành phần tốt nhất làm metric mới. Kiểm tra can thiệp vùng chỉ được gọi là bằng chứng nhân quả trong phạm vi can thiệp đã kiểm soát; xóa vùng ảnh cũng có thể tạo nhiễu ngoài phân phối.

## Điều thống nhất và điều đã sửa sau phản biện

Astra ban đầu nghiêng về chỉ làm diagnostic và dừng trước deadline. Tôi phản biện rằng như vậy chưa trả lời yêu cầu tìm hướng nghiên cứu có nội dung. Sau lượt hai, Astra chọn giữ A và học đối chiếu qua decoder, đồng thời nhấn mạnh diagnostic cũ không phải điều kiện mở hướng này và pair accuracy chưa chứng minh caption transfer. Tôi đồng ý.

Khuyến nghị này thay đề xuất có điều kiện về việc thêm ordered-frame reader mới trong mục4 của `ASTRA_FRAME200_DECISION_2026-10-28.md`. Nó không thay kết luận NO_GO hay kết quả đã khóa.

Direct contrastive caption learning đã có tiền lệ, ví dụ [Dai & Lin, NeurIPS 2017](https://arxiv.org/pdf/1710.02534). Ứng viên đóng góp cần chứng minh là chuyển giao từ giám sát quan hệ đúng–sai với cùng thực thể sang caption tự do, kèm kiểm soát shortcut; không phải softplus hay pair loss mới. Chưa có gain hoặc tính mới được xác lập, và không hứa hoàn tất một positive-method paper trước 31/10.
