# Báo cáo tuần từ 25 đến 31 tháng 10 năm 2026

BÁO CÁO KẾT QUẢ TUẦN

Thời gian từ ngày 25 đến ngày 31 tháng 10 năm 2026

1 Mục tiêu

Trong tuần, em tiếp tục nghiên cứu cách căn chỉnh thông tin thị giác với các cụm mô tả trong mô hình sinh caption. Mục tiêu là tăng CIDEr và giảm việc nhắc đến đối tượng không có trong ảnh so với mô hình nền A. Trọng tâm của tuần là kiểm tra mô hình có thực sự dùng thông tin được bổ sung khi sinh câu hay không, so sánh các phương án bằng đối chứng phù hợp và lựa chọn một hướng cụ thể để trình thầy.

Kết quả đáng chú ý là các nhánh C và C0 đã tăng CIDEr so với A trên tập đánh giá mở rộng 4.000 ảnh. Tuy nhiên, mức giảm ảo giác đối tượng chưa được xác nhận đủ chắc chắn và chưa có cơ chế căn chỉnh mới nào được chứng minh hiệu quả. Các công việc và giới hạn được trình bày đầy đủ dưới đây.

2 Công việc đã thực hiện

Kiểm tra chất lượng cặp caption dùng cho học ưu tiên

Sinh 10.000 caption ứng viên trên 2.000 ảnh TRAIN bằng mô hình A cố định. Sau lọc còn 9.570 câu hợp lệ và 1.526 ảnh có cặp ưu tiên theo quy tắc tự động. Phép chọn lý tưởng dựa trên thông tin đánh giá có thể tăng CIDEr từ 1,3305 lên 1,5092 trên tập này, nhưng đó chỉ là mức tiềm năng khi biết trước câu tốt, chưa phải kết quả của mô hình đã học.

Em cùng trợ lý kiểm tra 30 cặp được chọn sẵn. Sau khi em xác nhận và chỉnh nhãn, chỉ 13 cặp đồng ý với hướng ưu tiên tự động; 9 cặp ưu tiên ngược lại, 4 hòa và 4 chưa rõ. Quy tắc không đạt mức 27/30 đã đặt trước, nên dừng mở rộng ngân hàng ưu tiên và chưa dùng các nhãn này để huấn luyện.

Kiểm tra liên kết giữa prompt và các vùng ảnh

Phân tích đường attention từ các từ chỉ đối tượng trong prompt qua 32 query của Q-Former đến các patch CLIP. Trong 10.000 ảnh TRAIN, chỉ 138 ảnh đáp ứng bộ lọc chặt để có hai lớp đối tượng và hai box tách biệt. Attention của đúng cụm có trung bình 32,33% khối lượng trong box, cao hơn phân bố đều 14,38%; thứ tự theo trục giữa hai đối tượng đạt 74,82%. Tuy vậy, tâm attention chỉ nằm trong box ở 28,26% trường hợp. Kết quả cho thấy có tín hiệu không gian, chưa cho thấy định vị chính xác hoặc caption tốt hơn.

Thử điều chỉnh query theo vùng chứa đối tượng

Trên cùng 138 ảnh, so sánh mô hình gốc với phép điều chỉnh nhỏ theo query tập trung vào box đúng và một đối chứng lấy query từ vật khác. Việc tăng attention vào box chưa làm danh từ mục tiêu có xác suất cao hơn. CIDEr chỉ thay đổi rất ít và số caption bị CHAIR đánh dấu tăng từ 9 lên 10. Vì vậy, không dùng phép chọn query này làm tín hiệu hướng dẫn cho một lượt huấn luyện lớn hơn.

Thử huấn luyện với prompt không còn phù hợp và kiểm tra nhãn quan hệ

Chạy hai nhánh C/T về prompt không còn phù hợp, mỗi nhánh 1.250 bước trên TRAIN10k. Nhánh T giảm số caption bị CHAIR đánh dấu từ 52 xuống 44 trên dev1000 nhưng CIDEr giảm từ 1,2055 xuống 1,1782 và recall giảm quá giới hạn đã đặt. Đồng thời, phép kiểm role-interval trên 128 cặp sampled/greedy chỉ có 6 cặp tạo chênh lệch thưởng quan hệ khác 0. Phát hiện thêm rằng ride=0 trong V-COCO không đủ để phủ định câu tiếng Anh như “riding a skateboard”, vì dữ liệu có nhãn hành động skateboard riêng. Em dừng cả hai cấu hình thay vì nới quy tắc sau khi xem kết quả.

Các thí nghiệm căn chỉnh và quan hệ đã hoàn tất

Thử loss hỗ trợ thị giác với ba nhánh C H T

Huấn luyện ba nhánh C, H và T cùng 500 bước trên 2.000 ảnh. C là đối chứng thích nghi chung; H học xếp hạng điểm cuối; T là phương án ràng buộc mức cải thiện phải được hỗ trợ bởi thông tin thị giác tương ứng. T có CIDEr 1,2010, thấp hơn A 1,2055 và H 1,2087. Mức tăng nhỏ của H so với A chưa có khoảng tin cậy tách 0. Chỉ 2 trong 458 nhóm huấn luyện có mức tăng dương của T đạt tỷ lệ hỗ trợ 80%, nên chưa thấy ràng buộc đề xuất đạt được mục đích trong lượt thử này.

Chuẩn bị dữ liệu phát biểu quan hệ có đúng sai

Tạo bộ dữ liệu quan hệ từ VSR và ghép với box detector theo lớp đối tượng và IoU. Phiên bản v2 có 256 cặp huấn luyện trên 484 ảnh và 64 cặp đánh giá trên 128 ảnh; cả hai nhóm đều bao phủ 16 nhãn quan hệ. Các ảnh đánh giá được tách khỏi dữ liệu thích nghi tương ứng. Em đã kiểm tra 32 phát biểu và box, chốt 16 đúng và 16 sai. Đây là kiểm tra nhỏ có trợ lý hỗ trợ, không phải bảo đảm mọi nhãn của toàn bộ dữ liệu đều đúng.

Huấn luyện và kiểm tra tín hiệu quan hệ

So sánh RelPos và RelContrast sau 200 bước tiền huấn luyện. Trên 64 cặp giữ lại, RelPos chọn đúng ảnh cho cùng phát biểu ở 40/64 cặp, RelContrast ở 41/64 cặp. Chênh lệch một cặp chưa đủ chắc chắn; lợi ích trên nhóm vị trí đi kèm giảm điểm ở nhóm tiếp xúc hoặc gần nhau. Thí nghiệm này chưa sinh caption tự do, nên không có kết quả CIDEr hay CHAIR và chưa chuyển sang bước huấn luyện caption tiếp theo.

Thử dùng chung bộ đọc đặc trưng cặp đối tượng

Hai nhánh I/J hoàn tất 500 bước với cùng dữ liệu và khởi tạo. J tốt hơn I nhưng vẫn dưới A: CIDEr lần lượt là 1,1342, 1,0359 và 1,2055; số caption bị CHAIR đánh dấu là 61, 83 và 52. Tắt thông tin cặp trong các mô hình đã học còn làm CIDEr tốt hơn, cho thấy việc đưa thêm đặc trưng chưa có ích trong cấu hình này. Kết quả không xác định được nguyên nhân chỉ từ phép tắt nhánh.

Kiểm tra thêm caption quan hệ trên 64 ảnh thuộc 32 cặp. Trong 175 đánh giá khác nhau, có 17 đánh giá được em xác nhận; 158 vẫn là nhãn của trợ lý. A diễn đạt phát biểu mục tiêu trên 15/32 ảnh dương, I và J đều 13/32; cả ba đều có một phát biểu sai xác định trên 32 ảnh âm. Panel này chưa cho thấy J có lợi ích ngữ nghĩa so với A và không được gọi là đánh giá hoàn toàn bằng người.

Huấn luyện C G và đối chứng C0

Trên 4.000 ảnh TRAIN và 500 bước, C tiếp tục SCST, dùng KL để hạn chế lệch khỏi A và giữ xác suất cho các khái niệm được hỗ trợ. G bổ sung phạt các khái niệm âm tính yếu. C và G tăng CIDEr so với A nhưng G chưa hơn C rõ ràng và chưa giảm số câu ảo giác. Sau đó chạy C0 bỏ cả hai loss khái niệm, chỉ giữ phần tiếp tục SCST và KL. C0 vẫn tăng CIDEr, nên chưa thể quy lợi ích của C cho loss giữ khái niệm. Chi tiết được tách riêng trong các bảng kết quả.

Các công việc cuối tuần và tổng hợp kết quả

Đánh giá mở rộng và thử bộ nhớ quan hệ

Khóa checkpoint C/C0 rồi đánh giá cùng A trên đúng 4.000 ảnh bổ sung, không giao với dev1000 và TRAIN của lượt hiện tại. Không huấn luyện lại trên tập này. C và C0 đều tăng CIDEr so với A; C có ít câu ảo giác hơn theo điểm ước lượng, nhưng chưa qua điều kiện giảm hallucination đã đặt trước. Vì vậy, em giữ lại bằng chứng tăng CIDEr và không kết luận loss khái niệm có đóng góp riêng đã được xác nhận.

Tiếp tục thử bộ nhớ K/V ở hai nhánh D/E: D dùng value độc lập, E dùng chung value cho fact. Cả hai hoàn tất 500 bước trên cùng 4.000 ảnh caption. E có CIDEr 1,2210 và 55 câu bị CHAIR đánh dấu, chưa tốt hơn C0 cùng lượt 1,2307 và 52 câu. Tắt reader chỉ đổi 1/64 caption của D và không đổi caption nào của E trên panel nhỏ, dù logits có thay đổi. Kết quả cho thấy điểm biểu diễn quan hệ tốt chưa tự chuyển thành lợi ích ở câu sinh ra; nguyên nhân cụ thể vẫn chưa xác định.

Khôi phục kết quả và chuẩn hóa việc lưu thí nghiệm

Khôi phục báo cáo nhánh D từ output version 116 sau lỗi ghi kiểu số int64 ra JSON, không huấn luyện lại. Sửa lỗi kiểm hash manifest do khác byte xuống dòng; sửa phép so digest giữa wrapper có teacher và mô hình C0 riêng trong diagnostic mới. Giữ lại lịch sử lỗi và chỉ dùng các lượt có trạng thái hoàn tất để kết luận. Lưu cache và checkpoint thành dataset Kaggle, khóa đúng version nguồn và kiểm các hash cần thiết để tránh lấy nhầm output.

Cập nhật mã nguồn lên GitHub, ghi kết quả vào bảng theo dõi, lưu protocol trước triển khai và trạng thái sau khi kiểm tra. Thiết lập file RESEARCH_EXCHANGE để nhận phản biện từ Codex ở tài khoản khác. Các phép tổng hợp local chỉ đọc metadata, văn bản và số liệu đã xuất; không tự chạy mô hình hay notebook Kaggle. Các lượt mô hình trong báo cáo do người dùng chạy.

3 Kết quả thực nghiệm

Bảng dưới dùng cùng dev1000. CIDEr được ghi theo thang gốc khoảng 1,2, không nhân 100. CHAIRs là số caption bị công cụ CHAIR đánh dấu có đối tượng ảo giác chia cho số ảnh. Ký hiệu C/G thuộc nhóm concept mass, không phải nhánh C của các thí nghiệm khác. Các giá trị giữa tập 1.000 và 4.000 ảnh không được so trực tiếp vì khác cohort và cách tính IDF.

Nhánh trên dev1000 | CIDEr ↑ | Câu ảo giác / 1.000 ↓ | Recall ↑
A nền | 1,205513 | 52 | 45,80%
C giữ khái niệm | 1,225678 | 51 | 45,87%
G thêm phạt âm tính yếu | 1,229535 | 52 | 45,77%
C0 tiếp tục SCST và KL | 1,230741 | 52 | 46,05%
D bộ nhớ value độc lập | 1,228833 | 60 | 45,93%
E bộ nhớ value dùng chung | 1,221030 | 55 | 45,83%

C0 có CIDEr cao nhất theo điểm ước lượng trong nhóm này, nhưng chưa được chứng minh vượt C/G một cách chắc chắn. A và C0 cùng có 52 câu bị gắn cờ và 53 lần nhắc đối tượng ảo giác. CHAIRi của C0 giảm nhẹ do tổng số lần nhắc đối tượng tăng, không phải do giảm số lỗi. D/E không đạt điều kiện để mở rộng cấu hình.

Kết quả trên tập mở rộng 4000 ảnh

Mô hình | CIDEr ↑ | Câu ảo giác ↓ | Lượt nhắc sai / tổng | Recall ↑
A | 1,189771 | 239 / 4.000 | 254 / 6.484 | 46,81%
C0 | 1,204560 | 235 / 4.000 | 248 / 6.502 | 47,11%
C | 1,207042 | 230 / 4.000 | 243 / 6.477 | 47,03%

C tăng CIDEr 0,017271 so với A, khoảng tin cậy bootstrap 95% là [0,008819; 0,025853]. C0 tăng 0,014789, khoảng tin cậy [0,006063; 0,023629]. Đây là tín hiệu cải thiện CIDEr trên cùng tập đánh giá. Tuy nhiên, chênh lệch C−C0 chỉ 0,002481 và khoảng tin cậy còn chứa 0, nên chưa chứng minh loss giữ khái niệm tạo thêm lợi ích ngoài đối chứng tiếp tục huấn luyện.

Số câu bị CHAIR đánh dấu của C giảm từ 239 xuống 230 so với A, và số lượt nhắc sai giảm từ 254 xuống 243. Đây là mức giảm quan sát được, nhưng các khoảng tin cậy về hallucination còn chứa 0; kiểm định theo cặp cũng chưa đạt mức đã đặt. Kết quả mới có một seed và tập validation từng được sử dụng trong nghiên cứu, chưa thay cho kiểm chứng trên test độc lập.

Kiểm tra phản ứng ngữ nghĩa trên cặp ảnh chỉnh sửa

Tìm và kiểm tra các cặp ảnh gốc và ảnh chỉnh sửa thuộc tính. Các mô tả nguồn không được coi tự động là nhãn đúng: ở đợt đầu 20 mẫu chỉ có 17 cặp xem được và nhiều mô tả sai vật hoặc sửa lan. Đợt 30 cặp màu giữ được 3 cặp để em xác nhận; đợt mở rộng 40 cặp có 12 cặp được đề xuất rồi được em xác nhận. Tổng cộng chốt 15 cặp duy nhất, gồm phát biểu trước sửa, sau sửa, phát biểu còn đúng và một phương án sai. Nhãn có trợ lý đề xuất và em xác nhận, không phải đánh giá mù độc lập.

Chạy C0 cố định trên 30 ảnh của 15 cặp, giữ nguyên prompt gốc cho cả ảnh trước và sau. Mỗi cặp được chấm bằng xác suất các cụm cho sẵn và xuất thêm caption greedy thực tế. V2 hoàn tất đủ 15 cặp, qua kiểm tra 109 tensor, trạng thái mô hình không đổi và sanity cùng đầu vào. Worker mất khoảng 422 giây trên T4; đây không phải tổng thời gian hay quota của toàn notebook.

Chỉ số trên 15 cặp phát triển | Số cặp
Điểm của cụm thay đổi dịch chuyển đúng chiều | 14 / 15
Ưu tiên đúng cụm thay đổi ở cả ảnh gốc và sửa | 7 / 15
Ưu tiên đúng fact giữ nguyên trên cả hai ảnh | 9 / 15
Đồng thời đạt đúng phần thay đổi và giữ nguyên | 4 / 15
Caption tự sinh khác chữ giữa hai ảnh | 13 / 15

C0 có phản ứng với thay đổi ảnh, nhưng chưa luôn đổi preference đúng hoặc giữ đúng fact còn nguyên. Caption không nhắc màu không tự là caption sai; caption khác chữ cũng không tự là tốt hơn. Bộ 15 cặp đã xem và chọn có chủ đích, nghiêng về đổi sang màu xanh, nên chỉ dùng làm diagnostic. Không suy kết quả này thành tăng CIDEr, giảm ảo giác đối tượng hay bằng chứng đã xác định nguyên nhân lỗi căn chỉnh.

Tổng hợp tài liệu và phương pháp đề xuất

Đối chiếu các hướng với tài liệu về grounded generation, chỉnh sửa caption, học phản thực và giữ ổn định thông tin. Các tài liệu đã được dùng trong phân tích gồm Neural Baby Talk, các mô hình sinh từ có điều kiện vùng ảnh, TIger, DECap, LURE, Woodpecker, See or Guess, HALVA, MED và CTM kết hợp Stay. Phạm vi đọc từng tài liệu đã được ghi trong nhật ký; một số phần vẫn dựa trên ghi chú hoặc đoạn method đã truy cập, không coi tất cả là đã tái hiện đầy đủ.

Qua phản biện, rút các cách nhận tính mới chỉ từ tên module, việc dùng region để sửa từ hoặc việc cộng các loss thành một ma trận. Phương án CE + change + stable ở đầu ra được giữ làm giả thuyết, nhưng chưa có đường liên kết vùng–cụm tường minh. Phản hồi mới X018 đề xuất một cấu hình cụ thể hơn để trình thầy; cấu hình này mới ở mức thiết kế, chưa có code huấn luyện hay kết quả.

4 Hướng đề xuất và công việc tiếp theo

Hướng đề xuất để trao đổi với thầy

Em đề xuất nghiên cứu căn chỉnh vùng–cụm trong sinh mô tả ảnh khi thuộc tính của một đối tượng thay đổi. Phương pháp kế thừa mô hình CLIP → Q-Former → decoder và cơ chế grounded generation. Từ các patch CLIP, lấy đặc trưng vùng theo box detector. Khi sinh từ, một bộ chọn vùng phân phối trọng số cho các vùng ứng viên và nhánh caption gốc; đặc trưng của từng vùng trực tiếp tham gia tạo phân phối xác suất từ.

Phần cải tiến dự kiến là giữ liên kết giữa cụm mô tả với cùng đối tượng qua cặp ảnh, trong khi cho đặc trưng và từ chỉ thuộc tính thay đổi theo ảnh. Ví dụ xe buýt đổi đỏ sang xanh nhưng ô tô vẫn đỏ: cụm mô tả xe buýt cần tiếp tục đọc vùng xe buýt, còn cụm mô tả ô tô đọc vùng ô tô. Không ép đặc trưng xe buýt ở hai ảnh giống nhau và không bắt hai caption giống từng chữ.

Đường căn chỉnh dự kiến là cụm đang sinh → bộ chọn vùng → đặc trưng vùng → xác suất từ của chính decoder. Ngoài loss caption và giám sát vùng–từ từng ảnh, thêm ràng buộc correspondence giữa các đối tượng tương ứng ở hai ảnh. Phần lớn C0 được giữ cố định; các phép chiếu vùng, bộ chọn và nhánh đọc vùng mới là phần được học. Đây là cấu hình kế thừa có cải tiến huấn luyện dự kiến, chưa được xác nhận là mới so với toàn bộ tài liệu liên quan.

Các bước thực hiện tiếp theo

Trước hết, trình thầy mục tiêu, pipeline và phần cải tiến dự kiến; đối chiếu kỹ cơ chế correspondence qua ảnh chỉnh sửa với nghiên cứu gần nhất. Kiểm tra khả năng bổ sung box cho tối đa 6 cặp phát triển có hai vật rõ ràng, trước khi thu thêm dữ liệu. Bộ 15 cặp hiện tại chưa có nhãn box–cụm đầy đủ và không được chuyển thành tập test độc lập.

Nếu dữ liệu và phương pháp được duyệt, so sánh hai nhánh có cùng mô hình, nhãn và ngân sách: grounded generation học từng ảnh độc lập, và cùng mô hình có thêm ràng buộc correspondence qua cặp ảnh. Giữ C0 làm mốc. Kiểm tra việc thay riêng đặc trưng vùng có làm thay đổi đúng xác suất từ hay không, đồng thời theo dõi tỷ lệ dùng nhánh dự phòng để tránh mô hình bỏ qua vùng ảnh.

Đánh giá cả caption tự sinh và các điểm Changed/Stable trên tập tách theo ảnh nguồn, sau đó kiểm CIDEr, CHAIR, recall và độ dài trên ảnh thật. Nếu chỉ tăng điểm probe hoặc sửa màu mà không cải thiện câu sinh ra thì chưa đạt mục tiêu. CHAIR chỉ đo ảo giác đối tượng, không chấm đầy đủ sai màu hoặc sai quan hệ. Chỉ chốt ngân sách và lịch huấn luyện sau khi xác nhận dữ liệu và số giờ GPU còn lại.

Kết luận của tuần

Tuần này đã có bằng chứng tăng CIDEr từ việc tiếp tục huấn luyện với SCST và KL, đồng thời xác định được nhiều cấu hình chưa có ích để dừng đúng lúc. Mục tiêu giảm ảo giác đối tượng và đóng góp căn chỉnh riêng vẫn chưa được xác nhận. Hướng vùng–cụm qua cặp ảnh thay đổi thuộc tính đã có cơ chế và đối chứng cụ thể hơn để trình thầy; bước tiếp theo là kiểm khả thi dữ liệu và làm rõ khác biệt trước khi triển khai.

## Nguồn kiểm tra nội bộ

- [preference_candidate_audit/HUMAN_REVIEW_RESULT.md](experiments/preference_candidate_audit/HUMAN_REVIEW_RESULT.md)
- [attention_geometry_diagnostic/RESULT.md](experiments/attention_geometry_diagnostic/RESULT.md)
- [routing_utility_diagnostic/RESULT.md](experiments/routing_utility_diagnostic/RESULT.md)
- [stale_prompt_pilot/RESULT.md](experiments/stale_prompt_pilot/RESULT.md)
- [role_interval_alignment/RESULT.md](experiments/role_interval_alignment/RESULT.md)
- [semantic_grounding_method/RESULT.md](experiments/semantic_grounding_method/RESULT.md)
- [relation_frames_inventory/V2_RESULT.md](experiments/relation_frames_inventory/V2_RESULT.md)
- [relation_frames_inventory/HUMAN_AUDIT32_RESULT.md](experiments/relation_frames_inventory/HUMAN_AUDIT32_RESULT.md)
- [relation_frame_pretrain/RESULT.md](experiments/relation_frame_pretrain/RESULT.md)
- [shared_pair_readout/RESULT.md](experiments/shared_pair_readout/RESULT.md)
- [shared_pair_readout/SEMANTIC_PANEL_RESULT.md](experiments/shared_pair_readout/SEMANTIC_PANEL_RESULT.md)
- [concept_mass_grounding/RESULT.md](experiments/concept_mass_grounding/RESULT.md)
- [concept_mass_grounding/C0_RESULT.md](experiments/concept_mass_grounding/C0_RESULT.md)
- [concept_mass_grounding/EXPANDED_RESULT.md](experiments/concept_mass_grounding/EXPANDED_RESULT.md)
- [semantic_grounding_method/KV_DE_RESULT.md](experiments/semantic_grounding_method/KV_DE_RESULT.md)
- [semantic_grounding_method/RESPONSE15_C0_V2_RESULT.md](experiments/semantic_grounding_method/RESPONSE15_C0_V2_RESULT.md)
- [X018 phương pháp dự kiến](RESEARCH_EXCHANGE.md)

Word local: Bao_cao_tuan_31_10_2026.docx. 5 trang, giữ định dạng tuần trước, đã kiểm tra trực quan. Tài liệu tổng hợp, không tạo run mới hoặc ghi kế hoạch thành kết quả.
