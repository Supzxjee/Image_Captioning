# Cổng điều chỉnh ngữ nghĩa — kết quả V2, 10 epoch XE

Lượt chạy hoàn tất. Người dùng xác nhận Save Version **12**; notebook `kltnnnnnff` kế thừa từ mạch phiên hiện tại. Source mô hình `cc6a20e`; V2 chỉ đổi giới hạn toàn phiên sang 11,5 giờ và thư mục output. ZIP SHA `63647dd102dcee6c69f5ef20e99a7a8a331dd986ec85d05c735ce97125dbbc68`, đọc một lần và lưu text tại `../.review_cache/semantic_strength_xe10_v2/archive_text.json`.

Audit `review_semantic_strength.py`: **1.960 kiểm tra PASS** về source30pins, scope, byte seals của dữ liệu/feature/vocabulary/reference so với bản bank + prompt cũ, exposure/LR, caption/IDs, CIDEr arithmetic, CHAIR counts, NLL và thống kê cổng. Không chạy lại Torch, mô hình, scorer hoặc mở binary checkpoint/feature trên máy local. BLEU/ROUGE là điểm runtime báo; CIDEr saved-score means và CHAIR counts được tính lại từ dữ liệu đã lưu.

## So sánh trên cùng VALID5000

Mỗi cấu hình đều chạy từ đầu 10 epoch XE, 177.110 update, global batch32, 5.667.470 lượt caption trên113.287 ảnh TRAIN. Cùng lịch học và quy tắc chọn checkpoint bằng CIDEr VALID cao nhất trong10epoch, ưu tiên epoch sớm khi hòa. Không SCST, reranking hay TEST. Best cũ ởepoch9, best cổng mới ởepoch8; đây không phải hai snapshot cùng số update. Endpoint10 được báo thêm bên dưới.

| Chỉ số | Patch baseline199 | Bank + prompt cũ | Bank + prompt + cổng |
|---|---:|---:|---:|
| Epoch được chọn | 10 | 9 | 8 |
| CIDEr | 119,0942 | 120,5804 | 120,2586 |
| BLEU-1 | 77,2054 | 77,7238 | 77,5542 |
| BLEU-4 | 37,0724 | 37,4403 | 37,6161 |
| ROUGE-L | 57,3740 | 57,6802 | 57,5474 |
| CHAIRs | 5,38% | 4,58% | 3,80% |
| CHAIRi | 3,7842% | 3,1826% | 2,6585% |
| Recall | 43,0500% | 43,6527% | 42,8051% |
| VALID NLL | 2,101613 | 2,100591 | 2,096618 |
| Caption bị CHAIR đánh dấu | 269 | 229 | 190 |

Các metric caption nhân100; CHAIR/Recall là phần trăm; NLL giữ nguyên. So bản bank + prompt cũ, cổng mới: CIDEr−0,3218điểm, BLEU4+0,1759, CHAIRs−0,78điểm%, CHAIRi−0,5240điểm%, NLL−0,003972. Caption bị gắn cờ giảm39/5000 (khoảng17,03% tương đối). So baseline patch, vẫn tăng CIDEr1,1644điểm và giảm CHAIRs1,58điểm%, nhưng chưa có tăng thêm CIDEr từ cổng.

Tại cùng endpoint10: cũ→mới CIDEr120,4308→120,2095, CHAIRs4,42%→4,06%, CHAIRi3,0746%→2,8057%, NLL2,106570→2,106759. Vì thế không thể nói cổng tăng CIDEr chỉ bằng cách đổi checkpoint so sánh. Giữ best8 theo quy tắc đã khóa, không đổi lựa chọn theo CHAIR.

## Cổng đã học gì?

Tại best8, alpha được thống kê trên toàn bộ980.000patch VALID: trung bình0,490705, min0,000255/max0,999204;14,2839%patch có alpha<0,1 và14,8458% có alpha>0,9. Epoch1/2/8/10 có mean0,4220/0,4040/0,4907/0,4949. Trung bình gần0,5 không có nghĩa cổng đứng yên: phân bố trải rộng. Các accumulator hai rank đã được gộp và replay đúng từng epoch.

Bằng chứng này cho thấy cổng có biến thiên và không đóng hết toàn bộ ngữ nghĩa. Nó không xác nhận vùng nào được căn chỉnh đúng, không phải xác suất tin cậy và chưa cô lập việc dùng prompt. Residual weights có thể bù cho alpha, nên không diễn giải giá trị cổng thành mức đóng góp ngữ nghĩa trực tiếp.

## Mức chắc chắn và đánh đổi

Theo ảnh:125ảnh mất cờ CHAIR,86ảnh mới bị gắn, net−39. Đây là cờ ảo giác đối tượng, không phải xác nhận125caption hoàn toàn đúng. Bootstrap cặp2.000mẫu/seed42: CIDEr deltaCI95[−1,7331;+1,0966]điểm, CHAIRs deltaCI95[−1,34;−0,20]điểm%. Có điều kiện trên một seed/VALID tái sử dụng/checkpoint đã chọn, chưa điều chỉnh selection hay nhiều so sánh; không phải kiểm định tương đương.

Độ dài9,4262→9,3096từ; object mentions7384→7222, recalled unique6953→6818 và Recall giảm0,8476điểm%. Điều này gợi ý caption có thể bảo thủ hơn; chưa chứng minh nguyên nhân. Recall là chỉ số phụ nhưng phải báo đánh đổi, không thay mục tiêu chính là CIDEr và hallucination. NLL tốt hơn ở best8 không tự động đồng nghĩa CIDEr tốt hơn.

34.801.913tham số học, chỉ thêm1.025 so với bản cũ. Worker train+VALID+checkpoint7h33m23s, trước đây7h09m10s; stage patch32m38s, semantic56,71s. Không phải thời gian billing và không cô lập overhead của cổng khỏi biến thiên phần cứng/IO. T4×2/coreinit/alpha endpoint/DDP/beam guards đều PASS.

## Kết luận

**Mục đích tăng thêm CIDEr bằng cổng mới chưa đạt trên VALID.** Lợi ích quan sát được tiếp tục ở giảm ảo giác, với một số đánh đổi về độ đầy đủ. Không tuyên bố tốt hơn toàn diện hoặc đã giải quyết nguyên nhân căn chỉnh không tăng CIDEr. Bản này chưa chấm TEST/METEOR/SPICE; không trộn với TEST của hai cấu hình cũ.

Giữ `best.bin` epoch8 SHA `456681d45b8b9ce8e3d34f3452c5f6b7bebdef21fd2ad3e6f344790ccbe4ccf9` và `checkpoint.bin` epoch10 SHA `523063b6fd698a79c970f8de21553ad68ed0c0729bae462618685b84d813fb17`. Chưa tự chạy TEST, SCST, tăng epoch hay thêm một cơ chế khác. Không sửa Word.
