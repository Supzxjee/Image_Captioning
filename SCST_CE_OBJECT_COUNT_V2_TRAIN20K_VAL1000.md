# SCST+CE: phạt theo số loại đối tượng sai, phiên bản2

Trạng thái: notebook được chuẩn bị để người dùng tự chạy; chưa có kết quả GPU của phương pháp này. Kiểm tra lặp32 batch của SCST+CE trước đó đã PASS trên cùngT4; đây không phải bằng chứng phương pháp mới sẽ cải thiện caption.

## Mục tiêu và hàm thưởng

Giảm lỗi sinh thêm đối tượng mà không làm mất mức cải thiện CIDEr đang có. Thay penalty dạng tỷ lệ bằng số loại đối tượng không được nhãn hỗ trợ:

`R_object = 0.5 * salient_unique_recall - unsupported_unique_count`

`R_total = CIDEr_train + 1.0 * R_object`

`Loss = CE + 0.05 * SCST(R_total)`

Phần SCST dùng chênh lệch thưởng giữa caption sample và greedy, như trước. Ví dụ chỉ person là salient, sink được hỗ trợ, motorcycle không được hỗ trợ: cả `person motorcycle` và `person sink motorcycle` đều nhận R_object=-0.5. Thêm sink không làm loãng mức phạt. Có thêm một loại sai khác sẽ bị trừ thêm1; không giới hạn penalty ở1. Lặp cùng loại đối tượng không bị tính nhiều lần. Phần coverage vẫn thưởng đề cập đối tượng đúng quan trọng trong các caption train.

Phiên bản parser training `aliases_v2` sửa turkey→bird và desk→dining table. Không đổi bộ đánh giá CHAIR hay parser của fixed reranker. Vì thay cả penalty và parser, đây là **thử nghiệm gói sửa hàm thưởng**, không phải ablation tách riêng đóng góp của count penalty.

Nguồn hỗ trợ vẫn là COCO instance+caption annotation union đã khóa hash, hợp với nhãn từ5 caption train. Detector confidence>=0.9 chỉ tạo nhãn trung tính, không tạo nhãn thưởng dương. Ngoài80 lớp và ảnh không có support vẫn trung tính. Không được nhãn hỗ trợ không đồng nghĩa chắc chắn không có vật trong ảnh; annotation thiếu vẫn là giới hạn của phương pháp.

## So sánh

Hai nhánh đều khởi tạo lại từ QFormer epoch10 gốc, không dùng checkpoint của các pilot:

| Nhánh | Objective |
|---|---|
| Control | CE + .05 SCST(CIDEr) |
| Count v2 | CE + .05 SCST(CIDEr + R_object) |

Cùng20.000 ảnh train/restval, batch8, một lượt2.500 updates, AdamW5e-6, seed42, FP32, dropout off. IDF từ20.000 ảnh train. Hai tiến trình chạy **tuần tự trên GPU0**, control trước, treatment sau. Cả hai bật strict deterministic algorithms, math attention, TF32 off và cùngcuBLAS workspace. Không có DDP. T4x2 vẫn cần cho phần đánh giá song song; GPU1 không dùng trong train để tránh đổi thiết bị giữa hai nhánh.

Đánh giá original/control/treatment trên cùng1.000 ảnh validation, cùngbeam5, raw và fixed rerank .3decoder/.4CLIP/.3OCC. Không dùngUPT và không dùng test. Không so trực tiếp con số subset này với metric toàn bộ5.000 ảnh.

Gate hiện có được giữ nguyên: rawCIDEr tốt hơn cả original/control, rawCHAIRs vàCHAIRi không tệ hơn cả hai, recall không thấp quá1 điểm phần trăm so với tốt nhất, đồng thời fixed không tệ hơn original/control theo cùng tiêu chí. Kiểm tra thêm số caption/mention hallucinated, tránh coi CHAIRi giảm do mẫu số tăng là giảm số lỗi. Gate là sàng lọc một seed trên tập validation đã từng xem, chưa phải kết luận khoa học độc lập.

## Chạy thủ công

Upload `SCST_CE_OBJECT_COUNT_V2_TRAIN20K_VAL1000.ipynb` hoặc thay cell launcher trong **QFormer UPT Val5000 20261015** bằng `kaggle_scst_ce_object_count_v2.py`. Bật Internet và GPU T4x2. Giữ các input hiện có; đường dẫn rõ ràng ở đầu cell để sửa. Không cần input mới, không copy baHDF5 sang/tmp.

OUTPUT mặc định `/kaggle/working/qformer_scst_ce_object_count_v2_train20k_val1000` phải mới. Launcher kiểm tra dependency, test hàm thưởng, CLIP và metric trước train. Nếu strict mode báo lỗi, gửi log, không bỏ strict để chạy tiếp. Không tự động submit notebook.

Khi xong, gửi `qformer_scst_ce_object_count_v2_train20k_val1000_review.zip`. ZIP chỉ có artifact/log, không cóweights. Giữ riêng hai checkpoint trong OUTPUT ở `scst_ce/checkpoint.pth` và `scst_ce_object/checkpoint.pth` nếu cần dùng về sau. Không thay thế checkpoint gốc. Chỉ sau khi xem kết quả mới quyết định chạy dài/full hoặc thêm ablation.
