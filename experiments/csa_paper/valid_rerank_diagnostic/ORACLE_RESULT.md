# Oracle năm caption trên VALID — CPU189

Lượt COMPLETE,20.030 kiểm tra saved-data PASS. CIDEr từngảnh của rank0 và câu đã chọn khớp lịch sử; đúng5000VALID/nhánh và5candidate/ảnh. Chỉ chấm lại caption đã lưu, không train, sinh câu, thay bộ chọn hoặc dùng TEST.

| Cấu hình | CIDEr rank0 | CIDEr rerank hiện tại | CIDEr oracle5 | Oracle−rerank |
|---|---:|---:|---:|---:|
| CONTROL | 105.1821 | 114.2522 | 140.2472 | 25.9950 |
| PROMPT | 105.8365 | 113.8004 | 139.7454 | 25.9450 |

## Diễn giải

Oracle chọn câu có CIDEr cao nhất trong5câu bằng caption tham chiếu VALID. Đây là upper bound có thông tin tham chiếu, không phải phương pháp có thể dùng lúc suy luận và không phải kết quả mô hình mới. PROMPT còn25,945đ khoảng cách này, cho thấy cần ưu tiên chất lượng xếp hạng ứng viên trước khi mặc định phải đổi backbone/train lại. Không hứa có thể thu hồi cả26đ bằng bộ chọn thực tế.

PROMPT chọn đạt maxCIDEr (kể cả hòa)1504/5000ảnh=30,08%; CONTROL1497/5000=29,94%. Không gọi đây là accuracy; câu không đạt max vẫn có thể đúng thị giác. Phần lợi ích CIDEr so rank0 được bộ chọn khai thác là23,49%PROMPT và25,87%CONTROL, không phải tỷ lệ caption đúng.

Hai bộ5câu có oracleCIDEr khá gần (CONTROL140,2472/PROMPT139,7454), nên chưa có bằng chứng PROMPT có candidateCIDEr tốt hơn CONTROL; lợi ích giảmảo giác trước đó vẫn là kết quả riêng, không bị oracleCIDEr thay thế. Oracle không tối ưuCHAIR.

## Kiểm tra thành phần điểm, không dò trọng số

| Bộ chọn chẩn đoán | CONTROL | PROMPT |
|---|---:|---:|
| Chỉ decoder | 105.1821 | 105.8365 |
| Chỉ CLIP | 110.5978 | 110.5684 |
| Chỉ đối tượng | 108.7672 | 109.1750 |
| Kết hợp hiện tại | 114.2522 | 113.8004 |

Các probe argmax dùng điểm D/C/O có sẵn, không chạyCLIP mới và không triển khai policy mới. Kết hợp tốt hơn từng điểm riêng; không có cơ sở chỉ tăng trọng sốCLIP hoặc bỏ các nhánh còn lại.

## Ví dụ đã xem ảnh

COCO139917 (bé trai chơi game): rank0 CIDEr1,2097, selectedrank3=0,1964; oracleđúngrank0. Có ứng viên mô tả hành động phù hợp hơn nhưng bộ chọn ưu tiên câu lặp. COCO85338: oraclechỉ0,1754, các ứng viên đầy đủ lặp nặng, nên chỉ đổi bộ chọn không đủ cho ảnh này. COCO373857: oracle1,9346 nhưng cả5câu đều nhắc TV lặp; điểmCIDEr cao không chứng minh câu tự nhiên hoặc đúng số lượng vật.

## Quyết định đề xuất

Ưu tiên thiết kế một thí nghiệm cải thiện reranking có mục tiêu chất lượng/ngữ nghĩa và theo dõiCHAIR/recall, giữcheckpoint/candidatepool. Chưa tự đổi trọng số, lọc lặp hoặc huấn luyện ranker. Nếu muốn học bộchọn, supervision/fit phải trênTRAIN, chọncấu hình trênVALID; không dùngTEST làm nhãn huấn luyện. Trước đó automaticpreferenceaudit từngkhôngđạt; không tái sử dụng các cặp đó như humantruth. Reranker cải thiệnCIDEr không tự động chứng minh cải thiệncănchỉnh vùng.

## Nguồn và kiểm tra

Người dùng xác nhận SaveVersion189 của notebook chính `ducanh2403/2-vi-cai-kia-qua-nhieu`. Nguồn candidate vẫn CONTROL180/PROMPT60; sourceGit590be7, bundleSHA vàZIPsha trongORACLE_RESULT.json. Không tảiAPI lại; đọcZIPmộtlần vào cachetext. Audit sourceGit/bundleSHA,2×5000uniqueIDs,cùngrefs,5finiteCIDEr/ảnh,rank0/selectedper-image replay,aggregate/oracleties/progress; không chạyofficialscorer hoặc model ở máy tác giả.

- Oracle uses VALID references; not deployable, not achievable-gain guarantee, not visual truth.
- Selected-is-best counts allow CIDEr ties; not caption accuracy. Capture ratio is fraction of CIDEr headroom, not accuracy.
- CIDEr oracle does not optimize CHAIR, relation truth or linguistic quality.
- Same generation/weights; no new model training. VALID reused, one seed; no official scorer/model local.
- Component-only argmax probes are fixed diagnostics, no weight search or TEST tuning.

Đã ghi Sheet Kết quả mới A80:L87,42giá trị khai báo được đọc lại; mở rộnggrid81→91rows và chỉ thêmblockmới, không sửadata cũ. API values/format verification, không claim nativevisualQA. Không sửa Word.
