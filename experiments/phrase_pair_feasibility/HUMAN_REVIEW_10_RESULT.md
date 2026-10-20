# Audit 10 nhóm human đầu tiên — 20/10/2026

Đã kiểm tra file review do người dùng gửi: đúng schema, manifest hash, group IDs, toàn bộ enum và đủ trường cho nhóm1–10. Có một người tự báo cáo chấm, chưa có người thứ hai/adjudication. Bản JSON gốc lưu local; báo cáo public không ghi tên người chấm.

## Kết quả theo nhãn người dùng

| Tiêu chí | Số nhóm |
|---|---:|
| Đã chấm đủ trường | 10/10 |
| Ma trận hai chiều đối lập rõ | 9/10 |
| Đối lập rõ và cả4 box đúng | 9/10 |
| Thêm điều kiện cả hai event còn thấy bằng chứng ở224 | 7/10 |
| Trong7 nhóm trên: cùng ảnh, hai cặp mục tiêu | 4/7 |
| Trong7 nhóm trên: cả hai cặp có selected UPT objects khớp IoU≥.5 | 6/7 |

Đây là **tính khả thi dữ liệu theo nhãn human**, không phải semantic accuracy của model, không phải UPT raw recall, không phải chứng minh cache CLIP giữ đủ thông tin.

## Các nhóm cần tách riêng

- **Nhóm1 — bicycle posture:** ma trận `[[supported,contradicted],[supported,contradicted]]`. Cả hai event cùng hỗ trợ ngồi trên xe, không tạo đối chiếu2×2. Không dùng làm cặp loss hai chiều; không sửa nhãn để khớp mining hint. Có thể vẫn giữ từng event làm positive riêng sau kiểm tra nếu cần; chưa tự phân vào train. Assistant đã xem overlay ảnh này; người thứ hai ở mép ảnh có phần bị cắt nên chưa coi quan sát đó là adjudication độc lập.
- **Nhóm4 — chair posture:** ma trận đối lập rõ và box đúng, nhưng224 event1 được chấm `unclear`. Có ý nghĩa ngữ nghĩa, chưa đủ điều kiện nhìn rõ224 ở cả hai phía cho bộ sàng lọc hiện tại.
- **Nhóm5 — phone position:** ma trận đối lập rõ, box đúng, nhưng cả hai event224 đều `unclear`. Không loại thành nhãn sai; tách khỏi nhóm đủ chi tiết224.

**7 nhóm đáp ứng cả ba điều kiện:** 2,3,6,7,8,9,10. Mọi nhãn vẫn là self-reported human, không tự động được duyệt thành ground truth huấn luyện.

Nhóm9 có cả hai phone boxes không chứa tâm patch nhưng người chấm vẫn thấy bằng chứng ở224. Điều này minh họa số tâm patch không đồng nhất với khả năng nhìn ra qua tư thế/ngữ cảnh; không được áp quy tắc “zero center = không có thông tin”. Chưa có suy luận model để kiểm tra thông tin thực sự dùng được.

## Quyết định

**Đáng hoàn tất20 nhóm còn lại của audit hiện có; chưa cần mining thêm hoặc yêu cầu chấm hàng trăm nhóm, chưa huấn luyện.** Mở đúng trang review hiện tại, chuyển nhóm11, tiếp tục tới30; export JSON tích lũy. Không thay manifest, mô tả hay thứ tự nhóm sau khi thấy nhãn.

Lý do tiếp tục: đã có9 nhóm ngữ nghĩa đối lập/box rõ và7 nhóm nhìn rõ224, gồm4 nhóm cùng ảnh. Đây là tín hiệu khả thi ban đầu, chưa đủ quy mô cho fit/dev/probe.

**Giới hạn chọn mẫu:** 10 nhóm đầu có7 nhóm cùng ảnh do luật ưu tiên; cả20 nhóm còn lại là khác ảnh. Không ngoại suy tỷ lệ9/10 hoặc7/10 thành yield chung. Audit dùng13 ảnh duy nhất, caption train/restval; không phải độc lập với lịch sử backbone. Các cặp cùng ảnh có giá trị kiểm soát global context nhưng chưa chứng minh model giải được binding.

Sau30 nhóm mới quyết định tiếp: số nhóm đủ chi tiết, phân bố họ quan hệ, độ có mặt trong selected UPT JSON và công tạo nhãn thực tế. Phương án CE so với CE+cross vẫn là thiết kế đang cân nhắc, không phải thí nghiệm đã chạy. Chưa có cải thiện caption/CIDEr/CHAIR mới.
