# CPU190 — không đổi trọng số reranking

Lượt `ducanh2403/2-vi-cai-kia-qua-nhieu/190` hoàn tất. Giữ **0,3 decoder + 0,4 CLIP + 0,3 object consistency**. Không có cấu hình mới được chấp nhận và không có kết quả TEST mới.

Trong 66 bộ trọng số, chỉ bộ cũ đáp ứng đồng thời điều kiện trên TUNE2000 cho CẢ CONTROL và PROMPT: CIDEr không giảm, CHAIRs/i không tăng, recall không giảm. Vì thế lựa chọn khóa trước CHECK chính là bộ cũ. `check_accept=false` do không có cải thiện nghiêm ngặt, không phải run lỗi hoặc một bộ mới thất bại trên CHECK.

## CHECK3000 với bộ cũ

| Chỉ số | CONTROL | PROMPT |
|---|---:|---:|
| CIDEr ×100 | 115,85 | 115,21 |
| CHAIRs (%) | 4,33 | 2,90 |
| CHAIRi (%) | 2,92 | 1,83 |
| Object recall (%) | 43,87 | 44,88 |
| Caption có ảo giác | 130/3000 | 87/3000 |

PROMPT ít caption có ảo giác hơn trên phần này, nhưng CIDEr thấp hơn CONTROL khoảng 0,64 điểm. Đây là số của CHECK3000, không thay thế TEST5000 hoặc số fullVALID. CIDEr dùng IDF của corpus CHECK riêng.

## Vì sao không lấy bộ có CIDEr trung bình cao nhất?

Nếu bỏ điều kiện, bộ D=.3/C=.3/O=.4 có CIDEr trung bình TUNE cao nhất trong lưới, nhưng mức tăng trung bình chỉ khoảng **0,0313 điểm CIDEr×100**. CONTROL tăng 115,0936→115,4201; PROMPT giảm 114,8785→114,6146. Recall cũng giảm ở cả hai: 43,1871→42,9842% và 44,1704→43,8583%. Vì vậy bộ này không đạt tiêu chí đã chốt. Đây chỉ là giải thích từ TUNE đã lưu; không chọn lại theo CHECK và không triển khai bộ này.

## Kiểm chứng và giới hạn

- ZIP đọc một lần, text cache ignored tại `.review_cache/valid_weight_calibration_v1/archive_text.json` ngoài repo. [Audit](../audit_weight_calibration.py), [JSON](CALIBRATION_RESULT.json): **187073 checks PASS**, gồm nguồn 7 file ở b2ebae2, split/SHA, 10 bộ CHAIR5000, parity CHAIR lịch sử, 66 hàng TUNE, phép chọn/lockSHA, CHECK/counts và quyết định giữ bộ cũ. Runtime báo historical per-image CIDEr replay PASS; không chấm CIDEr/PTB độc lập local.
- CHECK là kiểm tra nội bộ trên VALID đã dùng trong phát triển; không phải test độc lập. Một cặp mô hình/seed, không có kiểm định ý nghĩa thống kê mới.
- Chỉ kiểm tra 66 bộ trọng số chung với ràng buộc đã định. Không chứng minh bộ cũ tối ưu toàn cục hoặc mọi cách reranking khác đều không hiệu quả.
- Không train, sinh lại caption, chạy mô hình hoặc official scorer local. Không sửa báo cáo Word.

**Quyết định:** đóng lượt hiệu chỉnh trọng số này, giữ pipeline và kết quả TEST đã có. Không cần chạy TEST lại vì quy tắc chọn câu không thay đổi. Bước cải thiện tiếp theo cần một giả thuyết về tín hiệu chấm hoặc chất lượng ứng viên, không tự mở thêm lưới/đổi split/nới tiêu chí sau khi thấy kết quả.
