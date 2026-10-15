# Pilot V2: học lợi ích so với fixed, có quyền giữ nguyên caption

**Đã chuẩn bị, chưa chạy.** Chạy trong notebook người dùng xác nhận: **QFormer UPT Val5000 20261015** (`ducanh2403/qformer-upt-val5000-20261015`). Trợ lý không khởi chạy Kaggle.

V1 đổi165 caption,76 tăng/77 giảm CIDEr/12 hòa sau PTB. Ba caption được sửa hallucination lại tăng CIDEr; phần giảm nằm ở các thay đổi khác. V2 kiểm tra cách học/chọn có thể hạn chế đổi câu bất lợi, không giả định phải hy sinh CIDEr để giảm hallucination.

## Thay đổi thực hiện

- Mỗi candidate được so với caption fixed của cùng ảnh. Mô hình nhận chênh representation candidate–fixed, dự đoán **gain CIDEr**, và neo gain của fixed đúng0.
- Học Huber regression trên gain; true gain âm có trọng số2, beta=.1. Đây là preference từ reference, không phải nhãn sai quan hệ đã xác minh. Không nhận reference/target làm input khi dự đoán.
- Giữ3 arm scores/global/regional như V1; khởi tạo head cuối0 nên bắt đầu giữ fixed. Không train lại Q-Former/CLIP.
- Đổi câu theo policy raw nếu gain dự đoán>0; policy guarded chỉ đổi nếu gain>mức hiệu chỉnh. Nếu chưa đủ bằng chứng, giữ fixed. Guard chỉ chọn giữa proposal raw và fixed, không tự tạo lựa chọn thứ ba.
- Inference không cần reference. Ngưỡng không được chọn theo validation hoặc test.

## Tách dữ liệu

Giữ cohort1792 ảnh và cache V1: fit1024 train/restval;256 ảnh tune cũ được chia bằng seed20261015 thành **tune128 chọn epoch** và **calibration128 hiệu chỉnh**, không trùng nhau; validation512 giữ nguyên chỉ để đánh giá sau cùng. Những ảnh calibration từng xuất hiện trong tune V1: đây là tách vai trò trong thử nghiệm mới, **không phải dữ liệu hoàn toàn chưa từng xem**.

V2 dùng IDF từ **fit1024 references** cố định cho target fit/tune/calibration, tránh làm đơn vị gain thay đổi chỉ vì số ảnh từng cohort khác nhau. Thư viện CIDEr gốc tính normalizer bằng số reference groups; helper dùng empty internal pairs để giữ số đó ở1024 khi chấm subset, không đưa các empty pairs vào calibration hoặc DF. Có test khớp scorer chuẩn trên fit corpus và bất biến khi tách subset. Epoch chọn theo proxy train-IDF CIDEr; final validation metrics vẫn dùng scorer chuẩn trên512 ảnh. Không so hai thang điểm như cùng một evaluation.

AdamW lr1e-3, weight_decay1e-4, batch8,10 epochs+epoch0, seed42. Chọn epoch tốt nhất trên tune128 theo policy raw, tie epoch sớm. Calibration không tham gia chọn epoch.

## Hiệu chỉnh quyết định đổi câu

Trên calibration128, mỗi ảnh lấy proposal mô hình chọn, không lấy oracle. Đo `error = predicted_gain − actual_gain_of_that_proposal`. Dùng order statistic `ceil((128+1)*.9)=117`, clamp correction>=0. Sau đó khóa correction trước khi chấm validation.

**Đây không phải bảo đảm90% các caption được đổi sẽ tốt hơn.** Quantile là hiệu chỉnh sai số thực nghiệm, không đảm bảo conditional/selective precision; calibration là train/restval nên có thể lệch phân phối validation. Không thay mức.9 theo kết quả validation.

## So sánh và gate

Xuất fixed và6 policies: scores/global/regional, mỗi model có raw và guarded dùng **cùng checkpoint**. Nhờ vậy raw-vs-guarded kiểm tra riêng tác động quyết định giữ câu. So với V1 có nhiều thay đổi loss/split/architecture nên không quy mọi khác biệt cho guard.

Primary policy đặt trước: `regional_guarded`. Gate mở rộng giữ yêu cầu: CIDEr>=fixed+.005; CHAIRs/i<=fixed; Recall>=fixed−.01; CIDEr hơn guarded scores và guarded global. Không tự mở full, không dùng test. Validation512 đã được phân tích lặp lại: mọi lợi ích ở đây là development evidence, chưa phải xác nhận độc lập.

## Chạy tiếp trong notebook hiện tại

**Thay cell launcher cũ**, tránh `Run All` chạy lại V1 trước V2. Dùng [notebook](CAPTION_GAIN_GUARD_V2_PILOT.ipynb) hoặc [cell đầy đủ](kaggle_caption_gain_guard_v2.py), Internet ON và GPU T4x2.

Cell ưu tiên cache `/kaggle/working/learned_caption_ranker_1024_512`; cũng kiểm đúng một mount path tường minh của output notebook hiện tại. Nếu cache còn đầy đủ, kiểm hashes1792 feature files rồi chỉ học các head. Compact review ZIP không có feature cache và không dùng thay output đầy đủ được.

Nếu cache không còn trong phiên mới hoặc chưa gắn input, cell tái sinh **beams và CLIP features đóng băng**, không train lại các head V1. Hai GPU chia việc tái sinh; các head nhỏ train trên GPU0. Giữ5 dataset cũ. Không dùng Kaggle API token, không quét toàn bộ Input hoặc tự chạy notebook khác.

Output mới `/kaggle/working/caption_gain_guard_v2`; không ghi đè output V1. Gửi **`caption_gain_guard_v2_review.zip`** sau khi `GAIN_PILOT_COMPLETE`. ZIP giữ head nhỏ, targets train/tune/calibration, locked fit-IDF, policies, per-candidate predicted gains và metric/CHAIR; bỏ feature cache lớn và checkpoint Q-Former.

## Kiểm tra local

19 tests PASS: gain fixed=0, fallback khi hòa/chưa đủ gain, gradient regional và padding, quantile từ proposal thực tế chứ không oracle, tune/calibration không trùng, fixed-IDF khớp scorer chuẩn và subset invariant, V1 zero-init/journal/locked-test vẫn qua. CLI help và syntax hợp lệ. Chỉ test local, chưa có caption metric V2 hoặc pilot training GPU.
