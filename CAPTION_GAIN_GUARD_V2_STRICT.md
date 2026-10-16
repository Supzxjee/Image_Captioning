# Chạy tiếp: gain guard với QFormer gốc đóng băng

Trạng thái: chuẩn bị để người dùng tự chạy; chưa có kết quả GPU. Đây là bước thực thi thiết kế gain guard đã có nhưng chưa chạy, không phải một phương pháp mới vừa được chứng minh vượt baseline.

## Đo đúng tác động của bộ chọn

Dùng QFormer epoch10 gốc để tạo beam5 cố định. Không dùng checkpoint SCST0.5 trong pilot này: generator train/tune/validation phải thống nhất và các input baseline đã có sẵn. Không train lại QFormer hay CLIP. Nếu thành công, việc chuyển bộ chọn sang candidate SCST cần một thử nghiệm riêng với feature cache tương ứng.

Ba nhánh `scores`, `global`, `regional` dự đoán gainCIDEr của từng ứng viên so với caption fixed. Caption fixed có gain0. Loss SmoothL1 trên gain, tăng trọng số lỗi ở ứng viên thật sự kém hơnfixed. Mỗi nhánh xuất cả lựa chọn raw và guarded từ cùng một checkpoint. Guard giữfixed khi proposal chưa vượt mức hiệu chỉnh.

`regional` dùng caption tokens và visual patches, chưa phải bộ kiểm chứng từng bộ ba quan hệ hay từng cặp instance. Không được coi kết quả tốt hơn là bằng chứng quan hệ đúng hoặc đóng góp mới về quan hệ. So sánh với `scores`/`global` để xem nhánh vùng ảnh có ích không.

## Dữ liệu và quyết định được khóa

- Fit1024 ảnh train/restval; tune128 chọn epoch; calibration128 khác tune để khóa ngưỡng; validation512 chỉ đánh giá sau cùng. Cohort1792 ảnh giống pilot trước.
- Target fit/tune/calibration dùng IDF từ fit1024. Không dùng oracle labels của validation1000 vừa chẩn đoán để huấn luyện.
- Mỗihead tối đa10epochs,AdamW1e-3,batch8,seed42; giữ epoch0 tái tạofixed. Calibration90% là quantile sai số proposal trêntrain, **không đảm bảo90% caption đổi sẽ tốt hơn**.
- Validation512 đã từng được xem và calibration từng thuộc tune củaV1. Đây là development evidence, không phải tập hoàn toàn mới.
- Primary `regional_guarded`:CIDEr>=fixed+.005,CHAIRs/i<=fixed,Recall>=fixed-.01,CIDEr>guarded scores/global. Không tự đổi gate theo kết quả hoặc chọn nhánh tốt nhất rồi gọi làprimary.

## Chạy thủ công

Upload `CAPTION_GAIN_GUARD_V2_STRICT.ipynb` vào **QFormer UPT Val5000 20261015**, thay cell launcher cũ, bật Internet + T4x2. Giữ5dataset input hiện có. Đường dẫn đầy đủ ở đầucell để sửa trực tiếp.

Ưu tiên cache đầy đủ `learned_caption_ranker_1024_512` trongworking hoặc mount notebook tường minh có trongcell. Kiểm đủ1792feature hashes trước tái sử dụng; ZIP review không chứa featurecache. Nếu cache không có, tạo lại frozen beams và CLIP features trên2GPU rồi chỉ train3head nhỏ trênGPU0. Không cần tìm versionoutput cũ hay thêmAPItoken. Tạo lại cache có thể tốn thời gian/dung lượng; không phải chỉ đọc ZIP SCST rồi train ngay.

OUTPUT mới `/kaggle/working/caption_gain_guard_v2_strict`; nếu đã có thì dừng để tránh ghi đè. Có strict deterministic runtime cho cáchead và worker tạo feature mới, lưu `runtime.json`; cache cũ giữ runtime khi nó được tạo, không biến thành strict chỉ do được đọc lại. Chưa kiểm chứng strict này trênGPU với pipeline ranker; lỗi operation sẽ được báo, không tự bỏstrict.

Gửi `caption_gain_guard_v2_strict_review.zip` sau `GAIN_PILOT_COMPLETE`. Gói này cóhead nhỏ, train/tune/calibration targets, ngưỡng đã khóa, dự đoán gain và cácmetric/CHAIR. Không đưa frozenCLIP/QFormer hay privatefeatures lớn vàoZIP. Giữ cáccheckpoint SCST trước đó riêng.

## Kiểm tra trước bàn giao

11tests gain/ranker PASS cục bộ:fixed gain0/fallback,gradient vàpadding,calibration theo proposal thực tế,fixed-IDF bất biến khi tách subset,tune/calibrationdisjoint,zero-initialization vàkhông đưa reference vào inference. Syntax và notebook/launcher khớp. Không có notebook được assistant submit; chưa đo chất lượng mới.
