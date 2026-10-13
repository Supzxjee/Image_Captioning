# Prompt đối tượng–quan hệ chọn lọc: prototype offline đầu tiên

Module captioning/selected_relation_prompts.py nhận scored detector pairs,
không nhận GT labels/annotation IDs/caption references. Entity ID tạo từ filename,
class và detector box để giữ binding từng instance. Subject/object text prompts và
structured relations lưu riêng; aggregated prompt chỉ phục vụ xem trước, chưa encode.

Policy giữ thresholds R50, nhận score>=threshold,cho phép nhiềurelation cùngpair,
dedup theo instance pair+predicate,max3relations/image. Khi không có relation,
dùng object-only fallback. Score-threshold margin chỉ làpriority deterministic
cho cap,không calibrated confidence giữarelationclasses. Không retune từ cohortR55.

## Lượt CPU đã chạy

Nguồn R55predictions,364GT-referenced matched detector pairs trên286ảnh.
Object inventory cũng giới hạn trongcặp đã chấm,không mọiobject của detector.

- 44ảnh cóobject-relation prompt,242ảnh object-only.
- 58claims giữ lại:43knownpositive,6knownnegative,9unknown.
- Cap loại2claims; không dùnglabels để lựa chọn chúng.
- Các lớp carrying/talking_on vẫnabstain dofrozen thresholds.

Nhãn chỉ đọc ở bước audit sau selection,ghi CSV review riêng. Không xóa6knownnegative
để giảvờ prompt đúng;9unknown cần kiểm tra. Đây chưa làfull inference prompt cache,
chưa captionquality hoặc noveltyproof.4selector tests riêng PASS; các
matching/residual tests của pipeline trước vẫn giữ, không diễn giải thànhmodelquality.

Xem trước cóclaim đángnghi nhưperson sitting_on pizza cùngperson eating pizza.
Không tự loại claim đó sau khi xem nhãn rồi báo như cải thiện trênevaluation độc lập.
Đây làđộng lực kiểmtra candidatepool/semanticcompatibility trong thiết kế tiếp.

## Artifacts

selected_prompt_cache.json:objects/instanceIDs/boxes/objectprompts/relations/scores/
thresholds/relationprompt/objectfallback theo filename.
selected_claim_review.csv:58claim,nhãn audit và cột human review trống.
prompt_summary.json:config,counts,checksum nguồn vàcheckpoint.

```python
import subprocess, sys
subprocess.run([sys.executable, 'build_selected_relation_prompt_prototype.py',
    '--predictions', 'PATH_TO/residual_detector_boxes_500_evaluation/predictions.json',
    '--evaluation-summary', 'PATH_TO/residual_detector_boxes_500_evaluation/frozen_evaluation_summary.json',
    '--output', 'PATH_TO/selected_relation_prompt_prototype_v1', '--max-relations', '3'], check=True)
```

Selector tests:GTlabels/IDs đổi không đổi output;fallback vàthresholdboundary;
dedup/multiple relations/cap;hai instance cùngclass không gộp.

## Trước khi nối caption model

Cần mở rộng candidate pool sang tất cả cặp person-object của detector trên nhóm ảnh
nhỏ,lưu toàn bộ detectorobject inventory và audit false/unknown acceptedclaims.
Tập matched hiện tại chỉ kiểm tra nhánh scorer/serializer,không chứng minh inference
fullpool tốt. Sau đó encode từngobject/relation prompt riêng theo instance để tránh
ghép nhiềurelation rồicắt token làm mất thôngtin. Chưa encodeCLIP hoặc sửaDecoder
trong lượt này,không dùngcacheprototype chofullcaptiontraining.
