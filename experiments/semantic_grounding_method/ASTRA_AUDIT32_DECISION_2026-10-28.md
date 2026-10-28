# Audit32 — chấp nhận cho exploratory pilot sau cập nhật trực tiếp từ human

28/10/2026, Asia/Bangkok. Quyết định hẹp theo human-audit receipt root chuyển; không đọc lại ảnh/raw/ZIP, không code model hoặc launch GPU.

## Quyết định

**AUDIT_GATE_ACCEPTED_EXPLORATORY.** Human đã sửa trực tiếp#16/#22 thành FALSE/contradicted, thay thế đánh giá uncertain trước đó. Effective audit:32/32confirmed,16supported/16contradicted,0uncertain,0definite source disagreement và0box issue. Có thể giữ recipe và frozen V2 cohort để tiếp tục preparation. Đây là chấp nhận scientific data-audit gate cho pilot, **không phải training approval**; CPU cache-copy receipt và ML preflight vẫn pending.

Nguồn quyết định: original human export SHA `0daa710272b2a403889f8c3eecbd34d5da09d25e2533c7b0561fe5d8db3f1d09` giữ nguyên. Supplement gồm#12explicit confirmation và#16/#22assertion revisions theo chat user “16 22 false nhe”, SHA `dbd59a82e62355ce9baa00917ca3b0d4ca43ff2aa62139c06e1182ad737e1f3b`. Root báo5provenance/override-guard tests pass. Không sửa raw export hoặc VSR manifest để ghi đè lịch sử review.

#32COCO104510 do human đổi AI-supported thành contradicted, khớp released FALSE; human judgment vẫn ưu tiên hơn AI suggestion. Review là AI-assisted/self-reported source audit, không independent gold hoặc bằng chứng 100% label precision cho corpus.

Hai revisions mới nhất đều khớp nhãn FALSE đã có trong source:

-#16,TRAIN,COCO497579,person next to horse khi đang riding.
-#22,HELD,COCO324483,suitcase below cat khi mèo ở trong vali.

Đây là explicit human adjudication, không phải assistant tự suy uncertain thành FALSE. Không có thay đổi nhãn training/evaluation trong manifest.

## Primary giữ nguyên; sensitivity theo uncertain không còn áp dụng

1. Giữ manifest V2 SHA `6ca6af23d0a31043fecdf33398f5b60098be07cbeea9fb91cbfec4b314c2cfa7`,256train/64held pairs, nhãn VSR và mọi primary gate. Không relabel/drop/replace examples, không chỉnh quota/seed vì audit.
2. Primary semantic result dùng toàn bộ64held pairs, cùng32/32strata và paired comparisons/CI đã khóa.
3. Đề xuất sensitivity loại pair chứa#22 vì uncertain trước đó được **rút lại** sau explicit human revision, trước mọi learned model scores. Không tạo eval63pairs hoặc train255pairs vì hai ca này. Các sensitivity/ablation cơ chế khác trong protocol không bị ảnh hưởng.
4. Sample32 audit qua completeness/consistency review; chưa chứng minh toàn cohort sạch hoặc phương pháp có gain. Scope pilot/data claims giữ đúng giới hạn review.

## Bước tiếp theo

Chờ CPU cache-copy receipt xác minh source SHA,612image-ID bindings, feature/prompt coverage và schema. Reuse original caches khi verify; không cần mới hóa CLIP hoặc thay prompt bằng GT/VSR text. Root có thể tiếp tục preparation/implementation theo phạm vi user đã duyệt; launch/training vẫn là bước riêng, chưa được quyết định này authorize.

Audit này không yêu cầu brainstorm, re-split hoặc sửa phương pháp. Tính mới, decoder use và caption improvement vẫn phải qua các thí nghiệm đã định; data-audit acceptance không phải method-success evidence.
