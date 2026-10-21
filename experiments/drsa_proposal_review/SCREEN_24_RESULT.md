# DRSA — sàng lọc 24 ảnh từ caption original đã lưu

**SCREEN COMPLETE — STOP: chưa đủ bốn ca sai quan hệ.** Không chạy model; mọi nhận định mới là assistant screening, chưa là human gold.

## Cách chọn và nguồn

Dùng `origin_predictions.json` của artifact A, 1000 caption; protocol khai báo checkpoint original SHA `5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75`. Arm A đi trực tiếp qua `base.build_memory`, không append auxiliary slots; origin được sinh trước train. Đây là provenance từ artifact/source, không phải tái kiểm độc lập weights trong lượt này.

Regex danh từ ván/điện thoại kết hợp từ quan hệ tạo 52 ứng viên. Khóa thứ tự SHA256 `drsa-screen-v1:<Karpathy imgid>`, lấy đúng 24 đầu trước khi xem ảnh. Không chọn theo reference/CIDEr, không dùng test. Join Karpathy imgid sang COCO ID bằng groups.json; không tạo tên ảnh trực tiếp từ imgid. Chi tiết và hash nằm ở SCREEN_24_SELECTION.json.

Đã xem đủ 24 ảnh qua ba contact sheet; xem ảnh đầy đủ thêm các COCO ID 474741,454509,13031,167271,253341,546229,430670,563882,41247. Ảnh công khai tải từ host COCO, hash từng ảnh giữ local trong image_receipt.json.

## Kết quả

| Phân loại | Số ảnh |
|---|---:|
| Quan hệ cốt lõi phù hợp | 17 |
| Chưa đủ bằng chứng để kết luận sai | 5 |
| Sai loại đối tượng | 1 |
| Ứng viên sai quan hệ rõ | 1 |

Không dùng 17/24 như accuracy: mẫu được chọn có chủ đích và chỉ kiểm quan hệ cốt lõi. Giới tính, số lượng, bối cảnh hoặc các chi tiết khác có thể vẫn sai.

## Hai ca hiện có, chưa khóa oracle

| COCO ID | Caption gốc / đoạn sai | Bằng chứng | Sửa tối thiểu dự kiến | Nội dung có thể giữ |
|---|---|---|---|---|
| 200984 (từ audit trước) | a man is holding a cell phone **to his ear**. | Điện thoại được cầm phía trước mặt, không ở tai. | a man is holding a cell phone **in front of him**. | Người và điện thoại. |
| 253341 (mới) | a boy is **sitting on a skateboard** in the street. | Người ngồi xổm trên nền, ván tách ở phía trước. | a boy is **crouching beside a skateboard** in the street. | Người, ván và nền cứng; street chưa chắc. |

Ở 200984, holding vẫn đúng; looking và holding không loại trừ nhau. Ở 253341, caption sửa giữ nguyên street để là thay đổi tối thiểu, không có nghĩa street đã được xác nhận. Hai bản sửa là đề xuất assistant. Protected content chưa phải protected contrast; chưa khóa actual tokenizer prefix/decision token.

Shortlist basis 8 ảnh cũ chủ yếu đứng/mang ván và điện thoại. Nó chưa trực tiếp bao phủ crouching-beside so với sitting-on; không dùng nguyên shortlist và tuyên bố đã phù hợp.

## Toàn bộ 24 ảnh

| Thứ tự | COCO ID | Caption original | Nhận định |
|---|---:|---|---|
| 1 | 207561 | a group of people riding surfboards on top of a wave. | supported_core_relation: Several people are surfing in the breaking waves. |
| 2 | 365711 | a man holding a surfboard on a beach. | supported_core_relation: Person visibly holds a surfboard on the beach. |
| 3 | 563882 | a group of people sitting on a bench with their phones | supported_core_relation: People are seated; the foreground person holds a phone. Bench versus linked chairs is not adjudicated. |
| 4 | 332087 | a man riding a wave on top of a surfboard. | supported_core_relation: Person is surfing on a wave. |
| 5 | 362005 | a man riding a skateboard up the side of a ramp. | supported_core_relation: Skateboarder performs a trick at the ramp; exact temporal direction is not inferred. |
| 6 | 98801 | a man standing on a beach holding a surfboard. | supported_core_relation: Standing person holds an upright surfboard on the sand. |
| 7 | 474741 | a man in a pink shirt and tie holding a cell phone | uncertain: A phone is not identifiable in the visible hand; absence of a visible phone does not establish a contradicted relation. Pink shirt and tie are visible. |
| 8 | 278967 | a man riding a wave on top of a surfboard. | supported_core_relation: Person and surfboard are in breaking surf; broad surfing description fits. |
| 9 | 240275 | a woman riding a wave on top of a surfboard. | supported_core_relation: Surfer is riding a wave; gender is not independently adjudicated. |
| 10 | 454509 | a man riding a wave on top of a surfboard. | supported_core_relation: Surfer and board are airborne during a wave maneuver; this does not contradict broad riding/surfing. |
| 11 | 13031 | a man riding a wave on top of a surfboard. | object_error: Visible vehicle is a jet ski with a shaped hull and handlebars, not a surfboard. Do not count category confusion as a pure relation error. |
| 12 | 362409 | a man riding a wave on top of a surfboard. | supported_core_relation: Person visibly surfs a wave. |
| 13 | 41247 | a group of people riding surfboards on top of a wave. | uncertain: Distant surfers are visible; some stand and some paddle. Evidence is insufficient to certify the complete group claim false. |
| 14 | 125322 | a man is standing on a bridge holding a cell phone. | supported_core_relation: Standing person holds a phone in front of the body at a railing. |
| 15 | 71877 | a man riding a skateboard on top of a cement block. | supported_core_relation: Person is on a skateboard at the edge of a concrete obstacle, performing a trick. |
| 16 | 167271 | a woman laying in bed while talking on a phone. | uncertain: Phone is held in front of the face while the person reclines. Speakerphone use cannot be ruled out by a still image. |
| 17 | 546229 | a group of young men standing next to each other on skateboards. | uncertain: Group contains people holding boards upright and people with feet on boards. A group-level non-universal reading remains possible; do not count as an unambiguous relation error. |
| 18 | 329498 | a person holding a surfboard in the water. | supported_core_relation: Person carries/holds a surfboard while walking in shallow water. |
| 19 | 430670 | a man riding a skateboard down a flight of stairs. | uncertain: Airborne skateboarder is beside stairs; a still frame does not safely establish temporal direction. |
| 20 | 253341 | a boy is sitting on a skateboard in the street. | relation_error_candidate: Person is crouching on the ground; the skateboard is separate in front of them, with no seated contact on the board. Full image inspected. |
| 21 | 542758 | a group of people standing on a beach with surfboards. | supported_core_relation: People stand on the beach with surfboards; no incompatible relation is asserted. |
| 22 | 386089 | a man in a suit and tie talking on a cell phone | supported_core_relation: Person holds a phone to the ear; broad talking-on-phone description fits visible posture. |
| 23 | 107430 | a man riding a wave on top of a surfboard. | supported_core_relation: Person visibly surfs a breaking wave. |
| 24 | 449402 | a man in a top hat is talking on a cell phone. | supported_core_relation: Person holds a phone against the ear while wearing a tall hat. |

## Điểm dừng và bước tiếp

Đã đạt giới hạn 24, chỉ bổ sung một ca sai quan hệ; tổng với audit trước là hai ứng viên, chưa đủ bốn. Dừng sàng lọc. Không thêm lỗi đối tượng/omission cho đủ số, không tự mở rộng ảnh, không báo gate 3/4 pass/fail khi thiếu dữ liệu.

Đã gửi Astra summary một lần, không yêu cầu đọc lại artifacts; xin chốt công khai phạm vi hai-ca exploratory hoặc tạm dừng cấu hình này. Chưa viết notebook hoặc chạy oracle. Đây là thiếu điều kiện thực nghiệm, chưa phải bằng chứng DRSA thất bại.
