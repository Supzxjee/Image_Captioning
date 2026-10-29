# Quyết định nghiên cứu sau I/J v2

28/10/2026. Đọc RESULT.md/json và phần implementation liên quan đến off; không đọc lại ZIP, không chạy neural/GPU. Nguồn kết quả: [RESULT](RESULT.md), commit 8024d1b81a07f026f5ded10905fec7756c4269be. Full-checkpoint/H5 digests vẫn là runtime reports như root đã nêu.

**Chốt: giữ A; dừng mở rộng recipe I/J 500 hiện tại; chấm hết panel 32 cặp đã khóa trước. Chưa chạy seed 2, full train, solver hoặc sweep.** Chỉ khi panel cho thấy lợi ích ngữ nghĩa đáng theo đuổi mới xét một phép inference chẩn đoán, trước khi quyết định sửa recipe.

## 1. Bằng chứng đã có

| So sánh | CIDEr difference; 95% paired CI | Kết luận trong run này |
|---|---|---|
| J−I | +.098336;[+.066233,+.130566] | Shared readout tốt hơn control matched |
| J−A | −.071306;[−.108043,−.036360] | Chất lượng theo CIDEr kém A rõ trên dev này |
| I−A | −.169642;[−.211611,−.130025] | Control cũng suy giảm mạnh |
| J on−off | −.034014;[−.054391,−.014389] | Tắt pair score/value cải thiện CIDEr trong J đã học |

CHAIRs: A 52/1000, I 83, J 61; recall: A .45804, I .46019, J .45435. J cũng chưa thắng A về các số noun-grounding này. CHAIR chưa có pairedCI trong report, nên chênh 9 caption là mô tả, không gọi statistically significant. Câu J ngắn hơn A: 9.370 so 10.213 words; chưa biết là gọn đúng hay né nội dung.

Held whole-tail ranking: I 44/64, J 46/64; predicate-only đều 43/64. Đây là likelihood có điều kiện target, không phải relation correctness của caption tự do; equal accuracy cũng không có nghĩa hai arm đúng cùng cases.

Matched inputs, init, schedule và code cho phép xem J−I là bằng chứng về lựa chọn readout **khi cả hai cùng được huấn luyện theo recipe này**. Nó bao gồm thay đổi gradient/coadaptation, không chỉ đổi attention lúc inference. Một seed và CI trên reuseddev chưa xác nhận robustness theo seed hoặc tập ảnh mới.

## 2. “Recipe chung thất bại” giải thích đến đâu?

**Được nói:** cả hai tổ hợp huấn luyện+reader đã thử đều chưa cải tiến baseline A. J hạn chế mức suy giảm so I. Shared marginals không bị phủ định bởi kết quả này, nhưng mục tiêu cải tiến caption so A chưa đạt.

**Chưa được nói:** SCST/KL sai, relation loss quá mạnh, visual features vô dụng, decoder forgetting là nguyên nhân chính, hoặc chỉ cần freeze decoder sẽ thắng. Chưa có Tune-only dưới cùng objective để tách tác dụng recipe khỏi việc học reader.

Off trong code không phải reader bypass hoàn toàn:

- Nó đặt pair score s=0 và pair values r=0.
- Entity values, priors và output projection vẫn chạy; với J, T khi đó phân rã thành alpha×beta nên entity contexts trở về các priors riêng.
- Hai decoder layer đã học vẫn giữ nguyên. Vì vậy J_off 1.168221 không phải kết quả một A-only finetune và không cô lập tác hại của appearance hay toàn bộ reader.

Reader on cũng không xấu hơn off ở mọi metric: J_on có 61 hallucinated captions so 69 của J_off, dù CIDEr thấp hơn. Do đó cần panel để biết đây là đánh đổi nội dung nào, thay vì chỉ gọi pair branch hoàn toàn gây hại.

Giả thuyết còn mở: adaptation trên 484 ảnh/loss assertion làm lệch ngôn ngữ; residual mới chưa khớp hidden/logit space; shortcut theo geometry/context; hoặc thông tin cặp không hữu ích trong setting này. Đây là giả thuyết, không phải các lỗi đã xác định.

## 3. Bước có giá trị quyết định nhất: chấm panel đã khóa

**Có, cần chấm.** Không tạo panel mới, không chọn cases J thắng và không dùng ranking thay thế. Không cần optimizer hoặc GPU để chấm captions đã xuất.

Giữ 32 pairs/16 mỗi stratum, 64 images, 192 captions A/I/J; trình bày ảnh, assertion q và captions được giấu arm/random thứ tự. Caption trùng nhau có thể chấm một lần rồi gán lại để giảm công. Không tiết lộ ảnh+ hay ảnh− cho người chấm trước khi họ đọc nội dung. Bất đồng/uncertain phải được giữ hoặc adjudicate; nếu chỉ model-assisted thì ghi đúng mức bằng chứng đó.

Cho từng caption, tách hai cột:

1. Có khẳng định q hoặc paraphrase cùng ý không? Tách khẳng định q, khẳng định quan hệ khác, không đề cập và uncertain. Không ép mọi giới từ là một predicate độc lập.
2. Nếu có nói quan hệ về cặp mục tiêu, quan hệ ấy đúng/sai/không xác định theo ảnh? Q đúng trên ảnh+ và sai trên ảnh−, nhưng một mô tả khác q không tự động sai.

Báo RC=P(khẳng định q|ảnh+), FR=P(khẳng định q|ảnh−), Y=RC−FR và coverage nói quan hệ về cặp đích; thêm correct/wrong/uncertain trong phần đã đề cập. Report cả numerator/denominator, paired J−A/J−I, 32 pair-bootstrap theo strata và uncertain bounds. Không dùng Y tăng nhờ im lặng ở ảnh− để che việc RC/coverage giảm.

Đây là phân tích endpoint đã hẹn trước, không phải vòng data audit mới. Không đặt threshold thắng sau khi biết caption panel. Quy tắc dưới đây là quyết định tài nguyên cho bước tiếp, không hồi tố thành preregistered success gate hay tiêu chuẩn accepted.

## 4. Đọc panel và chốt tiếp/dừng

- **Không có lợi ích ngữ nghĩa J so A; hoặc giảm sai chỉ do ít đề cập:** dừng recipe/hướng triển khai hiện tại. Ranking 46/64 và J>I không đủ cứu mục tiêu. Giữ J−I như kết quả tương đối có giới hạn, không quảng bá thành cải tiến A.
- **Panel rất ít quan hệ được nói hoặc uncertainty lớn:** ghi chưa đủ thông tin về relation generation; kết hợp CIDEr đã giảm rõ để dừng đầu tư thêm trong ngân sách hiện tại. Không biến “chưa biết” thành “thành công tiềm ẩn”.
- **J có lợi ích free-caption thật:** một tín hiệu đủ để xét diagnostic là Y_J>Y_A, RC/coverage không giảm so A, FR không tăng, và giữ dấu lợi ích sau xử lý uncertain; báo J−I, effect/CI kể cả CI rộng. Không bắt J thắng I ở mọi metric: J có thể giữ lợi ích ngữ nghĩa của I trong khi ít hại CIDEr hơn. Nếu coverage giảm thì phải ghi tradeoff cụ thể; chưa tự coi Y tăng là pass. Tín hiệu nhỏ bất định vẫn chỉ exploratory. Trường hợp này đáng **một diagnostic inference**, chưa đủ chạy full hoặc claim thắng A. Với CIDEr −.0713, cần trình bày rõ đây là semantic/quality tradeoff, không che bằng một scalar.

Không yêu cầu mọi subgroup phải thắng; báo cả hai strata và không dùng một subgroup thuận lợi để thay kết luận toàn panel. Panel 32 được tái sử dụng, không xác nhận độc lập hoặc seed stability.

## 5. Nếu panel có ích: một inference mới, không train

Chỉ thêm **J_bypass**: dùng nguyên checkpoint J sau 500 updates, nguyên adapted decoder và upstream của nó, nhưng đặt **toàn bộ residual reader output delta bằng0** trước vocab projection. Không chỉ dùng off hiện tại; không gọi teacher/original A để thay decoder, vì khi đó đã đổi câu hỏi.

Decode đúng dev 1.000 và cùng settings, xuất per-imageCIDEr+CHAIR/recall/length, pairedCI của J_bypass−A và J_on−J_bypass. Đây là protocol diagnostic mới sau kết quả, không là heldout confirmatory test. Chỉ cần một condition inference, không Tune 500 updates hay checkpoint/epoch selection.

Diễn giải để chọn việc có đáng sửa:

| Kết quả | Thông tin thu được |
|---|---|
| Bypass gần A, J_on kém bypass rõ | Phần residual đã học làm giảm CIDEr trong decoder hiện tại; chưa có lý do mở thêm train cùng cấu hình |
| Bypass kém A rõ, J_on tốt hơn bypass | Adapted decoder không giữ được baseline khi bỏ nhánh; reader bù được một phần. Đây là dấu hiệu coadaptation/drift, chưa chứng minh objective-alone gây hại |
| Cả bypass<A và J_on<bypass rõ | Cả decoder state và residual đang bất lợi theo CIDEr; dừng recipe, không chỉnh một scalar để tuyên bố đã giải quyết |
| CI rộng | Không gán nguyên nhân chắc chắn; không suy từ “không significant” thành bằng nhau |

“Gần A” cần report point/CI và một practical margin được chốt trước inference nếu muốn kiểm equivalence; CI chứa0 tự nó không chứng minh equivalence. Không cộng/trừ hai CI đã có để tạo CI bypass−A.

Tôi **chưa đề nghị một lần train sửa chữa lúc này**. Nếu panel hữu ích và bypass cho dấu hiệu decoder drift trong khi reader có ích, candidate sửa nhỏ nhất để bàn tiếp là khóa nguyên decoder A rồi học reader dưới objective caption thật. Nó là giả thuyết bảo toàn A, không kết luận từ bypass và không được auto launch. Lịch sử H/L cho thấy freeze có thể làm adapter bị bỏ qua; vì vậy không hứa cách này hiệu quả. Nếu điều kiện ấy không có, không chọn freeze chỉ vì nó dễ làm.

## 6. Ý nghĩa với bài đầu tiên

Hạ tham vọng novelty không làm kết quả kém baseline thành cải tiến. Bảng hiện tại hỗ trợ một quan sát thực nghiệm có ích: coupling đọc thực thể/cặp giảm tổn hại so đọc priors riêng trong recipe đã thử. Chưa có bằng chứng cải thiện A hoặc relation generation.

Sau panel ta có thể viết báo cáo thực nghiệm/negative finding và nêu tradeoff nếu có. Một bài tự nhận “cải tiến caption” cần bằng chứng ích lợi nhiệm vụ có coverage so A; J>I trên một control suy giảm mạnh chưa đủ. Tên venue và khả năng gửi/accepted vẫn chưa xác định.

**Thứ tự chốt cho root:** chấm panel hiện có → report counts/effects/tradeoff → dừng nếu không có lợi ích ngữ nghĩa; nếu có thì chỉ J_bypass inference → quyết định có một giả thuyết sửa đáng kiểm hay kết thúc. Không có lý do hiện tại để chạy seed 2, F hoặc full train.
