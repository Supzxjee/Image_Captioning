# Chuẩn bị pilot màu: sàng lọc30cặp

31/10/2026, theo user đồng ý phát triển3nhánh. [Thiết kế](SEMANTIC_RESPONSE_PILOT_PROTOCOL.md), [nhãn30](SEMANTIC_RESPONSE_COLOR30_REVIEW.json). Chưa training-ready, chưa code/training model/GPU.

## Đã chuẩn bị

- Từ MED SFT pinned38f969ac và VG image metadata đã lưu, lọc ứng viên có ít nhất2từ màu trong mô tả, có VG→COCO mapping vào Karpathy train/restval và không trùng1000dev C0 theoID. Có7319candidate names. Đó là text filter lấy ứng viên, không phải xác nhận truth/màu hoặc randomsample.
- Lấy30VG IDs đầu tiên duy nhất gặp trong tar sau lọc. Dừng tại offset23,621,120bytes (~22.5MiB), không tải cả5GB;30ảnh sửa và30ảnh gốc tải/giải mã đủ. Train indices, source IDs, source paths và60JPEG hashes lưu trong nhãn/manifest.
- C0dev được kiểm tra từcachedmetadata,1000/1000làval; ứng viên30khôngtrùng theoID. Chưa audit gầntrùng ảnh hoặc tất cả cohort lịch sử; không gọi là independent/test-cleared. Các pair chưa chia fit/heldout.
- Xem30cặp ở display tối đa550px/ảnh và224×224bilinear RGB. Không chạy CLIP hay decoder; “nhìn được ở224” là nhận xét trợ lý, không phải model success.
- **3provisional (#1,17,22),8conditional (#4,5,10,12,14,16,18,28),19reject.** #1 trùng feasibility20 #7, nên chỉ có2provisional mới, không cộng thành6cặp sạch độc lập. 0humanconfirmed.
- Provisional: #17 ván đỏ→xanh, Stable người mặc áo vàng; #22 mũ tím/hồng→xanh, Stable mèo đen-trắng; #1 cà vạt xanh→đỏ, Stable nhóm người trước tòa nhà. #22 tên màu cũ cần xác nhận. Các chi tiết/logo trên target nếu bị đổi không thuộc Stable.
- Các conditional có drift shape/identity/phụ kiện; #10sàn và#14cát là target nền, ngoài scope vật ban đầu, không tự mở rộng scope để dùngchúng. Nguồn #10 nói sai màu mèo; nhãn sơ bộ đã sửa sang màu sàn theo ảnh. Nhiều reject chỉ đổi scene/material hoặc mô tả không khớp.
- Các từ trong11candidate fact/contrast sets có trong savednativevocab10000 từcachedC0metadata; không load checkpoint. False alternatives do trợ lý kiểm tra ảnh cả2phía, không suy ra từ absence của reference. Chưa người dùng xác nhận hoặc dùngfit.

## Trang người dùng kiểm tra

Local: `review/med_color_pilot/index.html`, ignored khỏiGitHub cùng ảnh. Ba provisional hiện đầu; các mục khác gập lại. Có ảnhgốc/sửa, thumbnail224, Changedbefore/after, Stabletrue/false, lý do và mô tả nguồn táchriêng. Chọn một trong “xác nhận / cần sửa / loại”, có ghi chú, lưu cục bộ bằnglocalStorage và tải `med_color30_verified_review.json`. Mục không chọn vẫn human_confirmed=false; không tự coi nhãn trợ lý là nhãnngười.

## Kiểm tra kỹ thuật và quyết định

- Metadata assertions:30cocoIDs/30VGIDs duy nhất,30train/restval;60JPEGs decode được, sourcehashes lưu; page image/link refs tồn tại;11factsets khôngOOV.
- `node --check` PASS; mockDOM export test PASS:30records, chỉ mẫu chọnconfirm mới human_confirmed=true, needs_revision vẫnfalse, notesgiữlại. Chưa browserUI automation/actualclick; khôngmodel/CUDA/scorer.
- Chưa xuất notebook huấn luyện vì dữ liệu này chưa đủsize/labels và chưa chốt prompt treatment/mask/quota/schedule. Không lấy30cặp lặp lại nhiềuupdate rồi so với baseline lớn để claimđónggóp.
- Bước tiếp theo: user xem3priority và xuấtreview; sau khi chốt quyước nhãn mới quyết định có đáng mở rộng tập sạch hay không. Không đề nghị chấm hết30mục loại hoặc chạyGPU để bù chất lượngnhãn. Chưa chứng minhnovelty/tăngCIDEr/giảmCHAIR.
- Không ghi thêm sheetrow hiệu quảmôhình; artifactnày là chuẩn bịdata vàthiếtkế, khôngrun thựchiệm.

## Cập nhật xác nhận qua chat — 31/10/2026

Người dùng nói: “tôi xác nhận các nhãn đề xuất của 3 cái đó”, tham chiếu rõ ba priority #17,#22,#1 vừa được gửi. Đã ghi3humanconfirmations trong mainreview và xuất [confirmed3](SEMANTIC_RESPONSE_CONFIRMED3.json); không cần người dùng tải/gửi lạiJSON cho cùng xác nhận này. Changed/Stable/false-alternative texts và60sourceimagehashes giữ nguyên;27mục còn lại không đổi. #22 được xác nhận theo nhãn purple đã đề xuất, không tự chuyển sangpink.

Đây là assistant-proposed labels được user xác nhận tự báo cáo; không phải người chấm độc lập/blind. Có3VGsourceIDs duy nhất;#1vẫn là mẫu đã xuất hiện ởfeasibility20, không tăng sốpair độc lập. 3mẫu là ví dụ phát triển quyước nhãn, chưa đủfit/heldout, không có kết quảmôhình mới. Những dòng0humanconfirmed phía trên ghi trạng thái trước cập nhật này, không được dùng làm trạng thái hiện tại.

Metadata checksPASS:chỉ3confirmations,27records khác giữ nguyên, mọi label/sourceID/imagehash giữ nguyên, confirmedsubset3uniqueIDs, training_ready=false. Chưamodel/GPU; điều kiện trước notebook tại pilotprotocol vẫn áp dụng.
