# Mở rộng nhãn màu —40source mới

31/10/2026. User đã đồng ý mở rộng sau xác nhận3mẫu. Phạm vi hoàn tất lượt này: data-only preparation, khôngmodel/Torch/GPU/Kaggle/training. [Protocol/amendment](SEMANTIC_RESPONSE_PILOT_PROTOCOL.md), [40reviewrecords](SEMANTIC_RESPONSE_EXPANSION40_REVIEW.json), [pool15](SEMANTIC_RESPONSE_LABEL_POOL.json).

## Lấy mẫu và provenance

- MED SFT pinned revision `38f969ac98d7db856ee8d57b1a23c4032cfff09e`, metadata SHA256 `95fc6968cb6c3425c809beab085ebae6c3afea40ed017318edad3336e8641442` đã cache; không tải/đọc lại ZIP kết quả mô hình.
- Textfilter mới được đặt saucolor30: đúng2basic-colorwords, <=55words, có target noun, không dấu hiệu scene/sky/background/additional/size/count/material. Từ7319candidate names còn1113; đã loại mọi sourceVG đã xem trước đó. Filter không phải nhãn truth và không đủ đảm bảo màu target đúng.
- Lấy40sourceVGids duy nhất đầu tiên gặp trongarchive saufilter; dừng tại offset190,710,272bytes (~181.9MiB), dưới192MiB đã đặt. Không tải cảarchive5GB. Tải đủ40edited+40originalJPEG. Không đổi mẫu sau khi nhìn chất lượng hoặc theo kết quả model.
- Mọi source cóVG→COCOmapping train/restval, loạival/test/unknown vàC0dev1000 theoID. 40source không trùng các packetcũ; combinedpool15uniqueVG/cocoIDs. Chưa kiểm gầntrùng ảnh hoặc mọi eval lịch sử. BaseC0 đã huấn luyệnCOCO nên không gọi các source này “modelchưa từngthấy”. Chưa chiafit/heldout.

## Review và pool hiện tại

Xem40cặp ởdisplay tốiđa550px/ảnh và224×224bilinear RGB (thumbnails theo hình học resize native, khôngmodel encode). Kết quả assistantreview:

| Trạng thái mới | Số | ID |
|---|---:|---|
| Provisional, đề xuất nhãn đầy đủ |12|E02,E05,E09,E17,E18,E19,E21,E30,E33,E34,E37,E39|
| Conditional, chưa đưa vào pool |7|E01,E03,E07,E23,E35,E36,E40|
| Reject, chưa đưa vào pool |21|CácIDcòn lại|

Mỗi provisional có Changedbefore/after, Stabletrue và distractorStablefalse được tôi xét sai trêncả2ảnh. Toàn bộ từ trong12factsets có trongsavednativevocab10000; không đọc checkpoint/model. Mọi candidate vẫn human_confirmed=false.

Ưu tiên Stable cùng thuộc tính nhưng khác vật, thay vì chỉ câu tổngquát:

- E02: bus đỏ→xanh, taxi cạnh bus vẫnđỏ.
- E18: áo vàng→xanh, mũ vẫnđỏ.
- E30: xe tải xám→đỏ, đồnghồđỗxe vẫn xám.
- E33: laptop đen→trắng, áo người vẫnđen.
- E34: áo trắng→xanh, mũ vẫnđen.

Đây là ví dụ nhãn có thể kiểm tra ảnh hưởngchéo; chưa bằng chứng model đã học selectivity/grounding. Vài edit còn đổi texture/logo hoặc khuônmặt chútít; các chi tiết đó không được gánStable, và người dùng vẫn có thểloại cặp.

Nguồn có sai cả target/chiềuđổi: E04chữ tàu vốnđỏ; E19áo vốnđãcóchữ; E23suit vốnđãxanh, thứ đổi làcàvạt; E35thực tếáođỏ→xanh, ngược mô tảnguồn. Không tạo nhãn bằng cách chép sourcecaption.

Combinedpool hiện có **15sourcepairs =3user-confirmed trước đó +12assistant-only mới**. Không cộng7conditional/21reject hoặc mẫu cũtrùng. Nhãn user-confirmed cũ giữ nguyên cả wording/provenance; không tự gọi15làhumangold. Không đủ dữ liệu để chốt runtraining3nhánh hoặc hiệu quảbài báo.

## Artifact và kiểm tra

Localpage: `review/med_color_expansion/index.html` chỉ hiển thị12provisional mới, không yêu cầu xemlại3đãxácnhận hay40mẫu loại/conditional. Có ảnhgốc/sửa, thumbnail224,4nhãn mỗi cặp, nguồn táchriêng, lựa chọnconfirm/needs_revision/reject và ghi chú. Lưu nhập localStorage; xuất `semantic_response_expansion_verified_review.json`. CácIDmới cóprefixE để không nhầm #17cũ vớiE17mới. Payload xuất giữ3xácnhận cũ nguyênvẹn; mục mới không chọn không được tínhconfirmed.

Kiểm traPASS:80JPEGdecode;40sourceIDs vàpool15unique;12factsets khôngOOV;72image/linkreferences trongpage tồn tại; JavaScriptsyntax; mockDOMexport15records giữ nguyên3confirmationcũ, chỉ mẫu mới chọnconfirm mới được cộng, needs_revision/notesgiữđúng. Chưa browserUI automation hoặc neural/model test. Ảnh/rawmetadata/page ởreviewignored; chỉpushprotocol/labels/pool/report.

**Bước tiếp theo:** người dùng kiểm12cặp mới và gửiexport hoặc xác nhận qua chat cóIDrõ. Không xin xác nhận lại3mẫu cũ. Sau đó mới quyết định có đủ nhãn và đáng đầu tư mở rộng/train hay không; prompt-condition, trainablemask, quota vàschedule chưa chốt. Chưaxuất/chạy notebook; novelty vẫnUNKNOWN. Không tạo sheetrow hiệuquảmôhình từdatapreparation này.

## Xác nhận12cặp qua chat —31/10/2026

User đã được giải thích rõ confirm/needs_revision/reject và nói: “tất cả đều xác nhận 4 nhãn đề xuất”. Áp dụng cho12provisional đang hiển thị ởpage expansion (E02,E05,E09,E17,E18,E19,E21,E30,E33,E34,E37,E39), không áp dụng cho7conditional/21reject hoặc27mục khác củapacketcolor30.

Đã cập nhật mainexpansionreview human_confirmed_count=12 vàcombinedpool human_confirmed_count=15, assistant_only_count=0; [snapshot15](SEMANTIC_RESPONSE_CONFIRMED15.json). Các từ/ngữ, hướngmàu,ID vàimagehash không đổi;3confirmationcũ giữnguyên provenance. Đây là assistant-origin + human_self_reported confirmation, không phải15nhãn chấm độc lập/blind. Không cần export/uploadJSON để lặp lại xác nhận qua chat này.

Kiểm tra metadataPASS:chínhxác12recordmới đổi status,28recordexpansionkhác giữnguyên,3recordcũ giữnguyên,15uniqueVG/cocoIDs vàlabels/hashes không đổi. Tranglocal đã báo15confirmed và khóa các lựa chọn cũ để không xuất ngược thànhchưa xácnhận. 15cặp có thể làm reference cho kiểm tra dữ liệu/cơ chế bước đầu; chưa đủ để tuyên bố trainpilot cóýnghĩathốngkê hoặc tăngCIDEr/giảmCHAIR. training_ready vẫnfalse; điều kiện prompt/mask/split/near-duplicate/quota/schedule chưa tự coi đã xong. Khôngmodel/GPU/notebook.
