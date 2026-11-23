# Ablation chữ prompt M3: zero-text đã đủ10epochCE

23/11/2026. Người dùng xác nhận output **ducanh3/notebook7bc83c4c76/193**, nối từv192 bằngAPI. [Số liệu/audit](FULL_XE10_RESULT.json), [protocol trước chạy](PROTOCOL.md), [run1partial](RUN1_RESULT.md). Chưa sửaWord hoặc chạy mô hình/scorerlocal.

## Thiết kế và kết quả chính

Hai arm giữ full53.183.736tham số học, bankM1/M2, adapterM3, ROI, geometry/masks. Zero-text chỉ đặt object_text/relation_text củaM3 bằng0 trongtrain+eval; bankkhôngzero. Cả hai cóseed42 và lịch10CEepochs/global32/grouped/percaption/FP32/Accelerate. ChọnmaxVALIDCIDEr theoquytắcđãkhóa: zero best8, promptđúngảnh best9. Đây không phải môhình M2-only hoặc tắtadapter.

| Thang đo VALID5000 | Zero-text M3 | Prompt đúng ảnh | Prompt − zero |
|---|---:|---:|---:|
| BLEU-1 | 77.7949 | 77.9212 | +0.1263 |
| BLEU-4 | 37.8795 | 37.9421 | +0.0626 |
| ROUGE-L | 57.7151 | 57.8256 | +0.1106 |
| CIDEr | 120.3044 | 120.2955 | -0.0089 |
| CHAIRs (%) | 4.9600 | 4.0000 | -0.9600 |
| CHAIRi (%) | 3.4957 | 2.7915 | -0.7042 |
| Recall (%) | 42.4724 | 42.6984 | +0.2260 |
| NLL | 2.163914 | 2.164207 | +0.000293 |

**CIDEr điểm quan sát gần như bằng nhau:**120.3044 so120.2955, prompt−zero−0.0089. Không mô tả kết quả này là prompttăngCIDEr, cũng không gọi là kiểmđịnhtươngđương/noninferiority.

Cóprompt: CHAIRs4.96→4.00%, giảm0.96pp/tươngđối19.35%; captionảo248→200, giảm ròng48. CHAIRi3.4957→2.7915%, giảm0.7042pp/tươngđối20.14%; lượt nhắcđối tượngảo254→203. Tổngobjectmentions7266→7272, recalledunique6765→6801/cùng15928referenceunique,Recall42.4724→42.6984%. Meanwords9.2242→9.2932. Mức giảm ảo giác quan sát không đi kèm giảm tổng lượt nhắcđối tượng, recall hoặc độdàicâu.

METEOR/SPICE chưa đo trong lượtVALIDnày. Nhánhzero-text chưaTEST. Không so VALIDzero với TESTfull hoặc lấy các kết quảSCST/rerank thaychoablationCE.

## Kiểm tra cùng endpoint10

Đểkhôngchỉdựa vào hai epochbest8/9, giữ cảpoint epoch10 đãscored: zeroCIDEr119.2262/CHAIRs4.42%/CHAIRi3.1511%; full119.6139/3.86%/2.7537%. Prompt−zeroCIDEr+0.3877,CHAIRs−0.56pp,CHAIRi−0.3974pp. Chiều giảmCHAIR còn ởcùngepoch10; mức giảm nhỏ hơn tại cặpbest. Đây là kiểmtra bổsung đãcó từsavedendmetrics, không đổi tiêu chí chọnbest.

## Resume và audit

- Đúngparentv192latest1d7008c2…/best57d1c0b2…,48transportmetadataseals. Identity và41sourcepayload giữ nguyên. Prefix1420dòngtraininglog và7VALIDhistory giữ nguyên; logmới bắtđầu141477 (saucheckpoint141433),LRtiếp tụcđến1e-5,khôngrestartwarmup. Tổng177110updates/5667470events,epoch11cursor0/history10.
+- Bestmới8SHA0aa81a0c0168c64987b77e0c86203fa5c4f7ccb858de806ed8842130ce37e1e0; final10SHA80998038f14c8f4781336a96f8ccd4833ea23774b4567d83dead3e2479a286bb. Không thayfinal10chobest8đểlấyCHAIRtốthơn. Giữentireoutputv193.
+- ZIPCRC/SHA137files;source/hash/identity/evaluationseals/trainlogđạt.50000captionVALID/10cohort5000IDs, CIDErper-imagemeans/NLLweighted/meanlength táitổnghợpđạt.10000CHAIRcaption best8/end10 khớptext vàmọi sốđếm.
+- Runtimezero-textguard53.18M/182statetensors/init/ROI/geometry/DDP/beam/positivecontrolcopy giữ nguyên vàđạt. Guard làsyntheticpreflight, không là bằngchứng chất lượngcaption. Actualsource/runtimeuser-run hoànthành; binary không nằm trongreview nên khôngload/hash lại local.
+- Resume worker9885.29s≈2h45, chỉ phiênresume gồmtrain+VALID, chưa gồmsetup/scoring/phiênrun1, không phảiGPUbilling. Không chạyTorch/autograd/inference/scorerlocal.

## Diễn giải và giới hạn

Kết quả mộtseed trênVALID **ủnghộ lợi ích giảmobjecthallucination của thôngtin chữ M3 ngoài bank+ROI+geometry/adapter**. Mục tiêu duytrì chấtlượngcaption vàgiảmảo giác phùhợp với chiều sốliệu, nhưngchưaxácnhận trênTEST hoặc nhiềuseed. Chưađủđểkhẳngđịnh promptluôntăngCIDEr, formalnoninferiority, hoặc các mô tảhànhđộng/quan hệđúng; CHAIRchỉchấmđối tượng.

Cùng sốtham sốkhai báo/cấu trúc, nhưngzero-inputlàm các cộttextkhông hoạtđộng; không gọi làsameeffectivecapacity. Khởi tạo so vớiconstructorfull táidựng, fullseed42lịch sử dùnglocalFP16patchstage cònzerođọcH5trực tiếp; khôngbitwiseclaim hoặc bỏqua biếnthiêntraining. Hai bestepochkhácnhau đượcchọntheocùngquytắc; endpoint10 kiểmthêm vàkhôngthaytiêuchíchọn.

## Trạng thái

Sheet Kết quả mới!A323:D340 đãghi/đọclại đúng,formatmetadatachecked,nativevisual chưa xem. Ghi mốc v192partial tronglịch sử, v193 làcompletedresult.

Khuyếnnghị tiếp: chấmTEST5000cho chínhzero-textbest8v193 bằng cùng TESTcache/scorers và vẫnzeroM3text; chưađượcchuẩnbị/chạy tronglầnnhậnZIP này. M2-only đểsau, không tự mởrộngtraining/seed/SCST. Cóthểviết ablationVALIDnhưkếtquảbanđầu, giữgiớihạn rõràng.
