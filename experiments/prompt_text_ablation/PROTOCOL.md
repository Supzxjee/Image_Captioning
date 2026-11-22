# Full model với prompt text bằng 0 — 22/11/2026

## Quyết định đã chốt

Người dùng đồng ý ablation sau matchedbaselineTEST. Triển khai **lượt đầu tiên: full + zero text**, seed42/fresh10epochCE, để so với full prompt đúng ảnh đã có. Nhánh M2-only nằm sau trong kế hoạch, chưa chuẩn bị/chạy cùng lượt này. Người dùng báo25giờGPU cho tài khoản chạy. Không tự chạy notebook hoặc Torch/model/scorer local; chỉ giao notebook local cho người dùng chạyKaggle. Không sửa Word.

## Câu hỏi

Embedding chữ riêng ảnh trong M3 có giúp CIDEr và objecthallucination ngoài bank/M2, adapter, thông tin vùng và hình học không? Giữ nguyên toàn bộ53.183.736 tham số học của fullmodel; bankbuffers/ROI/masks/geometry không đổi. Zero cả object_text và relation_text của M3 trong TRAIN và mọi VALID/inference của nhánh này. Noun/verb semantic bank của M1/M2 **không** đặt0.

Đây là đối chứng nội dung chữ riêng ảnh M3, không phải mô hình không ngữ nghĩa, không phải tắtM3 và không phải M2-only. Zero làm các cột trọng số chỉ nhận đầu vào text0 không có tín hiệu cập nhật tương ứng; cùng số tham số khai báo không có nghĩa cùng dung lượng hoạt động hiệu dụng. Không dùng inference-onlyzero thay cho retrainmatchedcontrol.

## Thiết kế giữ cố định

- Dùng nguyên CSAPatchCaptioner và override duy nhất streams: tạo dict nông mới, zero_like hai tensor text rồi gọi streams gốc. Không thêm/bỏ module; giữ nguyên state keys/RNG constructor và thứ tự thực hiện mọi khối còn lại. Một điểm can thiệp bao phủ forward/encode/VALID và checkpoint inference sau này.
- Cùng constructor seed42, kích thước/rawCLIP196×768→512, decoder3parallel/dropout0.2, vocab10000, global32, image-grouped42+epoch; FP32/Accelerate1.15.0/T4×2, per-caption/no sharing/AMP/compile. Tái sử dụng runner directH5 đã dùng cho fullseed43/44 và baseline mới, pinnedancestor5eb012279e6fd0b5e47a56710ed5e3efa7449d2f.
- TRAIN113287/VALID5000,566747captionevents/epoch,17711updates/epoch,177110updates sau10epochs. Adam(.9,.99),eps1e-8,wd0,clip1; warmup1epoch8e-6→1e-4,cosine9epochs→1e-5. Giữ nguyên lịchVALIDbeam5/max20/no length normalization; best=maxVALIDCIDEr, tie giữ epoch sớm.
- Gắn bốn dataset cũ: visual-cache, vuthetam/mscoco-2014, csa-semantic-banks-v130, csa-prompt-full-karpathy-v175. Kiểmcachev130/v175 nguyên vẹn; không sửa hoặc tái tạo cache. ĐọcH5 trực tiếp, không chép fullpatch sangmemmap, không chạyCLIP/UPT/YOLO lại.
- Không nạp trọng số full/baseline đã học. Chỉ resume output của đúng zero-textmethod/source/seed/data/software, với đầy đủ optimizer/RNG/cursor; không resume checkpoint prompt đúng ảnh. Cap11.5h một invocation, không ETA/billing hoặc tự tăng cap; atomiccheckpoint500updates/phase/epoch, giữbest vàlatest/evaluations/receipts.

## Kiểm tra

Local: compile mọi cell/source, SHA/importclosure/undefinedglobals, mock intervention chỉ đổi hai keytext và không mutateinput, không xóaROI hoặc bank; preservation của train/data/scoring; resume identity khácfull. Không importTorch hoặc dùngautograd local.

Kaggle guard trước train: toàn bộ state tensor/parametercount khớp constructorfull seed42; số tham số53.183.736; text0 khi train/eval, ROIweights/geometry/masks unchanged. Trên **bảncopy** đặt promptoutputprojections thànhidentity để tránh kiểm vô nghĩa lúc adapteroutput ban đầu bằng0: thay text không đổi zero-model memory, zero-model khớp referencefull nhậntext0; referencefull cùngcopyweights nhạy với text. Syntheticguard không phải kết quả chất lượng. Original weights không đổi. Giữ guardDDPfirst32/tail27,causal/beam và64discarded timingsteps; mainworker khởi tạo lại fresh.

## Đánh giá và giới hạn

So fullzero với fullpromptseed42 cùng10epochs, primary trướcSCST/rerank. Đây là mộtseed trênVALID chọncheckpoint; chấmTEST đủ8metrics chỉ sau reviewcheckpoint và userduyệt. Không chọn checkpoint theoCHAIR/TEST. Báo rõ historicalfullseed42 dùnglocalFP16staging trong khi nhánh mớiH5trực tiếp, runtime nondeterminism không bảo đảm bitwise replay. Không tự chạy thêmseed/SCST/M2-only từ kết quảpilot hoặc metadataPASS.

Kỳ vọng cần kiểm chứng, không đặt trước chiều kết quả. Nếu hai nhánh gần nhau hoặc zero tốt hơn, báo đúng: chưa có bằng chứng textM3 giúp ích trong recipe này; không đổi giả thuyết sau khi nhìn kết quả. Không bỏ đối chứng vì CIDEr không thuận lợi. CHAIR chỉ chấm đối tượng, không tự chứng minh quan hệ/thuộc tính đúng.
