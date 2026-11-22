# TEST5000 baseline cùng decoder — người dùng duyệt22/11/2026

Sau kết quả VALIDv188, người dùng đồng ý chấm TEST. Chỉ chuẩn bị notebook và kiểm tra static/metadata/mock local; người dùng tự chạy Kaggle. Không train, không rerank, không chỉnh beam hoặc chọn lại checkpoint theo TEST.

Khóa `ducanh3/notebook7bc83c4c76/188`, best.bin epoch9, SHA `1eade1fca3b872091b2f2a345a04d8141a887d3fcd1e5772e4952a495c5f4e6b`. Snapshot đã học159399updates/5100723caption events; cursor epoch10/0, lịch sử9epochs, seed42. Không dùng checkpoint.bin cuối epoch10 SHAab38aaf7 hoặc các mô hình bank+prompt/baseline199.

Tái sử dụng cohort TEST5000, TRAIN vocabulary/patch extraction, beam5/max20/raw logprob/no length normalization và scorers của full model seed42/43/44. Source runner/score nền lấy notebook seed43 đã chạy; thay contracts/checkpoint/model bằng baseline cùng decoder. Mô hình dùng chính visual_only_model.py và core sources đã khóa trong lượt train. Hai worker đọc2500ảnh/rank,FP32; lưu cả5beam/ảnh và top1 trước CPU scoring. Reference TEST dùng chấm điểm và teacher-forcedNLL sau sinh câu; không tham gia chọn caption.

Chỉ gắn visual-cache và vuthetam/mscoco-2014. Không cần cache bank/prompt/detector/ảnh crop hoặc checkpoint Dataset mới. Mặc định tải ONLYbest.bin của exactv188 qua pinned KaggleCLI, Secret codex/KAGGLE_API_TOKEN có quyền nguồn; token không ghi log. Có CHECKPOINT_ROOT tùy chọn nếu đã gắn đúng file, vẫn kiểm SHA. Mock API phải kiểm owner/slug/v188 và chỉ một định nghĩa routing. Không fallback latest/versionkhác hoặc tự cài lại Torch.

Stage chỉ5000TESTpatch≈1.4GiB, khôngcopy fullTRAINcache. Cap2h gồm setup/scorers, không ETA/quota; giữ caption/beam/metrics từngphase nếu CPUmetric lỗi. Cácmodebasic/meteor/spice/chair/combine tách biệt, không tự decode lại. Chấm đủ BLEU1/4,METEOR,ROUGE-L,CIDEr,SPICE,CHAIRs/i; bổ sungRecall/NLL. So với fullseed42 top1 TEST đã có, không trộn VALID hoặc rerank.

Kiểm local: cell/sourceAST/hash/import closure, lock/cohort/receipt, accept đúng metadata và reject sai source/epoch/checkpoint/modelshape, effective API routing và preset không training. Không chạy Torch/inference/scorer/API thật. Chỉ báo CUDA, quyềnAPI và Java PASS sau lượt thực tế của người dùng. Không sửa Word hoặc thêm dòng kết quả Sheet khi mới chuẩn bị notebook.

Một seed baseline; full model nhiều tham số hơn. TEST từng được sử dụng trong phát triển lịch sử, không gọi đây là tập chưa từng thấy của toàn bộ dự án. Phép so sánh mới đánh giá phần bổ sung ngữ nghĩa toàn bộ, không chứng minh riêng embedding chữ M3.
