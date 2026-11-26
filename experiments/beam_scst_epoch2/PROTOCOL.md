# Beam SCST epoch 2 + TEST — quyết định 26/11/2026

Người dùng chốt chuẩn bị học tiếp epoch 2 và chấm TEST; tài khoản còn 6 giờ GPU. Agent chỉ chuẩn bị và kiểm tra static/metadata/mock, không chạy Kaggle, Torch, mô hình hay scorer local.

## Câu hỏi và cấu hình đã chốt

Học thêm đúng một lượt 113.287 ảnh TRAIN có tiếp tục cải thiện CIDEr không, và CHAIR thay đổi thế nào? Không hứa tăng tiếp từ kết quả epoch 1. Đây là thử tối ưu tiếp cùng mô hình; không suy riêng đóng góp prompt từ lượt này.

- Khởi đầu duy nhất: full `checkpoint.bin` của `ducanh3/notebook7bc83c4c76/199`, SHA256 `5b8d33fc0b26f2c77ad1cee4a58d3e4a718f13107a2d2bd9a0884c427faed88c`. Giữ model, Adam/moments/step và CPU/CUDA/NumPy RNG hai rank. Không dùng weights-only pilot hay output TEST200/benchmark.
- Chuyển rõ từ epoch 1 đã đủ 3.541 update/cursor113.287 sang epoch 2 cursor0, cumulative update3.541. Endpoint epoch2: cumulative7.082 update, 226.574 image events; riêng epoch2 đúng113.287 ảnh không thay thế, thứ tự cache gốc như epoch1. Không shuffle mới, reset Adam hoặc reseed sau khôi phục RNG.
- Giữ beam5/train-mode/mean reward baseline/20-token denominator, pure SCST CIDEr-D TRAIN, FP32/noTF32, T4×2 DDP, local4×accum4/global32, Adam4e-6/betas(.9,.99), không CE/clip gradients/AMP/compile/feature sharing. Dùng đúng hai tối ưu đã vượt guard: batched prompt transfer và deferred finite-logit checks; giữ finite loss/reward/grad guards.
- Reuse cache82/83/84 đã đóng Dataset và metadata/VALID; không trích feature hay copy lại full patch arrays. Giữ nguyên model/core/M3/reward/decoder. Preflight6 update bỏ, sau đó process train riêng khôi phục lại trạng thái gốc. Giữ checkpoint atomic định kỳ100 update và tại deadline, identity epoch/cursor rõ ràng để resume chính xác.

## Đánh giá và giới hạn

Endpoint cố định cuối epoch 2; không chọn checkpoint dựa vào TEST. Chấm VALID5000 top1 bằng đường đã chạy của epoch1 để so đúng cùng seed43. Sau đó TEST5000 sinh beam5 một lần, chấm top1 và rerank cố định 0,3decoder/0,4CLIP/0,3object, đủ BLEU1/4, METEOR, ROUGE-L, CIDEr, SPICE, CHAIRs/i, thêm Recall/NLL. Báo cả hai chính sách dù bất lợi; so với kết quả epoch1 TEST output200 và CE179 cùng seed đã lưu. Không tìm trọng số theo TEST, không tự epoch3.

TEST đã được dự án dùng trong phát triển; ghi rõ đây là đánh giá lặp lại, không gọi là holdout hoàn toàn chưa xem. CHAIR đo lỗi đối tượng, không chứng minh thuộc tính/quan hệ đúng. Một seed SCST chưa xác lập độ ổn định.

Hai namespace source riêng cho train/VALID và TEST để tránh csa_per_caption_test_data bị cache nhầm split. TEST checkpoint seal được tạo chỉ sau khi full epoch2 hoàn tất, dựa trên SHA thực tế export/receipt và kiểm tra lineage epoch2/7082/CE179/parent199. Không dùng SHA epoch1 cố định cho trọng số mới.

Cap mặc định5,75 giờ wall/phiên, không phải ETA hay cách tính GPU quota. Dự phòng100 phút sau train cho VALID/TEST/scorer; epoch2 train dự kiến gần3giờ từ speed sample, không bảo đảm đủ ngân sách. Nếu train hết cap, giữ full state và dừng, không chấm partial checkpoint. Nếu scorer lỗi/hết giờ, giữ caption/beam/CLIPscores/selection/basic metrics và weights để recovery CPU; không train/decode lại tự động. Notebook hỗ trợ resume bằng toàn bộ output epoch2 được gắn và kiểm tra identity/SHA; nếu chỉ có Save Version, cần source mới xác nhận để làm API resume riêng.

## Dữ liệu cần gắn

1. `ducanh3/csa-scst-full-karpathy-cache-v1`
2. `vuthetam/mscoco-2014`
3. `ducanh2403/visual-cache`
4. `ducanh2403/csa-prompt-test-cache-v181`
5. `ducanh2403/objectdetectionecache`

Internet và Secret `codex` hoặc `KAGGLE_API_TOKEN` có quyền tải nguồn ducanh3. Không cần gắn output199 riêng; API tải đúng version/path và SHA trước setup mô hình.

## Đã chuẩn bị và kiểm tra, 26/11/2026

[Notebook](BEAM_SCST_EPOCH2_FROM_V199_VALID_TEST_RERANK_2GPU.ipynb), [source manifest](SOURCE_MANIFEST.json), [9 kiểm tra host](CHECKS.json). Bản sao ở Downloads được xác minh cùng SHA. Các cell và80 payload compile/hash PASS; closure các import runtime được kiểm tra riêng hai namespace. Metadata transition/resume/tail7 và endpoint binding có trường hợp từ chối sai nguồn/partial; AST update giữ nguyên toán học của epoch1 ngoài batch helper đã kiểm chứng. Model/reward/inference/scorer/selector và speed candidate giữ đúng bytes nguồn đã chạy.

Mô phỏng API acquisition chỉ nhận đúng v199 full checkpoint; không gọi API thật. Mô phỏng pipeline dùng file fixture (không phải weights/caption/metric thực): đường thành công tới TEST comparison, partial/timeout train chặn TEST, timeout scorer giữ weights và beam fixture. Không import Torch hoặc metric scorer local. Actual API authorization, load binary/Adam/RNG/CUDA, scorer assets và chất lượng/thời gian toàn lượt chưa được agent chạy.

Kiểm tra Torch/NumPy phiên Kaggle ngay sau CUDA guard, trước setup scorer/cache; không tự cài lại runtime khi khác nguồn v199. Java/scorer smoke chạy trước lượt huấn luyện chính để phát hiện lỗi môi trường sớm. Không có kết quả thực nghiệm mới, không ghi dòng metric mới vào Sheet hoặc sửa báo cáo Word chỉ vì đã chuẩn bị notebook.
