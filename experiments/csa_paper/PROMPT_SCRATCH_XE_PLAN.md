# Đưa prompt vào từ đầu giai đoạn XE

05/11/2026. Người dùng đồng ý chuyển sang huấn luyện captioner từ đầu, sau khi pilot fine-tune161 cho chênh CIDEr +0,2251điểm có CI chứa0 và zero/roll/off không đổi0/64caption. Đây là giả thuyết về cách học, chưa chứng minh fine-tune là nguyên nhân prompt yếu hoặc scratch sẽ tăng điểm. Thay ưu tiên rerank172/56 bằng xây dựng đối chứng scratch; giữ nguyên các kết quả cũ.

## Đã chốt

- Hai nhánh: CSA đối chứng và CSA + object/relation prompt gắn vùng; giữ nguyên bank danh từ/động từ, backbone/cache CLIP và nguồn UPT. Không sao chép trọng số captioner/CSA/decoder/optimizer từ161,15epoch,CONTROL172 hayPROMPT56.
- CSA aggregation/alignment/fusion/decoder khởi tạo mới cùng seed42, cùng tensor đầu của phần chung giữa hai nhánh. Bank CLIP lấy từ semantic-bank Dataset130, kiểm SHA/shape/order; không phải trọng số captioner đã học.
- Prompt lấy từ chính ảnh, giữ liên kết instance/ROI và S/O đã map. Caption tham chiếu chỉ dùng XE và đánh giá. Stage XE chỉ lossCrossEntropy; chưa SCST hoặc objectreward. SCST+CE là bước sau khi có kết quả XE và người dùng chốt ngân sách.
- Có prompt từ lượt XE đầu: tính token/adapter ngay khi train bắt đầu. Giữ zero-output initialization đã kiểm để đầu ra step0 của hai nhánh tương đương; output projection nhận gradient ở update1, upstreamMLP bắt đầu nhận gradient khi projection mở. Không nhầm điều này với việc đợi CSA học xong rồi mới gắn adapter.
- Dùng haiGPU mỗi notebook và cùng feature sharing theo ảnh: encode mỗi ảnh một lần trong batch, gather memory cho các caption của ảnh đó. Phải kiểm cùng kết quả eval/gradient khi dropoutOFF và nói rõ feature sharing tạo tương quan dropout giữa caption khi train. Hai nhánh dùng cùng cơ chế để so sánh.
- Không tự chạy notebook/model/Torch/autograd/scorers. Người dùng tự chạy; tác giả kiểm metadata/NumPy/static và chuẩn bị user-run neural/DDP guards.

## Hai thông tin đang chờ người dùng

**Đã trả lời:** người dùng chọn **full Karpathy, chuẩn bị cache prompt đầy đủ trước**, và báo còn15giờGPU khi hỏi hai tài khoản; chưa dùng số này để tự chốt full10/15epoch hay billing. Phần lựa chọn bên dưới giữ làm lịch sử trước trả lời; pilot10k không phải hướng đang triển khai.

1. Quy mô: fullKarpathyTRAIN113287 (chuẩn bị thêm prompt toàn bộTRAIN trước), hay pilot cùngTRAIN10k trong5epoch trước. **Cache169 chỉ có TRAIN10k + VALID5k**, không được lặng lẽ coi phần113287 còn thiếu là no_objects hoặc dùng prompt của ảnh khác. Chưa chọn horizon full10/15epoch thay người dùng.
2. GiờGPU hiện còn của từng tài khoản. Không dùng15h trước172/56 làm quota hiện tại; không tự cộng thành30h hoặc launchfulltrain.

## Thứ tự thực hiện sau khi chốt quy mô

1. Nếu full: CPU map/cacheprompt cho113287TRAIN+5000VALID bằng đúngUPT/detector/ROI134, khôngTEST; dùng lại318vector169 khi exacttext/model/assetSHA tương thích, mã hóa chỉ text mới trênCPU nếu có. Bank từ vẫn giữ riêng. Reviewcoverage/receipt rồi lưuDataset.
2. Nếu pilot: dùng nguyêncache169, subsetTRAIN10k đúngcohort; mọicaption/reference của10k này được học qua từngepoch, không gọi500update làepoch. FullVALID5000. Không dùng kết quả nhỏ để so thẳng số fullCSA15.
3. Hai notebook từ cùng freshinit, XE learningrate/warmup phù hợp sốstep thực, cùngglobalbatch32/epoch/sampler/optimizer/gradclip, validation beam5/max20 cùnghorizon. Chốt sốepoch/caps trước userrun; không tự khẳng định đã hội tụ.
4. Ghi initdigest/RNG, exposure/tokens/TrainNLL/ValidNLL/CIDEr/CHAIR/recall; kiểm finitegradients, bankfrozen, no-prompt/closedparity, DDP scaling và feature-sharing trước train. Smokeweights bị bỏ trước freshinit chính.
5. Kiểm textzero/roll/adapteroff trên tập cố định có phân tầng object-only/relation/empty để khắc phục64sortedIDprobe cũ; không coi sensitivity là relationaccuracy. Chỉ sau đó quyết định SCST hoặc mở rộng.

Trạng thái hiện tại: **chuẩn bị CPU fullpromptcache118287ảnh trước**, tái sử dụng318vector169, chỉ encode text mới, không captionweights. FullXEepoch/cap sẽ chốt sau cache review và đối chiếu quota; chưa có notebook scratch train phát hành hoặc modelrun do tác giả. Không ghi điểm caption mới lên Sheet.

## Notebook CPU đã chuẩn bị

[CSA_PROMPT_FULL_KARPATHY_CACHE_CPU.ipynb](CSA_PROMPT_FULL_KARPATHY_CACHE_CPU.ipynb), source `e294c5c5fb2faed7e4d359636cd2e2f218a3b58e`/27filepins,14.7KB. AcceleratorNone/InternetON; gắn4Datasets: ROI134, UPTpromptfull-all-actions-v2, objectdetectionecache, sharedprompt169. Không cầnbank130/ảnhJPEG/COCOJSON/APISecret lúcchuẩnbị; không deserialize captionweights trong Dataset169.

19safe metadata/NumPy/static/mock fixturesPASS từ đúng27Gitblobs tải trong notebook;3codecellsAST/emptyoutputs/CPU/scopefull/inspect-before-conditional-CLIP PASS. Trong fixture có dòng “FULL…COMPLETE” của dữ liệu mô phỏng; **không phải actualfullcache/modelrun**. FrozenCLIP thực chỉ do người dùng chạy nếu cótextmới; kiểm8assetSHA,FP32/L2/no-truncation. 318vectors không mã hóa lại; kiểm nguyên15koldrecords. Nếu chưa đủ113287TRAIN/5000VALID hoặc thiếufile, không đánh dấuCOMPLETE.

Giữ toàn bộ `/kaggle/working/csa_prompt_full_karpathy_cache_v1`, gửi reviewZIP cùng tên `_review.zip`. Root có9payload trong `cache_receipt.json`; thư mục con `mapping` giữ chẩn đoán nên khi lưuDataset ở bước sau chỉ cần copy đúngcontract, không chọn vài file thủ công. Cap4hCPU là giới hạn dừng, không phảiETA/quota; actualcoverage/newtextcount/elapsed chờ outputngườidùng. Chưa chốt2notebookXEepoch10/15 hoặc SCST.
