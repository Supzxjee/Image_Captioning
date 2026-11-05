# CSA + prompt gắn vùng — matched pilot 500 update

05/11/2026. Tiếp tục phương án người dùng đã đồng ý; Dataset đầu vào xác nhận `/kaggle/input/datasets/ducanh2403/csa-prompt-pilot-inputs-v169`. Không tự chạy notebook/mô hình. Giữ cả bank danh từ và bank động từ bên trong checkpoint CSA; không thay bằng prompt.

## Khóa cấu hình trước khi chạy

- CONTROL và PROMPT đều khởi đầu **best161**, SHA `58ce5dc0e30ac26add9bac8e24a2e9e03b91532bfdeece588997395bf9cfc16b`. Không dùng checkpoint15/147 hoặc pilot prompt trước. Fresh Adam LR5e-6, betas(.9,.99), eps1e-8, không weight decay; FP32, dropoutOFF, grad clip1.
- Cả hai có wrapper giống nhau để không thay RNG khởi tạo. CONTROL tắt residual prompt và đóng băng adapter; PROMPT học CSA + hai adapter, output projections bắt đầu0. Không diễn giải đây là đối chứng bằng số tham số: thí nghiệm kiểm cả gói prompt/visualpair/adapter.
- TRAIN pool10.000 chọn hash `42:COCO_ID`; permutation seed42, lấy4.000 ảnh đầu, 500 update/global8 =4/rank trên T4×2. Luân phiên caption theo sampler cũ; ở lượt này mỗi ảnh thấy một lần. **Không phải full TRAIN epoch**, không phải 30k ảnh. Prompt riêng ảnh lấy từ cache169 đã map167; refs chỉ dùng CE/reward/metrics.
- Loss `CE + .05 SCST(CIDEr-D)`, sampled/greedy cùng policy hiện tại, fixed TRAIN113287-image IDF/5refs như161. Không object reward hoặc loss grounding mới. CE chuẩn hóa global token count, SCST chuẩn hóa global image count; hai rank có độ dài caption khác nhau.
- FullROI134 + dataset COCO JSON đúngSHA + Dataset169 chứa một trọng số161. Không JPEG, UPT inference, CLIP inference, private-output API hoặc Secret trong notebook GPU. Hai tài khoản cần được cấp quyền đọc Dataset chung; không chia sẻ token.
- Cả hai đo ORIGINAL (step0) và END (step500) trên **cùng VALID5000**, beam5/max20, toàn bộ reference NLL, CIDEr/B4/ROUGE/CHAIRs/i/recall/length/counts. Fixed endpoint, không chọn checkpoint sau nhìn VALID; checkpoint100step dùng bảo toàn tiến độ.
- Lưu cả5 beams/ảnh và scores ở mỗi pha. **Phân phối thành hai bước**: notebook này train + rank0 metrics + lưu ứng viên; lượt đánh giá tiếp dùng cùng fixedrerank `.3decoder/.4CLIP/.3object` cho hai nhánh, không đổi weights theo kết quả. Không gọi rank0 là caption đã rerank hoặc khẳng định đã so sau rerank khi chưa có lượt chấm đó. Không mở TEST.
- PROMPT sau END có probe64VALID ảnh cố định: normal / zero text / batch-roll text (giữ slot/box) / adapterOFF, cùng greedy/max20. Báo caption change và teacher NLL/logit difference; zero/roll không phải nhãn sai có kiểm chứng. Probe là sensitivity, không chứng minh quan hệ đúng hoặc lợi ích riêng của ngữ nghĩa text.

## Kiểm tra thực trước train, do người dùng chạy

Kiểm tất cả SHA cache/weight/ROI/COCO; cache318×512/L2/15k IDs và roi_count phải đúng stagedROI, không nhầm index UPT. Tiny CPU fixtures từ169 không thay thế những bước sau:

1. Nạp161 strict; closed adapter và all-empty prompt giữ hai memory và logits trên cùng batch thực. Rollout/rescore dùng tolerance rtol1e-4/atol2e-5 vì khác kernel FP32; ghi maxdiff, không ép beamID giống qua batchsizes.
2. Hai rank actualCUDA/DDP, hai optimizer step smoke trên TRAIN: rank0 giả lập empty slots, rank1 có object+relation; finite/nonzero output projection gradients và upstream gradients sau khi projection mở; đồng bộ parameters/gradients. Các trọng số smoke bị bỏ khi tiến trình kết thúc; worker train nạp lại nguyên161 và reseed, không dùng checkpoint smoke.
3. Đường normal/empty/adapterOFF an toàn padding. DDP chỉ đăng ký tham số trainable của từng arm, không bỏ mask bảo vệ NaN để có gradients.

## Ngân sách và tiêu chí đọc kết quả

Người dùng báo15h, chưa rõ tổng hay mỗi bên. Mỗi notebook có **parent STOPcap3h**, mọi subprocess bị chặn bởi deadline; worker dành thời gian cho VALID và lưu checkpoint, không tự retry/retrain. Hai notebook ×2GPU×3h tối đa12GPU-device-hours theo cấu hình, **không xác nhận cách Kaggle tính quota/billing**; không hứa hoàn tất trong cap. Projection từ25updates thực sẽ dừng/bảo toàn nếu không kịp. Chuẩn bị, smoke và metrics cũng tính trong parentcap.

Đọc END PROMPT so END CONTROL trước, rồi so cả hai ORIGINAL. Báo chênh lệch và CI theo ảnh khi đã có hai ZIP; mục tiêu CIDEr tăng, CHAIRs/i không xấu hơn, recall không giảm quá0.5điểm%. Mức +1CIDEr×100 là practical signal để cân nhắc, không phải ngưỡng p-value hoặc bảo đảm. Chỉ mở rộng sau kiểm cả sensitivity và chất lượng/rerank; nếu không hơn đối chứng, không tự tăng epoch. Chưa có gain/novelty hoặc quan hệ đúng được chứng minh.

## Phát hành trước lượt chạy

Hai notebook `CSA_PROMPT_CONTROL_PILOT500_2GPU.ipynb` / `CSA_PROMPT_PROMPT_PILOT500_2GPU.ipynb`: nguồn `fc54bdea66479e66c8e7662831be167325ffef8e`, 25filepins, mỗi notebook14.7KB. Tất cả3codecell ASTPASS, execution_countnull/outputs rỗng, đúngT4×2/sharedmount/không APIsecret/phaseorder. Năm safe fixture metadata/NumPy/ASTPASS từ chính Gitbundle độc lập, kiểm thiếu cachefile/saiweights/split/ROIcount/schedule/scaling. SHAcoreCSA giữ `139edbb1…`. Xem `PROMPT_PILOT_AUTHOR_CHECKS.json`.

Tác giả chưa chạy CUDA, Torch/autograd, metric scorer hay notebook. Kiểm thực nằm trong notebook người dùng. Chưa có gain, chưa rerank. Nếu gián đoạn: giữ cả folder/checkpoint/optimizer/RNG/cursor, gửi review trước; notebook không tự resume/restart và không tự coi partial500 là hoàn tất. Việc có output SaveVersion không tự chứng minh đủ500update/VALID; trạng thái phải qua worker +score complete.
