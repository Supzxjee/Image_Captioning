# Hai nhánh CSA từ đầu — full Karpathy, 10 epoch XE

Người dùng chốt06/11/2026:10epoch, báo15giờGPU khi hỏi mỗi bên. Không chạy notebook/model thay người dùng.

| Notebook | Cơ chế |
|---|---|
| [CONTROL](CSA_PROMPT_SCRATCH_CONTROL_XE10_2GPU.ipynb) | CSA với bank danh từ/động từ; adapter prompt bị khóa và bypass |
| [PROMPT](CSA_PROMPT_SCRATCH_PROMPT_XE10_2GPU.ipynb) | Cùng CSA, thêm adapter nhận CLIPtext + ROI đối tượng; quan hệ nhận text + ROI chủ thể/đích + geometry rồi cộng residual vào hai memory |

Mỗi notebook T4×2, InternetON, SaveVersion→Save&RunAll. Có thể chạy song song, không lấy outputCONTROL để khởi tạoPROMPT. Gắn cùng3Dataset: `csa-full-roi-karpathy-v134`, `csa-semantic-banks-v130`, `csa-prompt-full-karpathy-v175`. Path175 đã được người dùng cung cấp; notebook vẫn kiểm14fileSHA/fullcohort/shape/norm và liên kếtROI trước train. Cần chia sẻ quyền Dataset175 cho Ngân nếu DatasetPrivate. Không cần APISecret/checkpointcaption/ảnhJPEG/JSONCOCO riêng.

CSA vàdecoder khởi tạo mới commonseed42. CLIP/UPT/semanticbanks dùng cache frozen. Không lấy trọng số captioner đã huấn luyện. So `common_init_sha256` của hai kết quả; source/config/data/seed phải khớp để quy kết chênh lệch cho prompt.

Mỗi epoch113287TRAIN/566747captionevents, global32,17711updates (batch cuối27chia16/11).10epoch177110updates/5667470events. Cùng grouped sampler/image-feature sharing trên từng rank, AdamLR warmup3epoch8e−6→1e−4/betas(.9,.99)/FP32/clip1. Sharing thay tương quan encoderdropout giữa các caption của cùng ảnh; hai nhánh cùng dùng sharing, CSA10cũ chỉ tham khảo.

Adapter zero-output ởstep0; outputprojection học từupdate1, upstream nhận gradient khi projection mở. Ảnh không có prompt map được dùng residual0, không phải nhãn âm. Prompt có quan hệ trên28190TRAIN/1214VALID, không phải mọi ảnh. Không có guarantee prompt giúp CIDEr/CHAIR.

## Kiểm tra và lưu kết quả

22safe NumPy/AST/mock fixturesPASS từ đúng50Gitblobs nguồn8420162; hai notebook18,8KB/3codecells/emptyoutputs/matchedsetup-only-ARM/CLIflagsPASS. Model/adapter/oldfullworker không thay đổi. Không tác giả chạy Torch, CUDA, DDP hoặc scorer. Neuralfixtures3adapter+2reader và4DDPguards(closed/opened ×mixed32/tail27) chạy trên Kaggle trướctrain; weights kiểm tra bị bỏ, worker chính khởi tạo lại. Tolerance loss5e−5; gradientmaxabs3e−5/relativeL2.005, không đòi bit equality qua batchshape.

Mỗi epoch đủ5000VALID, originalper-imagebeam5/max20/rawlogscore; lưu CIDEr/NLL, captions và SHA của captions. CHAIR/BLEU/ROUGE chấm từ caption đã lưu sautrain; chưa có TEST/SCST/rerank. Chọn best theoVALIDCIDEr/tieearlier; giữ cả endpoint vàbest để so cùnghorizon. Sau kết quảXE sẽ kiểm textzero/roll/off theo strata và quyết định SCST, không coi sensitivity là precision quan hệ.

ParentSTOPcap10,5giờ là giới hạn dừng, không phải ETA hoặc bảo đảm10epoch hoàn tất. Latest checkpoint chứamodel/Adam/cursor/RNG hai rank mỗi1000update/trước&sauVALID; best riêng. Giữ toàn bộoutputfolder, gửi ZIPreview kể cả khi dừng. Khi resume: mount toàn bộoutput cũ, điền `RESUME_ROOT`, cùngarm/notebook/source/software trênoutput mới. Khôi phục và kiểmSHA các captionsVALID cũ để không mất kết quảbest nếu nó thuộc phiên trước; không tự restart hoặc dùng checkpointarmkhác. Dataset/mapping không cần tạo lại.

Output: `/kaggle/working/csa_prompt_scratch_CONTROL_xe10` và `...PROMPT_xe10`. [Author checks](PROMPT_SCRATCH_XE10_AUTHOR_CHECKS.json). Hai notebook chưa được tác giả chạy; chưa có điểm caption mới.
