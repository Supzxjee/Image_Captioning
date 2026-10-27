# Decoder-side pre-Q readout — ứng viên pilot, chưa train

**Cập nhật27/10:** phương pháp chính đã chuyển sang
[explicit joint phrase–region semantic alignment](../semantic_grounding_method/PROPOSAL.md)
để đáp ứng yêu cầu đề tài. P/Q thuần source-readout ở tài liệu này được giữ làm
đối chiếu thiết kế/ablation nguồn sau này, không còn là pilot chính được đề xuất.

2026-10-27. Astra chọn giữ giả thuyết đọc token trước Q-Former sau khi root phản
biện SD-RPR. Đây là thiết kế để xét thực nghiệm, không có kết quả gain hoặc novelty
đã xác lập. Không triển khai ML/notebook hoặc launch GPU từ attachment Argon.

## Câu hỏi được kiểm

Với A giữ nguyên, cùng một readout phụ thuộc trạng thái decoder, **đọc lại196 token
trước Q có giúp sinh caption tốt hơn đọc lại32 Q sẵn có không?** Đây là so sánh
nguồn token trong một cấu hình/budget. Nó chưa chứng minh Q đã làm mất semantic
information, raw768 tốt hơn projected512 hoặc attention là grounding đúng.

## Module và dữ liệu

Giữ toàn bộ A frozen/eval, original prompt và vocabulary10000. Lấy đúng hidden
state h[B,T,512] mà A đưa vào fc_out. Không gọi h là thuần ngôn ngữ.

P: Z[B,196,512] lấy từ **ReLU(A.vis_projection(V_raw[:,1:]))** ở eval, ngay trước
Q-Former; bỏ CLS. Source này gồm frozen projection+ReLU đúng code A, không chỉ
linear projection, và không bypass768→512. Q: Z[B,32,512] là visual_memory/Q của
chính A. Original memory[prompt;Q] và masks giữ nguyên ở cả hai nhánh.

q=LN0(h)Wq; K=LN0(Z)Wk; V=LN0(Z)Wv; dimensions512→128. Readout softmax(qKᵀ/√128)V,
sau đó Wo128→512. Tất cả4 linear **không bias**, LN không affine:262144 tham số.
Wo=0 lúc đầu, Wq/k/v khởi tạo bình thường, hai nhánh dùng cùng initial tensors.

Logits=fc_out_A(h)+F.linear(readout,fc_out_A.weight,bias=None).
Không cộng fc_out.bias lần thứ hai. Output residual0 khôi phục logitsA tại
khởi tạo với cùng shapes/masks/runtime và tensors hữu hạn; đây không là bảo đảm
caption distribution/CIDEr giữ nguyên sau train.

Không topK, entropy gate, scalar gamma, orthogonal subtraction, MLP chỉ nhận h,
hard lexical foils hoặc nhãn âm suy từ random image/thiếu reference.

## Objective phải được khóa trước implementation

SCST trên phân phối sau residual: sample multinomial temperature1 và greedy của
current student; sequence-sum policy loss, tính EOS, mask tokens sau EOS/PAD.
Reward dùng đúng audited CIDEr-D + object_count_v2 đã pin trong dự án; không dựng
training IDF riêng cho2000 ảnh. Cần kiểm exact corpus/hash/tokenizer hiện có khi
đóng gói, không suy rằng chỉ cần gọi tên CIDEr là cùng reward.

Cộng .1 KL(A || student), mean valid token trên **cùng sampled prefixes**; target
phân phối A và prefix IDs detach. A là anchor frozen. No CE warm-up trong pilot
này. KL giảm áp lực rời A nhưng không bảo đảm preservation; không diễn giải các
regression trước thành bằng chứng CE luôn làm phẳng SCST.

Đây vẫn dùng COCO references khi train qua reward. Inference không có references,
GT boxes, V-COCO hoặc external verifier. Same loss ở P/Q; không lặp confound đổi
patch source đồng thời bỏ auxiliary loss như bản Argon.

## Pilot và tiêu chí

Đề xuất:2000 hash-first ảnh từ locked TRAIN hợp lệ,2epochs,global batch8→500updates,
AdamW1e-4,weight_decay.01,clip1,seed42,end-only. Cùng module/init/data/order/source
hashes/reward/optimizer, chỉ đổi Z. A baseline không train lại. Atlas20 không làm
train set, checkpoint selector hoặc gate; dev1000 đã dùng nhiều lần.

P−A CIDEr≥.01 và paired95%CI lower>0; P−Q CI lower>0. CHAIRs/i không xấu, recall
giảm không quá.005; báo độ dài câu. Chưa đổi sang+.006 theo lý do variance chưa đo.
CHAIR chỉ là object safeguard, không xác nhận predicate-scope semantic correctness.

Nếu P≈Q cùng tăng, chỉ có evidence cho readout/adaptation, không cho riêng bypass.
Nếu P>Q nhưng không hơn A, chưa đạt mục tiêu. No-Go chỉ dừng recipe này, không là
architecture ceiling. Cần đối chiếu prior trước mọi paper/novelty claim.

## Việc cần làm khi được duyệt triển khai

CPU synthetic checks về parameter count/grad/frozen state; user-run GPU parity A
và zero-init logits/greedy cùngshape; đúng source pre-Q ReLU và Q; effective loss
trên sampled distribution; EOS/global-DDP normalization; source/checkpoint cache
hash và train/dev disjointness. Chưa chạy các checks này trên module vì module
chưa được viết. Toy ARGON_MATH_CHECK không thay thế chúng.

Đề xuất cap chung preparation30 phút và mỗi arm train+end45 phút T4×2, đo throughput
sớm để dừng nếu vượt; không phải ETA3–5phút. Chưa xác nhận quota hiện tại. Không
cần dựng ảnh che/crop mới hoặc thêm human-label atlas để qua gate. Giữ nguyên
baseline pipeline, chưa thêm full-scale training/rerank/seed sweep.
