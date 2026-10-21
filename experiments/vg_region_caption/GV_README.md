# GV — chỉ đưa gradient phrase phụ vào bộ đọc ảnh

Một lượt pilot mới theo quyết định Astra sau khi đủ A/G/T. G có CIDEr1.208400647, A1.205512599, T1.185746927; G−A chưa qua gate, G−T có CI dương. **Giả thuyết decoder bị ảnh hưởng xấu bởi văn phong/nhiễu VG chưa được chứng minh.**

## Chạy

Import `VG_REGION_CAPTION_GV_2GPU.ipynb`, chọn **T4×2**, Internet ON, **Save & Run All**. Chỉ chạy một notebook GV; không chạy lại A/G/T.

Giữ nguyên sáu đầu vào và đường dẫn ở cell đầu như G:

- Q-Former baseline: `qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth` (**checkpoint đầu của G**, không dùng checkpoint G đã train).
- Legacy `prompt-cache/prompt_clip_tokens_cache.pt`.
- CLIP `visual-cache`.
- COCO `mscoco-2014/dataset_coco.json`.
- Object `objectdetectionecache/objectdetectioncache.json`.
- Hidden dataset phẳng `/kaggle/input/datasets/ducanh2403/upt-hidden-pairs-caption11k`.

VG payload và lịch10.000phrase/3.307ảnh đã nhúng, không cần dataset VG riêng. Dữ liệu,mask,seed42,globalbatch8,1250updates,LR5e-6,AdamW,dropout policy,caption/evaluator đều như G. Stage vẫn như G/T; **chưa tối ưu hoặc cam kết rút ngắn I/O**. T đã mất nhiều thời gian chép cache dù huấn luyện chỉ8.31phút; thời gian phiên mới phụ thuộc storage/runtime.

Đầu ra: `/kaggle/working/vg_region_caption_GV_review.zip` gửi lại; giữ riêng `/kaggle/working/vg_region_caption_GV/checkpoint.bin`.

## Thay đổi duy nhất trong học

| Gradient | Visual projection + Q-Former | Decoder |
|---|---|---|
| Caption CE/SCST | Như A/G | Như A/G |
| `.1 * phrase_CE` | Có | Không |

Nhiệm vụ chính vẫn CE trong200updates đầu, rồi CE+.05SCST(CIDErD+.5object_count_v2). PhraseCE gồm EOS, mean trên token nonPAD mỗi rank như G, xuất hiện từ đầu.

`visual_only_model.py` dùng `torch.func.functional_call` với **tham số decoder hiện tại detached**, strict parameter/buffer matching. Không copy decoder cố định, không đưa phrase vào visual encoder, không no_grad toàn decoder và không tắt requires_grad của caption decoder. Autograd vẫn đi qua các phép tính decoder về visual memory. Không thêm tham số. Tham khảo API: https://docs.pytorch.org/docs/2.11/generated/torch.func.functional_call.html .

Caption inference như A/G, không dùng bbox VG. Decoder vẫn có thể thay đổi gián tiếp qua nhiệm vụ caption khi visual memory thay đổi; không gọi đây là bảo vệ hoàn toàn decoder hoặc sửa sạch nhãn VG.

## Kiểm tra trước train trong notebook

- Cùng giá trị phrase loss và gradient visual như G ở cùng trọng số/đầu vào, với cả decoder synthetic nhỏ và `CaptionDecoder` thực (khởi tạo ngẫu nhiên, không download pretrained).
- Phrase-only backward tạo gradient projection/Q-Former, **decoder.grad phải None**, giữ nguyên identity/flags/value tham số.
- Combined CE hoặc CE+SCST+.1phrase cho gradient decoder khớp caption-only; decoder vẫn có gradient caption.
- Auxiliary dùng đúng giá trị decoder sau khi tham số thay đổi, không snapshot cũ.
- Hai GPU, một backward/update, thử CE và CE+SCST, tham số hai rank đồng bộ sau optimizer step; decoder là `CaptionDecoder` thực kích thước nhỏ.

Tests này **chưa chạy local**, vì máy không có torch/h5py và user tự chạy notebook. Trên local chỉ kiểm tra artifact/syntax/hash: source/data gốc của G giữ nguyên byte-for-byte trong bundle GV, chỉ thêm bốn source GV; notebook736.675bytes, unexecuted, dưới1MB.

## Gate khóa trước

- GV−A CIDEr≥.01, paired bootstrap95% lower>0, GV>original, CHAIRs/i không tăng, recall giảm tối đa.01 soA.
- Muốn quy lợi ích cho gradient routing cần GV−G paired95% lower>0. Nếu chỉ GV>A mà GV−G bất định, chưa kết luận cơ chế.
- Sau gate mới review mù nhỏ để kiểm tra nội dung ảnh; CHAIR không chấm quan hệ.
- Nếu không vượt A hoặc tăng factual errors: dừng nhánh VG region-CE ở cache/quy mô này. Không tự thử freeze khác,trọng số khác,seed khác để cứu kết quả.
- Development đã dùng nhiều lần, một seed và weak VG vẫn là giới hạn; chưa phải kết quả test độc lập/novelty.
