# COCO routing interface smoke — kết quả nhận ngày 22/10/2026

**TECHNICAL PASS trên bốn ảnh TRAIN / hai T4. Chưa có kết quả caption gain hoặc causal grounding.** Root chỉ audit ZIP, không chạy lại model. Người dùng chạy notebook.

## Artifact và tính toàn vẹn

- ZIP: `coco_routing_interface_smoke_review.zip`, 19.337 bytes, 12 files.
- SHA256: `c9dd065a1fb563327896ad5c3f22c18eef3e3f324fb79f5ace7d209e7cc7f466`.
- Source observer, worker và train4 masks khớp cả protocol hash lẫn bundle đã giao trong notebook.
- Model source khớp Git blob tại project pin `6acd3c9f0553d4d12d46684bbe7dcb8c38033d6b`; input receipt đúng checkpoint original `5f8eead3…`, legacy prompt `d7910e9b…`, COCO JSON `2fd999…`.
- Cả hai rank có state hash trước và sau giống hệt nhau: `21c212f904b5dd2419b8f9fddbe50c5e1f4229ca5081e3a4efb17e786c89170b`.
- **0 parameter mới, 0 optimizer updates**. Không CIDEr, CHAIR, dev/test hoặc checkpoint selection.

## Kiểm tra kỹ thuật

| Kiểm tra | Kết quả |
|---|---|
| CPU và CUDA: observer probabilities so với public MHA | PASS cả hai worker, bool và additive padding |
| Memory/mask/logits so với baseline | PASS; max absolute logits difference **0.0** ở hai rank |
| Greedy token IDs | Trùng baseline theo assertion worker |
| Không đổi state/parameter identity | PASS |
| Full/empty masks, padding, Q=patch+CLS mass | PASS |
| Đổi noun/future tokens không đổi dự đoán trước noun | PASS trên các span mẫu |
| Gradient đến decoder/Q-Former/projection/query tokens | Hữu hạn và khác zero |
| CLIP backbone/cache | Backbone không load; cached input không nhận gradient |

Tolerance khóa trước `rtol=1e-5, atol=1e-6`, không thay sau chạy. Bản nhỏ không kèm logits, model/cache hoặc stage tensors; memory/greedy/gradient checks được xác minh qua **worker source khớp + receipt + log**, chưa được root replay độc lập trên tensor.

Gradient norm rank0/rank1: decoder last cross-attention in-proj **2.820823 / 3.696064**; Q-Former last cross-attention **0.805762 / 1.444668**; visual projection **3.075302 / 6.688585**; learned queries **1.201578 / 2.981333**. Đây là gradient của loss định tuyến qua graph quan sát dùng chung tham số, không phải bằng chứng caption đang dùng đúng vùng.

## Raw routing masses tại vị trí dự đoán noun

| COCO ID | Span | Decoder→Q | Q→spatial path | Q→CLS path | Trong bbox union |
|---|---|---:|---:|---:|---:|
| 64627 | man | 0.466712 | 0.464437 | 0.002275 | 0.379216 |
| 529017 | people | 0.602530 | 0.597479 | 0.005050 | 0.066096 |
| 468966 | man | 0.641654 | 0.632061 | 0.009593 | 0.330792 |
| 580507 | surfboard | 0.380487 | 0.370714 | 0.009774 | 0.089039 |

Đây là khối lượng xác suất đường attention đã định nghĩa, **không phải accuracy, confidence hoặc điểm chất lượng caption**. Các union có diện tích và số instance khác nhau. Không dùng bốn số để kết luận noun nào grounded tốt hơn. Legacy mass và CLS mass không bị bỏ rồi renormalize. Root kiểm lại quan hệ mass và tính lại route loss từ bbox supports: rank0 **1.037957**, rank1 **2.567662**, khớp receipt trong tolerance.

## Runtime và chi phí

Torch **2.11.0+cu128**, transformers **4.57.1**, Tesla T4×2; FP32 eval, deterministic strict, math SDP, MHA fastpath/flash/memory-efficient SDP tắt theo runtime. Synthetic tests có cả CPU và GPU.

Hai worker báo khoảng **1,35 giây mỗi rank** cho đoạn đã đo sau setup; peak allocated **0,30549 GiB**, reserved **0,31836 GiB** mỗi GPU. Không tính thời gian clone, dependency, hashing/cache staging, setup model hoặc synthetic checks; **không phải tổng thời gian notebook hoặc upper bound cho training**.

Log có pip dependency warnings với gradio/diffusers không dùng ở smoke; target imports và worker vẫn hoàn tất. Không sửa môi trường ứng dụng không liên quan chỉ để xóa warnings.

## Quyết định

Đạt gate kỹ thuật để xem xét bước dữ liệu đầy đủ: nguồn COCO instances train+val, clean noun-span policy, class-union/patch-size và đối chứng vị trí. Không tự chuyển PASS này thành pilot hoặc claim cải thiện caption. Đã gửi summary cho Astra, không yêu cầu đọc lại raw ZIP.

Rủi ro còn nguyên: token ambiguity, hộp person quá rộng, residual/legacy/CLIP context ngoài attribution, và loss có thể ép bỏ thông tin hữu ích từ legacy/CLS. Việc quan sát đúng xác suất và có gradient chỉ xác nhận có thể tối ưu ràng buộc, chưa xác nhận ràng buộc có ích.

Chi tiết và hash: `SMOKE_RESULT.json`; audit không import torch: `audit_smoke_review.py`.
