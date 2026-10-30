# K/V D/E pilot — endpoint hoàn tất, không đạt gate mở rộng

31/10/2026. [Audit JSON](KV_DE_RESULT.json), [code audit](audit_kv_de_results.py), [spec khóa trước](RELATION_KV_DE_PILOT_SPEC_2026-10-30.md). Hai ZIP chỉ đọc một lần, text cache trong `review/kv_D_endpoint_loaded.json` và `review/kv_E_endpoint_loaded.json` ignored/local. Không chạy lại mô hình/HDF5/checkpoint/scorer local.

## Kết quả chính trên cùng reused dev1000

| Nhánh | CIDEr | Caption có hallucination | Hallucinated mentions / object mentions | Object recall | Mean words |
|---|---:|---:|---:|---:|---:|
| C0 raw, cùng runtime | 1.230741 | 52 | 53 /1646 | .460498 | 10.044 |
| D: independent values | 1.228833 | 60 | 61 /1637 | .459268 | 9.922 |
| E: shared fact values | 1.221030 | 55 | 56 /1630 | .458346 | 9.916 |

**Gate tiếp tục FAIL.** E không có point gain CIDEr so D/C0, không giảm false captions so C0; semantic screen và recall/length guards PASS không bù được các điều kiện đó. Không promote E, không tự chạy full/thêm epochs/seed hoặc tuning để cứu claim.

Paired image bootstrap10000 draws, seed20261030:

| So sánh | ΔCIDEr [95%CI] | ΔCHAIRs [95%CI] | Sửa lỗi / gây lỗi mới | Exact McNemar p |
|---|---|---|---|---:|
| D−C0 | −.001908 [−.016762,+.012857] | +.008 [+.003,+.014] | 0 /8 | .0078125 |
| E−C0 | −.009711 [−.026108,+.006888] | +.003 [−.003,+.009] | 3 /6 | .5078125 |
| E−D | −.007803 [−.016504,+.000553] | −.005 [−.010,−.001] | 5 /0 | .0625 |

E có point signal giảm hallucination so D (5 câu), nhưng vẫn nhiều hơn C0. Percentile bootstrap CHAIRs E−D không chứa0 trong khi exact McNemar p=.0625; báo cả hai, không chọn phép kiểm thuận lợi để gọi thắng. CIDEr CIs chứa0 không phải bằng chứng tương đương/no-effect. Kết luận là cấu hình chưa đạt screen đã chốt, không bác bỏ mọi value-sharing.

## Semantic diagnostic và đường tới decoder

99held facts/98images,16observed same-query groups:

| Điểm đánh giá | Macro family AUROC | Same-query concordance |
|---|---:|---:|
| D trước/sau effective content | .832842 | .895833 |
| E trước projection | .837727 | .895833 |
| E sau effective projection | .824002 | .833333 |
| Actual decoder C0 query likelihood | .737231 | .697917 |
| Actual decoder D | .739221 | .697917 |
| Actual decoder E | .739221 | .697917 |

Điểm effective của E qua operational thresholds .60/.60, nhưng không hơn D về point scores. Không có paired semantic significance claim; CI của các estimates và sparse bootstrap support nằm trong JSON. Hai head/score tốt không chứng minh mọi câu quan hệ đúng hoặc đã có useful caption grounding.

Reader-off trên64dev images giữ adapted decoder: D đổi1caption, E đổi0. Logits vẫn thay đổi: mean KL D≈6.13e−5, E≈3.99e−5; contribution norms/gradients khác0. Vì vậy không gọi branch hoàn toàn bị bỏ qua; chỉ thấy ảnh hưởng lên greedy captions ở panel này hạn chế. D/E khác nhau47/1000captions sau training, nên cũng không phải hai endpoint giống hệt nhau.

Quan sát phù hợp với **khoảng cách giữa auxiliary representation và quyết định sinh câu**. Đây là vấn đề cần giải thích tiếp, chưa xác định nguyên nhân: sparse P3 coverage, supervision/readout mismatch, anchor hoặc optimization chỉ là các giả thuyết. Không chữa tự động bằng tăng loss/epochs, cũng không lấy gradient nonzero hoặc construction invariance làm semantic success. Relation-caption truth/generic direction/same-class instance grounding vẫn NOT_ESTABLISHED/NOT_MEASURED.

## Tính hợp lệ và khôi phục D116

- Cả4workers có ENDPOINT_500_COMPLETE, cùng500updates/globalbatch8, exact4000caption exposures và auxiliary schedules; rank states/frozen tensors/runtime/pooled text/init/data identities khớp. 520metadata/audit assertions PASS, không phải520neural experiments.
- Initial digest cả hai `5746ba6d4f414a51c41b62ce787f60c0e517a29c5ea48a57a04089263855e508`; frozen digest `89c5548adab445f1c73f0fbe5863071ede5b25f3f52e4ab846b7b086cb5082b7`. C0 captions và per-image scores giống hệt giữa hai runtime.
- D worker core1402633, E744264b: model/training/data/preflight source hashes giống; chỉ shared JSON writer và evaluator khác do reporting fix, cùng thêm tests trước GPU. Không đổi scientific recipe. E source `.py` archive khớp Git blobs; source receipts đều kiểm.
- D116 recovery đã hoàn tất: summary_created=true, training_repeated=false, model_loaded=false/GPUfalse. 36text artifacts trong recovery khớp original download hashes, checkpoint hash trong recovery/summary giống nhau. Root không đọc binary checkpoint độc lập; hashes từ user-run recovery/worker và full-source validation là provenance hiện có.
- D checkpoint SHA `599f52a24c4568baa53116db0f39247705967c03199cc834fc8c0df89cdd4f6e`; E `f3e45dc512a766a3e345db3c1fda2a212346918f1e894a8ca1ceddd8203cb1ee`. Giữ để audit, không promote.
- GPU worker elapsed max khoảng323s D/305s E; training rank0 khoảng205s/192s. Đây là worker timing, không tổng wall toàn notebook hoặc quota/billing. Không lấy20h báo trước116 làm ngân sách hiện tại.

Giới hạn: một seed, reused development1000, không independent test; CHAIR chỉ object hallucination. Không trộn rawCIDEr của cohort4000 với1000. Official scorer không rerun local; root kiểm counts/ratios/CI/identities trên artifacts đã lưu.

## Quyết định và phối hợp

Giữ C0 làm reference của lượt này; đóng pilot current instantiation theo gate. Không yêu cầu train lại D do lỗi JSON đã được recovery. Gửi facts qua `RESEARCH_EXCHANGE.md` để Codex2 phản biện đường representation→decoder và đề xuất một quyết định/hypothesis có thể kiểm chứng; không bắt bên kia phải cứu K/V. Chưa có run, nguồn nhãn, loss hoặc budget mới được duyệt.
