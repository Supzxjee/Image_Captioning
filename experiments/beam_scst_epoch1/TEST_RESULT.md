# Beam SCST v199 — TEST5000 + rerank, output v200

Ngày kiểm tra26/11/2026. Người dùng xác nhận output `ducanh3/notebook7bc83c4c76/200`; trọng số vẫn là endpoint v199, SHA`3b03acc29f025b07dc358846b30618ae9c83f8634bc12bf7905e9d908b971f24`.

**SCST tăng rõ CIDEr; rerank giảm phần CHAIR tăng sau SCST. So với CE + rerank cùng seed43, CIDEr cao hơn5,8147 điểm; CHAIRs cao hơn0,02 điểm phần trăm.** Không coi chênh lệch nhỏ này là bằng chứng tương đương hay không kém hơn.

| Thang đo | CE top-1 | CE + rerank | SCST top-1 | SCST + rerank |
|---|---:|---:|---:|---:|
| BLEU_1 | 77.7612 | 77.9099 | 80.1978 | **80.8697** |
| BLEU_4 | 37.7903 | 37.4171 | **40.2639** | 39.6176 |
| METEOR | 28.8134 | **29.4109** | 28.8231 | 29.1738 |
| ROUGE_L | 57.9128 | 58.2496 | **59.1989** | 58.9878 |
| CIDEr | 121.9843 | 124.3639 | 129.6063 | **130.1785** |
| SPICE | 21.8257 | 22.5706 | 22.6560 | **23.2493** |
| CHAIRs | 4.0600 | **3.6600** | 5.0600 | 3.6800 |
| CHAIRi | 2.8035 | **2.4509** | 3.3435 | 2.4847 |
| Recall | 43.7127 | 44.9320 | 46.6023 | **46.8754** |
| NLL | **2.1781** | **2.1781** | 2.9789 | 2.9789 |

Caption metrics nhân100; CHAIR/Recall tính phần trăm. NLL không nhân100. NLL đo cùng frozen captioner nên không đổi theo chính sách chọn câu.

## Diễn giải

SCST top-1 so với CE top-1: CIDEr121,9843→129,6063 (+7,6220); CHAIRs4,06→5,06% (+1,00 điểm%,203→253caption), CHAIRi2,8035→3,3435%. Điều này tái hiện sự đánh đổi đã thấy trên VALID, không chứng minh SCST tự giảm ảo giác.

Rerank trên cùng checkpoint SCST: CIDEr129,6063→130,1785 (+0,5722); CHAIRs5,06→3,68% (−1,38 điểm%,giảm27,27%tương đối), CHAIRi3,3435→2,4847% (−0,8588 điểm%). SPICE22,6560→23,2493 và METEOR28,8231→29,1738 tăng, nhưng BLEU4 40,2639→39,6176 và ROUGE59,1989→58,9878 giảm; không phải mọi chỉ số đều cải thiện.

So sánh đồng chính sách: CE+rerank124,3639CIDEr/3,66%CHAIRs/2,4509%CHAIRi; SCST+rerank130,1785/3,68%/2,4847%. CIDEr tăng5,8147, nhưng METEOR giảm0,2371; CHAIRs tăng0,02điểm% (183→184caption), CHAIRi tăng0,0338điểm%. Nhìn theo điểm ước lượng, đây là ứng viên chất lượng cao với CHAIR gần mốc CE+rerank; chưa kết luận tương đương, có ý nghĩa thống kê hay lợi ích riêng prompt.

Rerank đổi hạng3.096/5.000ảnh (61,92%); histogram hạng0/1/2/3/4=1904/1322/937/639/198. Có102ảnh hết cờ CHAIR,33ảnh có cờ mới, giảm ròng69. Đây là cờ tự động, chưa được người chấm xác nhận. Caption trung bình8,534→8,9746từ; không có caption≤2từ.

## Nguồn và kiểm tra

Full seed43 bắt đầu CE v179bestepoch8; beam-SCST một113287TRAINepoch/3541updates ởv199. Cùng checkpointv199 sinh25.000beam TEST một lần; chọn top1 và rerank cố địnhdecoder0,3/CLIP0,4/object0,3 với normalization/threshold đã khóa. Không dùng caption tham chiếu để chọn câu, không tuningTEST. Caption và scores ởv200; mốc CEtop1 từv181 và CE+rerank từ`ducanh3/notebook7bc83c4c76/183`.

Đã đối chiếu90ZIPfiles/37sourceSHA, endpointSHAreceipt, TEST5000đúngID và táchVALID, refs byte-identical CE, CLIPmodel/tokenizer/revision và selector khớpCE. Kiểm tra25.000beam/25.000CLIPpairs, score/worker hashes, top1khớpbeam0 và replay cả5.000lựa chọn bằng hàm rule-based SHA-verified. Tính lạiCIDErmean từscore từngảnh, CHAIRcounts từcaption gốc, NLLtừhaiworker; METEORartifactđủ5000giátrị/scalarkhớp; basicBLEU/ROUGEartifacts khớp kết quả. SPICEaggregate có trong kết quả hoàn tất, không cóper-imageSPICE trongZIP đểtính lại.

Top1 CHAIR253caption/262nhắcảo/7836nhắcđối tượng; rerank184/195/7848; referenceunique15746. CLIPmodel/JPEGbinary và weightSHA là receipt runtime, không tải/lặp mô hình local. Worker decode tối đa213.42s,CLIPworker94.62s; không phải tổng notebook/quota.

Agent không chạy Torch/mô hình/Java/CIDEr/CHAIRscorer local; chỉ đọc cachetext và tổng hợp/replay lựa chọn từ điểm đã lưu. No formal bootstrap/equivalence test run. Một seed và TEST đã được dự án dùng trong phát triển; chưa quy đóng góp riêng cho prompt hay beam baseline, vì recipeSCST thay nhiều yếu tố. CHAIR chỉ đo đối tượng, không suy rộng sang thuộc tính/hành động/quan hệ.

**Khuyến nghị:** giữ SCST+rerank như ứng viên tăng chất lượng, CE+rerank như mốc ít ảo giác. Không tự động học epoch2, retune rerank theo TEST hoặc đổi kiến trúc từ kết quả này.

Sheet kết quả: A462:E481; đã ghi/đọc lại100ô. Native Google Sheets phần chú thích đã xem; screenshot phần đầu bảng không chụp được, đã kiểm tra metadata định dạng và giá trị.
