# Brief cho một lượt Astra Ultra — chưa gửi

Ngày2026-10-22. Người dùng chỉ có thể dùng1–2lượtUltra. Tài liệu này tổng hợp facts đã root kiểm toán để bạn quyết định nghiên cứu trong một lượt; **không cần đọc lại ZIP/repo/Sheet để xác minh lại**. Chỉ tra cứu tài liệu nghiên cứu gốc khi cần cho một lập luận cụ thể, và nói rõ có đọc fulltext hay chỉ abstract. Không dùng lượt thứhai chỉ để diễn đạt lại proposal.

## Yêu cầu nghiên cứu

Đề tài: căn chỉnh ngữ nghĩa–thị giác cho imagecaptioning, trước đây định hướng prompt đối tượng–quan hệ. Người dùng cần cải thiện so baseline mạnh và cơ chế/đóng góp căn chỉnh có thể kiểm chứng. Không bó buộc vào CSA/UPT hoặc gán mỗi từ vào một bbox. Bạn có quyền bác bỏ các quyết định trước đây của chính bạn; giải thích bằng chứng khiến đổi quan điểm. Root thực hiện và phản biện, người dùng chạy Kaggle.

Trong **một lượt**, hãy:

1. Chẩn đoán nút thắt khả dĩ nhất; phân biệt điều được chứng minh, giả thuyết còn mở, thông tin chưa có. Đừng gán mọi thất bại cho decoder bỏ qua prompt nếu chưa có phép thử xác nhận.
2. So sánh tối đa3hướng khác nhau thực chất, gồm cơ chế tác động lên caption, supervision/nguồn dữ liệu và rủi ro. Không bắt buộc tạo3hướng nếu bằng chứng chỉ đủ choít hơn.
3. Chọn một hướng ưu tiên hoặc đề xuất phép đo quyết định trước khi chọn kiến trúc. Có quyền dừng CSA hay giữ lại; không lựa chọn chỉ vì vừa triển khai xong.
4. Với hướng chọn, xác định input/output/module/gradient và công thức loss đủ triển khai, positives/negatives/labelcoverage và hành vi trên ảnh mới. Nếu chưa có dữ liệu cho loss, nói rõ, không dùng pseudo-label như chân lý.
5. Đưa phép thử nhỏ nhất có thể bác bỏ giả thuyết, đối chứng cần thiết, gate tiếp tục/dừng, và cách diễn giải kết quả âm. Nếu thiếu một fact then chốt, nêu phép đo lấy fact đó thay vì mở hàng loạt nhánh hoặc giả định có dữ liệu.
6. Tự phản biện phương án chọn. Tách kế thừa/engineering khỏi phần có khả năng mới; không tuyên bố novelty nếu chưa kiểm tra priorart. Không cần viết code.

## Pipeline và nguồn lực hiện có

- CLIPViT-B/16 frozen,last_hidden_state197×768 (CLS+196patches),cacheFP16đọcFP32. Ảnh resize224vuông bilinear,khôngcentercrop.
- Projection768→512→QFormer2layers,32learnedqueries512dim. Baseline không dùng prompt-conditionedQFormer.
- LegacyCLIPtextpromptcache giữ nguyên: promptprojection và prompt→visualattention+gate tạo groundedlegacytokens. Decoder đọc `[grounded_legacy;32queries]` để sinh caption. Đây không đồng nghĩa từng từ có box verified.
- Checkpointoriginal đã qua SCST+CE+objectreward ở giai đoạn trước, không phải model CE yếu. Các pilot phía dưới bắt đầu từ cùng checkpoint đó; A là đối chứng continuation mạnh trên cùng train/dev. Không trộn các số greedydev này với beam5/test5000 lịch sử.
- A:train10k,globalbatch8,T4×2,seed42,1250updates;200CE rồi CE+.05SCST;rewardCIDErD+.5object_count_v2;fullTRAIN IDF. AdamWLR5e-6,clipglobalnorm1,evalmode nhưng gradients bật.
- Object reward dùng TRAINCOCO instance+reference union, salientuniqueobjectrecall và penaltyunsupportedcount; detectorhighconfidence có thể neutralizeunannotatedclasses,không tạo positiveGT. Captionrefs dùng khi train và khi eval, không có ở inference.
- Có UPT fullCOCO predictioncache và promptcache allactions; có hiddenpaircachetrain10k/dev1000, tối đa8slots1024dim,cặp người→vật với detection>.2,khôngverbthreshold. Train6423/10000cópair,dev654/1000cópair. Đây là dự đoán, không GT; VCOCO coverage chỉ một phầnCOCO. Quan hệ đa nhãn, unannotated không tự là negative.
- Có VGregionphraseannotations,trong audit có weak/ambiguous/wrongregionphrases; không GT sạch. Các patchCLIP đã contextualized; regionmask không cô lập hoàn toàn ngữ cảnh. Crop/region224 có thể mất chi tiết phân biệtquan hệ.
- T4×2 một notebook; đôi khi có hai tài khoản/notebook, không mặc định nhiềuGPU hoặc ngân sáchgánnhãn lớn. Quota còn lại hiện chưa biết. Không tự chạy GPU/notebook local.

## Bằng chứng đã kiểm toán trên cùng dev1000

| Arm | CIDEr | CHAIRs | CHAIRi | Recall |
|---|---:|---:|---:|---:|
| Original |1.189065508 |.058 |.035452323 |.456809099 |
| A continuation |1.205512599 |.052 |.032475490 |.458038733 |
| H: thêm UPT hiddenpair memory |1.201123876 |Không cần cho lập luận hiện tại |— |— |
| G: VG regionphrase auxiliaryCE |1.208400647 |.053 |.033067973 |.456194282 |
| T: cùng phrase nhưng allpatches |1.185746927 |.052 |.032635468 |.455272057 |
| GV: G nhưng aux không gradient trực tiếp vào decoderparams |1.193220147 |.053 |.033027523 |.458653551 |
| CSA hai pha bật |1.199997684 |.051 |.031116534 |.459268368 |
| CSA hai pha tắt sau train |1.212947915 |.048 |.029625151 |.462649862 |

- H−AΔ−.004388723,paired95%CI[−.023219654,+.014609599]. Off H1.202134333; tháo memory làm48/1000caption đổi,swap làm36/1000đổi. Nonzerogradient không chứng minh grounding.
- G−AΔ+.002888048,CI[−.017606965,+.023640698]. G−TΔ+.022653720,CI[+.001459794,+.044562693]. G>T không chứng minh G>A; có thể giảm hại từ phrase không localize.
- GV−AΔ−.012292452,CI[−.032403839,+.007695362];GV−GΔ−.015180500,CI[−.035034832,+.003657362]. Decoder vẫn cập nhật captionCE/SCST, aux functional_call detached decoderparams vẫn truyền gradient về visualmemory. Globalclipactive2500/2500ranksteps nên có tác động gián tiếp lên scale cập nhật decoder. Đã dừng recipeVGregionCE này.
- Các diagnosticpair/phrase trước có nhiều kết quả âm; không dựa vào việc overfit vài mẫu để khẳng định caption tổng thể tốt hơn. Không có số định lượng bổ sung đã xác minh trong brief này để suy rộng.

## CSA cụ thể đã thử — tránh nhầm với tái hiện fullpaper

Nguồn *Cascade Semantic Prompt Alignment Network for Image Captioning*, IEEE TCSVT2024,DOI10.1109/TCSVT.2023.3343520. OfficialCrossmodalGroup/CSA-Net commit0cc570ba7c50ef08c46a21439fe73ef0a5df97ab.

- GlobalRegionCLIPnounbank4764×1024/verb975×1024,đóngbăng. Mỗi bank learnedSFA thành150prototype512dim.
- Giữ nguyên authorcore2objectstages+1relationstage: similarity/pooling,salienceweight,SA/CA/FFN/scalargatedfusion;23,161,644params. Source rawdot/temps.1/.05,khác diễn giải cosine trongpaper; không âm thầm sửa math.
- Input32QFormerlatentqueries512 thay150detectorregions gốc. Vo/Vr32tokensmỗiloại,decoderhiệncó đọc `[M_A;Vo;Vr]`;không dùng originalparallelCSAdecoder. Authorrel_mask sumfeatures==1 được giữ đểparity,không dùng padding;64tokenthêm đềuvalid.
- Smoke xác nhận engineeringparity trên cùng authorforward:outputmaxabs0,N32/150,93paramgradtensor checks;GPUbackwardbankfrozenbypasscorrect. Không haiimplementation độc lập, không chứngminh grounding.
- Pha1:1250CE,baselinefrozen,coreLRwarmup100→1e-4. Pha2:1250joint,base5e-6/core2e-5,CE200rồi objectiveA. Freshoptimizer/RNG/sampler giữa pha;bankfrozen. Nhiều hơn A1250core-onlyupdates.
- Audit12source,phase1baselinebấtbiến,phase1final=phase2initial,phase2joint đúng lịch,2ranksstatehash khớp,fullfinite. Commonruntime/inputgroups/traintargets/originalpredictions/per-imageCIDEr/CHAIR/evalreferencehash khớpA. HistoricalA không lưuactualsamplerorder/RNG; chỉ so đượcpolicyseed/source.
- Stagingtensorlayoutkhác do bỏ UPT không dùng;ZIPkhôngchứa checkpoint/stagetensors để rootrehash độc lập. Runtime đãhashcheck. Không có căn cứ lỗi chạy hiện tại để giải thích sạch mọi kết quả âm.

CSA comparisons:pairedbootstrap20kimagesresampling,seed42:

| So sánh | ΔCIDEr | CI95% |
|---|---:|---|
| On−A |−.005514915 |[−.024690383,+.013728802] |
| Off−A |+.007435316 |[−.012887746,+.028659773] |
| On−Off |−.012950231 |[−.029562424,+.002663415] |

On/off267captionkhác. CHAIRhallucaptionA52,on51,off48. **Chưa có bằng chứng vững on kém A/off hoặc off hơn A**;tất cảCItrênchứa0. OffkhôngphảiA;không được chọn off saukhinghìnsốrồigọi phươngpháp thắng. Chưa làmhumanblindreview mới.

CSAallocatedpeak1.334GiB/GPU,reserved1.377GiB. Staging~63.9phút;pha1train2.04phút,pha2train8.66phút,khônggồmbootstrap/fullTRAINidf/generation. Không suy fulltrainingtime bằng phépnhân4stepssmoke.

## Tiêu chí quyết định và điều chưa biết

Gate đã đặt trước:soAΔCIDEr≥.01,paired95%lower>0,beatOriginal,CHAIRs/i không tăng,recalldrop≤.01;đạt sốmới blindfactualreview. CSAon/offđều chưađạt. Dừngrecipe2500steps này theo gate,không kếtluận mọiCSA/cănchỉnh khôngkhảthi. Ngânsáchkhôngđảmbảohộitụ,devđãdùngnhiều,1seed,bootstrapkhôngđo seedvariance/selectionbias.

Tôi muốn bạn tìm **điều kiện thay đổi thực chất** để kỳ vọng kết quả khác. Tránh chỉ đổi gate/weight/addmodule rồi chạy lại; cũng không mặc định phải lặp fullpublishedsystem,thuê thêmGPU/gánnhãn lớn. Nếu cần phân biệt decoderconditioning với semanticrepresentation/evidencequality, hãy thiết kế phép đo quyết định rõ thay vì tuyên bố một nguyên nhân đã được chứng minh.

Đầu ra cần một quyết định và kế hoạch có thể thực hiện, không danh sách ý tưởng dài. Root sẽ kiểm lại assumptions và triển khai; chưa viết code trong lượt này.
