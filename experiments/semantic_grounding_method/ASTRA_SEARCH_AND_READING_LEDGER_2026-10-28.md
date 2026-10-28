# Tra cứu và đối chiếu phương pháp: 20 bài bổ sung

Ngày 28/10/2026. Tài liệu phục vụ quyết định nghiên cứu, không phải systematic review. Quyết định mới ở [bản đề xuất đi kèm](ASTRA_METHOD_RECOMMENDATION_AFTER_LITERATURE_2026-10-28.md); metadata, loss, giới hạn và độ sâu đọc từng bài ở [CSV](ASTRA_LITERATURE_20_2026-10-28.csv).

## Phạm vi và cách kiểm nguồn

Sàng lọc ban đầu 20 bài, sau đó thêm 3 bài gần cơ chế dự kiến: GOT, Sinkformers, OTF-CBM. Giữ 20 bài trong bảng cuối; chuyển GroundCap, AHNPL và OPERA sang danh sách bổ trợ dưới đây. Đây là 20 tài liệu bổ sung cho các lượt trước, không dùng lại CSA-Net, IIT, Strict IIT, RAVEL, Shen2020, MTTSNet hay See-or-Guess để đủ số. Có 19/20 bài từ 2022–2026; GOT 2020 giữ lại vì gần cơ chế, dù ngoài khoảng ưu tiên.

Áp dụng nature-academic-search / multi-source-search. Phiên này không có công cụ MCP học thuật chuyên biệt, nên fallback sang web trên CVF, ECVA, PMLR, AAAI, NeurIPS, OpenReview, IEEE và arXiv. Dùng bài của tác giả hoặc proceedings làm nguồn phương pháp; kết quả từ trang tổng hợp chỉ là đầu mối. Không có truy cập trực tiếp Scopus/WoS/Crossref trong lượt này.

Khử trùng theo DOI khi có; nếu thiếu DOI thì chuẩn hóa title, gộp bản arXiv với bản proceedings, giữ URL cả hai khi hữu ích. Đã kiểm 20 title không trùng và các DOI được ghi không trùng. DOI để trống là chưa xác minh, không phải bài không có DOI. Không gán Q1/Q2 cho hội nghị; quartile của TIP chưa xác minh.

**Độ sâu đọc được ghi theo mục đã mở, không đồng nghĩa đọc hết mọi trang/phụ lục.** Tám bài cốt lõi đã xem phương pháp/công thức và một phần thực nghiệm/ablation. GRiT, Groundhog, Hard Positive Truth cũng được đọc các mục phương pháp/thực nghiệm như bổ trợ. Các bài còn lại chỉ được dùng ở mức abstract; không mượn chi tiết loss chưa đọc để kết luận tính mới.

Các sửa metadata sau kiểm nguồn: GRiT là ECCV2024; HALC là ICML2024. C²LIP có record CVPR2026 trên CVF. OTF-CBM chỉ xác nhận được bản arXiv2026, chưa xác minh venue được chấp nhận. Không nâng thông tin venue từ trang bên thứ ba thành kết luận.

## Bảng bằng chứng

| ID | Bài và nguồn gốc | Năm; nơi công bố | Phạm vi đã đọc | Toàn văn |
|---|---|---|---|---|
| N01 | [GRIT: Faster and Better Image Captioning Transformer Using Dual Visual Features](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/2605_ECCV_2022_paper.php) | 2022; ECCV | **Cốt lõi**: Sec.3.2 cross-attention; Sec.3.3 XE/SCST; Sec.4.3-4.4 and Table 1 | [Full text](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960165.pdf) |
| N02 | [GRiT: A Generative Region-to-text Transformer for Object Understanding](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10356.pdf) | 2024; ECCV | Sec.3.1-3.3 Eq1; Sec.4.3-4.6 Table3 | [Full text](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10356.pdf) |
| N03 | [Visual Cluster Grounding for Image Captioning](https://ieeexplore.ieee.org/abstract/document/9785461) | 2022; IEEE Transactions on Image Processing | Abstract; công thức chưa xác minh | Chưa lấy được full text |
| N04 | [Cycle-Consistency Learning for Captioning and Grounding](https://ojs.aaai.org/index.php/AAAI/article/view/28363) | 2024; AAAI | **Cốt lõi**: Methodology pp5537-5538 Eq1-9; Tables3-6 | [Full text](https://ojs.aaai.org/index.php/AAAI/article/download/28363/28711) |
| N05 | [GROUNDHOG: Grounding Large Language Models to Holistic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_GROUNDHOG_Grounding_Large_Language_Models_to_Holistic_Segmentation_CVPR_2024_paper.html) | 2024; CVPR | Sec.2.1-2.3; Sec.3; Sec.4.1 and Tables3-7 | [Full text](https://arxiv.org/html/2402.16846v1) |
| N06 | [Graph Optimal Transport for Cross-Domain Alignment](https://proceedings.mlr.press/v119/chen20e.html) | 2020; ICML | **Cốt lõi**: Sec.2.1-2.4 Eq3; Sec.4.1; Sec.4.2 captioning and Table 4 | [Full text](https://proceedings.mlr.press/v119/chen20e/chen20e.pdf) |
| N07 | [When and Why Vision-Language Models Behave like Bags-of-Words, and What to Do About It?](https://github.com/mertyg/vision-language-models-are-bows) | 2023; ICLR | Abstract; công thức chưa xác minh | [Full text](https://arxiv.org/pdf/2210.01936) |
| N08 | [Multi-Grained Vision Language Pre-Training: Aligning Texts with Visual Concepts](https://proceedings.mlr.press/v162/zeng22c/zeng22c.pdf) | 2022; ICML | Abstract; công thức chưa xác minh | [Full text](https://proceedings.mlr.press/v162/zeng22c/zeng22c.pdf) |
| N09 | [The Hard Positive Truth about Vision-Language Compositionality](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02149.pdf) | 2024; ECCV | Sec.4.1-4.3; Sec.5.1-5.2 Table2 | [Full text](https://arxiv.org/html/2409.17958v1) |
| N10 | [Contrasting Intra-Modal and Ranking Cross-Modal Hard Negatives to Enhance Visio-Linguistic Compositional Understanding](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Contrasting_Intra-Modal_and_Ranking_Cross-Modal_Hard_Negatives_to_Enhance_Visio-Linguistic_CVPR_2024_paper.html) | 2024; CVPR | Abstract; công thức chưa xác minh | [Full text](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_Contrasting_Intra-Modal_and_Ranking_Cross-Modal_Hard_Negatives_to_Enhance_Visio-Linguistic_CVPR_2024_paper.pdf) |
| N11 | [Sinkformers: Transformers with Doubly Stochastic Attention](https://proceedings.mlr.press/v151/sander22a.html) | 2022; AISTATS | **Cốt lõi**: Intro Eq1; Sec.3 Eq3-4; Sec.6.1-6.4 Tables1-2 (not all theory proofs) | [Full text](https://proceedings.mlr.press/v151/sander22a/sander22a.pdf) |
| N12 | [No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](https://openaccess.thecvf.com/CVPR2026?day=2026-06-07) | 2026; CVPR | **Cốt lõi**: Sec.3.1-3.2 Eq4-8; Sec.4.1; Sec.4.4 Table 5 | [Full text](https://arxiv.org/html/2603.25722v1) |
| N13 | [FLAIR: VLM with Fine-grained Language-informed Image Representations](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_FLAIR_VLM_with_Fine-grained_Language-informed_Image_Representations_CVPR_2025_paper.pdf) | 2025; CVPR | **Cốt lõi**: Sec.3.1-3.4 Eq1-2; Sec.4.2-4.3; Sec.4.4/AppendixC Table 7 | [Full text](https://arxiv.org/html/2412.03561v1) |
| N14 | [FineCLIP: Self-distilled Region-based CLIP for Better Fine-grained Understanding](https://proceedings.neurips.cc/paper_files/paper/2024/file/3122aaa22b2fe83f9cead1a696f65ceb-Paper-Conference.pdf) | 2024; NeurIPS | Abstract; công thức chưa xác minh | [Full text](https://proceedings.neurips.cc/paper_files/paper/2024/file/3122aaa22b2fe83f9cead1a696f65ceb-Paper-Conference.pdf) |
| N15 | [TripletCLIP: Improving Compositional Reasoning of CLIP via Synthetic Vision-Language Negatives](https://proceedings.neurips.cc/paper_files/paper/2024/hash/39781da4b5d05bc2908ce08e43bc6404-Abstract-Conference.html) | 2024; NeurIPS | Abstract; công thức chưa xác minh | [Full text](https://proceedings.neurips.cc/paper_files/paper/2024/file/39781da4b5d05bc2908ce08e43bc6404-Paper-Conference.pdf) |
| N16 | [HALC: Object Hallucination Reduction via Adaptive Focal-Contrast Decoding](https://proceedings.mlr.press/v235/chen24bi.html) | 2024; ICML | Abstract; công thức chưa xác minh | [Full text](https://arxiv.org/pdf/2403.00425) |
| N17 | [Bridging Vision and Language Concepts through Optimal Transport Semantic Flow](https://arxiv.org/abs/2606.26891) | 2026; arXiv; accepted venue unverified | **Cốt lõi**: Sec.3.1-3.4; Sec.4.1-4.2 and Sec.4.5 Tables5-6 | [Full text](https://arxiv.org/html/2606.26891v1) |
| N18 | [Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://openaccess.thecvf.com/content/CVPR2024/html/Leng_Mitigating_Object_Hallucinations_in_Large_Vision-Language_Models_through_Visual_Contrastive_CVPR_2024_paper.html) | 2024; CVPR | Abstract; công thức chưa xác minh | [Full text](https://arxiv.org/pdf/2311.16922) |
| N19 | [Grounding Multimodal Large Language Models to the World](https://openreview.net/attachment?id=lLmqxkfSIw&name=pdf) | 2024; ICLR | Abstract; công thức chưa xác minh | [Full text](https://openreview.net/attachment?id=lLmqxkfSIw&name=pdf) |
| N20 | [Multi-Modal Hallucination Control by Visual Information Grounding](https://openaccess.thecvf.com/content/CVPR2024/papers/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.pdf) | 2024; CVPR | **Cốt lõi**: Sec.3; Sec.4.1 Eq1-3; Sec.4.2 DPO setup; Sec.5.1 Table 1 and part5.3 | [Full text](https://arxiv.org/html/2403.14003v1) |

**Thiếu toàn văn:** Visual Cluster Grounding N03 chỉ có abstract/metadata; chưa so sánh được công thức của bài với ứng viên mới. [PubMed](https://pubmed.ncbi.nlm.nih.gov/35635813/) xác nhận DOI 10.1109/TIP.2022.3177318. PDF CVF của FLAIR lỗi truy cập ở lượt đọc, dùng bản arXiv của tác giả; PDF ECVA Hard Positive Truth timeout, dùng HTML arXiv. Không coi các lần lỗi là đã đọc.

## Tám bài cốt lõi: điều thực sự rút ra từ phương pháp và thực nghiệm

### N01 GRIT

Grid và region được đọc qua cross-attention riêng; tác giả thử nhiều cách hợp nhất, gồm nhánh song song có cổng. Huấn luyện XE rồi SCST. Table 1 có ablation từng nguồn. Bài cho một đối chứng tốt về đọc nhiều nguồn; chưa đủ để gọi việc thêm nhánh đối tượng/cặp là đóng góp mới.

### N04 CyCo

Hai chu trình tái tạo caption và box; loss gồm CE và GIoU+smoothL1. Các bảng 3–6 khảo sát ghép nhiệm vụ, kể cả caption toàn ảnh. Cycle consistency khuyến khích hai tác vụ tương thích; một câu quan hệ sai nhưng tái định vị được vẫn có thể qua chu trình. Đây là phản ví dụ của tôi, không phải kết quả bài báo.

### N06 Graph Optimal Transport

WD căn chỉnh node và GWD căn chỉnh cấu trúc cạnh dùng chung transport plan; cộng vào task loss. Captioning có thật trong §4.2/Table 4. Vì vậy “ghép đối tượng–quan hệ bằng OT” đã có tiền lệ; delta phải chỉ rõ vai trò của phân phối chung trong forward sinh caption.

### N11 Sinkformers

Eq.3–4 thay row-softmax bằng chuẩn hóa hàng/cột; §6 so sánh nhiều số vòng lặp. Trong NMT, tác giả chỉ thay attention encoder, nêu khó khăn mask nhân quả ở decoder. Ta xử lý cặp đối tượng tại một bước sinh, không chuẩn hóa qua các vị trí từ tương lai; solver vẫn là phần kế thừa.

### N12 C²LIP

Eq.4–8 nối loss sigmoid toàn ảnh với positives ở mức noun phrase và pooling cục bộ theo concept. Table 5 tách các thành phần. Đây là bằng chứng có thể học nhiều mức khái niệm mà không chỉ tăng hard negatives. Mô hình được huấn luyện lại encoder; bài không chứng minh decoder A dùng được thông tin ấy.

### N13 FLAIR

Text làm query đọc patch; có loss nhiều positives. Table 7 kiểm cách tạo negatives: cần giữ cùng text điều kiện khi đổi ảnh, tránh học so text với chính query. Caption tự hồi quy chưa có toàn cụm tương lai; forward của ứng viên vì thế chỉ dùng prefix.

### N17 OTF-CBM

§3 dùng UOT có KL biên, inverse OT và flow matching; §4.5 có ablation classification. KL mềm, background và concept bottleneck đều có tiền lệ. Không bê semantic flow sang A: chưa có bằng chứng về caption, còn thêm một giai đoạn học. Chưa tái lập hay kiểm độc lập số liệu của bản preprint.

### N20 M3ID

Eq.1–3 dùng chênh lệch log-prob có/không điều kiện thị giác; §5.1 kiểm caption. Tác giả lưu ý ít phụ thuộc thị giác có thể hợp lý cho từ chức năng. Với A, prompt vốn đã phụ thuộc ảnh: bỏ raw visual stream chưa tạo nhánh vô điều kiện đúng nghĩa. Đây là giới hạn khi chuyển cơ chế.

## Tổng hợp theo cơ chế và ba chặng của bài toán

1. **Tìm vùng:** region/cluster/mask cho biết bằng chứng nằm ở đâu. GRiT và Groundhog mở rộng đơn vị grounding; đối tượng không bắt buộc một bbox ứng với một từ. Grounding tốt chưa xác định đúng predicate.
2. **Biểu diễn ý nghĩa:** concept positives, hard positives và graph alignment xét tổ hợp/mối liên hệ. Chúng không tự bảo đảm caption decoder dựa vào chúng.
3. **Đưa vào quyết định sinh:** dual-stream reader, distillation và contrastive decoding đặt bằng chứng gần luồng sinh hơn. Cần kiểm caption tự do; attention đẹp hoặc assertion likelihood tốt mới là chỉ báo trung gian.

Suy luận nghiên cứu của tôi: đáng thử một cấu trúc đọc chung cho thực thể và tương tác, trong đó evidence quan hệ có thể cập nhật lựa chọn thực thể. Đây là giả thuyết kiến trúc từ khoảng trống giữa ba chặng, không phải kết luận rằng các bài đã chứng minh A mắc lỗi do ba reader bất nhất. A hiện chưa có ba reader ấy.

## Ba bài đã sàng lọc nhưng không giữ trong 20 cốt bảng

- [GroundCap](https://arxiv.org/abs/2502.13898),2025: dataset grounding entity/action, chỉ đọc abstract; ít gần phép đọc decoder dự kiến hơn GOT.
- [AHNPL](https://ijcai.org/proceedings/2025/605),IJCAI2025: perturbation và adaptive negatives, chỉ đọc abstract; cơ chế loss không được kiểm sâu.
- [OPERA](https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_OPERA_Alleviating_Hallucination_in_Multi-Modal_Large_Language_Models_via_Over-Trust_CVPR_2024_paper.pdf),CVPR2024: decoding với attention penalty/rollback, chỉ đọc abstract; kiến trúc MLLM khác A.

Không tính bài distillation2020 dưới đây vào 20. Dùng làm nearest prior cho ứng viên thứ hai: [More Grounded Image Captioning by Distilling Image-Text Matching Model](https://arxiv.org/pdf/2004.00390), §3.3 Eq.15–19 và §4.1. Bài đã có KL hướng caption attention theo teacher tại noun và reward matching trong SCST. Chưa xác minh venue trong lượt này; không gán một venue dựa trí nhớ.

## Nhật ký truy vấn chính

Các truy vấn dưới đây phản ánh đường tìm, gồm cả giả định venue sai đã được sửa sau kiểm nguồn; không phải tất cả kết quả của truy vấn đều được chấp nhận làm bằng chứng. Đợt đầu tập trung GRIT, GRiT, cluster grounding, CyCo/Groundhog; raw log đợt 1 không được lưu trọn nên không dựng lại câu query từ trí nhớ.

Đợt 2:

- `"When and why" "Bags-of-Words" vision language ICLR 2023`
- `"Multi-Grained Vision Language Pre-Training" ICML 2022`
- `"compositional" "vision-language" "hard negatives" CVPR 2024 2025`
- `"grounded image captioning" "2023" "2024" IEEE transactions`

Đợt 3:

- `"No Hard Negatives Required" "Concept Centric" CVPR 2026`
- `"FLAIR" "Localized" "Image" "Text" CVPR 2025`
- `"FineCLIP" "Fine-Grained" CVPR 2025`
- `"TripletCLIP" NeurIPS 2024`

Đợt 4:

- `"HALC" "Object Hallucination" CVPR 2024`
- `"OPERA" "Over-Trust" CVPR 2024`
- `"Mitigating Object Hallucinations" "Visual Contrastive Decoding" CVPR 2024`
- `"Kosmos-2" "ICLR 2024" grounding`

Đợt 5:

- `"HALC" "ICML" 2024 decoding`
- `"OPERA: Alleviating Hallucination" CVPR 2024`
- `"GroundHog" CVPR 2024 grounded`
- `"Multi-Modal Hallucination Control by Visual Information Grounding"`

Đợt 6:

- `site:openaccess.thecvf.com/content/CVPR2026 "No Hard Negatives Required"`
- `site:openaccess.thecvf.com/content/CVPR2024 "Contrasting Intra-Modal"`
- `site:proceedings.mlr.press "HALC"`
- `"Visual Cluster Grounding for Image Captioning" pdf Jiang`

Đợt 7:

- `"Kosmos-2" site:openreview.net/forum`
- `"GROUNDHOG" site:arxiv.org/abs`
- `"Cycle-Consistency Learning for Captioning and Grounding" arxiv`
- `"HALC" site:proceedings.mlr.press/v235/`

Đối chiếu delta cuối:

- `"Graph Optimal Transport for Cross-Domain Alignment"`
- `"Sinkformers" doubly stochastic attention`
- `"Bridging Vision and Language Concepts through Optimal Transport Semantic Flow"`
- `"More Grounded Image Captioning by Distilling Image-Text Matching Model"`
- `"image captioning" "structured attention" joint subject object relation`
- `"captioning" "unbalanced optimal transport" relation attention`

Hai query cuối chỉ là collision check có giới hạn. Các kết quả remote sensing/audio/UOT khác chưa đọc sâu không được tính vào 20 và không dùng để tuyên bố đã phủ hết prior. Không thấy trùng khít chưa phải bằng chứng tính mới.

## Những gì chưa làm

Chưa triển khai module/loss; chưa chạy neural/local/GPU; không đọc lại ZIP. Chưa chứng minh novelty bằng exhaustive review, chưa kiểm quota, chưa cam kết thời gian train, chất lượng caption hay khả năng được chấp nhận bài báo. Các file nghiên cứu này không đổi NO_GO của frame200.
