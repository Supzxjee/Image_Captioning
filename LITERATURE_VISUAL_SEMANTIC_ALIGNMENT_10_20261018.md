# 10 nghiên cứu căn chỉnh thị giác–ngôn ngữ, sàng lọc ngày 18/10/2026

Phạm vi 2024–2026; loại bản chỉ có arXiv chưa xác nhận công bố. Đã đối chiếu
metadata và abstract trên nguồn tác giả/nhà xuất bản/proceedings. Riêng CSA-Net
đọc thêm implementation chính thức ở commit0cc570b. Chưa đọc toàn văn cả10 bài;
không gọi đây là systematic review hoặc novelty audit đã hoàn tất.

## Tiêu chí nơi công bố

Q1/Q2 là quartile tạp chí, không quy đổi sang hạng hội nghị. Danh sách gồm1 bài
tạp chí Q1 và9 bài hội nghị A*; không gán Q1 cho proceedings của các hội nghị.

- TCSVT: Q1 theo cột Quartile(JIF) trong IEEE Title List tháng1/2025, dùng dữ liệu
  JCR2023 phát hành6/2024. Đây là bằng chứng cụ thể theo kỳ đó, không khẳng định
  mọi ngành/mọi năm đều cùng quartile. [IEEE](https://open.ieee.org/wp-content/uploads/IEEE-Title-List-January-2025.pdf).
- [CVPR](https://portal.core.edu.au/conf-ranks/604/),
  [ICML](https://portal.core.edu.au/conf-ranks/1121/),
  [NeurIPS](https://portal.core.edu.au/conf-ranks/98/): A* theo ICORE2026;
  cả ba có A* ở CORE2023. Không có workshop/Findings trong danh sách.

## Danh sách chính thức

1. **Li et al., Cascade Semantic Prompt Alignment Network for Image Captioning
   (CSA-Net)**. IEEE TCSVT34(7):5266–5281,2024; online/DOI2023.
   DOI [10.1109/TCSVT.2023.3343520](https://doi.org/10.1109/TCSVT.2023.3343520).
   [Tác giả xác nhận](https://kkzhang9.github.io/),
   [code chính thức](https://github.com/CrossmodalGroup/CSA-Net).
   Đối tượng/ngữ nghĩa vùng→ngữ nghĩa quan hệ→caption decoder; phù hợp trực tiếp
   nhất với đề tài. Code load bank noun/verb RegionCLIP, tổng hợp thành semantic
   vectors, visual-dependent weighting, cross-attention và response gates theo tầng.
   Không mô tả nó như mỗi ảnh được cung cấp GT triplets. Chưa xác minh đầy đủ losses
   hoặc toàn bộ paper từ chỉ file csa_net.py.

2. **Bica et al., Improving fine-grained understanding in image-text pre-training
   (SPARC)**. ICML2024, PMLR235:3974–3995.
   [Proceedings](https://proceedings.mlr.press/v235/bica24a.html).
   Gom patch thưa theo token và contrastive sequence-wise cục bộ, kết hợp loss
   toàn cục. Liên quan trực tiếp giả thuyết token/cụm↔vùng; paper còn đánh giá
   faithfulness/captioning. Không suy ra phiên bản nhỏ của ta sẽ tái lập gain.

3. **Jing et al., FineCLIP: Self-distilled Region-based CLIP for Better Fine-grained
   Understanding**. NeurIPS2024 Main Track.
   [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3122aaa22b2fe83f9cead1a696f65ceb-Abstract-Conference.html).
   Self-distillation từ global tới local và regional contrastive với region-text
   pairs, giữ global consistency. Hữu ích khi đặc trưng CLIP vùng chưa đủ giàu nghĩa.
   Không phải chỉ cắt bbox rồi cosine đã tương đương phương pháp của họ.

4. **Patel et al., TripletCLIP: Improving Compositional Reasoning of CLIP via
   Synthetic Vision-Language Negatives**. NeurIPS2024 Main Track.
   [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/39781da4b5d05bc2908ce08e43bc6404-Abstract-Conference.html).
   Dùng hard-negative caption và ảnh tổng hợp tương ứng, huấn luyện contrastive
   luân phiên. Liên quan kiểm tra khác biệt quan hệ/thành phần; không đồng nhất
   với việc tùy tiện swap mọi predicate rồi gọi là negative thật.

5. **Xiao et al., FLAIR: VLM with Fine-grained Language-informed Image
   Representations**. CVPR2025:24884–24894.
   [Proceedings](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_FLAIR_VLM_with_Fine-grained_Language-informed_Image_Representations_CVPR_2025_paper.html).
   Sub-caption sampling và text-conditioned attention pooling trên local tokens.
   Gần ý dùng cụm quan hệ để truy vấn vùng ảnh. Chủ yếu alignment/retrieval;
   không phải bằng chứng direct decoder caption sẽ tăng CIDEr.

6. **Chunyu Xie et al., FG-CLIP: Fine-Grained Visual and Textual Alignment**.
   ICML2025, PMLR267:68777–68793.
   [Proceedings](https://proceedings.mlr.press/v267/xie25k.html).
   Long captions, region-text supervision và hard negatives. Có1.6B caption-image
   pairs,40Mboxes trên12Mimages và10Mhard negatives: không phù hợp tái pretrain
   nguyên mô hình bằng ngân sách T4 hiện có. Học nguyên tắc hoặc thử pretrained.

7. **Shaoan Xie et al., SmartCLIP: Modular Vision-language Alignment with
   Identification Guarantees**. CVPR2025:29780–29790.
   [Proceedings](https://openaccess.thecvf.com/content/CVPR2025/html/Xie_SmartCLIP_Modular_Vision-language_Alignment_with_Identification_Guarantees_CVPR_2025_paper.html).
   Căn chỉnh modular để xử lý mức chi tiết và các khái niệm bị trộn. Hữu ích cho
   việc caption chỉ nói một phần ảnh; không ép mọi token tương ứng một vùng độc lập.
   Identification guarantees chỉ theo giả định paper, không phải UPT luôn đúng.

8. **Choi et al., GOAL: Global-local Object Alignment Learning**.
   CVPR2025:4070–4079.
   [Proceedings](https://openaccess.thecvf.com/content/CVPR2025/html/Choi_GOAL_Global-local_Object_Alignment_Learning_CVPR_2025_paper.html).
   Local Image-Sentence Matching và Token Similarity-based Learning, giữ ngữ cảnh
   toàn cục khi khớp vùng/câu. Đánh giá image–lengthy-text retrieval; chưa đủ để
   coi nó là phương pháp phát hiện sai quan hệ người–vật hay caption hallucination.

9. **Zohra et al., b-CLIP / β-CLIP: Text-Conditioned Contrastive Learning for
   Multi-Granular Vision-Language Alignment**. CVPR2026:680–689.
   [Proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Zohra_b-CLIP_Text-Conditioned_Contrastive_Learning_for_Multi-Granular_Vision-Language_Alignment_CVPR_2026_paper.html).
   Caption/câu/cụm truy vấn patch bằng cross-attention; contextualized contrastive
   loss xử lý semantic overlap. Gần ý alignment đa mức của ta; cụm nghĩa chồng lấp
   không nên tự động thành negative độc quyền.

10. **Chunyu Xie et al., FG-CLIP2: A Bilingual Fine-grained Vision-Language
    Alignment Model**. ICML2026, PMLR306:139736–139758.
    [Proceedings](https://proceedings.mlr.press/v306/xie26y.html).
    Region-text/long-caption supervision và Textual Intra-modal Contrastive loss
    phân biệt caption gần nghĩa; bilingual English/Chinese. Không suy diễn hỗ trợ
    tiếng Việt từ kết quả bilingual này. Chỉ học thêm loss không tự tạo novelty.

## CSA-Net gần đến đâu?

Gần trực tiếp theo bài toán và cấu trúc: nguồn ngữ nghĩa đối tượng/quan hệ, align
với đặc trưng vùng, rồi hỗ trợ decoder. File code chính thức đã đọc:
[csa_net.py@0cc570b](https://github.com/CrossmodalGroup/CSA-Net/blob/0cc570ba7c50ef08c46a21439fe73ef0a5df97ab/models/caption/csa_net.py).

Khác biệt hiện tại: CSA dùng semantic banks noun/verb RegionCLIP; UPT cache ta
có detected instance-pair/action scores. Đây mới là khác biệt đầu vào, chưa phải
đóng góp căn chỉnh mới. Q-Former hoặc thay DETR/YOLO cũng không đủ khẳng định mới.
Không khẳng định CSA không có cơ chế kiểm tra một lỗi nào đó trước khi đọc toàn văn.

## Nhận định để brainstorm, không phải kết luận của paper

Ưu tiên đọc sâu CSA-Net→SPARC→FLAIR/β-CLIP→TripletCLIP. Ba câu hỏi cần trả lời:

1. Cách học correspondence mới khác CSA/SPARC/FLAIR ở đâu, thay vì thêm generic
   cross-attention hoặc gate và đổi tên?
2. Có phân biệt đúng instance và đúng interaction khi class names không đổi?
   Hai mệnh đề có thể cùng đúng, ví dụ hold và ride, nên phải quản lý ambiguity.
3. Gain alignment có chuyển thành caption tốt hơn với cùng checkpoint/budget/data?
   Token heatmap hoặc retrieval tốt hơn không tự chứng minh caption giảm hallucination.

Đề xuất cũ chỉ gồm text query→patch attention→decoder memory đang thiếu novelty
so với các công trình này. Chưa nên triển khai module lớn trước khi đối chiếu
losses, supervision và failure modes cụ thể. Không thay code caption trong lượt tìm.

## Nhật ký nguồn

Academic MCP chuyên dụng không có trong bộ tools phiên này. Dùng web proceedings,
Crossref API cho CSA DOI/year/authors, trang tác giả và GitHub chính thức. CVF
web-reader trả403 ở một số trang; đã truy cập lại cùng trang bằng Python urllib,
đối chiếu citation metadata/abstract trực tiếp. ICORE và IEEE quartile được mở
nguồn gốc. Không dựa vào ResearchGate/summary site để xác nhận phương pháp.

FineLIP CVPR2025 đã sàng lọc nhưng không nằm trong10bài cuối: ưu tiên thay bằng
CSA-Net vì liên quan trực tiếp captioning hơn. Không bịa journal quartile hoặc
quy đổi hạng hội nghị. Chưa có kết quả thực nghiệm mới để ghi bảng metrics.
