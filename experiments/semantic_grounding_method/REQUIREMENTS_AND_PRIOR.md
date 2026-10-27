# Yêu cầu phương pháp căn chỉnh ngữ nghĩa — 2026-10-27

Người dùng cần một phương pháp có cơ chế học căn chỉnh để phát triển bài báo;
readoutP/Q chỉ cóSCST+KL chưa đáp ứng phần giám sát semantic tường minh. Đang chốt
method với Astra; không triển khai ML, notebook hoặc GPU từ tài liệu này.

## Các điều kiện phương pháp phải đáp ứng

1. Đơn vị semantic được gọi đúng: groundable noun mention/phrase hoặc event,
   không gọi tất cả tokens là đối tượng và không coi mọi động từ có một bbox riêng.
2. Có đích học liên kết với vùng, kèm source/positive/unknown policy. Caption
   reference không nhắc tới một vật hoặc V-COCOaction0 không tự là phủ định.
3. Phân biệt attention có thể học với explicit grounding supervision và grounding
   đúng đã đo. L_align phải có mục tiêu rõ hơn “attention map trông hợp lý”.
4. Nếu chỉquery theo prefix, không thể đòi cùnginput ở “a person holding a” trỏ
   duy nhất vào hai vùng khác nhau tùy reference chưa được nhìn thấy. Cần word/
   concept-conditioned grounding hoặc mô hình hóa nhiều đích khả dĩ. Không có
   ground-truth future word tại inference.
5. Chính vector từ đường đã được giám sát phải ảnh hưởng phân phối caption cuối;
   không chỉ dùng một localizer phụ để vẽ ảnh giải thích rồi bỏ khỏi generation.
6. Loss caption/SCST phải tối ưu output thực tế. Zero-init/frozen A chỉ giúp anchor,
   không bảo đảm giữ CIDEr, fluency hoặc triệt tiêu mọi language shortcut.
7. Tối thiểu cần đối chứng cùng head/lịch/data có-vs-không L_align để biết tác
   dụng của alignment, không chỉ thêm tham số. Giữ cả baseline A.
8. Nguồn label/splits phải không lẫn atlasval/test, không đổi IDF sau chọn subset.
   Grounding phải có chỉ số riêng; CHAIR không đo predicate-role correctness.
9. Mới so với các thử nghiệm dự án không đồng nghĩa mới trong literature.
   Giám sát attention, contrastive matching, cycle và SCST đã có tiền lệ.

## Nguồn ghép noun–box sẵn có, đã kiểm metadata

`experiments/coco_grounding_feasibility/review/noun_box_train10k_v1_1/dataset`
có17.904 spans,5.602 region masks trên4.495ảnh/71classes. Receipt SHA:
`bde873ba77124333488b263a052419dc80e412c1e089178cdf423d2836e1b146`.
Ngày27/10, root kiểm lại6payloadSHA và mọi span.mask_key có trong maskbank.
Source policy chỉTRAIN10k, receipt weak_supervision_not_gold=true; không modelrun.

Đây là ghép tự động dựa trên mention/uniqueclass và COCOinstance annotation,
không phải human phrase-instance gold. Mask có normalizedxyxy và196patch fractional
areas. Span ghi caption_index, token IDs và vị trí decoder. Các artificial/depiction
contexts và aliases mơ hồ có policy loại sẵn. Dùng nguyênpolicy đã khóa, không nới
sau nhìn atlas để tăng coverage. Quan hệ hành động/thuộc tính của cả phrase chưa
được các noun–box labels này xác nhận.

## Prior sát nhất đã đọc phần liên quan

| Công trình | Cơ chế liên quan | Ranh giới với đề xuất mới |
|---|---|---|
| [Zhou etal., More Grounded Image Captioning, CVPR2020](https://arxiv.org/pdf/2004.00390), method3.3/Eq15–19 | POS-SCAN regularizes caption attention; có cả công thức dùng grounding labels; tiếpSCST+localmatchingreward | Không gọi alignment loss+SCST tự nó là mới |
| [Ma etal., Learning to Generate Grounded Visual Captions without Localization Supervision, ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123630341.pdf), method3.2 | Decode→word-conditionedlocalize→reconstruct,shareddecoder | Cycle hoặc giải quyết việc attention chưa biết future word đã có tiền lệ |
| [Mavroudi etal., Weakly-Supervised Generation and Grounding..., CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Mavroudi_Weakly-Supervised_Generation_and_Grounding_of_Visual_Descriptions_With_Conditional_Generative_CVPR_2022_paper.pdf) | Joint descriptions/latent word-region alignments, conditional generative model; explicit discussion of word-agnostic grounding | Kiểm abstract/search-extractedmethod lần này; PDFtool403. Không nhận đã loại hết overlap |
| [CSA-Net, TCSVT2024](https://doi.org/10.1109/TCSVT.2023.3343520) | Semantic prototypebanks, object→relation alignment, caption objectives; local PDF/source audit đã có | Kế thừa bank/attention không phải mới; CSA pilot hiện tại chưa quaA |

Đây là đối chiếu prior có mục tiêu, không exhaustive novelty clearance. Academic
MCP không được mount; dùng primary arXiv/CVF/ECVA. Nguồn sáu payload được kiểm
local bằngmetadata/CPU; không chạy neural inference hay training.
