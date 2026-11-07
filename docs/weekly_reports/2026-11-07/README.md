# Báo cáo tổng hợp ngày07/11/2026

**Bản hiện tại sau sửa08/11:6 trang dọc, mỗi module có hình và phần giải thích cách hoạt động, sau đó là pipeline tổng thể và các bảng kết quả. Bảng đã đảo chiều, điểm tốt nhất tô đậm, không còn tên CSA/mũi tên lên xuống/đoạn nguồn ở cuối.**

Bản Word cập nhật từ báo cáo06/11, giữ sơ đồ overview và phần huấn luyện. Tổng3trang: portrait–landscape–portrait. Trang2 có hai bảng đủ8metrics: NoCSA10/CSA10 vàCONTROL/PROMPT trước/sau rerank (48giátrị). Trang3 giữ VALID vàprobe tách riêng, giới hạnCI/seed/epoch/adapter/TESTexposure; mốcCSA15điểm117.06được ghi là thamkhảo và3metriccònthiếu. Khôngquy toànbộ tăngđiểm cho riêngprompt. Bảncũ được giữ nguyên.

- `Bao_cao_nghien_cuu_07_11_2026.docx`: báo cáo giao người dùng; bảncopytrongDownloads cùngtên.
- `update_report.py`: cập nhật từDOCX cũ vàJSONaudited tại experiments/csa_paper.
- Nguồn số: PLAIN_CSA10_EXTRA_RESULT.json, PROMPT_TEST_EXTRA_RESULT.json, PROMPT_TEST_RESULT.json, PROMPT_SCST_RESULT.json. Khôngchấm lại metric.

QA: so48ôTESTkhớpgiátrịJSON đãaudit;3sections/4tables/1diagram. render_docx.py được thử nhưng thiếuLibreOffice; đã chẩnđoánlỗiPATH, dùngMicrosoftWordCOMxuấtPDF vàPopplerrenderPNG thaythế. Đãxem đủ3trang bảncuối, khôngtràn/bịcắt bảng/chữ hoặc trangtrắng. Fontbody12 vàtable11; trangngang dànhbảng8metric. PDF/PNG giữlocalchoQA, khôngdelivertrung gian. Không chạyTorch,Java,scorer hoặcKaggle; khôngthêmSheetrow cho việcdàntrang báo cáo.


## Pipeline update 07/11

Replaced page1 overview with the user-approved three-module main figure (modules/00-overall.png, small annotations removed). Caption and module identifiers M1/M2/M3 in two existing explanatory paragraphs updated; all4 table XMLs and section settings unchanged. Report remains3 pages. Script insert_pipeline.py applies this local edit after update_report.py; original backup outsideGit under .review_cache.

render_docx.py attempted: LibreOffice soffice.exe absent. Fallback Microsoft Word COM exported the complete3-page PDF; Word Quit returned RPC cleanup error after successful export/close. Poppler rendered all3 pages and each was inspected: image/caption fit, tables and paragraphs intact, no clipping/blank pages. PDF/PNGs kept private for QA. Final Word copied to Downloads original name and *_pipeline.docx. No model/scorer/metric/Sheet changes.


## Portrait result tables

User requested portrait page2, metrics down rows and configurations across columns. Transposed only the two TEST result tables: 8 metric rows ×2/4 configurations. All48 numeric strings, prose, other two tables and pipeline image preserved. All3 sections portrait; no landscape page remains. Table font11pt retained, vertical cell padding30twips and paragraph spacing0 avoid an orphaned note page. Final3-page WordCOM/Poppler render inspected page by page. Builder sequence: update_report.py → insert_pipeline.py → transpose_result_tables.py. Final Word in Downloads original name, *_pipeline.docx and *_bang_doc.docx. No new model/metric/Sheet run.


## Best-result emphasis and descriptive names 08/11

User requested best numeric cells bold, arrows removed, and no CSA name in report. highlight_best_results.py selects within each of4 tables: quality metrics maximize, CHAIR/counts minimize, ties all bold. Exactly22 numeric cells bold; all metric strings preserved. Up/down arrows and their former legend removed. Method names in prose/headings/configurations are now descriptive bank-alignment terms, including historical comparison; architecture and provenance unchanged. Visible source link points to experiments root. Original font formatting retained. WordCOM+Poppler final3 portrait pages inspected. Downloads *_hoan_chinh.docx and previous aliases refreshed. Builder order now update_report.py → insert_pipeline.py → transpose_result_tables.py → highlight_best_results.py. No models/scorers/Sheet edits.


## Source note removal 08/11

Removed exactly the two final paragraphs requested: source/version note and GitHub link. Other paragraph XML and all4 table XMLs unchanged, including22 bold numeric winners. Source provenance remains in repo research files. Final WordCOM/Poppler3 pages visually inspected; renderer log notes missing LibreOffice. Downloads *_da_sua.docx plus previous aliases refreshed. Run remove_report_sources.py last if rebuilding.


## Missing module figures corrected 08/11

User correctly noted previous report contained only overall diagram. add_module_figures.py now inserts M1 and M2 onpage1, M3 onpage2, then overview/training onpage3, results onpages4–5. Module detail comes before overall as requested; section/figure numbering updated. Existing4 result-table XMLs including bold winners unchanged. All4 embedded PNG hashes match latest cleaned figures. All5 portrait pages rendered in Word/Poppler and visually inspected, no clipping/blank pages; the dedicated M3 page intentionally retains whitespace for readable full-width drawing. Final Downloads *_du_module.docx and previous aliases refreshed. Builder step add_module_figures.py runs last, guarded against duplication. No model/metric/Sheet changes.


## Module operation explanations 08/11

Added8 paragraphs under module captions: M1 POS/lemma/CLIP templates/512d fixed shared banks; M2 learned150-vector aggregation, cosine relevance, self/cross attention with Q/K/V roles, norm/FFN/gate, exact two-object plus relation cascade; M3 UPT human–object phrases, CLIP text, class/IoU matching, slot definition and originalVo inputs, linked masked attention and residual, unmatched ROI/no-delta and detector-dependence limitation. Source code read only, no model execution. Original text remains in order, all4 table XMLs and22 bold winners unchanged.

Each module stays on one page: M1 page1, M2 page2, M3 page3; overview/training page4, results pages5–6. All6 WordCOM/Poppler pages inspected. Downloads *_giai_thich.docx and prior aliases refreshed. explain_module_operation.py runs last and guards against double insertion.


## Six review annotations applied 08/11

Title/footer changed to Báo cáo nghiên cứu tuần13; header date and opening summary removed. Module is a valid technical term; section heading changed to Các khối chức năng của mô hình for readability. Two paragraphs after M2 caption explain V from CLIP patches/YOLO boxes/ROI pooling/768→512 projection and generic X = V/V1/Vo for the three alignment stages. M3 Only slots... labels removed in vector source and exports and embedded figure refreshed. All4 table XMLs, numbers and22 bold winners retained. All6 portrait WordCOM/Poppler pages inspected, plus M3 PDF render. Downloads Bao_cao_nghien_cuu_tuan_13.docx and prior aliases refreshed. apply_week13_comments.py is final rebuild step. No model/metric/Sheet change.
