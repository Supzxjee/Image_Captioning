# Báo cáo tổng hợp ngày07/11/2026

Bản Word cập nhật từ báo cáo06/11, giữ sơ đồ overview và phần huấn luyện. Tổng3trang: portrait–landscape–portrait. Trang2 có hai bảng đủ8metrics: NoCSA10/CSA10 vàCONTROL/PROMPT trước/sau rerank (48giátrị). Trang3 giữ VALID vàprobe tách riêng, giới hạnCI/seed/epoch/adapter/TESTexposure; mốcCSA15điểm117.06được ghi là thamkhảo và3metriccònthiếu. Khôngquy toànbộ tăngđiểm cho riêngprompt. Bảncũ được giữ nguyên.

- `Bao_cao_nghien_cuu_07_11_2026.docx`: báo cáo giao người dùng; bảncopytrongDownloads cùngtên.
- `update_report.py`: cập nhật từDOCX cũ vàJSONaudited tại experiments/csa_paper.
- Nguồn số: PLAIN_CSA10_EXTRA_RESULT.json, PROMPT_TEST_EXTRA_RESULT.json, PROMPT_TEST_RESULT.json, PROMPT_SCST_RESULT.json. Khôngchấm lại metric.

QA: so48ôTESTkhớpgiátrịJSON đãaudit;3sections/4tables/1diagram. render_docx.py được thử nhưng thiếuLibreOffice; đã chẩnđoánlỗiPATH, dùngMicrosoftWordCOMxuấtPDF vàPopplerrenderPNG thaythế. Đãxem đủ3trang bảncuối, khôngtràn/bịcắt bảng/chữ hoặc trangtrắng. Fontbody12 vàtable11; trangngang dànhbảng8metric. PDF/PNG giữlocalchoQA, khôngdelivertrung gian. Không chạyTorch,Java,scorer hoặcKaggle; khôngthêmSheetrow cho việcdàntrang báo cáo.


## Pipeline update 07/11

Replaced page1 overview with the user-approved three-module main figure (modules/00-overall.png, small annotations removed). Caption and module identifiers M1/M2/M3 in two existing explanatory paragraphs updated; all4 table XMLs and section settings unchanged. Report remains3 pages. Script insert_pipeline.py applies this local edit after update_report.py; original backup outsideGit under .review_cache.

render_docx.py attempted: LibreOffice soffice.exe absent. Fallback Microsoft Word COM exported the complete3-page PDF; Word Quit returned RPC cleanup error after successful export/close. Poppler rendered all3 pages and each was inspected: image/caption fit, tables and paragraphs intact, no clipping/blank pages. PDF/PNGs kept private for QA. Final Word copied to Downloads original name and *_pipeline.docx. No model/scorer/metric/Sheet changes.
