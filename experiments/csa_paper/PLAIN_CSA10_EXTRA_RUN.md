# Chạy bổ sung8chỉ số choNoCSA10 vàCSA10

Notebook: [PLAIN_CSA10_TEST_EXTRA_CPU.ipynb](PLAIN_CSA10_TEST_EXTRA_CPU.ipynb).

1. Import vào bất kỳ tài khoảnKaggle nào. Chọn **Accelerator None**, **Internet ON**.
2. Không cầnSecret/API/Dataset/ảnh/checkpoint. Save & Run All một lượt; notebook chấm cảhai baseline.
3. Gửi `/kaggle/working/plain_csa10_TEST_extra_v1_review.zip`. Giữ toàn bộfolderoutput nếu bất kỳphase lỗi, không chạy lại model.

## Tệp dữ liệu

Snapshot512496bytes từtextcache đãaudit:5000caption NoCSA10,5000captionCSA10 và25010captionthamchiếu (4990ảnh×5+10ảnh×6). Canonicalrefs vàIDs khớp nguyên nội dung TEST184/61. Nguồn: `PLAIN_CSA10_EXTRA_INPUTS.json`, SHA rawcaption/metadata/result vàZIP riêng mỗi baseline; nguồn là snapshot đã xác minh, không giả định privateAPI hoặc version45 là lượt TEST. Trọng số NoCSA từtrain45,CSA từ139; cảhai10epochs/beam5/max20. Không thay câu hoặc dùng thamchiếu đểchọnbeam.

## Chấm và xuất

Dùng nguyên bộscorer `score_prompt_test_extra.py` pin a58d5b9 đã chạy thành công185/62 (byte-identical); wrapper chỉ chuẩn bịhai bộcaption cũ. COCO1.2/PTB/Java8/CoreNLP3.6.0 và smoke trướcfull. Chấm3metricmới BLEU1/METEOR/SPICE; replayBLEU4/ROUGE/CIDEr khớp1e-7. CHAIRs/i giữaudit cũ sau khiSHA snapshotkhớp, không tínhCHAIRmới. Result raw/×100 vàCSV chỉCOMPLETE khi đủ8metric/cả2×5000ảnh. Khôngchấm thêmCSA15/SCST trongnotebook này.

4hCPU STOPcap, khôngphảiETA; heartbeat30s; logs/cache giữkhi lỗi. TảiJava/CoreNLP (~403MB) cóprogress; khôngtốnGPU.

## Kiểm tra tác giả

4safe stdlib/metadata/mock fixtures PASS và lặp trong exactGitbundle pin c6306eb (5files inclsnapshot). Snapshot full5000/25010,1853NoCSA/388CSAshortcaptions và nguồnmetric5giátrị cũ được so đúng. 3filecodeAST/emptyoutputs/CPU/noAPI/noTorch vànotebook6.4KB PASS; scorer chungkhôngđổi. MockPTB chỉkiểm traI/Oshape, khôngphảiJava thật. KhôngauthorlocalJava/scorer/model/API/notebookrun. Kết quảthực tếchờuser chạy.

Sheet Kếtquảmới A44:L47+note49 đãghi pending,10metriccũ và6ô3metricmới đểtrống;48cells đọc lạiPASS. Không đổi bảngC/P34–41. APIformatQA, khôngclaimnativevisual. KhiusergửiZIP sẽaudit/ghikếtquảvàbổsungbảngbàibáo; không mởthêmtraining.
