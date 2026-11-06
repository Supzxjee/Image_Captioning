# NoCSA10 / CSA10: bổ sung đủ8metrics trênTEST5000 — 07/11/2026

Người dùng duyệt ưu tiênNoCSA10 vàCSA10. Một notebookCPU chấm cả hai, không cầnAPI/tài khoản chéo/Dataset/ảnh/checkpoint. Dùng đúng5000caption đãaudit và25010refs nguyên gốc. Snapshotgzip nhỏ trênGitHub (khôngảnh, khôngbinaryweights) khóaSHA; nguồn rawchairinput/metadata/result vàZIPSHA ghi trongmanifest. Refs/IDs đã so bằng vớiTEST184/61 hiện tại, không thay câu hoặc thứ tự theođiểm.

Tái sử dụng scorer COCO1.2/Java8/CoreNLP vàrunnerCPU đã dùng thành công ở185/62, chỉ đổi bộcaption đầu vào. Chấm BLEU1/METEOR/SPICE, replayBLEU4/ROUGE/CIDEr khớp1e-7. CHAIRs/i mang từaudit cũ, không chấm mới. Lưu đủ8chỉ sốraw/×100 vàCSV;complete chỉ khi cả2cohort5000 đủphase. Smoke trướcfull;4hCPU STOPcap/heartbeat30s/preserveoutput/cache/log, không tựretry. Tác giả chỉstatic/metadata/stdlibmock, khôngJava/scorer/model/localtraining. Chưa chấmCSA15/SCST hoặczero/shuffle/VALID tronglượt này.

So sánhNoCSA10/CSA10 là cùng10epochs/beam5/max20, kháccapacity/encoder; NoCSA còn sinh nhiều câu ngắn nênCHAIRs thấp riêng lẻ không chứng minh chất lượngtốt hơn. Không coi là đối chứng riêng nội dungprompt hay novelty. Chưa có3metricmới cho hai baseline cho tới khi user chạy notebook.
