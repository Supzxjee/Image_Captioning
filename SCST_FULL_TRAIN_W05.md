# Full train SCST+CE với object reward0.5

Trạng thái: notebook được chuẩn bị để người dùng tự chạy; chưa có kết quả GPU toàn bộtrain. Không bảo đảm phương pháp vượtbaseline. Đây là mở rộng dữ liệu của hướng đang có tín hiệu, không phải một đóng góp khoa học mới tự thân.

## Huấn luyện

Hai nhánh đều khởi tạo từ **QFormer epoch10 gốc**, không tiếp tục checkpoint pilot20k:

| Nhánh | Objective |
|---|---|
| Control | CE + .05 SCST(CIDEr_train) |
| Object0.5 | CE + .05 SCST(CIDEr_train + .5 R_object) |

`R_object = .5 * salient_unique_recall - unsupported_unique_count`. Parser `aliases_v2`, nhãn từtrain, detector>=.9 chỉ tạo nhãn trung tính; giữ các quy tắc cũ. Không đổi prompt,decoder,beam5 hay fixed rerank .3decoder/.4CLIP/.3OCC. Không dùng gainranker đãfail.

**113.287 ảnh train/restval**, một lượt, batch8, **14.161 cập nhật mỗi nhánh**, AdamW5e-6, seed42, FP32, dropout off. Hai nhánh chạy tuần tự trênGPU0 với strict runtime; evaluation dùngT4x2. KhôngDDP và không tự động tăngepoch. Thời gian dài hơn pilot20k đáng kể; chưa có phép đo end-to-end để hứa thời gian hoàn tất.

CIDEr reward dùngIDF của **toàn bộ113.287 ảnh train**. Vectorreference tính khi cần, cacheLRU256images thay vì giữ sẵn tất cả để hạn chếRAM; DF và normalizedtrainreferences vẫn nằmRAM. Kiểm thử chứng minh cache này cho score giống hệt bản giữtoànbộ trên corpus nhỏ, kể cả sau eviction. Đây không phải IDF thay đổi theo batch. FulltrainIDF khác20kIDF nên không quy mọi khác biệt chỉ cho sốupdate.

Lưu checkpoint ởupdate**7.081** và cuốiupdate14.161. Snapshot giữa lượt ghi rõcompletedupdates; checkpoint cuối có sốupdate trong training_summary; optimizerstate không được lưu, nên **không hỗ trợ resume training giữa lượt**. Nếu runtime lỗi, giữ cácsnapshot/log có sẵn và gửifailureZIP trước khi quyết định chạy lại. Không ghi đè OUTPUT cũ.

## Chọn checkpoint trước đánh giá phần còn lại

Giữ chính1000 ảnhvalidation đã dùng trongpilot làm tậpchọncheckpoint.4000ảnhvalidation còn lại không tham gia chọn trong lầnchạy này; chúng có thể đã xuất hiện trong báo cáovalidation5000 trước, **không gọi là hoàn toàn chưa từng thấy**. Không dùngtest.

1. Train xong cả hai nhánh, đánh giáoriginal,mid,end của mỗi nhánh trên1000 ảnh.
2. Mỗi nhánh chọn bằng **fixedCIDEr** cao nhất, vớiCHAIRs vàCHAIRi không cao hơnoriginal,recall>=original−.01. Thứ tự hòa:original trước,mid trướcend. Nếu không checkpoint nào cải thiện đủ theo quy tắc, giữoriginal làmfallback. Raw là số phụ, không dùng để đổiquy tắc sau đó.
3. Ghi `locked_selection.json` chứa lựa chọn vàSHAcheckpoint **trước** khi chấm4000ảnh.
4. Chỉ đánh giáoriginal vàhai lựa chọn đã khóa trên4000ảnh, cảraw/fixed; báo riêng1000và4000. Không lấy trungbìnhCIDEr của hai corpus vìIDFkhác. Không tự chọn lại checkpoint dựa vào4000.

Gate cuối: fixedCIDEr của nhánhobject lớn hơn cảcontrolvàoriginal,CHAIRs/i không cao hơn cảhai,recall không thấp quá1điểmphầntrăm so vớitốt nhất. Không tự promote nếugatePASS; cần kiểmtra độổnđịnh và đánhđổi. Nếu fallback vềoriginal thì ghi đúng điều đó, không gọibaseline là một môhìnhđược cải tiến.

## Chạy

Trong **QFormer UPT Val5000 20261015**, thay cell/notebook cũ bằng `SCST_FULL_TRAIN_W05.ipynb` hoặc `kaggle_scst_full_train_w05.py`. Bật **Internet + GPU T4x2**, giữ5datasetinputhiệncó. Đườngdẫn tườngminh ởđầucell đểsửa, không quétInput hoặc tải nguyênHDF5sang/tmp.

OUTPUT mới `/kaggle/working/qformer_scst_ce_object_w05_fulltrain`. Cell kiểmhashinput,dependency,test,CLIP/metricsmoke; runner kiểmđủcachevisualtrain+val,CHAIRcoverage vàtrainlabels trước train. Khôngassistant submitnotebook.

Khi xong gửi **`qformer_scst_ce_object_w05_fulltrain_review.zip`**. ZIPkhôngkèmweights. Giữriêng:

- `scst_ce/checkpoint_step_7081.pth` và `scst_ce/checkpoint.pth`
- `scst_ce_object/checkpoint_step_7081.pth` và `scst_ce_object/checkpoint.pth`
- `locked_selection.json` đểbiếtcheckpointnàođượcchọn.

ReviewZIP vẫn được đónggói nếu train/eval lỗi sau khi protocol đượcghi, có `failure.json`. Nếu dừng thủcông/killedprocess thìarchivecóthểchưađượctạo. Không gọiINCOMPLETE là kếtquảso sánh.

## Kiểm tra chuẩn bị

50testsCPU PASS:7fullprotocol/lazycachetests cộng43SCST/anchor/object/count/replayregressions. Kiểmtra lazyreward không đổiDF/score, train/restvalbaođủ vàtestloại,split1000/4000khôngtrùng,baselinefallback,guardCHAIRcảhai vàrecall,tiegiữsớm. Syntax và notebook/launcherkhớp. Không chạyfulltrain trênmáyngườidùng hoặcKaggle; bộnhớ/thờigiantoànpipelineGPU chưađượcđo.
