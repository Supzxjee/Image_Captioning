# Đánh giá test5000: QFormer gốc và checkpoint fulltrain object0.5

Trạng thái: chuẩn bị notebook thủ công, chưa chạy GPU hay có metric test mới. Chỉ inference/evaluation, không train hay chỉnh hệ số.

## Các lựa chọn đã khóa trước test

- OriginalQFormer epoch10 SHA256 `155106910d49715f8065a444baca3321bb5e668a1f684ea3c78f860de8a39012`.
- Object0.5: checkpoint cuối fulltrain đã được chọn trênvalidation1000. FilelegacySHA256 `5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75`; SHAfileZIPgốc `6225193761e2519994e9922376db9a83b7b0e05b0cbe2cef251cdf166ce7804c`. Việc chuyển định dạng trước đã kiểm109tensor vàtoànbộmetadata giống nhau;hashfile khác do serialization.
- Beam5,candidate5; fixedrerank .3decoder/.4CLIP/.3OCC;khôngUPT,khôngGainGuard. Báo cảraw/fixed cho2checkpoint;primaryfixed. Không chọn lại môhình theo kết quảtest.
- Strictdeterministic inference,TF32off,mathattention vàCLIPrevision đãkhóa. Originalvàselected được tính lại trên cùng5000ảnh, không so trực tiếp với metric trên4000validation.
- SCST+CE được quy tắcvalidation chọnfallback vềoriginal, nên không tạo một hàngcontrol trùngbaseline rồi gọi làđối chứngđộc lập.

Karpathytest5000 là split test đã từng được đánh giá trong lịch sử đềtài. Lần này chỉ khóa cấu hình cho phép so sánh mới, **không gọi là tập test hoàn toàn chưa từng xem**. Không chỉnh hệ số/checkpoint/reranker theo kết quả lần này.

## Chạy thủ công

Trong **QFormer UPT Val5000 20261015**, thay cell/notebook cũ bằng `SCST_OBJECT_W05_LOCKED_TEST5000.ipynb` hoặc `kaggle_scst_object_locked_test5000.py`. Bật Internet vàT4x2. Tênnotebook cũ không quyết địnhsplit; runner lấy đúng `split=='test'` và5000IDduynhất.

Giữ5datasetcũ vàthêm dataset checkpoint mới:

```text
/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth
```

Đườngdẫninputrõràng ởđầucell đểsửa trực tiếp. Không dùngoutput versioncũ vàkhông quétInput. KhôngcopyHDF5sang/tmp. Repo nằm `/tmp/Image_Captioning-scst-locked-test` đểkhông đưa hàngtrămfilemãnguồn vàoOutput.

Launcher kiểmdependency/testsuite/CLIP/metric trước inference. Runner kiểmSHA5fileinput,selectedcheckpointmetadata,cachevisualtest5000,detectorcoverage vàCHAIRcoverage trướcchạy. Originalvàselected lầnlượt sinhbeam, mỗi lần chiaảnh trên2GPU;CLIPscoring dùngGPU. KhôngDDP/optimizer/lossbackward/training.

OUTPUT mới `/kaggle/working/qformer_scst_object_w05_test5000_locked`. Nếu đãcó thì chọnOUTPUTmới, khôngghiđè. GTđượcghi `test_gt.json`;worker dùngtestdataframe, khôngvalidation.

Khi `LOCKED_TEST_COMPLETE`, gửi **`qformer_scst_object_w05_test5000_locked_review.zip`**. ZIPchứa protocol,runtime,predictions,beamjournals vàmetric/CHAIR, khôngweights. Checkpoint vẫn nằm datasetđãlưu. Nếu lỗi saughi protocol, đónggói `failure.json` khi cóthể;INCOMPLETE khôngphảikếtquảtest.

## Kiểm thử cục bộ

12testsCPU PASS:5tests testpartition/metadata vàmockedintegration kiểmđúngsplit,testGT,metadata,testworker vàvalidationdefaultkhôngđổi;7testsfullprotocol/lazyrewardregression. Tests dùngfixture, khôngphảisuyluậnQFormer5000ảnh. Syntax vànotebook khớplauncher;chưa đo thời gian hoặc metricGPU. Người dùngtựrun;assistant khôngsubmitnotebook.
