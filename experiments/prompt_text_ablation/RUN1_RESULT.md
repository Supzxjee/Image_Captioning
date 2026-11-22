# Zero-text run1 — dừng có checkpoint, chưa đủ10epoch

22/11/2026. User cung cấpversion192, tiếp tục notebook ducanh3/notebook7bc83c4c76 theo nguồn gần nhất. [Audit](RUN1_RESULT.json), [resumeAPI](FULL_ZERO_PROMPT_TEXT_RESUME_V192_API_2GPU.ipynb), [quy tắc continuation](RESUME_V192_PROTOCOL.md).

Nguyên nhân xác định từ traceback hai rank: **TimeoutError: Saved before parent/worker cap**. Parent chỉ bọc thành CalledProcessError. Không có bằng chứng lỗiCUDA/model; CUDA/zero-text/DDPguardsđạt, toàn bộ logNLL/lr/gradnormhữu hạn. Safetydeadline có dự phòng trước parentcap11.5h; không gọi đây là hếtquota hoặc runtimechínhxác11.5h.

Snapshotlatest epoch8/cursor17456/update141433/captionevents4525821. ĐãVALID7epochs; còn255updatescuốiepoch8 +17711epoch9 +17711epoch10 =35677updates. Cần resume **latest**, đồng thời giữbest7/history, khôngtrainfresh hoặc dùngbest7 để bỏ phần đãtrain.

| Epoch đãVALID | CIDEr×100 | NLL |
|---|---:|---:|
| 1 | 97.0869 | 2.686091 |
| 2 | 107.9783 | 2.430391 |
| 3 | 112.2586 | 2.325646 |
| 4 | 114.2332 | 2.268999 |
| 5 | 117.5600 | 2.224523 |
| 6 | 117.9951 | 2.193350 |
| 7 | 119.4168 | 2.174477 |

Best tạmepoch7/CIDEr119.4168. Chưa chấmCHAIR/qualitybest-end do runner chỉ gọi sau đủ10epochs. Không kết luận prompt chữ có/không có lợi bằng cách so best7 này với fullbest9. NhánhM2-only vẫn đểsau; khôngSCST/TEST trong continuation.

Audit đạt:108fileZIPCRC/SHA,41sourcekhớpnotebook,35000captionVALID/7cohort5000ID cốđịnh, mọi evaluationhash/CIDErmean/NLL theo token/meanlength vàtraininglogSHA.1420dònglog, lastloggedupdate141377; latestcheckpoint141433 có56updateschưađếnmốcghi100, không phải mất bước. Hai rankguard cùng53.183.736params/182statetensors/initialstate vàtextinvariance; positivecontrolmemorydelta0.55382 trêncopy, original khôngđổi. Syntheticguard không là kết quả chất lượng.

Latestreceipt SHA1d7008c2…; bestreceipt57d1c0b2… . Reviewkhông chứa binary nên chưaload/hashcheckpointthật local. Verifierresume kiểm binarySHA từAPI/mountedoutput trước khi runnerload. Cần giữ optimizer/RNG/source/software/data như cũ, khôngresetlịchwarmup. Notebookresume khôngthay bất kỳ41trainingpayload/source_receipt; wrapperAPI riêng,48metadataseals, cap6h khôngETA/billing. Fourstatic/mocktestsPASS;actualAPI/binaryrestore/CUDA/user-runpending.

Sheet Kết quả mới!A311:D321 đãghi/đọclại theo trạngtháipartial, formatmetadatađãxem, chưavisualnative. Không sửaWord, khôngchạyTorch/model/scorerlocal. Giữv192fulloutput;ngườidùng chạyresumeđể hoànthànhlịch10epochđãduyệt.
