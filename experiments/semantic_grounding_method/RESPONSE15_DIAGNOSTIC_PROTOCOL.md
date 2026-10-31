# C0 response15 — user-run diagnostic, chưa huấn luyện

User đã đồng ý notebook kiểm tra decoder sau khi xác nhận15cặp. Phạm vi: C0 frozen, pinnedCLIP, 30ảnh,60câu nhãn; không backward/optimizer, không SCST/CEtraining, khôngCIDEr/CHAIR trên15cặp.

Input: ZIP nhỏ gồm15cặpJPEG nguyênbytes đã được xác nhận, filenames COCO để lookup đúng prompt, nhãn, native-vocab hash và source hashes. Không tái tạo prompt từ nhãn hoặc caption tham chiếu. Không public ảnh lênGitHub; người dùng upload gói private vàoKaggle. Gói cómanifest tên riêng để tự tìm root dùKaggle giải nén.

Checkpoint C0 version112 SHA256 5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d. Strict native state-dict load, frozen, tensor digest phải bằng969345610f41fd61efc40db59e7665d85ebb01c746146b2c1997a3e866813ee4. KhôngfallbackA/C hoặc checkpoint khác. Native vocab lấy từcheckpoint và so fullmappinghash đã kiểm. Legacy prompt file SHA d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d, lấy15prompt theo exact COCO filename. Thiếu/ambiguous thì dừng.

Conditioning: original native prompt được giữ NGUYÊN cho hai ảnh trong pair, cảscore vàgreedycaption. Prompt không cập nhật theoedit; edited side là fixed-original-prompt intervention, không gọi fullnativepipeline chạy trênảnhmới. JPEGgốc làVGsource đượcuserxem, không claim byteparity vớiCOCOvisualcache cũ. Train/restvalsource cũng không được gọi chưa từngC0thấy.

CLIP openai/clip-vit-base-patch16 revision57c216476eefef5ab752ec549e440a49ae4ae5f3, nativeResize224×224bilinear/mean/std, fp32/noautocast; last_hidden_state→fp16→fp32 mô phỏng cacheprecision. Khôngcrop/CLIPProcessor tựđổi preprocessing.

Score: mỗi cặp Changed/Stable tìm longest common wordprefix, teacher-force từngcâu trêncảhaiảnh. Mean logprob chỉ từ chỗhai câukhác nhau tới cuối, khôngEOS; giữ cùngcandidate/context khi soảnh. Multiwordtail dùngprefixriêngđúngthứtự. Đây là contrast của normalizedspanloglikelihood, không phải probability hiệu chuẩn hoặc whole-sentence logodds.

Báocáo d_changed(original), d_changed(edited), chênhlệch, hai phía xếp đúng vàdirection đúng; Stable preference hai phía vàđộbiếnđộng. Không đồngnhất “deltađúngchiều” với “đúngcảhaiảnh”. Greedycaption max30tokens: ghi câu tựsinh vàcờtextchanged, không tự chấm semanticcorrect bằng substringmàu. XuấtHTML ghépảnh/caption/nhãn/điểm đểuser/rootđọc.

Sanity trước15pairs: same-image+same-prompt pair phải cho logitclose rtol1e-5/atol1e-5 vàgreedycaptions giống nhau. Dùng toleranceFP32 thay cho bitexact giữa các batch. C0modeldigest trước/sau không đổi, tấtcảrequires_gradFalse vàgradNone. Khôngdiagnosticzero/blankprompt rồi gọi đó làC0native. Runtime1GPU, khôngDDP;30ảnhkhôngcần2worker. NếuKagglechọnT4×2, chỉGPU0dùng. GPUinferencecap1800s, notebookrunprocesscap2400s; capkhôngphảiETA/quotabalance. Người dùng tựSave&Run, agentkhôngsubmit.

Notebook giữsourcecommit+sourcefilehashes, logheartbeats, checksmetadata trướcmodel, reviewZIP ngay cảkhi fail, statusINCOMPLETE khi không đủ15records. Khôngxuấtcheckpoint/KEEPgiả saufail. Localchỉmetadata/NumPy/static/mock; actualCUDA/modelchưa kiểm chođếnuserchạy.
