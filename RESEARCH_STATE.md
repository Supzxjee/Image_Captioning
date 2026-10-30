# Trạng thái nghiên cứu — đọc trước khi tiếp tục

Cập nhật: 30/10/2026, giờ Việt Nam. File này là bản tóm tắt để tiếp tục công việc; protocol và artifacts được dẫn bên dưới mới là nguồn chi tiết. Không cần đọc toàn bộ lịch sử chat.

## Mục tiêu và cách làm việc

- Tăng CIDEr và giảm hallucination so với baseline mạnh A; cần đóng góp căn chỉnh thị giác–ngữ nghĩa có bằng chứng cho bài báo đầu tiên tại tạp chí đại học trong nước.
- Cộng sự phải phản biện, không chiều ý. Astra định hướng, agent phản biện/triển khai, người dùng chốt hướng và tự chạy Kaggle.
- Không tự chạy mô hình/Torch/autograd local hoặc notebook Kaggle. Kiểm tra metadata/NumPy/static/mock được phép; ghi rõ giới hạn. Quy trình tại [AGENTS.md](AGENTS.md).

## Hiện đang ở đâu

- User đã duyệt giữ K/V làm ứng viên chính, nearest-check rồi pilot D/E nhỏ. Quota mới: khoảng **20 giờ GPU mỗi tài khoản**. Không tự chạy notebook/model; user chạy.
- [Nearest-method closure](experiments/semantic_grounding_method/KV_NEAREST_METHOD_CLOSURE_2026-10-30.md): đủ cơ sở chuẩn bị pilot như một cải tiến kế thừa hẹp; K/V memory, shared values và group averaging đã có prior. Không chứng nhận firstness/novelty hay hiệu quả. [Giả thuyết](experiments/semantic_grounding_method/RELATION_KEY_FACT_VALUE_HYPOTHESIS_2026-10-30.md): D giữ directional K/V; E giữ K và ghép hai equivalent fact views chỉ ở V; cùng dữ liệu/loss/slot/đường caption.
- Đã chuẩn bị [bước CPU chuẩn bị](experiments/semantic_grounding_method/RELATION_KV_CPU_PREPARATION_PROTOCOL.md), chưa bàn giao D/E training. Chỉ official VSR TRAIN + Karpathy train/restval; `contains↔inside`, `touching↔touching`; tránh frame-sensitive labels chưa có code mapping. Nhãn inverse là transport theo giả định literal, không thêm nhãn độc lập. Top6 predicted boxes được GT match; không GT ROI substitution/unknown negatives.
- Raw visual cache có123287ảnh theo metadata nguồn; 15k là staged cohort cũ. Chưa kiểm coverage ID/thực thi HDF5 copy trên Kaggle. Data gates có thể fail; không tự nới ontology/threshold để tăng count.
- NumPy math5, synthetic metadata8, mocked source3 checks PASS (16); chưa neural/Torch/HDF5/model/CUDA. Static package3checks PASS; tổng19checks, raw Git export16checks PASS. CPU notebook `PREPARE_RELATION_KV_CPU.ipynb` pin core `bfc54da`, chưa user-run. Chi tiết nguồn/split ở [pilot preparation](experiments/semantic_grounding_method/RELATION_KV_PILOT_PREPARATION.md).
- Bước tiếp: user chạy CPU notebook, gửi review ZIP; kiểm actual matched facts/MP groups/cache integrity rồi khóa objective/cohort và viết D/E. Chưa có ETA GPU hay run full được duyệt.
- Các hướng cũ CDR-AEV/shared-referent/region-lexical giữ làm lịch sử tại các proposal/decision trong `experiments/semantic_grounding_method`; không tự quay lại hoặc gọi đã được chọn. K/V là ứng viên hiện tại, chưa có kết quả model.

**Expanded4000 đã hoàn tất đúng fixedcore6645673; không dùng hotfixoverlay. [EXPANDED_RESULT.md](experiments/concept_mass_grounding/EXPANDED_RESULT.md), [JSON](experiments/concept_mass_grounding/EXPANDED_RESULT.json), [audit](experiments/concept_mass_grounding/audit_expanded_results.py). Kết quả/audit đãpush`aa7c09b`, sheetR217hàng244đãghi/đọclại xác nhận. C có CIDEr1.207042/230falsecaptions; A1.189771/239; C0 1.204560/235. CIDErC−A+.017271 CI[+.008819,+.025853]; C−C0+.002481 CI[−.001357,+.006263]. HallucinationC−A/C−C0 không qua bootstrap/exactMcNemargates(.306/.180). JointgateFAIL; theo protocol dừngclaim/đầu tư đợt hiện tại, không tuning hay tăngn tiếp. Đã gửi facts cho Astra, chờ quyết định/trao đổi user cho hướng mới hoặc đổi scope; chưa triển khai nhánh mới.**

- 4000uniqueprimary+16devsentinels(excluded),2ranks2000/model,alltensor/checkpoint/source identitiesPASS,paritymismatches0. Metadata/CHAIRcounts/ratios/20kCI kiểm lại, khôngneural/HDF5/officialscorerlocal. ZIP đọc1lần tại `review/expanded_loaded.json` ignored. One seed/reusedval/postpilot; khôngindependenttest/pooled5000.
- C falsementions254→243 vàfalsecaptions239→230 vsA là positivepointsignal thật, CHAIRi không chỉmẫu số. Nhưng95%CIvẫnchứa0 nên chưa chứngminh giảmhalu hoặcaddedbenefitkeep. Không gọi là no-effect/equivalence/harm.
- Checkpoint dataset đã đượcuser-run dùng đúng path `/kaggle/input/datasets/ducanh2403/concept-mass-c-c0-v111-v112`(C.bin,C0.bin), hashesfixed. Prep1447.8s,workermax181.25s; quota14h là báo cáo trước lượt này, KHÔNG coi là hiện tại. LỗiCRLFtrướcđó đã được sửa trước evaluation; [receipt](experiments/concept_mass_grounding/MANIFEST_LF_FIX.md) giữ lại lịch sử.

- [C0_RESULT.md](experiments/concept_mass_grounding/C0_RESULT.md), [C0_RESULT.json](experiments/concept_mass_grounding/C0_RESULT.json), [audit_c0_results.py](experiments/concept_mass_grounding/audit_c0_results.py). Review text cached ignored tại `experiments/concept_mass_grounding/review/C0_loaded.json`, ZIP đã đọc1lần.
- C0 CIDEr1.230741, CHAIRs52/1000, hallucinated mentions53/1646,recall.460498,words10.044. A52caption/53mentions: chưa giảm hallucination; CHAIRi giảm do mẫu số tăng.
- C0−A CIDEr+.025229, paired95%CI[+.006253,+.043992]; C−C0−.005063,[-.014020,+.003438]. Gain vẫn có khi keep=0; chưa chứng minh lợi ích riêng của preservation. Không kết luận preservation gây hại hoặc hai nhánh tương đương.
- Same4000uniqueTRAIN/order/ref/seed42,500updates,batch8,init/frozen/input/tensor/payload hashes;2rankpreflightPASS,Aexact. Effective KL.1 keep0 negative0, zero weighted aux gradient ở46logged rank snapshots.274metadata assertionsPASS; không chạy/load model/H5 local.
- [C0_PROTOCOL.md](experiments/concept_mass_grounding/C0_PROTOCOL.md), [CONCEPT_MASS_C0_2GPU.ipynb](experiments/concept_mass_grounding/CONCEPT_MASS_C0_2GPU.ipynb). Package7c8b602/corebb84be8; ablation được chọn sau pilot C/G, test chưa dùng.
- Giữ `/kaggle/working/concept_mass_C0_v2/checkpoint.bin`, SHA256 `5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d`.
- Người dùng xác nhận nguồn checkpoint: C = notebook version111, C0 = version112 của notebook đang dùng `ducanh2403/2-vi-cai-kia-qua-nhieu`. Notebook SAVE_CONCEPT_C_C0_V111_V112_DATASET.ipynb sẽ tải đúngversion vàfullSHA, khônglatest. Dataset mount đã xác nhận qua lượt evaluationcomplete. User báo14hGPU trước lượt chạy, giờ cần số mới nếu đề xuấtGPUtiếp.
- A checkpoint `/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin`, SHA256 `630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901`.
- Kết quả C0 và audit đã push ở commit `5efa385`; sheet R213 đã xác minh. Đã gửi Astra số liệu và giới hạn, không yêu cầu đọc lạiZIP. Quota24h là báo cáo cũ, không coi là hiện tại.

## Bằng chứng mới nhất: C/G đã hoàn tất

[RESULT.md](experiments/concept_mass_grounding/RESULT.md), [RESULT.json](experiments/concept_mass_grounding/RESULT.json), [audit_results.py](experiments/concept_mass_grounding/audit_results.py). Commit `a33a537`; registry R211. Hai ZIP đã đọc một lần; text cache ignored tại `experiments/concept_mass_grounding/review/C_loaded.json` và `G_loaded.json` trên máy này.

| Cùng fixeddev1000 | A | C | G |
|---|---:|---:|---:|
| CIDEr | 1.205513 | 1.225678 | 1.229535 |
| Caption có ảo giác /1000 | 52 | 51 | 52 |
| Hallucinated mentions / object mentions | 53/1632 | 52/1632 | 53/1633 |
| Object recall | .458039 | .458654 | .457731 |

- C/G đủ500updates,4000uniqueTRAIN; cùng init/frozen/input/tensor identities; preflight cả2rankPASS; A historical caption mismatches0. Metadata audit185assertionsPASS; không load checkpoint/H5 binaries local.
- C−A CIDEr +.020166, paired95%CI[+.002931,+.038610]; G−A +.024023,[+.006684,+.042807]. G−C +.003857,[-.003236,+.011555]. Một seed/dev đã tái dùng, chưa dùng test.
- Không chứng minh giảm hallucination; G không có lợi ích riêng rõ so với C. G có211weak-negativeevents, không phải zero-event. Neutral-prefix separation G cũng không hơn C. Negative là nhãn yếu từ annotation/reference absence + detector veto, không phải verified visual absence.
- Quyết định: giữ C như ứng viên tăng chất lượng; dừng mở rộng G hiện tại. C0 kiểm xem preservation có đóng góp ngoài continuation+KL không. Chưa được gọi C là phương pháp grounding đã xác nhận.

## Bước tiếp theo và điều chưa chốt

1. Dừng mở rộng/tuning C/G/C0 trên cohort này theo protocol đã chốt. Giữ C/C0 làm empirical quality candidates, giữ A làm reference; chưa promote positive preservation thành phương pháp grounding đã có hiệu quả.
2. Astra đã chốt đóng vòng C/G/C0, không thêm run để cứu gate: [EXPANDED_RESEARCH_DECISION.md](experiments/concept_mass_grounding/EXPANDED_RESEARCH_DECISION.md). Giữ C như practical quality candidate, C0 là đối chứng; chưa đạt dual goal/added alignment benefit. Trình user duyệt scope bài thực nghiệm “Đánh giá bảo toàn xác suất khái niệm trong tinh chỉnh mô hình sinh mô tả ảnh”, hoặc trao đổi lại thời hạn nếu giữ mục tiêu contribution mới đã chứng minh. User không chấp nhận đổi scope thành bài so sánh và đã yêu cầu brainstorm lại một đóng góp căn chỉnh; chưa duyệt candidate/code/GPU mới.
3. Candidate [region–lexical](experiments/semantic_grounding_method/REGION_LEXICAL_ALIGNMENT_PROPOSAL_2026-10-30.md) đã được Astra phảnbiện lại: [nearest-methodreview](experiments/semantic_grounding_method/REGION_LEXICAL_NOVELTY_REVIEW_2026-10-30.md). SỬA/thu hẹpclaim, chưa duyệttriểnkhai; cần usercân nhắc contribution adaptation có thểchứngminh với matchedcopy/globalcontrols và Argon. Không promote frozenhost+normalization thành novelty. Chưa code/budget/GPU mới. Khi báo cáo cũ không bỏ C0/gateFAIL hoặc cộng rawCIDEr1000/4000.
4. [Argon CDR-AEV decision](experiments/semantic_grounding_method/ASTRA_CDR_AEV_DECISION_2026-10-30.md): BỎ fullcandidate, không mởpilotveto hoặc orthogonalization/coverage đểcứunovelty. Cần giảthuyết cănchỉnh tiếp theo có semanticunit/evidence/captiondecision và supervision rõ; chưa chọn phươngpháp thaythế. Proposalpointer cũ không tựđộng thành hướngGPU đượckhuyếnnghị.
5. [Proposal mới có điều kiện](experiments/semantic_grounding_method/SHARED_REFERENT_CAPTION_SELECTION_PROPOSAL_2026-10-30.md): Astrachọn hướngwholecaption/sharedreferentgrounding đểpháttriển, chưauserchốt. Minimumđềnghịfit1verifier, soBIND/INDsameweights/features/pool plusC0greedy/pool-base; actualC0headroom/parse&sourcecoverage chưađo, khôngdùngoldbeamoraclecheckpointkhác. Độmớihẹp/cópriorrấtsát, cầnrootphảnbiệnclaimadaptation; chưacode/GPU/extraction.

## Các điểm phối hợp cần nhớ

- Repo GitHub: `https://github.com/Supzxjee/Image_Captioning`, branch main.
- Sheet `Experiment Registry`: `https://docs.google.com/spreadsheets/d/1XZ5iuyCs5JP8VE6dT7AYdVTiY1_nK4C_QdXs5ppw6FA/edit`, sheetId797499113. R213 kết quả C0 ở hàng240 đã ghi và đọc lại xác nhận; R212 ở hàng239; luôn đọc metadata/sentinel mới trước khi ghi.
- Chat Astra: `01a15544-6a7a-7903-a6e1-23e7b9f6caa6`, host local, tên `Đề xuất hướng căn chỉnh ngữ nghĩa–th`. Người dùng đã ủy quyền trao đổi nghiên cứu trong lịch sử phiên.
- A có prompt gốc; không gọi A là no-prompt. I/J và các nhánh trước chưa vượt A; chi tiết nằm trong thư mục thực nghiệm, không cần lặp mọi nhánh để tiếp tục C0.
