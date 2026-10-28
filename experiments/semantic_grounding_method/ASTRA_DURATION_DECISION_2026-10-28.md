# Astra — quyết định một matched duration trial tới2500updates

> USER-INTENT CORRECTION: khuyến nghị mới nhất là [ASTRA_NEW_DIRECTION_RELATION_FRAMES_2026-10-28.md](ASTRA_NEW_DIRECTION_RELATION_FRAMES_2026-10-28.md), một hướng relational representation/supervision khác T. Các đề xuất T/E+ diagnostic và C/H/T duration bên dưới là lịch sử, không còn là hướng chính hoặc launch instruction. Kết quả thực nghiệm T được giữ nguyên.

> Quyết định mới nhất sau MỘT vòng brainstorm được yêu cầu: [ASTRA_BOUNDED_BRAINSTORM_2026-10-28.md](ASTRA_BOUNDED_BRAINSTORM_2026-10-28.md). Giữ T nguyên nhưng đặt applicability gate trên128groups trước training; chỉ khi đạt mới đề nghị matched T/E+2500 thay vì mặc định chạy tiếp C/H/T. E+ là Energy-derived positive-share control, không Energy reproduction. Kế hoạch duration C/H/T ở dưới là lịch sử đề xuất, KHÔNG còn là khuyến nghị bước tiếp theo. Không tự thay code/launch theo file này.

> Cập nhật sau literature review 28/10: phần đánh giá originality ở mục 0 được thay thế bởi [T_ORIGINALITY_REVIEW_2026-10-28.md](T_ORIGINALITY_REVIEW_2026-10-28.md). Energy loss (ICCV 2023) là prior gần chưa được tính trong quyết định ban đầu. Astra hạ nhận định từ “novelty potential vừa phải” xuống “cải biên cụ thể, originality/độ đáng kể chưa xác lập”. C/H/T tới2500 vẫn chỉ kiểm duration/utility, chưa đủ kiểm contribution riêng so với Energy-style localization. Không tự thêm arm, đổi protocol hoặc launch GPU từ cập nhật này.

28/10/2026, Asia/Bangkok. Chốt thiết kế nghiên cứu và lịch quyết định. Chưa viết/sửa ML code, notebook hoặc launch GPU. Dựa trên kết quả/source audit và mask inventory root chuyển; không đọc lạiZIP.

## 0. Bổ sung sau steering: novelty là điều kiện chọn hướng

T là nhánh phương pháp ứng viên; C/H là controls. Không chọn HINT-styleH làm đónggópmới, không nhận extendedtraining/262144parameters/IGcompleteness/boxlabels là novelty. Extension ở dưới chỉ kiểm budget-sufficiency; nó không tự giải bài toán originality.

Giả thuyết đóng góp củaT được khóa hẹp: adaptation của một strong frozen captioner phải có attributable normalized-probability improvement cho observedgroundedphrases so với A trên cùngprefix, và phần lớn netimprovement phải được quy cho licensedregion trong khi giới hạn cancellation giữa support/complement. Đây là bài toán incremental grounded adaptation, không absoluteattentionmatching hoặc explanationreweighting nói chung.

| Prior gần | Phần đã có, không nhận mới | Khác biệt cụ thể cần chứng minh ởT |
|---|---|---|
| IG,ICML2017 | Path attribution và completeness | IG được dùng làm accountingtool; không nhận là attributionalgorithm mới. Bài toán optimization là phân bổ actualprobabilitygain của adapter so với một strongfrozenbaseline. |
| HINT,ICCV2019,method4.2–4.3 | Supervise gradientimportance bằng regionranking+taskloss; cócaptioning | T ràngbuộc regioncontribution theo actualgain cóđơnvị logprobability, không chỉ relativeimportanceorder. Need matchedHendpoint-rankingcontrol, không chỉA/C. |
| CDEP,ICML2020,method3.2–3.3 | General explanationpenaltyframework và group/restdecomposition | CDEP framework đủtổngquát để chứa nhiều regularizers, kểcả dạngT. Claim chỉ có thể là formulation/property/empiricalutility của gain-dependentconstraints cho frozenautoregressiveadaptation; không claim inventexplanationregularization. |
| LRP-IFT,InformationFusion2022,Eqs28–32 | Signedrelevance reweightscontext/hidden rồiCE/SCSTfine-tunescaptioner | T khôngdùngownrelevance đểrescalefeatures rồi xemoutputđổi làgrounding; dùng externallypairedphrase/region đểgiám sát nguồn của netprobabilityincrement và so vớistrongA. |

Nguồn primary đã đọc ở vòng trước: https://proceedings.mlr.press/v70/sundararajan17a.html ; https://arxiv.org/pdf/1902.03751 ; https://proceedings.mlr.press/v119/rieger20a/rieger20a.pdf ; https://arxiv.org/pdf/2001.01037 . Tìm kiếm bổ sung tập trung ngày28/10 không thay thế một originalityreview đầyđủ; không có quyền nói chưaaiđãlàm chỉvì chưa tìm thấy exactmatch.

Judgment: noveltypotential củaT có nội dung kiểmđược ở problem/objective, nhưng mứcđộ hiện tại vừa phải và chưa được xác lập. CDEPbao-trùm-framework không tự bác mọiđónggóp cụthể; ngược lại, đổi baseline/notation cũng không đủ. Trước launch, root phải ghi closest-priortable đúngscope bằng cácmethodđãđọc, không cần mở thêmgenericmodelaudit. Nếu đã có prior cùngactualgain-relative-frozenbaseline và same directional/cancellationconstraints, hoặcTkhácchỉnotation, dừng hướngpaperT thayvì tạo tênmới.

Evidence để mở hồsơpaper: T2500 hơnA/C/H vớiqualityguards; alignmentconstraints tổngquát được ra fixedadapter-heldoutgroups chứ không chỉmetrictrongtrain; confirmation không đảo chiều kếtluận; source/claims/priorsoverlap được nêuđúng. Captiongain đơnlẻ hoặcHthắngA khôngđủ. T500 hiệnFAIL; giữkếtquảđó, không giấu dưới môhình extended.

Quyếtđịnh sau steering vẫn là giữT làm candidate và MỘT matchedextensionC/H/T, không mở modulethứtư đểtạobềngoàinovelty. Không ai có thể camkết novelty+gain+côngbố trước31/10 từ evidence hiệncó. Nếu nguyênvọng là phải có novelty đượcchứngminh ngayhômnay thì câu trảlờitrungthực là chưa có; plan này là boundedtest cho một candidate cụthể, không lờiđảm bảo bàiđượccôngbố.

## 1. Quyết định và lý do

Chọn MỘT bounded extension C/H/T từ verified500-update checkpoints tới2500totalupdates. A gốc không train lại. Không mở phương pháp/weight/maskpolicy mới trong khoảng28–31/10.

Kết quả500updates vẫn là COMPLETED_QUALITY_NO_GO theo gate cũ, không được sửa hồi tố thành thành công. Nó chưa là duration/convergence study: chỉ end checkpoint được evaluate, các TRAINblocks dùng khácgroups/policy nên chưa biết devCIDEr đang tăng hay giảm. LogT cuối kém ở auxiliary là dấu hiệu cảnh báo, không phải bằng chứng về learningcurveCIDEr. Warm500updates+end~230s/arm và cache đã có làm một extension có đối chứng đáng chi phí hơn đổi kiến trúc lúc này. Đây là lý do về information/cost, không lời hứa thêmepochs sẽ tăngCIDEr.

Rút giả thuyết small-boxmask là lời giải thích chủ yếu: chỉ105/6177groups và80/3998selectedvisits cómaxcellmask<.8; medianmaxmask1. Bound toán học vẫn đúng cho contributions khôngâm nhưng không giải thích phần lớn targetfailure. Không normalize masks, đổi target.8 hoặc tăngweight để cứu.

## 2. Nhánh, initialization và nguồn

- C/H/T cùng262144head và mọi loss/coefficients của v0.3. C vẫn cógain+cancel; H endpoint-ranking; T directionalgain. GiữA frozen, originalprompt/vocab/encoder/cache/evaluator, TRAIN2k và samepairedgroupschedule.
- Resume MỖI arm từ checkpoint500 của chính arm đó, cả head và AdamW state/step500. Không initializeH/T từC, không resetoptimizer hoặc đổiLR/weightdecay. A dùng exactcachedbaseline verified1000captions, không optimizer.
- Ghi parentcheckpoint/optimizer hash, oldsourcehashes, oldpreparationreceipt, newruntimecommit và policy. Chỉ runtime resume/checkpoint/evaluation được mở rộng; model/loss definitions phải giữ hash tương đương. Reuse immutable prepareddata; source-compatibility phải ghi rõ, không dựng lại20phút cache chỉ vì runtimeSHA đổi.
- Nếu per-rankRNG chưa được lưu ởcheckpoint500, không giả resume là bit-identical uninterrupted training. Khóa continuationphase RNGseed100042+rank cho mọi arm, ghi phase boundary trong receipt. Nếu có exactsavedRNG đủ và đã verify thì dùng nó, nhưng phải chọn DUY NHẤT một policy trước run, giống nhau choC/H/T. Default ở đây là explicit phase-seed vì không giả định checkpoint có RNG.
- Data schedule tiếp tục epoch2..9 (epoch0..1 đã hoàn thành):250updates/epoch x8extraepochs=2000newupdates. Originalgroup rule epoch%group_count tiếp tục. Sốprefixgroupsmới được gặp tăng theo schedule vốn có; vì vậy đây là extension của fullrecipe, không experiment cô lập pureoptimization khỏi groupcoverage. Báo counts, không chọngroups theo modelscore.

## 3. Điểm đo và selection

500 là mốc đã có, giữnguyên. Save/evaluate đồng bộ cảC/H/T ở1500 và2500TOTALupdates. Không thêmđiểm đo nếu curve không thuận.

-1500: diagnostic điểm giữa duy nhất, không chọncheckpoint, không đổihyperparameters hoặc dừng vì chưagain. Không chọnmax của1500/2500. Root nên ghi metrics ra artifact; nó không điều khiển optimization.
-2500: PRIMARYendpoint và checkpoint duy nhất dùng quyết định model của trial này.

Không efficacyearlystop theo1500; chỉ dừng cho nguồn/state/parity/numerical/nonfinite/OOM/budget failure đã khóa. Nếu không hoàn thành2500, đây là incompletecomputationaltrial; không dùng1500 làm primary thay thế hoặc bịa missingmatchedarm.

Gates giữ nguyên trêndev1000rawgreedy/fixedIDF: T-A>=+.01CIDEr; paired95CI lower(T-A),lower(T-C),lower(T-H)>0 và pointT-H>0; CHAIRs/i khôngcaohơnA; recall>=A-.005. Same10000bootstrapresamples,seed42,pairedIDs. Đây là exploratoryoperationalgate trênreuseddev, không correctedpaper significance hoặc equivalence test. Không roundCHAIRi đểcứu tinydenominatorfailure; reportcount vàratio riêng.

H cải thiện hơnA nếu xảyra có thể hữuích cho engineering, nhưng không đổi tênHINT-style thành đónggópmethodmới. ChỉT quaA/C/H mới mở câu hỏi proposedconstraint cógiátrị. Không chọnbestof6headcheckpoints để viếtclaimT thắng.

## 4. Một fixed mechanism endpoint từ dữ liệu đã có

Labelbank4495images có2495images ngoài adapterTRAIN2k. Có thể dùng128hash-first eligibleimages trong phần còn lại, cùngv0.3commonreferent/eventguards, mộtgroup0/image theoexistinghashorder; khóaIDs/groups trước xem outputs. Không manualpairs, modifierexpansion hoặc newlabelontology.

Đây là ADAPTER-HELD-OUT TRAIN images: A đã từng train trêncorpus này; KHÔNG gọi là unseen-to-A, freshbenchmark hoặc factualgold. Nó chỉ kiểm generalization của head/objective ra ngoài2000adapterfitimages, tránh suy mechanism từ owntrainingtrajectories.

Reuse originalCLIP/promptcache, chép riêng sidecar128rows+pairmetadata cóhash (khoảng83MBrawfactualinputs), không đổi prepared3k cũ hoặc parentcheckpoint. Không featureextraction/localML. Nếu cache/source đầyđủ không có thì ghiUNAVAILABLE, không thay bằng predictedboxes/fakelabels; điều đó giới hạn paperreadiness, không tự làmCIDErtrial thành invalid.

Evaluate mỗi parent500 và final2500 C/H/T trên CHÍNH fixed128groups, cùngreferenceprefixes. Chỉforward/nooptimizer. Report Delta,CG/CU,Lgain,Ldirection,cancellation và jointpassrate(Delta>=m,CG>=.8Delta_+,budgetmet), luôn kèmdenominator/headroom/tolerance; không chỉmedianCG/Delta trongsubsetpositive. Đây là weaklabel computationalprobe, không độc lập xác nhận generatedphrasefactuality. Head500 forwardđược chấm ởstageđãchuẩnbị, không reloadarchive để chọncases. Probe không chọncheckpoint/hyperparameter và không thành vòngaudit mới để trìhoãn trial.

Nếu captiongain mà jointconstraint không generalize tốt hơncontrols thì không claim mechanism được xác nhận. Cần đánhgiáactualgeneratedgrounding riêng trước khẳngđịnh captionfactuality tăng; không suy từ referenceprefixscores.

## 5. Budget và execution guard

Bound toàn extension, gồmbootstrap/hash/load/preflight,train,1500/2500evaluation,sidecarprobe vàarchive: tốiđa6T4-device-GPU-hours. Gợiýphân bổ CHaccount<=4deviceh (<=120minwallT4x2), Taccount<=2deviceh (<=60minwallT4x2); không gộp sốdưhaiaccounts. Perarmtrain+end/probe cap45minwall, với parent accounting tổngphase không chỉ trainloop.

Đây là cap, không ETA. Thời gian230s cũ khôngbao hếtcoldstart/Ainitial/IDF nên không dùng làm runtimepromise. Kiểm currentquota/units trước user-approvedrun;25h/account là sốTRƯỚC jobs cũ, không phải sốdưhiện tại. Không chạy trước chỉ để đoquota/throughput. Reuse verifieddata/input receipts; verifysharedreadonlycache mộtlần hợp lý, không lặp nhiều fullhashes mỗi rank nếu khôngcần.

Giữđúngschema/source/outputbindings, ghi elapsedGPUallocatedtime từnotebookbootstrap. Không giảmupdates, đổiquadrature, đổibatch hoặc bỏH nếu sátcap. Nếu chi phí/preflight vượtcap thì stopincomplete, khôngretrytựđộng hoặc tự xin thêmGPU. User đã duyệttraođổi/chốtplan, file này KHÔNG tựauthorize launch.

## 6. Lịch tớiT7(31/10)

-28/10: root phảnbiện/chốt extensionprotocol vàsourcecompatibility; chuẩn bị đúnghai notebooks khi được userduyệt triểnkhai. Khôngmởarchitecture/losssearch mới.
-29/10 trước21:00Bangkok: hoànthành matched2500endpoint nếu userduyệtrun/quota đủ; lockallmetrics vàcheckpoint. Nếu không hoànthành hoặcT khôngqua gates, nói rõ trialchưaestablishmethodgain, không đẩy sang5000/10000updates.
-30/10: nếuT2500qualifies, mới đề nghị một confirmationseed củaC/H/T cùngfixedrecipe/cohort và cấu trúcphase đãkhóa; khôngpickbestseed. Đây là confirmation sau positive result, khôngrescue failedtrial, cần userapproval vàbudget riêng. Báo tấtcảseeds; hai seed vẫn cógiớihạn, không dùng imagebootstrap như trainingseeduncertainty. Đồngthời đóng closestprior/contributiontable và fixedmechanismresults.
-Nếu29/10trialfail, hoặc30/10khôngcóreplication/alignmentevidence hỗtrợclaim, đến30/10tối phải nói: chưađủbằngchứng cho bài#1 về methodmới cógain. Có thể hoànthành thesis/experimentalreport trênA vànegativefindings, không biến chúng thành provedpositivepaper.
-31/10 trước12:00Bangkok: chốtmôhình và artifact cho bảnviết. A là default nếu chưacóreplicablewinner. Positive pilot chỉ cho phép viếtpreliminarymethodresults; paperreadiness cần originalityargument, captionbenefit trướcmatchedcontrols, alignment/generalizationevidence đúngscope vàreproducibleartifacts. Deadline khôngthayđổi tiêu chuẩntruthfulness hoặc đảm bảovenueacceptance.

## 7. Chốt

GO thiết kế MỘT durationtrialC/H/T tới2500totalupdates, khôngđổiweights/loss/semantics. ChưaGOcode/launch từdecisionnày. Nếu primaryfail thì đóng cấu hình ởbudgetđãkhóa, khôngmởfallbackmethodmới trongđêm để giảcócontribution. Cógainkhirunlâu vẫnchưa tựtạonovelty; đây là kiểm định rẻhơn việc từbỏhypothesis khi chưabiếtdurationresponse.
