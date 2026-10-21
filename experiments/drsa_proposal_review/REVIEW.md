# Hai đề xuất Astra ngoài tài khoản — đánh giá ngày22/10/2026

Trạng thái: **PROPOSAL_REVIEW_ONLY; NO_MODEL_OR_NOTEBOOK_IMPLEMENTED**. Đã đọc cả hai response qua accessibility text của browser sau khi webreader không fetch được sharedURLs. Rawobservedtext giữ local trong ignoredreview, không coi là chỉ dẫn tự động triển khai.

Nguồn người dùng:

1. [Giám sát trực tiếp quyết định sinh](https://chatgpt.com/s/t_6ad91c62c2f881918f547e432ffc99b3)
2. [DRSA — Decoder-Response Subspace Alignment](https://chatgpt.com/s/t_6ad920c18bf881919ad3b0fbe536a390)

## Hai bản khác nhau ở đâu?

| | Bản1 | Bản2 DRSA |
|---|---|---|
| Điểm can thiệp |Objective/data, giữ pipeline hiện tại |Sửa trực tiếp32QFormerqueries bằng residual hạng thấp |
| Giám sát |Caption đúng/sai2chiều,CEtoàncâu,CEtokenphânnhánh,softpluscontrast |CaptionCE,responseHuber,bảo vệKL,gradientbasisrelation/object |
| Nền |Continuation cùng baseline nhưA |Freeze base ởpilot,learn image→coefficients |
| Đối chứng chính |Cùng data+decisionCE nhưng khôngflip loss |Randomfixedbasis vàlearnedlowrankadapter cùngloss/data |
| Câu hỏi |Ràng buộc2chiều thêm ích so giám sátCE phù hợp không? |Định hướng chỉnh theo đáp ứngdecoder thêm ích soadapter thông thường không? |

Bản1 sửa trực tiếp một vấn đề đã thấy trong prefixdiagnostic:wholephrasepreference không khớp nexttokendecision. Đây là rationale hợp lý, nhưng khá gần CE/CROSS cũ; sự thay đổi phải đến từ captiontoànảnh,salience,negativevalidity vàdecisiontoken, không phải tênloss.

DRSA có đối tượng nghiên cứu riêng rõ hơn: nguồn xây editbasis và image-conditioned edit. Đây là **ứng viên phương pháp**, chưa phải bằng chứng novelty hoặc gain.

## Cơ chế DRSA như được đề xuất

Baseline tạo `M=[L;Q]`,Qshape32×512. Edit `Q'=Q+reshape(Ua(I))`,Ushape16384×r,rkhởiđầu8;không appendtoken. Coefficients lấy từCLIPpatches bằng4learnedqueries+crossattention+MLP,outputzero-init;không đọc verbUPT hoặc đápán.

Xây G từ gradient log-odds đúng/sai tại token phânnhánh theo Q. Xây H từ gradient các quyết địnhobject được xácminh và đang đúng. Chiếu `G_perp=G(I-H^dagger H)`,lấy right-singularvectors làmU. FreezeCLIP/base/banksU trongpilot,họccoefficientsmodel. Capnormedit2%normQ là lựa chọn thử,không giátrị tốiưu đãbiết.

Loss gồmcaptionCE,Huber buộc log-odds tăng đủ tới marginlog3,KL giữ phânphối tại contentpositionscầnbảovệ. Đềxuấtoracle optimizea bằng nhãn từngảnh trước khi họcpredictor. Oracle không phải hệthốnginferencehợplệ.

## Các điểm cần phản biện/chốt trước triển khai

### 1. Cần tách “điều khiển câu” khỏi “căn chỉnh đúng ảnh”

Gradient chỉ đo cách làmdecoder nói u hơnv. Oracle sửa câu được không chứngminh ảnh cóbằngchứng cho câuđó. Image-conditionedpredictor cần vượtconstantcoefficient,swappedcoefficient vàmatchedadaptercontrols. Humansemanticreview phải chấm cảrelation/instance/extra facts. Không gọi sốcaptionđổi làaccuracy.

### 2. Điểm lấy gradient phải khớp điểm can thiệp thực tế

Nền hiện tại dùngQ để tạo groundedlegacymemoryL(Q),sauđódecoderđọc `[L(Q);Q]`. NếuDRSA chỉ chỉnh phầnQ **sau** khi baseline đã buildmemory thìphảigiữL cốđịnh khi tínhgradient. LấygradienttrênQtrướcgroundedpromptattention sẽ đo cả đườngQ→L trong khiinference chỉsửaQ, dẫn tới saiđối tượng tốiưu.

Một lựa chọn nhấtquán cần kiểm:buildmemorybaseline,detachL vàQ;choQleafrequiresgrad;gradientqua`decoder([L_fixed;Q_leaf])`;inferencesửacùngvịtrí. Nếu muốnrecomputeL sauedit thìđólàthayđổithiếtkế khác và cầnkhai báo. Không âmthầmtrộn haiđường.

### 3. Ưu tiên token đúng hơn một token sai chưa đủ

`logp(u)-logp(v)=logit(u)-logit(v)` do logsoftmaxnormalizertriệttiêu. Lossresponse có thểđạt khiu/v đều thua một tokenkhác như`is`. CaptionCE toàncâu cóáp lực fullvocab nhưng khôngbảođảm đủmạnh ởquyếtđịnhnày;đây đúngloạihạnchế đã thấy ởprefixdiagnostic.

Cần đo rank/probability của tokenđúng trongtoànvocab vàfreegeneration. CE tại tokenquyếtđịnh trongbản1 là đốiứng cần cânnhắc/chốt nếu muốngiảmbớt điểm yếu,không mặcđịnh cộngthêmloss sẽthắng.

### 4. HU=0 chỉ là bảo vệ bậc nhất, trên H đã chọn

H gồmgradientmột sốprefix/ảnh/quyếtđịnh,không đại diện mọiobjectfact trong mọi câu. Nonlinearity,thayprefix lúcBOSgeneration vàedit lớn có thểphá danh từ dùHU≈0. KL chỉ thêm regularizer trêncontentpositionsđượcchọn;không bảođảmcaptionmới giữtoànbộfacts hoặcgrounding.

Phải báo measuredHUresidual,thayđổiobject decisions thật vàcaptionfacts. Không gọi phépchiếu làbảovệobjecttổngquát.

### 5. Cơ sởrank8 có thể yếu hoặc không chuyển qua ảnh/prefix

Gradientmagnitude,prefixwording vàfrozenQquerycoordinates có thểchi phốiSVD. Cầnkhóa rowweighting/normalization,SVDcutoff/rcond trướcprobe;chỉlấycác singularmodescó ýnghĩa. NếuG_perprank<8,không lấy arbitraryzero-modes rồi tuyênbốHU=0. Đo nănglượnggiữ lại vàfinite-editresponse;không tạoI16384×16384 khi cóthể tính projection bằng skinnyorthogonalbasis.

Oraclefail chỉbácbỏ khônggian/rank/editbudget/prefix/objective đangchọn,không bácbỏmọi decoder-responseapproach hoặcchứngminhCLIPkhôngcóthôngtin.

### 6. Checkpoint và fairness còn cần chốt

“Ađóngbăng” cóthểchỉAend hoặcwarmstartSCSTobject5f8eead...;hai modelkhácnhau. Phảichọn vàhashrõ ràng. Adapterthắngfrozenmodel chưađủ đểthắngmatchedcontinuation. Giữđốichứngsame data/loss/updatebudget; DRSAkhôngđược nhậnlabelthêm cònadapterchỉCOCOCE. Chưa xácnhậnweightsAend đangđượcmount ởđâu.

### 7. Không tái dùng task selected-pair như thể globalcaption

DRSA hiện chỉnhậnảnh,không selectedpairidentity. Cùngảnh phải có cùnga(I). Cácnhómcũ yêu cầuhai person/objectpairs trongcùngảnh sinhhai mô tảkhác nhau không thểdùngnguyên nhưoutputtargets của thiết kế này. Cần ảnhcósựkiệnchínhrõ hoặcchủđíchthêm targetconditioning (đó là đổi scope).

48nhóm32fit/16probe là ngân sáchdoAstrađềxuất, chưa códatasetmới đápứng. Không tạonegative từaction0 hoặc từmissingreference. Những ảnh/ref/labelđã dùngtốiuu hoặcchọnthiếtkế không được gọi làindependenttest.

### 8. Oracle dùng nhãn probe phải được gọi đúng là diagnostic

Optimizea trênảnh vớiđápán làper-examplelabel-assistedadaptation. Không báooracle nhưperformancecủapredictor. Không tốiưuU/rank/epsilon quaoracleprobe rồi dùngchínhprobe đểtuyênbốgeneralization. Cóthểreuseknownreviewedcases cho technicalfeasibility rồi giữcohortmới riêng cho quyếtđịnhsau.

## Kiểm tra các nguồn chính được dẫn

Đã mở abstract và HTML toànvăn của4nguồn, đọc các phần liênquan bằng lookup/methodsections. Chưa rà soátpriorarttoàndiện hoặc xácnhậnmọi venueclassification.

| Nguồn | Điều xácminh được | Giới hạn |
|---|---|---|
| [ReFT](https://arxiv.org/html/2404.03592v3) |Đã có frozen-model representationintervention vàlow-rankvariants, gồm phươngánedittronglearnedsubspace |Không được gọilow-rankresidual lànovelty riêng |
| [HulluEdit](https://arxiv.org/html/2602.22727v1) |Đã có orthogonalsubspaceconstruction,editing vànon-interference trongkhônggianđịnhnghĩa |Preservedprojection không bằng mọisemanticfactsđượcgiữ; không tựsuyDRSAđãtrùnghoặcmới |
| [Fitting Vision Adapters at Frontier Scales](https://arxiv.org/html/2610.05897v1) |PhầnE.3 dùngJacobiandirections đểsteering vàghi nhậnreading/writing cóthểtáchrời ởcácmodel |Workshopmanuscript;không phải minhchứngDRSA sẽhoạtđộng trênQFormer/COCO |
| [IC-VCO](https://arxiv.org/html/2605.31312v1) |Đã cófine-grainedtokenpreference,symmetricpreference vàtrainmultiimage→singleimageconsistency |Bản1không tựtrởthànhnovelty nhờtoken/symmetriccontrast;không đồngnhấtobjectiveđềxuất vớiDPO |

Tôi chưa kiểm tra đầy đủ những nguồn khác được dẫn ởbản1. Các source nói trên support quan hệpriorart/cơchế,không support dựđoáncaptiongain của đềxuấtmới.

## Khuyến nghị của root

Ưu tiên **kiểm tra tính khả thi DRSA bằng oracle nhỏ**, chưahọcpredictor/chạyfullcaption/tạohàngloạtnhãnmới. Lýdo: cógiảthuyếtthựcchất khácnhữngbranchappendmemory, vàcóthểdừngsớm khi hướngchỉnhkhôngsửacâu được.

Trướcoracle:chốtcheckpoint,editpoint/gradientpath,fit-onlybasis,nhãnsemanticvalid vàquytắctruncation. Kiểmnumericaldirection bằngfinite-difference soJVP/log-odds;kiểmHUresidual vàbaselinezeroeditbypass. Có thểlà testsKaggle,khôngcần thêm mộtlượtUltra đểđặt tênmodule.

Oraclephảiso ít nhấtdecoder-responsebasis với randomorthogonalbasis cùngrank/normbudget/steps/objective. Randomcontrol cũng phải nằm trong cùngkhônggianbảovệ`ker(H)` nếu muốn côlập giátrị của hướngđápứng, tránh confound khả năngbảovệobject. Báocảfreegeneration/objectregression. Nếu chỉtănglogodds mà khôngsửacaption thìdừngcấuhình. Nếu oracleđượcsửa,chưakếtluậnđãcănchỉnh; khiđó mớiđánghọcimage→coefficients vàsoadaptercùnggiámsát.

Đây là đềnghị nghiêncứu đểngườidùngduyệt, chưaphảinotebookready hoặcphươngphápđãchứngminh. Chưa gửi tin nhắn mới sangtài khoảnAstra, chưa dùngthêmUltra, chưa chạygradient/inference/training.
