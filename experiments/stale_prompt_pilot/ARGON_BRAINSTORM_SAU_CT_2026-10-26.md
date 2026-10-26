# Brainstorm độc lập sau C/T — căn chỉnh ngữ nghĩa–thị giác cho image captioning

Bạn là cộng sự nghiên cứu độc lập. Tôi cần bạn đề xuất một hướng thực nghiệm có khả
năng cải thiện caption và có đóng góp căn chỉnh ngữ nghĩa–thị giác bảo vệ được cho bài
báo. Hãy kiểm tra giả định, phản biện có bằng chứng và nói thẳng khi chưa đủ dữ liệu.
Không đồng ý để chiều tôi, không đổi tên kỹ thuật có sẵn rồi gọi là novelty.

Tài liệu này là bản tóm tắt có thể đọc độc lập, cập nhật26/10/2026. Tôi sẽ gửi thủ công qua Gemini Argon;
bạn không cần đọc chat cũ, raw ZIP hoặc cả repository. Astra đang điều phối nghiên cứu,
cộng sự Codex triển khai và kiểm chứng. Bạn được quyền bác bỏ đề xuất của Astra/Codex.
Đừng mặc định phải phát triển tiếp phương án nào đã có.

## 1. Đề tài, mục tiêu và giới hạn thực tế

Đề tài: căn chỉnh ngữ nghĩa–thị giác trong sinh mô tả ảnh (image captioning).
Cần một phương pháp có đóng góp hợp lý và caption tốt hơn baseline mạnh đã có.
Dự kiến chốt hướng/thực nghiệm và bắt đầu viết bài báo #1 cuối tuần31/10–01/11/2026.
Đó là mục tiêu thời gian, không phải bằng chứng có thể chắc chắn đạt paper/gain.
Giải thích kết quả âm không tự thay thế yêu cầu có phương pháp.

Hai tài khoản Kaggle, mỗi notebook có T4×2; mỗi notebook chạy một nhánh. Người dùng tự
chạy notebook, không chạy model/training local và không tự launch GPU. Số giờ GPU còn
lại hiện chưa xác nhận; không mặc định số50 giờ từng báo trước các lượt vừa chạy còn nguyên.
Ưu tiên một phép thử có sức phân biệt cao, chi phí thấp; không mở hàng loạt ablation
trước khi biết hướng có triển vọng. Đề xuất thời gian phải tách I/O,cache,train,scoring.

Train có reference captions để học CE/SCST. **Inference trên ảnh mới không có reference,
COCO ground-truth boxes/segmentation hay nhãn quan hệ chuẩn.** Không dùng những thứ đó
như đầu vào inference. Có thể dùng supervision train nếu nói rõ nguồn và hạn chế.

## 2. Baseline A và tài nguyên đã có

A là checkpoint Q-Former captioning đã tối ưu bằng SCST+CE+thưởng đối tượng và huấn
luyện tiếp, không phải baseline yếu/chưa train. **A có prompt gốc**, không gọi A là
mô hình không prompt. Kiến trúc tóm tắt:

- CLIP ViT-B/16 visual cache197×768, projection vào512.
- Q-Former32 queries,2 layers.
- Nhánh semantic prompt dùng CLIP text tokens20×512; attention giữa prompt và visual
  queries, gate/normalization; Transformer decoder3 layers sinh caption.
- Có cache visual,cache embedding prompt gốc và JSON text gốc đã xác minh binding.
  Không tự thay text prompt UPT mới vào cache gốc của A rồi coi như cùng hệ thống.

A SHA256:630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901.
Nguồn dữ liệu chính MSCOCO2014/Karpathy. Locked TRAIN10k và development1000 giữ nguyên
cho các pilot gần đây. Nhiều lượt đã xem dev1000,val5000,test5000; không gọi một trong
các tập ấy là fresh unseen test chỉ bằng cách đổi tên/chia lại. Cần kế hoạch reporting
trung thực và final evaluation phù hợp nếu phát triển được phương pháp.

Tài nguyên sẵn có:

- Checkpoint A, visual-cache và original semantic prompt cache.
- UPT full COCO cache và prompt all-actions đã tạo. Đây là predicted HOI/object
  evidence, **không phải ground-truth**; V-COCO là bộ annotation chuẩn nhưng không phủ
  mọi ảnh/quan hệ COCO. Không tuyên bố UPT tốt hơn ground truth vì cover nhiều ảnh hơn.
- COCO train references,instances/segmentation; có thể khai thác supervision train.
- Một số tài nguyên/pilot Visual Genome,phrase pairs,region evidence,CSA-inspired
  core đã thử. Chưa có caption gain đáng tin cậy trước A để làm kết luận tích cực.
- Cache2.000 ảnh cho thí nghiệm bốn điều kiện bên dưới đã lưu làm notebook Output.

Chúng ta không bắt buộc semantic alignment phải là mỗi từ ↔ một bbox. Có thể ở mức
object,phrase,event hoặc quan hệ, nhưng phải chỉ ra bằng chứng nào và decoder dùng
bằng chứng đó như thế nào. Attention đẹp hơn không tự là caption tốt hơn.

## 3. Những gì đã thử: chỉ rút kết luận đúng phạm vi

Nhiều pilot UPT hidden features,V-G region/noun evidence,native resolution,CSA-inspired
alignment đã không cho gain rõ trước A. Đây không phải bằng chứng rằng mọi phương pháp
alignment đều vô ích hoặc tất cả thất bại do frozen decoder. Không có full ablation
cho phép khẳng định một nguyên nhân chung.

Diagnostic attention trước C/T:

-138/10.000TRAIN đủ điều kiện hai noun classes/unique instance và boxes tách nhau;
  298phrases. Phrase→query→patch maps tốt hơn image-mean và derangement về box mass
  và axis order. Tuy nhiên centroid trong box chỉ28,26%; mẫu eligibility hẹp,nhãn yếu.
- GT-max-query oracle có headroom về localization. Nhưng frozen intervention A/T/W
  tăng raw target box mass trên cả137 ảnh thực sự can thiệp (mean+.02233), vẫn làm
  noun utility/selectivity giảm. CIDEr T−A trên138 ảnh chỉ+.00201,CI95[-.00392,+.00889],
  chỉ3 caption thay đổi. Đây là can thiệp attention cụ thể, không bác bỏ mọi teacher
  hoặc mọi cách co-adapt decoder. Không chọn riêng ca dương sau gate fail để cứu recipe.

## 4. Thí nghiệm mới nhất: stale semantic prompt bốn điều kiện

Giả thuyết: nếu prompt đẩy decoder gọi tên đối tượng khi bằng chứng ảnh suy yếu,
hạn chế tác động đó có thể cải thiện caption ảnh thật.

Chọn2.000 ảnh TRAIN metadata/hash-first, một class/ảnh,không chọn theo model score.
Target có trong prompt còn giữ trong20 tokens,có reference first-noun one-token alias
và COCO segmentation hợp lệ. Union **mọi annotated instance cùng class**, loại cả
image–class nếu có crowd/invalid segmentation. Mask area1%–40%,fillRGB(123,117,104).
P0 thay mọi direct mention target bằng `object` trong đúng retained window P1;
không lộ thêm suffix do thay từ ngắn hơn,không overflow20 token; encode lại toàn chuỗi.
P1 giữ cache gốc. I0 encode lại pixels bằng frozen CLIP,không zero contextualized tokens.

| Điều kiện | Ảnh | Prompt |
|---|---|---|
|11|I1 gốc|P1 gốc|
|10|I1 gốc|P0 trung hòa target|
|01|I0 che annotated target|P1 gốc|
|00|I0 che annotated target|P0 trung hòa target|

Tất cả dùng cùng reference prefix trước first mention. s_vp là log tổng xác suất
one-token aliases đã khóa của target. Nó không bao phủ mọi từ đồng nghĩa hay phrase.
COCO mask **không xác nhận class đã vắng mặt/không thể nhận diện**. Panel vẫn thấy
silhouette người,bò,outline surfboard và context; có thể còn unannotated instances.
Khi ảnh khó hơn, dựa nhiều hơn vào prompt có thể là hành vi hợp lý.

Frozen A trên128 ảnh TRAIN hash-first:

- b1=s11−s10 mean0.748723; b0=s01−s00 mean1.647003.
- interaction b1−b0 mean−0.898280,paired image bootstrap95%CI[-1.122926,-.689220].
-113/128 ảnh có b0>0.1 đã khóa;112/128interaction âm.
- Mean absolute alias probabilities: p11=.609655,p10=.411095,p01=.477996,p00=.218779.

Có tín hiệu prompt boost trên ảnh che, không chỉ log-ratio ở xác suất cực nhỏ. Nhưng
chưa xác nhận hallucination, causal overreliance bất hợp lý hoặc đúng hướng regularization.

### C/T pilot đã chạy xong và đã kiểm chứng

C = CE factual +.1KL(p_A||p_student)+.1CE neutral prompt +.1L_vis.
L_vis=.5×([.2−s11+s01]_+ + [.2−s10+s00]_+).
T = C +.1L_stale; L_stale=[s01−stopgrad(s00)−.1]_+.

Hai arm dùng cùng2.000aux images,4 forwards,seed/order,không lọc hinge-active.
TRAIN10k một pass,1.250 updates,globalbatch8,auxbatch8 mỗi5 updates.
Main CE dùng đủ5 references/ảnh và global valid-token normalization.
Mở last Q-Former layer,prompt projection/MHA/gate/LN và toàn decoder (~29Mparams).
CLIP,visual projection,các phần còn lại và A anchor frozen;eval-mode có gradient.
AdamW LR5e-6,betas(.9,.999),eps1e-8,weight_decay.01,foreach=False,clipnorm1.
Không SCST/scheduler/sweep/AMP/interim-dev checkpoint selection;end-only.

Audit hai ZIP:

- Đủ1.250 updates; mỗi rank dùng đúng1.000aux rows;rank final tensors đồng nhất, frozen state
  và anchor giữ nguyên;source hashes, cohort, policy,cache/stage identity khớp.
- **A predictions,per-imageCIDEr và toàn bộ CHAIR outputs C/T trùng hoàn toàn**.
-14runtime source hashes/arm verified;metrics tổng hợp lại từ saved outputs.
  Checkpoint/HDF5 bytes không nằm trong review ZIP nên không verify trực tiếp local.

| Model | CIDEr ↑ | CHAIRs ↓ | CHAIRi ↓ | Object recall ↑ | Mean words |
|---|---:|---:|---:|---:|---:|
|A|1.205513|52/1000=5.20%|53/1632=3.2475%|1490/3253=45.8039%|10.213|
|C|1.186127|55/1000=5.50%|56/1631=3.4335%|1486/3253=45.6809%|10.265|
|T|1.178205|44/1000=4.40%|45/1597=2.8178%|1466/3253=45.0661%|10.305|

Raw greedy trên cùngdev1000,PTB/CIDEr dùng fixed full reference set như A/H trước đó;
CHAIR là automatic object metric,không kiểm đúng quan hệ hay factual truth toàn câu.
10.000paired image bootstrap samples,seed42:

| So sánh | ΔCIDEr | 95%CI |
|---|---:|---|
|C−A|−.019385|[-.036395,-.002481]|
|T−A|−.027307|[-.048591,-.005977]|
|T−C|−.007922|[-.024298,+.008236]|

T giảm8 caption bị CHAIR flag nhưng mất24lượt unique correct object/image;recall giảm
.7378 điểm%,vượt guard.5 điểm%.CHAIRs T−A CI[-.018,+.001] còn cắt0. Không chỉ do caption
ngắn vì meanwords tăng nhẹ. Không có post-training four-condition probe để khẳng định
nội bộ alignment/reliance đã cải thiện hoặc thất bại.

Gate khóa trước: T−A>=+.01CIDEr,both CI lower(T−A,T−C)>0,CHAIRs/i không xấu,recall
không giảm quá.005. **Không đạt**. GiữA; không full, sweep hoặc thêm seed để cứu recipe này.
C cũng giảm nên chưa quy toàn bộ regression cho stale term;T−C CI cắt0.

Chi phí thực: train+end greedy C178.68s,T194.22s;staging main HDF5 C62.84 phút,T18.87 phút.
Bottleneck là I/O,không phải vài phútoptimizer. Main stage~7.1GB nằm trong/tmp,có thể
đã mất khi đổi session,không nằm trongreviewZIP. Đừng đề xuất nhiều lượt chạy bằng giả định
có sẵn stage hoặc nghĩ tổngnotebook chỉ3 phút. Cacheaux2.58GB đã lưu.

## 5. Astra vừa đề xuất gì — cần phản biện độc lập

Astra dừng mở rộng C/T và đề xuất **một đối chứng continuation-only C0**:

L_C0 = CE +.1 KL(p_A||p_C0).

Giữ đúng initializationA,trainable29M,AdamW,seed/order,TRAIN10k,1.250 updates; bỏ mọi
auxiliary losses,đánh giá end-only dev1000.

Mục đích: tách continuation/training regime khỏi auxiliary recipe. Nếu C0 cũng giảm,
regime đã gây regression dù khôngaux; nếu C0 hơnC,có thêm bằng chứng chống tổ hợpaux.
CI cắt0 là chưa rõ,không phải chứng minh ổn định. Ưu tiên reuse main stage đúngSHA nếu
còn; nếu mất phải nêu chi phí dựng lại. Không tự launch GPU.

**Astra nói rõ C0 không phải phương pháp mới hay paper direction đã có gain.** Đây
là đề xuất kiểm soát điểm mù trước lần thêm module/loss tiếp. Bạn có quyền đồng ý,
bác bỏ,đề xuất chạy song song về mặt thiết kế hoặc yêu cầu kiểm tra khác có giá trị hơn.
Đừng mặc định mọi hướng mới phải chờ nhiều vòng control; cũng đừng bỏ qua confound.

## 6. Việc tôi muốn bạn brainstorm ở vòng này

Trả lời bằng tiếng Việt,khoảng1.500–2.000 từ,tối đa3 hướng thực chất. Đừng chỉ liệt kê
paper,buzzwords,đổi tên loss hoặc thêm một head mà không giải thích cách ảnh hưởng
caption. Không bắt buộc kế thừa stale loss,CSA,UPT hay frozen decoder; được phép đổi
mức biểu diễn,giao diện hoặc cơ chế học nếu hợp tài nguyên.

1. **Chẩn đoán ngắn:** đâu là bằng chứng,đâu là suy đoán? C0 có đáng chạy trước không,
   và chính xác nó trả lời/không trả lời câu hỏi nào? Có kiểm tra zero-GPU rẻ hơn không?
2. **Tối đa3 hướng:** mỗi hướng phải có giả thuyết về nút thắt khác nhau,cơ chế cụ thể
   để semantic content dựa vào visual evidence và decoder thực sự sử dụng nó. Nói
   khác biệt thực chất với recipe đã fail; không suy diễn vì model frozen là nguyên nhân.
3. **Chọn1 hướng:** sơ đồ tensors/module,trainable/frozen,loss có công thức/ký hiệu,
   nguồn positive/negative/supervision,và inference trên ảnh mới thực hiện thế nào.
   Thuyết minh vì sao có khả năng tăng CIDEr mà không chỉ giảm object recall.
4. **Novelty audit:** phần kế thừa là gì,phần có thể mới là gì? Chỉ ra2–3 primary works
   gần nhất và khoảng cách cụ thể; nếu chưa truy cập/đọc hoặc chưa kiểm tra được
   trích dẫn,hãy nói chưa xác minh. Không tự gán venue/rank,Q1/Q2/fulltext đã đọc.
   Bạn được đề nghị cần thêm paper trước khi claim,nhưng đừng coi tênmodule mới là evidence.
5. **Thử nghiệm nhỏ nhất để quyết định:** baseline/control bắt buộc,cỡ train/dev,
   seed/budget/endpoint/guard,early-stop,Go/No-Go và chi phístaging táchtrain.
   Không dùngtest để chọn phương pháp,không đổi metric/split sau kết quả.
   Nếu cần unlock decoder hay SCST hãy nêu rõ lợi ích/rủi ro và cách giữ đối chứng.
6. **Phản ví dụ/shortcut:** mô hình có thể đạtloss mà bỏprompt,đổi từ ngoàialias,
   học artefact,memorize language prior hoặc dùngnguồn không có lúc inference như thế nào?
   Chọn một phép kiểm có sức bác bỏ cao nhất để phân biệt captiongain với alignment.
7. **Chốt hành động:** một bước tiếp theo,không phải một chuỗi 10 lượt chạy. Nêu kết quả nào sẽ làm
   bạn đổi ý hoặc dừng. Nếu không có hướng đáng tin trong ngân sách,hãy nói rõ và đề xuất
   điều chỉnh phạm vi thay vì hứa một tên phương pháp sẽ cho paper cuối tuần.

Bạn có thể khác ý Astra. Chúng tôi cần một kế hoạch có thể sai và kiểm chứng được,
không cần một câu trả lời bảo vệ mọi lựa chọn cũ. Không yêu cầu bạn viết code ở vòng này.

## 7. Nguồn kiểm chứng nếu muốn đọc thêm (không bắt buộc)

Báo cáo C/T có số liệu và audit: 
https://github.com/Supzxjee/Image_Captioning/blob/ab825f6/experiments/stale_prompt_pilot/RESULT.md
Bộ module/runtime đã static-review:
https://github.com/Supzxjee/Image_Captioning/tree/4707b98/experiments/stale_prompt_pilot
Frozen preparation evidence:
https://github.com/Supzxjee/Image_Captioning/blob/2c14eeb/experiments/stale_prompt_preparation/RESULT.md

Các link là tài liệu của thí nghiệm này,không thay cho peer-reviewed evidence về novelty.
Nếu không mở được link,context ở trên đủ để brainstorm; đừng giả vờ đã đọc code/paper.
