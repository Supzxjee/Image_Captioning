# Phrase decoder: CE đối chứng với CE + loss hai chiều

**Đã chuẩn bị, chưa chạy. Người dùng tự chạy Kaggle; mỗi notebook một nhánh trên cả2 T4.**

| Notebook | Nội dung |
|---|---|
| `PHRASE_DECODER_CE_2GPU.ipynb` | Positive CE sinh cụm có xác nhận |
| `PHRASE_DECODER_CROSS_2GPU.ipynb` | Cùng CE; thêm0.25 logistic đối chứng từupdate61 |

Import hai notebook vào hai tài khoản nếu muốn chạy song song, Internet ON, T4×2, Save & Run All. Không tự launch notebook từ script local. Mỗi file dưới40KB, code/data đã nén nhúng, không cần tải thêm source. Dataset private cần được chia sẻ cho tài khoản bạn.

## Inputs — ba dataset đã có

1. **Nền SCST+CE+object reward đã dùng khởi tạo A–F**:
   `/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth`
   SHA256 `5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75`.
   **Không dùng checkpoint F/bin SHA4938…**. Đuôi `.bin` được phép nếu bytes/hash đúng nền, nhưng F/bin vẫn sai checkpoint.
2. `visual-cache`: `/kaggle/input/datasets/ducanh2403/visual-cache`.
3. MSCOCO2014: `dataset_coco.json` **và images/** tại mount đã dùng trước.

Chỉ sửa các path ở cell đầu nếu mount khác. Không cần legacy prompt-cache, prompt UPT JSON, object detections, CHAIR, V-COCO/raw UPT cache hoặc CLIP model download. Boxes và human targets được nhúng, không lấy predicate UPT làm input. Dataset JSON vẫn cần để tái tạo đúng vocabulary warm start.

## Protocol khóa trước khi chạy

-21 nhóm human khả thi đã review,42 event views/38 ảnh. **14fit/7adaptation holdout**, không trùng COCO ID qua hai phần. Fit14:4skate/5surf/4phone/1bicycle; holdout7:2skate/2surf/3phone. Hai nhóm cùng ảnh trong mỗi phần; không có bicycle holdout.
-Hash và quota theo family, không chọn theo lỗi/model performance. Near-duplicate independence chưa xác nhận. Mọi ảnh là caption train/restval, có thể từng thấy bởi backbone; đây không phải test độc lập.
-300 updates, globalbatch8nhóm (4/rank;16events/32teacher sequences toàn cục); cùng schedule/hash/seed42. Fit14 được lặp nhiều lần: có nguy cơ overfit cao, chủ đích là kiểm tra kỹ thuật chứ không hiệu quả đại diện.
-60 updates đầu CE giống hệt. Lưu hash model và optimizer tại60 để kiểm tra hai nhánh còn khớp. CROSS thêmloss từ61; CE giữweight0 nhưng vẫn tính4combos như CROSS để compute gần nhau.
-AdamW: evidence1e-4, adapter1e-5; decay.01, clip1. FP32 strict controls, dropoutOFF. Các hyperparameters là lựa chọn ban đầu cho sàng lọc, không phải tối ưu đã chứng minh. Không grid search, không chọn checkpoint theoholdout, không SCST trong phase này.

## Module cố định cả hai nhánh

```text
Cached CLIP [197,768]
 ├─ frozen projection + frozen QFormer → G[32,512], không legacy/text
 └─ patches[196,768] + hai box + geometry8D
          → sáu visual queries S/O/R×4 → Z[6,512]

[G,Z] [38,512] → decoder cũ + rank8 K/V updates → cụm BOS…EOS
```

Evidence dùng key/value768→512 khởi tạo từ projection nền, position/membership, geometry chuẩn hóa **chỉ fit**, sáu query học được, role embedding và LayerNorm. S attendboxS, O attendboxO, bốnR attend hình chữ nhật bao cặp. Đây là lựa chọn giữ cố định, **không phải so hard mask với soft mask**. Nearest-patch fallback vẫn cần ở một số box nhỏ, phải ghi nhận giới hạn.

Chỉ evidence encoder và low-rank K/V của cross-attention ba decoder layers được học; decoder gốc/QFormer/projection gốc frozen. Không đưa text/class/action score vào memory; teacher-forced prefix là target ngôn ngữ thông thường. Không có text residual, matcher phụ, selector, crop mới, END_CHUNK hay nối câu. Cached patches đã qua CLIP self-attention toàn ảnh; mask không bảo đảm thông tin chỉ cục bộ.

Đây là thích nghi sinh mô tả có target pair, **chưa là pipeline UPT thực tế hoặc full-caption generation**; không được gọi nó đã giải quyết salience hay novelty.

## Loss được chốt

Với mỗi nhóm có2events và2descriptions, tạo đủ4 teacher-forced combinations. Positive index từng event lấy từ ma trận human, có thể đảo hướng, không mặc định đường chéo theo thứ tự mining.

-CE positive: mean token NLL, **gồm EOS**, bỏPAD/BOS.
-Score đối chứng: mean word logP của cảcụm, **bỏ EOS/PAD/BOS**.
-Cross: trung bình `softplus(score_negative − score_positive)` qua hai events.
-CE arm: `CE + 0 × cross`; CROSS arm: `CE + scheduled(.25) × cross`.

Khi score độc lập ảnh, ưu tiên một câu cố định không làm cảhai chiều cùng đúng. Tuy vậy nền ảnh/geometry/template vẫn có thể là shortcut. Bài thử còn xuất fit-only geometry1NN cùng family để xem data có dễ giải bằng hình học hay không; không có thêm nhánh neural training.

## Preflight và đầu ra

Notebook kiểm tra3 data tests (parent reconstruction check chỉ chạy ởlocal;2 checks còn lại chạy với packet),3 synthetic CPU model tests, và2GPU synthetic DDP smoke trước training thật. Synthetic tests dùng weights random, không đọc checkpoint/cache thật. Sau đó mới verify checkpoint/data hashes/vocab/38 ảnh và stage khoảng vài chục MiB, không copy toàn cache.

Các model/DDP/GPU checks **chưa chạy local vì runtime local không có PyTorch**; chúng là điều kiện bắt buộc trên Kaggle. Local đã chạydata tests/AST/payload roundtrip và xác nhận hai notebook chứa cùng source+manifest. Chưa benchmark thời gian/peak GPU, không hứa tốc độ.

Outputs:

-`/kaggle/working/phrase_decoder_ce_21_review.zip`
-`/kaggle/working/phrase_decoder_cross_21_review.zip`

Gửi cảhai ZIP. Chúng chứa scores/caption trước–sau, on/off, ảnh+box HTML review, history, source/protocol/runtime/hash. Không chứa features hoặc tensors.

Giữ riêng `adaptation.bin` nếu cần tái dùng: **chỉ tham số học mới**, cần source baseline/loader `phrase_decoder_adaptation_v1`; không phải checkpoint caption thông thường và không thay được F/nền.

## Cách đọc và quyết định

`compare_reviews.py` đọcZIP, tái tính scoring và chỉ so sánh khi checkpoint/data/vocab/stagedtensors/source/schedule/runtime, initialpredictions và model+optimizer tạistep60 khớp. Không execute code từZIP/load tensors. Bootstrap theo7groups chỉ mô tả; không chứng minh power, độc lập ảnh hayseed uncertainty.

Mục tiêu đầu là có thêm đáp ứng đúng ở **cụm greedy và đúng cặp**, không chỉ agreement/loss. Không chấm bằng khớp động từ: paraphrase hợp lệ, câu chung chung, wrong instance, extra false claim và truncation phải phân biệt qua ảnh. Caption on/off thay đổi không tự động tốt hơn.

-Nếu không fit nổi: kiểm tra giao diện/gradient/input, chưa kết luận ảnh không có thông tin.
-Nếu fit tốt/holdout kém, hoặc CI rộng: kết quả chưa đủ; không tự tăngweight/updates để cứu trên cùngholdout.
-Nếu CROSS chỉ hơn ởscoring: chưa scale phương án sinh caption.
-Nếu hai nhánh ngang nhau: chưa có giá trị tăng thêm của loss đối chứng; không thêm module chỉ để tạo khác biệt.
-Nếu cụm tốt hơn: milestone tiếp theo phải kiểm tra UPT proposals thật và caption matched trên quy mô phù hợp. Không so điểm phrase với CIDEr1.205513 của A, không tuyên bố vượt baseline.

Không mở rộng việc chấm nhãn lúc này. Đây là bước dùng21nhóm hiện có để kiểm tra kỹ thuật và định hướng, không phải bảo đảm đóng góp mới hoặc cải thiện caption.
