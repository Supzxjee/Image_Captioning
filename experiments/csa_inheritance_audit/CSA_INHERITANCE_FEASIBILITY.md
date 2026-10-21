# Kiểm tra khả năng kế thừa CSA-Net — 22/10/2026

**Kết luận: có thể kế thừa khối căn chỉnh ở mức source/tensor; chưa thể gọi là tái hiện toàn CSA-Net hoặc duyệt chạy pilot.** Mã lõi và hai semantic bank có sẵn. E2 hiện tại của mình là CSA-inspired, không phải bản tái hiện khối gốc.

Không chạy code tham chiếu, model, inference hoặc training. Đã đọc PDF16trang do người dùng cung cấp, mã chính thức và config/loss; kiểm tra SHA/Gitblob, AST và header/storage của hai bank bằng `pickletools`, không unpickle hay `torch.load`. Chưa kiểm tra tương đương số học/gradient.

## Nguồn được khóa

- Bài báo: Jingyu Li et al., *Cascade Semantic Prompt Alignment Network for Image Captioning*, IEEE TCSVT34(7),5266–5281,2024; [DOI](https://doi.org/10.1109/TCSVT.2023.3343520). PDF người dùng có SHA256 `63ae0db0ee4a1eeeb41a4a241a5ff186bae342caf180fe7e5b2d2875d86575eb`.
- [Repository chính thức](https://github.com/CrossmodalGroup/CSA-Net/tree/0cc570ba7c50ef08c46a21439fe73ef0a5df97ab), commit `0cc570ba7c50ef08c46a21439fe73ef0a5df97ab` (HEAD tại lần kiểm tra). Tree API không bị truncated.
- [Lõi CSA](https://github.com/CrossmodalGroup/CSA-Net/blob/0cc570ba7c50ef08c46a21439fe73ef0a5df97ab/models/caption/csa_net.py), SHA256 `d6bc6d8072c03cb7d42bc8ffcc48e48cefb03b1b128e081b608902d845b6ded1`.
- Config, detector, caption decoder, engine và prototype hashes nằm trong `CSA_INHERITANCE_FEASIBILITY.json`. Raw PDF text/source/banks chỉ lưu local trong `review/`, không đẩy lên repo của mình.

## CSA-Net thực sự nhận gì và làm gì?

| Nguồn bài báo | Mã chính thức | Đầu vào → đầu ra / ý nghĩa |
|---|---|---|
| PDFp4–5, Eq(1)–(2) | `csa_net.py:55–57`; `detector.py:56–60` | Detector Swin/deformableDETR cho region feature `B×150×512`. Hai bank RegionCLIP chung cho mọi ảnh: noun `4764×1024`, verb `975×1024`. Không phải prompt UPT riêng từng ảnh. |
| PDFp5, Eq(3)–(5) | `object_semantic_agg_k/v`, `relation_semantic_agg_k/v`; forward70–80 | MLP tạo trọng số theo các từ, softmax theo chiều số từ; chiếu1024→512, tổng hợp mỗi bank thành150prototype512D. Prototype là hỗn hợp học được, không tương ứng một-một với một từ/đối tượng. |
| PDFp5–6, Eq(6)–(8) | `csa_net.py:82–89`,101–107,117–122 | Mức phù hợp giữa prototype và visual tokens → soft pooling trên vùng → trọng số prototype theo ảnh → semantic tokens được nhấn mạnh. |
| PDFp6, Eq(9)–(11) | SelfAtten138–173; CrossAtten175–211; gate/out_proj tại mỗi stage | Self-attention visual+FFN, cross-attention visual queries→semantic keys/values+FFN, gate vô hướng theo vùng và chiếu đầu ra. |
| PDFp6, Eq(12) | `csa_net.py:101–130` | Hai stage object trước, một stage relation sau: verb bank được chọn dựa trên visual representation đã tăng cường object. Trả `Vo,Vr`, mỗi tensor `B×N×512`. |
| PDFp6, Eq(13)–(17) | `cap_generator.py:90–177`, nhánh `parallel` | Decoder đọc hai stream object/relation bằng cross-attention riêng và gate, không phải chỉ nối memory rồi một cross-attention như nền hiện tại. |
| PDFp7, Eq(18)–(19) | `caption_engine.py:318–493` | CaptionXE rồi RL/CIDEr. Không thấy một loss bbox↔word có nhãn riêng. ReleasedRL dùng nhiều beam và reward baseline trung bình beam; khác greedySCST + CEanchor + object reward hiện tại. |

Nói cách khác, kế thừa căn chỉnh ở đây không cần ta tự gán nhãn mọi liên kết bbox–từ. Tín hiệu caption huấn luyện các phép chọn/fusion của khối; việc attention hoặc prototype được kích hoạt **chưa tự chứng minh grounding đúng**.

### Hai bank đã xác minh có thật

| Bank công khai | Shape header | Bytes | SHA256 |
|---|---|---:|---|
| `coco_caption/coco_nouns_emb.pth` | `4764×1024`, FloatStorage | 19.514.104 | `9d2a783f6db5c5afa055484bd41a86ba1feb5365af791dbda8875c5687479ed2` |
| `coco_caption/coco_verbs_emb.pth` | `975×1024`, FloatStorage | 3.994.360 | `7fc0eb7812bbbe0a004490330318dd64a0ea1f61e7a00dc3102a0fdf61bb40fb` |

Tổng23,51MB file. ZIP CRC, kích thước storage và shape metadata khớp; chưa load tensor bằng torch. Với hai bank này, chưa có lý do phải chạy RegionCLIP lại chỉ để dựng cùng đầu vào. Bài báo nói pool được tạo từ vocabulary COCOtrain; header tensor không đủ tự xác minh danh sách từ/thứ tự hoặc dữ liệu dùng để tạo bank.

## Những phần không được âm thầm coi là giống nhau

1. **Cosine và dot product:** Eq(6) ghi cosine có chuẩn hóa, nhưng source dùng tích vô hướng, không có explicit L2normalize trong khối. Tái hiện công thức và tái hiện releasedcode là hai quy ước khác nhau ở điểm này.
2. **Nhiệt độ:** source dùng0,1 khi pooling vùng và0,05 khi softmax prototype; Eq(8) chỉ ghi softmax(R). Cần khóa quy ước.
3. **Gate:** Eq(11) đặt G ở nhánh self; source đặt gate ở nhánh cross và thêm out_proj. Đảo gate có thể là tham số hóa tương đương khi đổi trọng số, nhưng không phải cùng phép tính tại cùng tham số. Không tự sửa rồi tuyên bố numerical parity.
4. **Mask:** source trả `sum(relation_features)==1`; chưa thấy giải thích đây là padding validity. Với Q-Former32query đều hợp lệ, không được dùng biểu thức này như bằng chứng grounding/validity đã đúng.
5. **Decoder:** source tính alpha2 bằng `fc_alpha1` dù có khai báo `fc_alpha2`, và chia tổng cho sqrt2. Đây là điểm đáng hỏi/xác minh, không kết luận tác giả đã dùng đúng bản source này để ra mọi số liệu.
6. **Recipe:** bài báo/config/README có khác nhau về batch,LR,warm-up và lịch freeze. Originalpaper dùng15XE+30SC, warm-up3epoch,2A800,batch32,beam5. Không kỳ vọng trực tiếp điểm145,2 trên pilot Q-Former1250updates/rawgreedy.

Nếu chọn kế thừa code, nên giữ phép tính nguồn rõ ràng rồi kiểm tra output/gradient tham chiếu. Các sửa về file loading/buffer/device phải ghi lại và kiểm tra không đổi toán học, thay vì sửa thuật toán kèm theo mà không khai báo.

## Có thể gắn vào nền hiện tại đến mức nào?

Q-Former của mình trả `B×32×512`; CSAcore nhận `B×N×512` và không ép N=150 trong phép tính. **Tương thích shape là bằng chứng tính khả thi kỹ thuật, không chứng minh query có ngữ nghĩa region như detector gốc.**

| Phạm vi kế thừa | Giữ được | Thay đổi / cách gọi đúng |
|---|---|---|
| OSA (hai stage object) | Nounbank, aggregation,alignment,SA/CA/FFN/gate | Q-Formerquery thay region; decoder/loss nền giữ. Gọi “kế thừa OSA trên Q-Former”. Chưa có relation stage. |
| FullCSAcore (2object+1relation) | Hai bank và toàn chuỗi căn chỉnh | Có hai output Vo,Vr. Cần chốt cách decoder nền sử dụng cả hai; concat vào một memory không giữ nguyên parallel decoder. Gọi “thích nghi lõi CSA trên nền Q-Former”. |
| Toàn CSA-Net | Detector/region semantics,CSA,paralleldecoder,vocab/recipe gốc | Chưa sẵn sàng từ assets/source hiện có; cần bổ sung phụ thuộc và nguồn checkpoint đã xác minh. |

**Khuyến nghị riêng:** nếu chọn nền kế thừa cho đề tài đối tượng–quan hệ, nên bàn với Astra về fullCSAcore, không gọi nhánh noun-only là đã căn chỉnh quan hệ. OSA nhỏ hơn là một phạm vi thành phần hợp lệ, không phải đầy đủ bài báo. Chưa chọn readout,khởi tạo,warm-up hoặc viết notebook trong lần audit này.

### Vì sao chưa chạy nguyên CSA-Net được ngay?

- `detector.py` import `models.detection.det_module`, nhưng **tree chính thức đã khóa không có thư mục đó**.
- Config gọi `detector_checkpoint_4ds.pth`; file không có trong tree, chưa xác minh đường tải bên ngoài. Không thấy link fullcaptioncheckpoint trong README đã kiểm tra.
- Cache `all_splits.h5` là đường dẫn config, chưa có cache tương ứng đã xác minh. CLIP197patch hoặc UPTpair1024vector hiện có không phải CSAregion150×512.
- Điều đó không cản việc đọc/tách lõiCSA độc lập với detector; nhưng cản tuyên bố chỉ clone repo là tái hiện được toàn hệ thống. Không kết luận các tài nguyên đó không tồn tại ở nơi khác.

## E2 của mình trước đây có phải bản kế thừa này không?

Đối chiếu `captioning/models.py:122–190` và `TRAIN_E2_RELATION_V2_1.md`:

- E2 nhận prompt/object tokens theo ảnh từ CLIP và cache; không có hai bank RegionCLIP toàn cục của tác giả.
- Không có bước semantic aggregation từ No/Nr về150prototype như gốc.
- Selector E2 dùng object-context pooled để chấm prompt; không phải weighted regional pooling của CSAcore.
- Một query-to-semantic attention cộng residual gate512chiều; không phải chuỗi2object+1relation với self/crossFFN và gate scalar.
- Không có hai output object/relation cùng parallel adaptive decoder gốc.

Vì vậy E2 là **CSA-inspired** theo source đã kiểm tra. Audit này không đọc lại toàn bộ checkpoint lịch sử và không tự gán tính đúng/sai cho các artifact chưa xem.

## Chi phí và gate trước khi đề nghị user chạy

- Theo kích thước layer khai báo, **fullCSAcore thêm23.161.644tham số**, OSA2stage thêm14.865.046. Đây là phép đếm từ source, chưa instantiate bằngtorch.
- FP32parameter+gradient+2Adamstates tương ứng khoảng370,59MB(fullcore) hoặc237,84MB(OSA), **chưa tính activation,decoder,CLIP/cache,temporaries và khác biệt mixedprecision**. Đây không phải số đo peakGPU hoặc cam kết vừaT4.
- Không cần trích lại semantic bank; có thể dùng cacheCLIP hiện tại nếu chấp nhận công bố rõ adaptation đầu vào. Fullsystem vẫn cần detector/feature khác.
- Chưa đo runtime, chưa đưa số giờGPU. Một core khởi tạo mới14,9–23,2Mparams không được mặc định đã học đủ sau1250updates ởLRbase5e-6.

Trước một pilot: chọn releasedcode hay paper-formula làm chuẩn; kiểm tra output/gradient cùng tensor với chuẩn, nhiệt độ/mask/normalization, bankfrozen/buffer/device, baseline bypass, đườngdecoder dùngVo/Vr và không có captionreference trong inference. Sau đó mới chốt khởi tạo/warm-up/budget và đối chứng A tương ứng.

**Kế thừa giúp có nền phương pháp và đối chứng; không tự tạo novelty hoặc bảo đảm gain.** Đóng góp mới phải là thay đổi giải quyết một hạn chế cụ thể và tốt hơn phiên bản kế thừa trong đối chứng công bằng. Chưa chạy model hoặc có kết quả caption mới trong audit này.
