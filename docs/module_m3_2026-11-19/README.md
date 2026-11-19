# M3 và sơ đồ tổng thể M1–M2–M3

- `m3_prompt_visual_alignment`: hình chi tiết M3, gồm chuẩn bị prompt riêng ảnh, nhánh đối tượng và nhánh quan hệ.
- `pipeline_m1_m2_m3`: sơ đồ tổng thể nối bank, đặc trưng thị giác, prompt, hai memory, decoder và reranking.
- Mỗi hình có bản `.svg` chỉnh sửa được, `.drawio` gồm các đối tượng riêng và `.png` xem trước. Bản sao ở Downloads cùng tên.

## Đọc M3 theo luồng

UPT cung cấp đối tượng, quan hệ và box. Các box được ghép với YOLO theo lớp và IoU; box YOLO đã ghép quyết định trọng số chồng lấp trên lưới patch 14×14. CLIP Text Encoder ViT-B/16 mã hóa các cụm prompt và chuẩn hóa L2 thành `t_o`, `t_r`.

Nhánh đối tượng dùng trọng số `W_o` để lấy đặc trưng `v_i` từ memory `V_o` của M2. Nối `t_o`, `v_i` và tọa độ UPT `b_i` (4 chiều), rồi qua Linear1028→512, ReLU, Linear512→512 để tạo slot. Cross-attention dùng `Q=V_o`, `K=V=slots`, chỉ cho phép patch đọc những slot hợp lệ có box chồng lấp patch đó. Cộng phần cập nhật vào `V_o` để có `V_o′`.

Nhánh quan hệ lấy hai đặc trưng `v_s`, `v_t` từ cùng `V_o` gốc. Nối với `t_r` và vector hình học tương đối `g` (8 chiều), rồi qua Linear1544→512, ReLU, Linear512→512. Attention dùng `Q=V_r` và các slot quan hệ làm K/V; mask là hợp vùng hỗ trợ của hai đầu quan hệ. Cộng phần cập nhật vào `V_r` để có `V_r′`. `g` không đơn thuần là nối hai tọa độ box: nó chứa độ lệch tâm, tỷ lệ kích thước, độ chồng lấp và khoảng cách.

Hai memory `V_o′`, `V_r′` đều có196×512, đi riêng vào hai nhánh cross-attention của decoder ba lớp. Patch không có slot hợp lệ giữ nguyên qua residual; các slot đệm không đóng góp vào kết quả.

M1 xây bank chung từ caption TRAIN; M3 dùng prompt riêng của ảnh đang xét. Hai loại thông tin này không thay thế nhau. Rerank dùng điểm hỗ trợ đối tượng từ detector; detector bỏ sót không xác nhận rằng đối tượng thực sự vắng mặt.

## Kiểm tra

Đã đối chiếu code được dẫn ở DESIGN.md, kiểm tra XML SVG/draw.io, AST generator,35/19 connector chỉ dùng đường thẳng hoặc gấp khúc vuông, kích thước PNG và SHA bản sao Downloads. Đã mở ảnh render, chỉnh tràn chữ, bỏ đường giao nhau trong tổng thể và tách đường hình học khỏi đường trọng số patch. Không chạy mô hình, notebook Kaggle hoặc sửa báo cáo.


## Bản sửa theo bố cục mẫu của Tâm

Ưu tiên bản trong `tam_style/`: `pipeline_tam_style` là tổng thể, `m3_tam_style` là hai adapter trình bày theo cột dọc. Các bản SVG/draw.io/PNG cùng tên có trong Downloads. Bố cục mượn cách đặt ảnh phía trên encoder bên trái, các nhánh ngữ nghĩa ở trên, memory qua hàng giữa, decoder và caption ở dưới; không thêm thành phần Fact-RAG hay thay cơ chế mô hình. M3 nhận embedding chữ đã chuẩn hóa từ cache; L2 normalization không mở rộng thành hộp riêng ở sơ đồ tổng thể. Hai nhãn Image I cùng chỉ một ảnh.

Đã mở render, chỉnh nhãn nằm trên đường dây và tách rõ prompt chữ khỏi box trước CLIP Text Encoder. Kiểm tra XML/AST,27 connector ở tổng thể và24 connector ở M3 đều vuông góc, bản sao Downloads trùng SHA. PNG tổng thể2100×1185, M32100×1215.


## Bản hiện tại: ký hiệu trong block và ví dụ cụ thể

Ưu tiên `data_blocks/pipeline_data_blocks` và `data_blocks/m3_data_blocks` (SVG/draw.io/PNG, bản sao cùng tên ở Downloads). Đây là lần sửa theo yêu cầu đưa Vo/Vr/tensor/QKV vào block và thêm nội dung ví dụ. Tổng thể có bank TRAIN với ví dụ từ đã kiểm tra, prompt riêng ảnh với ví dụ cụm đối tượng/quan hệ, hai loại memory và ví dụ caption ứng viên/cuối. Các cụm prompt/caption đều được ghi example/illustrative: không phải kết quả inference của ảnh minh họa. Không bịa vector số hoặc gọi patch tokens là object tokens.

M3 giữ hai cột, mỗi phép toán một block; công thức pooling, input/output slot và Q/K/V nằm trong block tương ứng, residual addition là block riêng. Kiểm tra ở `data_blocks/metadata_checks.json`:26/24 connector vuông góc, không đoạn dây nào xuyên block không liên quan,26 nhãn toán học nằm trong block, các bản sao trùng SHA. Đã mở render và sửa chữ/chỉ số chạm viền. Không đổi mô hình, số liệu, notebook hay báo cáo.


## Bản phác bố cục đang chờ phản hồi

Người dùng đã chê bản data_blocks chưa đẹp. Bản `layout_proof/pipeline_layout_proof` chỉ sửa tổng thể: chữ thường nhỏ hơn, ô xử lý thấp hơn và phần lớn nền trắng, đường mảnh, ví dụ trong hai ô bank/prompt. Chưa mở rộng lại M3 và chưa được người dùng duyệt. SVG/draw.io/PNG cùng tên ở Downloads; đã mở render, chỉnh một đoạn dây bị lệch dưới1pixel sau thu ô, kiểm XML/AST,27 đường vuông góc (gồm2gạch tiêu đề), vị trí nhãn trong block và SHA bản sao. Không sửa pipeline thuật toán hoặc báo cáo.


## Tổng thể mở rõ M2–M3 theo phương án người dùng đồng ý

Bản mới: `expanded_overview/pipeline_m2_m3_expanded` (SVG/draw.io/PNG; bản sao cùng tên ở Downloads). M2 tách noun/verb aggregation, hai lượt object alignment, relation alignment và gated update. Các khối alignment gói cosine reweighting, self-/cross-attention cùng normalization/FFN hiện có; không biến encoder thành chỉ một cross-attention. H/C lần lượt là visual/aligned contexts sau xử lý; T1 là noun prototypes đã reweight, được chuyển sang lượt đối tượng thứ hai.

M3 gồm hai hàng, mỗi hàng5phép toán: box pooling, concatenation, Slot MLP, masked cross-attention, residual addition. Shared visual context chứa Vo/Vr; mỗi operator lấy đúng trường được ghi trong block, cả hai pooling lấy Vo gốc. Attention/residual object dùng Vo, relation dùng Vr. Metadata prompt/ROI độc lập với memory; geometry không đi qua CLIP Text Encoder. Valid mask bao gồm slot hợp lệ và hỗ trợ không gian; patch không có slot hợp lệ có delta bằng0 trong implementation.

Kiểm tra `expanded_overview/METADATA_CHECKS.json`:54connector vuông góc, không xuyên block không liên quan, một wire crossing được tách bằng gap thẳng tại1780/532.5,29nhãn toán học nằm trong block, SHA bản sao đúng. Đã xem render và sửa vị trí wire ở tiêu đềM2/đệm chữattention. Không chạy mô hình, Kaggle, scorer hoặc sửa báo cáo. Người dùng duyệt phương án mở cơ chế, chưa duyệt chất lượng hình cuối.
