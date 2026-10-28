# Quyết định dữ liệu v2 — pair-first, image-disjoint

28/10/2026. Dựa trên kiểm kê CPU của root; không đọc lại raw/ZIP/ảnh, không code model/GPU. File này sửa **thiết kế cohort trước learning**, không đổi kết quả coverage v1 hoặc tạo quyền launch.

## Quyết định

**Chọn một protocol mới, không dừng hướng vì quota8x4 và không gọi v1 đã pass.** Giữ `DATA_COVERAGE_NO_GO` cho exact policy v1. Protocol v2 giữ cùng nguồn, label/vote policy,16relations, unique GT và predicted-box class/IoU>=.5; đổi đơn vị chọn held-out từ ảnh rời sang cặp phản chứng.

Lý do: với split ảnh80/20 độc lập, một cặp có cả hai đầu ở held-out chỉ có xác suất xấp xỉ.2^2=.04 trước các phụ thuộc. Quota8cho từng predicate không phải power analysis;8cases cũng không đủ làm claim chắc về predicate đó. Coverage của v1 cho thấy GT uniqueness không phải bottleneck:4399/4399records unique,3358records pass detector;297/72pairs pass trong fixed pools. Chỉ thất bại độ phủ theo lớp, không thiếu tổng paired supervision. Tuy nhiên v1 còn có thể lặp ảnh giữa các cặp; v2 sẽ kiểm soát luôn đơn vị độc lập của evaluation.

## Cohort v2: một quy tắc cố định, không tìm seed

1. Dùng **toàn bộ detector-matched records** thuộc master cohort đã khóa; không giới hạn vào fixed pools v1. Không đổi labels, votes, IoU, top12 hoặc thêm raw data. Lập danh sách paired statements bằng cùng exact-statement grouping và one-to-one pairing policy: một record không được nhân lên thành nhiều independent examples. Không Cartesian inflation.
2. Held-out không chứa ảnh của caption-adapter TRAIN2k; master exclusions caption dev/val/test giữ nguyên. A gốc có thể đã thấy các ảnh trong pretraining; đây là **adaptation-held-out**, không unseen-to-A.
3. Chọn **64 held-out pairs,128 distinct images**. Mỗi ảnh xuất hiện trong tối đa một selected held-out pair. Sau đó **loại mọi record của128ảnh này khỏi relational TRAIN**, kể cả những statement khác không được chọn để đánh giá. Các cross-boundary pairs bị bỏ, không kéo cả connected component vào eval hoặc cho một đầu quay lại train.
4. Chia evaluation thành hai strata có ý nghĩa với hypothesis:
   -32pairs **projective/containment**: `on, under, above, below, on top of, beneath, in, inside, within, outside, contains`.
   -32pairs **contact/proximity**: `touching, next to, near, far from, far away from`.
   Mục tiêu là có kiểm tra quan hệ vị trí/bao chứa và một nhóm contact/proximity; không để touching thay thế toàn bộ semantic scope. Không giả mọi predicate ở nhóm đầu đều là quan hệ toán học asymmetric, và không tạo role-swap negatives.
5. Chọn bằng một constrained selection trên metadata:64edges;32edges/stratum; mỗi image degree<=1; không ảnh TRAIN2k. Phải đồng thời giữ được256train pairs ngoài held-out images, và mỗi relation có mặt ở primary held-out có ít nhất một train pair của relation ấy; không vô tình gọi unseen-predicate test là ordinary held-out. Trong các nghiệm, ưu tiên số relation labels được đại diện lớn nhất, sau đó dùng hash ordering cũ làm tie-break/minimum-rank cost. **Không thử nhiều hash seeds.** Đây là stratified sampling theo data/labels, không chọn theo caption scores, truth logits hoặc model errors.
6. Từ các records/images còn lại, chọn **256 relational train pairs** bằng hash/one-to-one policy, giữ cả16labels nếu có. Report counts từng relation và distinct image/statement counts. Không còn gate16train/8held cho bốn labels; không xóa labels ít mẫu để làm đẹp macro accuracy.

Root có thể dùng matching/constraint solver CPU để kiểm khả thi. Một greedy heuristic không tìm được nghiệm không đủ chứng minh dữ liệu vô nghiệm. Chỉ thực hiện **một** thiết kế v2 này; nếu constraints không khả thi hoặc train còn<256pairs, báo `DATA_V2_INCOMPLETE/NO_GO` đúng nguyên nhân, không đổi seed/quota/IoU lần nữa để đạt số.

## Gate và cách diễn giải mới

- Data gate là64valid node-disjoint held-out pairs,32/32strata,256valid train pairs và image disjointness với cả relational/caption-adapter training. Các numerical/source/schema checks giữ nguyên.
- Primary semantic probe: trung bình pair-ranking accuracy của hai strata, trọng số.5/.5. Đồng thời report pooled64accuracy và mọi per-relation denominator; relation ít mẫu chỉ descriptive, không claim generalization cho tất cả16predicates. Vì32/32, pooled accuracy bằng balanced-strata mean trong cohort này.
- Bootstrap paired trên64held-out pairs; endpoint images không lặp giữa các pairs. Đây vẫn là pilot uncertainty conditional trên cohort/seed, không population estimate hoặc official VSR benchmark score. V2 được cân strata nên không đại diện prevalence tự nhiên.
- Gate pretrain cũ có thể giữ ở **>=70%** primary ranking, hơn RelPos>=10pp và CI lower(diff)>0; ngưỡng này là operational gate, không power claim. Không tune nó theo checkpoint. Full caption gates/controls không đổi chỉ vì sửa cohort.
- Sparse per-predicate coverage giới hạn claim; không thay bằng phát biểu “đã kiểm đủ4relations”. Để paper cần cohort lớn hơn hoặc official VSR evaluation với protocol/supervision split minh bạch.

## Việc root nên làm tiếp

Chỉ kiểm kê v2 CPU, lưu manifest mới và feasibility report, freeze trước bất kỳ learned model scores. Giữ nguyên inventory/BOX_COVERAGE_RESULT của v1 và ghi rõ vì sao protocol thay đổi. Nếu v2 đạt, prerequisite dữ liệu để chuẩn bị notebook được giải quyết; feature/pretrain/training vẫn cần đúng phạm vi user duyệt và quota GPU hiện tại. Không mở lại T hoặc một brainstorm khác từ quyết định này.
