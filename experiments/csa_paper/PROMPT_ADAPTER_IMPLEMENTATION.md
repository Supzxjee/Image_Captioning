# Triển khai thử CSA với prompt đối tượng–quan hệ

05/11/2026. Người dùng đồng ý triển khai phương án R249 và yêu cầu thêm tab kết quả trong Sheet cũ. **Không tự chạy notebook hoặc mô hình.** Dùng [thiết kế đã thảo luận](REGION_LINKED_PROMPT_CSA_PROPOSAL.md), giữ nguyên kết quả117.06 làm đối chứng lịch sử không có UPT prompt.

## Bước đã triển khai

- `prompt_adapter_data.py`: join instance UPT với ROI theo nhãn+IoU≥.5, tái dựng đúng thứ tự confidence-sort/clip/cap150 của cache134; loại collision/ambiguity, không dùng preview/GT/reference để tạo prompt. Giữ cả hai instance cùng lớp và thứ tự S/O. Có packing8object/3relation slots để nối module.
- `prompt_adapter_model.py`: wrapper quanh captioner hiện có; object/relation residual adapters gắn text với ROI/cặpROI, cập nhật trực tiếp Vo/Vr. Output projection zero-init và empty-row handling giữ đường base. Chỉ viết code/đọc source; chưa có model execution.
- `CSA_PROMPT_ROI_AUDIT_CPU.ipynb`: notebook đầu tiên cần chạy. CPU, InternetON, không cần checkpoint/ảnhJPEG/APIsecret/bank130; chỉ cần ROI134, UPT prompt cache và detectorJSON đã chốt. Audit TRAIN10k + fullVALID5000, không TEST. Hash chính xác nguồn/dữ liệu, kiểm box/mapping rồi xuất summary + examples + file liên kết đầy đủ để giữ.
- `test_prompt_adapter_data.py`: local-safe fixture/static checks. `test_prompt_adapter_model.py`: neural fixture tests **chỉ dành người dùng chạy ở bước model smoke sau**, không được gộp vào test local/CPU audit notebook. Static PASS không đồng nghĩa moduleCUDA PASS.

## Vì sao notebook đầu chỉ audit CPU

UPT và detector tạo ROI134 là hai nguồn khác nhau, chưa có số đo mapping. Phương án đã chốt yêu cầu biết có đủ cặp gắn vùng trước khi khóa cache/tensor/trainer. Vì vậy không viết notebook train500update dựa trên giả định index hai detector giống nhau, không tự hạ IoU để đủ mẫu. Audit trả số liệu coverage và ví dụ, **không tự cấp trạng thái training-ready**. Nếu mất nhiều cặp, cần bàn việc pool UPT boxes từ patch cache gốc; không chạy lại CLIP tự động.

Đây là bước phụ thuộc dữ liệu thật, không phải một vòng xin duyệt lại phương pháp. Sau khi người dùng gửi reviewZIP, tiếp tục chuẩn bị frozenCLIP text cache và hai notebook matchedCONTROL/PROMPT từ checkpoint161; dùng CE+.05SCST đã thảo luận. Thiếu quota hiện tại được hỏi song song; không tái dùng giờGPU cũ hoặc tự launch/train để đo.

## Đầu vào và đầu ra

Ba input mounts mặc định:

1. `/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134`
2. `/kaggle/input/datasets/ducanh2403/upt-prompts-full-all-actions-v2/prompt_cache.json`
3. `/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json`

Mặc định output `/kaggle/working/csa_prompt_roi_audit_v1`, giữ nguyên folder. Gửi `/kaggle/working/csa_prompt_roi_audit_v1_review.zip`. `prompt_links.jsonl` toàn bộ15000images và `prompt_texts.json` ở folder đầy đủ, ZIP nhỏ chỉ chứa summary/receipt/cohort/examples/source/log. Không tải/xem ảnh hoặc khẳng định prompt thật chỉ từ mapping tốt.

## Tình trạng đối chứng và kết quả

R249 chưa có điểm CIDEr/CHAIR mới. Hai nhánh500update chưa được chạy; số500×8=4000imageevents, không phải fullTRAINepoch. Khi đo sẽ so cùng lịch/train/control và VALID5000 trước/sau cùng fixedrerank, kèm prompt-off/text-swap sensitivity. Giữ nhận định kết quả cũ D/E cho thấy prompt có thể bị bỏ qua; gradient hoặc attention không chứng minh caption dùng prompt đúng.
