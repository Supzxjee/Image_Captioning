# Dán vào một cell MỚI của phiên Kaggle còn thư mục Working; tự bấm Run.
# Không Restart nếu muốn giữ cache dở. Cell này nạp và chạy notebook sửa lỗi.
ARM_TO_RUN = "C"  # Đổi thành "G" nếu đang sửa nhánh G.
import hashlib, json, urllib.request
if ARM_TO_RUN not in ("C", "G"):
    raise ValueError("Chỉ chọn C hoặc G")
commit = 'e56b562895c7e9321d75938cdda8c46cbf54d167'
expected = {'C': 'ba4633442c0d25d9f23ea4769be6b880ddc73e2284e7d6c0cfa043a5c0007c07', 'G': '21ff37a244edb29da5c3b8d7b73bcb25e846a1fb5188fbc8bafbd18ee30244a3'}
url = f"https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{commit}/experiments/concept_mass_grounding/CONCEPT_MASS_{ARM_TO_RUN}_2GPU.ipynb"
raw = urllib.request.urlopen(url, timeout=60).read()
if hashlib.sha256(raw).hexdigest() != expected[ARM_TO_RUN]:
    raise ValueError("Notebook khác bản đã khóa; dừng")
notebook = json.loads(raw)
namespace = {"__name__": "__main__"}
for index, item in enumerate(notebook["cells"]):
    if item["cell_type"] == "code":
        print(f"Running pinned recovery notebook cell {index + 1}", flush=True)
        exec(compile(item["source"], f"concept-recovery-cell-{index+1}", "exec"), namespace)
