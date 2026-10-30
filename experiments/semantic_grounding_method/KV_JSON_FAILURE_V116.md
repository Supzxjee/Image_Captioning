# Lỗi JSON ở lượt D116 và khôi phục không train lại

31/10/2026. User báo `TypeError: Object of type int64 is not JSON serializable` tại `kv_data.save`; log có KEEP checkpoint.bin. User xác nhận phiên kết thúc và còn Saved Output version116 của ducanh2403/2-vi-cai-kia-qua-nhieu. Chưa đọc review ZIP hoặc kiểm worker artifacts, không suy endpoint COMPLETE chỉ từ tên checkpoint.

Root cause xác định trong source: `paired_false` nhận NumPy flag arrays; `sum(np.bool_)` tạo `numpy.int64` cho repaired/introduced. JSON chuẩn không chấp nhận kiểu này. Các test cũ chỉ đưa Python lists nên bỏ sót đường dữ liệu thật. `2**n` còn có thể tràn số NumPy nếu discordance>=64; ép counts sang Python int sửa cả hai. Đây là lỗi triển khai/reporting do root, không phải model/method thất bại.

Sửa ở root `kv_data.canonical`: hỗ trợ NumPy scalar bằng `.item()`, vẫn reject arrays/objects và nonfinite floats; JSON native/fact identity không đổi. `paired_false` cast counts sang int ngay khi tính. In summary log dùng JSON đã ghi thay vì encode riêng một lần nữa. Không đổi model, loss, data, RNG, schedule, endpoint hoặc checkpoint selection.

Checks:4regression tests với real NumPy arrays/scalars và100discordances PASS;1synthetic full cached-summary test1000rows PASS. Không Torch/model/HDF5/scorer/GPU execution. Static PASS cũ không bắt được lỗi thực này; không gọi các checks mới là CUDA/model proof.

Recovery notebook CPU pin đúng Output116, tải saved metadata/predictions/CIDEr-per-image/CHAIR và checkpoint bytes (chỉ hash, không load). Yêu cầu hai worker ENDPOINT_500_COMPLETE và đủ cohort/artifacts trước khôi phục. Chỉ kiểm metadata của prepared dataset, không đọc lại H5. Giữ nguyên source download, copy sang recovery folder; validate cached captions/IDs/counts/finite, recompute aggregates/paired stats/semantic bootstrap, save summary và provenance riêng với original/recovery commits +rawartifact SHAs. Không gọi prepare/worker/training, không fallback latest. Thiếu GPU endpoint/artifact thì STOP và báo thiếu, không auto retrain.

User phải tự chạy CPU recovery; chưa xác nhận thực tế khôi phục xong. D116 giữ nguyên checkpoint/output. Future D/E notebooks được repin cùng serialization fix và chạy5regression/synthetic checks trước GPU. Không yêu cầu người dùng chạy lại nhánh đã hoàn tất GPU chỉ để đồng nhất source hash của thay đổi reporting; đối chiếu whitelist diff/init/checkpoint/data khi nhận ZIP.
