# Lưu checkpoint E3 vào notebook riêng

1. Tạo notebook Kaggle mới, đặt tên `E3 Checkpoint Store`. CPU là đủ.
2. Import file `E3_CHECKPOINT_STORE.ipynb` từ repo này.
3. Add Input output của `kltn-mscoco`, chọn version TRAIN E3 thành công, có thư mục `pair_grounded_relation_32q_2l_dual_gpu_3ep/checkpoints`. Version validation không chứa checkpoint. Notebook mới có thể chọn input version riêng với notebook diagnostic hiện tại.
4. Chạy cell và Save Version → Save & Run All. Cell kiểm tra metadata E3 epoch 3, sao chép checkpoint nguyên vẹn và kiểm tra SHA256. Không tải file `.pth` về máy hoặc upload thành dataset.
5. Trong notebook diagnostic, Add Input output của `E3 Checkpoint Store`. Copy File Path của checkpoint rồi thay `CHECKPOINT` bằng đường dẫn đó.

Output của notebook mới:

```text
e3_checkpoint_store/
  model_h1_2_crossattn_epoch_3.pth
  manifest.json
  train_history_h1_2.json  # nếu có trong output train
```

Nếu slug của notebook là `e3-checkpoint-store`, đường dẫn checkpoint dự kiến:

```python
CHECKPOINT = Path('/kaggle/input/notebooks/ducanh2403/e3-checkpoint-store/e3_checkpoint_store/model_h1_2_crossattn_epoch_3.pth')
```

Dùng Copy File Path để xác nhận mount thực tế. Giữ nguyên các input prompt V2.1, grounded relation cache và visual cache đã chốt.
