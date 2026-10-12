# E3 checkpoint store: source version 194

Import `E3_CHECKPOINT_STORE.ipynb` into a NEW Kaggle notebook. CPU is sufficient; enable Internet.

No Add Input from kltn-mscoco is required. The cell uses `kaggle kernels output ducanh2403/kltn-mscoco/194 --file-pattern ...` to retrieve only E3 epoch 3, explicitly selecting version 194.

For this private notebook, create an API token at https://www.kaggle.com/settings/api. Add it as `KAGGLE_API_TOKEN` in Add-ons > Secrets and enable access for this notebook. Do not paste the token into the code or chat.

Run and Save Version / Save & Run All. Output: `e3_checkpoint_store/model_h1_2_crossattn_epoch_3.pth` and `manifest.json` with source version 194. Metadata and copy SHA256 are verified.

Then Add Input this independent notebook output in the diagnostic notebook and use Copy File Path to replace CHECKPOINT. No local .pth download or dataset upload is required.

Official documentation: https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md and https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md
