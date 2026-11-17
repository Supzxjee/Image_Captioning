# Full FP16 XE epoch

Run one fresh batch-32 Karpathy epoch on two T4s with the same grouped order, mounted H5, feature sharing and ordinary tail27 as the completed FP32 reference. Keep the common FP32 preflight and a three-hour cap. Train with Accelerate FP16 autocast and gradient scaling. Evaluate both teacher-forced NLL and the existing beam decoding in FP32 on VALID5000 with the same metric implementation.

Check all 17,711 attempts and 566,747 caption events, track effective optimizer steps and nonfinite norms on both ranks, and preserve the checkpoint and scaler state. Compare one-epoch train time, CIDEr and NLL with the historical FP32 reference. The epochs run in different Kaggle sessions and represent one initialization. Do not infer TEST quality or 10-epoch convergence.
