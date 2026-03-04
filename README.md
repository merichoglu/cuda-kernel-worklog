# CUDA Kernel Optimization Worklog

GPU: NVIDIA Quadro P2200 — Pascal SM 6.1 — 174 GB/s — ~4.3 TFLOP/s FP32

---

## Matrix Multiplication

| Step | Description       | GB/s | GFLOP/s | Occupancy | Notes |
|------|-------------------|------|---------|-----------|-------|
| 01   | Naive             |      |         |           |       |
| 02   | Tiled shared mem  |      |         |           |       |
| 03   | Vectorized float4 |      |         |           |       |
| 04   | 2D blocktiling    |      |         |           |       |

---

## Reduction (Sum)

| Step | Description          | GB/s | GFLOP/s | Occupancy | Notes |
|------|----------------------|------|---------|-----------|-------|
| 01   | Naive                | 27.90 | 6.98   |           | 16% of peak BW |
| 02   | Interleaved          |      |         |           |       |
| 03   | Sequential addressing|      |         |           |       |
| 04   | Warp primitives      |      |         |           |       |

---

## Softmax

| Step | Description          | GB/s | GFLOP/s | Occupancy | Notes |
|------|----------------------|------|---------|-----------|-------|
| 01   | Naive row-wise       | 48.90 | 12.23  |           | 28% of peak BW |
| 02   | Online stable        |      |         |           |       |

---

## Layer Normalization

| Step | Description          | GB/s | GFLOP/s | Occupancy | Notes |
|------|----------------------|------|---------|-----------|-------|
| 01   | Naive                |      |         |           |       |
| 02   | Fused (Welford)      |      |         |           |       |
