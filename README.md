# SWIN-S + APB (Adaptive Parameter Bypass)

So sánh hiệu năng **Swin-S Transformer baseline** vs phiên bản tích hợp **APB Layer** trên dataset ImageNette-320.

---

## Cấu trúc repo

```
SWIN-S/
├── swin_s_apb_complete.ipynb   # Notebook chính: Swin-S + APB trên ImageNette
├── rebuildapb.ipynb            # Notebook thử nghiệm APB với CIFAR-10
├── BAO_CAO_KET_QUA_SWIN_S.md  # Báo cáo kết quả
└── BAO_CAO_KET_QUA_SWIN_S_Duy.pdf
```

> **Thư mục `data/` không được lưu trong repo** (quá lớn ~337 MB).  
> Dữ liệu sẽ **tự động tải về** khi chạy notebook lần đầu — không cần chuẩn bị gì thêm.

---

## Cách chạy

### 1. Clone repo

```bash
git clone https://github.com/NgoTranNgocDuy/SWIN-S.git
cd SWIN-S
```

### 2. Cài dependencies

```bash
pip install torch torchvision numpy requests
```

### 3. Chạy notebook

Mở `swin_s_apb_complete.ipynb` và **Run All Cells**.

- Lần đầu chạy: CELL 6 sẽ tự tải ImageNette-320 (~333 MB) từ fast.ai về thư mục `data/`
- Các lần sau: dùng lại dữ liệu đã có, không tải lại

### Chế độ chạy nhanh (QUICK_RUN)

Ở đầu notebook (CELL 1) có tùy chọn:

```python
QUICK_RUN = True   # Demo nhanh trên CPU (~15 phút, 200 ảnh)
QUICK_RUN = False  # Full training trên GPU (Kaggle/Colab, 20 epoch)
```

---

## Yêu cầu

- Python 3.8+
- PyTorch 1.12+
- GPU (khuyến nghị) hoặc CPU (dùng QUICK_RUN = True)

---

## Dataset

[ImageNette-320](https://github.com/fastai/imagenette) — subset 10 class của ImageNet, được tải tự động bởi notebook.
