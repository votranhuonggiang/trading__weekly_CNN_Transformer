**Objective**
Giữ `base only` làm baseline sạch, rồi thử model changes theo thứ tự ít rủi ro nhất đến nhiều rủi ro hơn. Mỗi lần chỉ đổi một thứ.

**Locked Baseline**
- Labels: `v2 30/40/30`
- Feature packs: `base`
- Execution: `top-5`, weekly full reset
- Score: `P(Buy) - P(Avoid)`
- Notebook/data package: pin về `base only` trước khi chạy candidate đầu tiên

**Experiment Order**
1. `model_exp_01_attention_pooling`
   - Thay `x.mean(dim=1)` bằng learned attention pooling
   - Không đổi CNN, không đổi transformer depth
   - Reason:
     - mean pooling đang làm phẳng toàn bộ 60 ngày
     - stock signal thường không phân bố đều theo thời gian

2. `model_exp_02_multiscale_cnn`
   - Thay single-kernel CNN bằng multi-branch kernels, ví dụ `3/5/9`
   - Concat channel rồi project về `d_model`
   - Giữ pooling/head như baseline nếu chưa kết hợp với exp 01
   - Reason:
     - cần bắt local pattern ở nhiều độ dài khác nhau

3. `model_exp_03_residual_cnn_block`
   - Thêm residual CNN block và normalization
   - Giữ transformer/head như baseline
   - Reason:
     - tăng stability, giữ raw signal tốt hơn

4. `model_exp_04_checkpoint_by_trading_metric`
   - Giữ architecture baseline
   - Đổi best-checkpoint selection từ `val_macro_f1` sang metric trading-aligned
   - Ưu tiên:
     - `validation top-5 return`
     - hoặc `validation sharpe`
   - Reason:
     - hiện train/eval objective lệch mục tiêu trading

5. `model_exp_05_aux_ranking_head`
   - Giữ 3-class head
   - Thêm auxiliary head cho regression/ranking target
   - Chỉ thử nếu 4 bước trên chưa đủ

**Decision Rules**
- Mỗi candidate phải có output riêng.
- Không combine candidate architecture với candidate khác trước khi candidate đơn lẻ chứng minh được giá trị.
- Nếu candidate thua baseline rõ ràng ở full realistic-cost run, loại.
- Nếu candidate thắng ở sample nhưng thua nặng ở full, không promote sang branch chính.

**Practical Recommendation**
- Bắt đầu bằng `model_exp_04_checkpoint_by_trading_metric` hoặc `model_exp_01_attention_pooling`.
- Nếu muốn thay đổi code kiến trúc trước, chọn `attention pooling`.
- Nếu muốn thay đổi ít code nhất trước, chọn `checkpoint_by_trading_metric`.
