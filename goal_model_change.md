Goal:
Chuyển từ feature phase sang model phase trên baseline v2 `base only`. Từ thời điểm này, giữ nguyên labels v2 `30/40/30`, giữ nguyên feature set `base only`, giữ nguyên execution rule `top-5` với score `P(Buy) - P(Avoid)`, và chỉ thử các thay đổi ở phần model/training-selection để cải thiện performance.

Success criteria:
- Có baseline `base only` được rebuild sạch cho local/Colab.
- Có một kế hoạch model-change rõ ràng, mỗi lần chỉ đổi một thứ.
- Mỗi candidate model có output so sánh với baseline v2 theo cùng format.
- Chỉ đánh dấu một candidate là đáng chạy full khi nó cải thiện rõ trên các metric trading chính so với baseline tương ứng.

Scope:
- Được phép thay đổi:
  - `src/model.py`
  - `src/train.py`
  - notebook / script cần thiết để chạy model candidates
  - reporting / summary script cho model phase
- Không được thay đổi:
  - label definition
  - feature set ngoài `base only`
  - execution rule
  - benchmark logic
  - decision gate nếu chưa có lý do kỹ thuật rõ ràng

Fixed baseline:
- Labels: `v2_buy_top30_hold_mid40_avoid_bottom30`
- Feature packs: `base`
- Execution: top-5, weekly full reset
- Score: `P(Buy) - P(Avoid)`

Model-change order:
1. Attention pooling thay cho mean pooling
2. Multi-scale CNN
3. Residual CNN block + normalization
4. Checkpoint selection theo trading metric
5. Auxiliary ranking/regression head nếu 4 bước trên chưa đủ

Validation loop:
- Sau mỗi thay đổi code:
  - `python -m py_compile` cho file đã sửa
- Sau mỗi candidate model:
  - rebuild / run với cùng baseline `base only`
  - xuất summary CSV/Markdown
  - so sánh trực tiếp với baseline hiện tại
- Không gộp nhiều thay đổi model trong cùng một candidate.

Stop rules:
- Dừng nếu thay đổi chạm vào features, labels hoặc execution rule.
- Dừng nếu một model candidate không cải thiện gì trên metric trading quan trọng so với baseline.
- Dừng và báo lại nếu bắt đầu cần đổi quá nhiều thành phần cùng lúc để giải thích kết quả.

Output format:
- File đã sửa
- Candidate model đang thử
- Baseline metrics
- Candidate metrics
- Kết luận giữ / bỏ candidate
