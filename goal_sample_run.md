Goal:
Thiết lập và dùng sample-run pipeline cho chiến lược CNN-Transformer v2 để sàng lọc các hướng **model change** trước khi chạy full. Từ thời điểm này, khóa baseline về `base only`, giữ nguyên labels v2 `30/40/30`, giữ nguyên ranking score `P(Buy) - P(Avoid)`, giữ nguyên weekly rebalance/backtest logic, và chỉ cho phép thay đổi ở phần model / training-selection.

Mục tiêu của phase này là tìm model candidate có performance **tốt hơn baseline base-only**. Không còn dùng điều kiện cứng `Sharpe > 2.5`.

Success criteria:
- Có một chế độ sample run riêng, chạy nhanh hơn full run đáng kể.
- Sample run giữ nguyên logic cốt lõi của v2:
  - label definition
  - feature set `base only`
  - score = `P(Buy) - P(Avoid)`
  - weekly rebalance / backtest logic
- Chỉ thay đổi model/training-selection; không thay đổi labels, features, execution rule hoặc benchmark rule.
- Có output so sánh giữa sample candidate và baseline v2 base-only theo cùng format.
- Có bảng summary cho mỗi candidate với ít nhất các metric:
  - total_return
  - annualized_return
  - annualized_volatility
  - sharpe
  - sortino
  - max_drawdown
  - win_rate
  - avg_turnover
- Chỉ đánh dấu một candidate là `worthy for full run` khi nó đồng thời tốt hơn baseline base-only ở các metric decision đã khóa.
- Nếu không candidate nào tốt hơn baseline, dừng ở sample run và không chạy full.

Scope:
- Được phép thay đổi:
  - `src/model.py`
  - `src/train.py`
  - notebook / script phục vụ sample-run model candidates
  - reporting / summary script
- Không được thay đổi:
  - feature set ngoài `base only`
  - label definition
  - logic score của v2
  - top-k / weighting / execution rule
  - benchmark logic
  - format output chính của baseline v2

Context / materials:
- Repo: `C:\Users\votranhuonggiang\OneDrive\ドキュメント\Python\MiQuant\Training\CNN_Transformer_Forecast_HOSE`
- Tài liệu tham chiếu:
  - `architecture_v2.md`
  - `results.md`
  - `model_change_plan.md`
- Baseline cần so sánh:
  - `base only`
  - labels `v2_buy_top30_hold_mid40_avoid_bottom30`
  - feature profile `v2_base_only`
- Sample run phải dùng v2 làm chuẩn đối chiếu, không dùng v3.

Sample run design:
- Sample time window:
  - train: giai đoạn ngắn hơn nhưng liên tục theo thời gian
  - val: 1 giai đoạn liền sau train
  - test proxy: 1 giai đoạn ngắn sau val
- Sample universe:
  - chỉ giữ nhóm cổ phiếu thanh khoản cao nhất
- Training budget:
  - epochs ít hơn full run
  - patience nhỏ hơn
- Mục tiêu của sample run:
  - so sánh tương đối giữa các model candidates
  - không dùng sample run để kết luận final production performance
- Model changes phải được test theo từng candidate riêng:
  1. baseline model
  2. attention pooling
  3. multi-scale CNN
  4. residual CNN block + normalization
  5. checkpoint selection theo trading metric
  6. các bước tiếp theo chỉ khi candidate trước có lý do kỹ thuật rõ ràng
- Không được gộp nhiều thay đổi model cùng lúc nếu chưa có kết quả sample riêng.

Validation loop:
- Sau mỗi thay đổi code:
  - chạy `py_compile` cho các file Python bị sửa
- Sau mỗi candidate model:
  - xác nhận labels và feature profile vẫn là baseline `base only`
  - chạy sample train + backtest
  - xuất summary CSV/Markdown
  - so sánh trực tiếp với baseline base-only
- Sau mỗi vòng sample:
  - phải có bảng xếp hạng candidate theo cùng một bộ metric
  - phải nêu rõ candidate tốt nhất hiện tại là model variant nào
- Không được kết luận hoàn thành nếu chưa có bảng so sánh metric.

Decision gate:
- Một candidate chỉ được coi là `worthy for full run` khi đồng thời:
  - `total_return > baseline`
  - `win_rate > baseline`
  - `sortino > baseline`
  - `annualized_volatility < baseline`
  - `max_drawdown > baseline`
- `sharpe` vẫn phải được báo cáo và dùng để đánh giá, nhưng không còn là điều kiện cứng `> 2.5`.

Checkpoints:
1. Xác định baseline base-only
   - Done when:
     - ghi rõ baseline metrics từ output hiện có
   - Validate by:
     - tạo bảng baseline metric chuẩn để dùng cho mọi so sánh
2. Khóa phạm vi model-only
   - Done when:
     - xác nhận rõ chỉ được đổi model / training-selection
   - Validate by:
     - không có thay đổi labels / features / execution
3. Lập danh sách model candidates
   - Done when:
     - có danh sách rõ ràng các candidate theo thứ tự test
   - Validate by:
     - mỗi candidate có tên, mô tả, và rationale ngắn gọn
4. Tích hợp sample mode vào model pipeline
   - Done when:
     - có thể chạy sample bằng 1 lệnh hoặc 1 cell rõ ràng
   - Validate by:
     - tạo ra output prediction/performance hợp lệ
5. Chạy candidate experiments trên sample
   - Done when:
     - mỗi candidate có summary metrics riêng
   - Validate by:
     - có file tổng hợp để so với baseline
6. Quyết định có chạy full hay không
   - Done when:
     - mỗi candidate được gắn trạng thái:
       - reject
       - keep for full run
   - Validate by:
     - áp dụng đúng decision gate so với baseline

Stop rules:
- Dừng nếu sample run không còn phản ánh đúng logic v2.
- Dừng nếu thay đổi chạm vào labels, features, score rule, top-k rule hoặc benchmark logic.
- Dừng nếu candidate không tốt hơn baseline base-only trên các metric decision đã khóa.
- Dừng nếu thay đổi bắt đầu lan quá rộng sang nhiều thành phần cùng lúc.
- Dừng và báo lại nếu output hiện tại không đủ để dựng baseline chuẩn.

Output format:
- Báo cáo cuối cùng phải gồm:
  - các file đã sửa
  - sample config đã dùng
  - baseline metrics
  - danh sách model candidates đã test
  - metrics của từng candidate
  - candidate tốt nhất hiện tại
  - candidate nào đáng chạy full
  - candidate nào bị loại và vì sao

Current phase status:
- Feature phase đã dừng.
- Active baseline hiện tại là:
  - feature profile: `v2_base_only`
  - feature packs: `base`
  - labels: `v2_buy_top30_hold_mid40_avoid_bottom30`
- Từ đây chỉ đi tiếp theo hướng model change.
