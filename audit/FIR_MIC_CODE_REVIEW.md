# FIR-MIC: kiểm tra code và kết quả GitHub

## Nguồn và phạm vi

Đã fetch trực tiếp repository `https://github.com/1101909/FIR-MIC-MMREC-Cold`.
Code mới nhất được kiểm tra: nhánh `agent/fix-semco-evaluator`, commit
`5587b8a`. Bộ kết quả: `kaggle-results-gridsearch-v1`, commit
`b80b23f4438a2f9688a9495a902bb5ed05e85f9b` (3 dataset × 10 seed).
Nhánh `main` là phiên bản cũ hơn. Không coi bản main là source của thực nghiệm grid-search.

Đã đọc đường chạy FIR-MIC, lớp Model kế thừa, preprocessing/teacher, split và
sampling, controlled baselines, adapter SEMCo/CLCRec, grid search, rebuild,
post-hoc gate, launcher/resume, thống kê và phân tích RQ3. Chưa kiểm toán từng
dòng trong toàn bộ thư viện upstream của các baseline.

Snapshot bằng chứng nằm trong `fir_mic_v1_evidence.json`, được tái tạo bằng
`scripts/audit_fir_mic_results.py`. Đây là phân tích kết quả đã có; không dùng
các test score này để chọn hyperparameter mới.

## Kết quả thực tế

| Dataset | FIR-MIC NDCG@10 | Baseline mạnh nhất theo NDCG@10 cũ | Gain | p paired t, Holm | p Wilcoxon, Holm |
|---|---:|---|---:|---:|---:|
| Baby | 0.026241 | SEMCo: 0.019769 | +32.74% | 0.00167 | 0.03516 |
| Clothing | 0.045571 | Per-item: 0.044260 | +2.96% | 0.03616 | 0.05078 |
| Sports | 0.032713 | Per-item: 0.032557 | +0.48% | 1.0 | 1.0 |

Không đúng khi nói mọi kết quả đều không có ý nghĩa thống kê. Sports là điểm
yếu rõ nhất; Clothing cải thiện thứ tự đầu bảng nhưng giảm Recall@10/20.
Các p-value chỉ đo biến động theo training seed trên một split cố định,
không đại diện cho biến động giữa các tập dữ liệu hoặc temporal split mới.
Không đổi kiểm định/correction chỉ để đạt p < 0.05.

## Nguyên nhân có bằng chứng

1. **Đóng góp router yếu.** Sports tắt `lambda_z` ở 8/10 seed qua validation.
   NDCG@10 của bản bỏ router là 0.033020, cao hơn full 0.032713.
   Route AUC trung bình: Baby 0.5014, Clothing 0.6948, Sports 0.5572.
   Đây là chẩn đoán cơ chế; không chứng minh một thay đổi cụ thể chắc chắn tăng metric.

2. **Nhãn stay phụ thuộc độ sắc nét và số intent.** Code dùng `current @ q_target`.
   Với hai phân bố giống nhau và đều trên K intent, nhãn là 1/K. Nó có thể là
   xác suất hai lần lấy mẫu trùng intent, nhưng không tương đương độ giống nhau
   của hai phân bố. Do router được diễn giải là stay/switch, đây là giả định
   thiết kế cần ablation. Bản v2 thử cosine, cho identical distributions = 1;
   không diễn giải cosine là xác suất đã được hiệu chỉnh.

3. **Target trùng trong batch bị coi là negative.** CE đường chéo không gom các
   cột cùng item ID. Với các cột giống nhau, mô hình phải chia xác suất giữa
   các bản sao; đây không phải bằng chứng mọi cột trùng có gradient đẩy nhau,
   nhưng tạo mục tiêu không khớp nhận dạng item và loss floor không cần thiết.
   Sửa thành numerator logsumexp trên tất cả target ID trùng, cùng cách
   controlled baseline đã dùng. Positive chưa quan sát vẫn là vấn đề riêng.

4. **Chuẩn hóa score không nhất quán.** Torch train dùng sample std, NumPy
   inference dùng population std; batch một mẫu tạo NaN. Đã thống nhất
   population std và epsilon. Việc chuẩn hóa trên batch lúc train và toàn
   catalog lúc inference vẫn là giả định cần ablation, chưa được giải quyết
   chỉ bằng sửa ddof. Train weights 0.1/0.1/0.2 và validation-selected weights
   cũng vẫn khác nhau.

5. **Router phụ thuộc batch đối với user một item.** Một history dài 1 nhận
   repetition=1 khi đứng riêng nhưng repetition=0 khi chung batch với history
   dài hơn. Đã làm thống nhất; giữ mode v1 khi nạp checkpoint lịch sử.

6. **Tái lập và resume chưa kín.** Validation DataLoader có thể tiêu thụ RNG
   toàn cục, làm thay đổi dropout so với retrain không validation. Đã cô lập
   Torch RNG của callback và generator đánh giá. Trong 30 log lịch sử,
   selected/confirmatory NDCG ghi nhận chênh lệch bằng 0, vì vậy chưa thể quy
   mức giảm hiệu quả quan sát được cho lỗi RNG. Resume cũ chỉ so một phần
   config hoặc sự tồn tại file; v2 thêm fingerprint source, feature, sample,
   seed, epoch ceiling và batch size. Không dùng lại v1 dưới nhãn v2.

7. **So sánh baseline thiếu kiểm soát đặc trưng.** Controlled baselines lấy
   block mean ảnh xuống 256 chiều; FIR-MIC dùng ảnh nguyên chiều. Baseline
   cũng chỉ dùng 2.000 validation samples để chọn checkpoint. Đã thêm chế độ
   ảnh nguyên chiều và toàn validation vào launcher v2. Các khác biệt về
   kiến trúc, learning rate grid và training budget vẫn phải công khai;
   chưa thể gọi mọi baseline là đã tối ưu với ngân sách tương đương.

8. **Post-hoc gate có fallback sai.** Khi không có cấu hình đạt guardrail,
   code cũ vẫn chọn một cấu hình vi phạm. Bản sửa luôn cho phép giữ nguyên
   ranking gốc bằng gate hằng 0.5. Gate tối ưu hard-proxy validation vẫn có
   thể không tăng overall test; cần giữ mục tiêu đánh giá được xác định trước.

9. **Thống kê có lỗi biên, không giải thích p lớn của Sports.** Hai dãy giống
   nhau từng cho t-test NaN và effect size infinity. Sửa p=1, effect=0 cho
   trường hợp null hoàn toàn; kiểm tra metric finite và bản ghi seed trùng.
   Không sửa p-value lịch sử hay nới correction. Strongest baseline theo
   test hiện là so sánh mô tả; nghiên cứu xác nhận nên chốt primary baseline
   và primary metric trước khi đánh giá.

## Hướng khắc phục và giới hạn

Bản local v2 sửa lỗi triển khai và thêm cosine router target để kiểm chứng.
Chưa push hoặc thay đổi repository trên GitHub. Chưa có bằng chứng bản sửa
vượt baseline trên cả ba dataset.

Nên chạy ablation theo thứ tự: v1; v1 + numerical/duplicate/RNG/batch fixes;
thêm cosine router target; full với router tắt; anchor-only. Dùng cùng split,
đặc trưng gốc, số seed và candidate catalog. Chọn cấu hình bằng validation;
không sửa target definition hoặc negative set để ưu tiên FIR-MIC trên test.
Vì test v1 đã được xem để phát triển v2, cần một temporal holdout mới cho
kết luận xác nhận mạnh hơn; tăng seed không tạo ra test set độc lập.

Nếu Sports vẫn không cải thiện, xem lại giả thuyết future intent thay vì
chỉ tăng epoch: future-only NDCG@10 cũ là 0.002980, anchor-only 0.032224.
Nhãn next là phân bố item_q đang học, có thể đổi trong training; teacher
warm-only và entropy/gradient diagnostics cần được kiểm tra qua validation.
Chuỗi pseudo-cold xóa mọi item cùng fold `item_id % 5` với target, làm thay
đổi chuyển tiếp quan sát được; đó là augmentation, không phải một mô hình
được fit độc lập ngoài fold. Cần ablation augmentation và mô hình ít intent
trên validation trước khi đưa thêm thay đổi kiến trúc vào bản chính.

Lệnh chạy lại trên Kaggle sau khi đưa bản sửa lên môi trường đó:

```bash
python kaggle/run_gpu.py --data-dir /kaggle/input/datasets/toanktx/mmrec-cold --datasets baby clothing sports --seeds 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 --epochs 20 --fir-mic-max-epochs 500 --batch-size 256 --semco-batch-size 256 --grid-patience 5 --output-dir results_v2 --run-controlled-content-baselines --run-official-cold-baselines --extended-ablations
```

Đây là lệnh tái chạy protocol hiện có với fixes, chưa phải chứng minh tuning
budget ngang nhau. Runtime local hiện có PyTorch CPU, không có CUDA; chưa
thực hiện lại 30 lượt GPU. Kiểm thử local kiểm tra correctness và tái lập,
không được dùng thay cho bảng thực nghiệm mới.
