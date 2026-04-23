Dưới đây là việc chuyển đổi toàn bộ 3 sơ đồ kiến trúc thành định dạng văn bản (Text/Markdown) kết hợp các luồng logic (Logic Flow). Định dạng này cực kỳ thân thiện với máy tính (Machine-readable), rất phù hợp để bạn copy/paste vào các AI Coding Assistant (như Claude, ChatGPT) để chúng hiểu rõ cấu trúc dữ liệu và sinh code PyTorch chuẩn xác.

---

### SƠ ĐỒ 1: MODALITY ENCODERS (Bộ mã hóa Đa phương thức)

**Mục tiêu:** Chuyển đổi dữ liệu sinh học thô thành các vector đặc trưng có cùng số chiều **$d$**.

* **Luồng 1: Sequence Stream (Trình tự Axit Amin)**
  * `[Input]` Chuỗi trình tự Axit Amin.
  * `[Encoder]` Mô hình ESM-2 (Pre-trained, đóng băng trọng số).
  * `[Pooling]` Lớp Multi-head Attention Pooling (giúp tập trung vào motif quan trọng).
  * `[Output]` Tensor **`h_seq`** (Kích thước: **$1 \times d$**).
* **Luồng 2: Structure Stream (Cấu trúc Hình học)**
  * `[Input]` Chuỗi 3Di tokens.
  * `[Encoder]` Mô hình ProstT5 (Pre-trained, đóng băng trọng số).
  * `[Pooling]` Lớp Multi-head Attention Pooling.
  * `[Output]` Tensor **`h_3di`** (Kích thước: **$1 \times d$**).
* **Luồng 3: Context Stream (Ngữ cảnh Mạng lưới)**
  * `[Input]` Đồ thị PPI (Protein-Protein Interaction) từ cơ sở dữ liệu STRING.
  * `[Encoder]` Thuật toán Node2Vec / GraphSAGE.
  * `[Pooling]` Trích xuất thẳng vector tĩnh đã huấn luyện trước.
  * `[Output]` Tensor **`h_ppi`** (Kích thước: **$1 \times d$**). Nếu protein không có dữ liệu PPI, khởi tạo bằng tensor toàn số 0.

---

### SƠ ĐỒ 2: ADAPTIVE FUSION (Cơ chế Dung hợp Thích ứng)

**Mục tiêu:** Tự động đánh giá và kết hợp các luồng dữ liệu, xử lý tính trạng khuyết thiếu dữ liệu (Missing Modality).

* **Bước 1: Nối Tensor (Concatenation)**
  * `[Operation]` Nối 3 vector đặc trưng lại với nhau.
  * `[Formula]` `H_concat = Concat(h_seq, h_3di, h_ppi)`
  * `[Output]` Tensor `H_concat` (Kích thước: **$1 \times 3d$**).
* **Bước 2: Mạng cổng (Gating Network)**
  * `[Input]` Tensor `H_concat`.
  * `[Operation]` Chạy qua mạng Multi-Layer Perceptron (MLP) nhỏ, theo sau là hàm Softmax.
  * `[Formula]` `[alpha_seq, alpha_3di, alpha_ppi] = Softmax(MLP(H_concat))`
  * `[Output]` 3 giá trị trọng số **$\alpha$** có tổng bằng 1.
* **Bước 3: Dung hợp (Fusion)**
  * `[Operation]` Tính tổng có trọng số (Weighted Sum) của các luồng đầu vào.
  * `[Formula]` `Z_protein = (alpha_seq * h_seq) + (alpha_3di * h_3di) + (alpha_ppi * h_ppi)`
  * `[Output]` Tensor **`Z_protein`** - Biểu diễn vector cuối cùng của Protein (Kích thước: **$1 \times d$**).

---

### SƠ ĐỒ 3: CLASSIFICATION & CUSTOM LOSS (Phân loại & Hàm mất mát)

**Mục tiêu:** Đưa ra xác suất dự đoán cho các nhãn GO và tính toán sai số tuân thủ logic sinh học.

* **Nhánh 1: Xây dựng Không gian Ngữ nghĩa (Semantic Space)**
  * `[Input]` Văn bản định nghĩa của **$C$** lớp Gene Ontology (GO terms).
  * `[Encoder]` Mô hình LLM BioBERT.
  * `[Output]` Ma trận trọng số GO cố định **`W_go`** (Kích thước: **$d \times C$**).
* **Nhánh 2: Tính toán Xác suất (Logits/Prediction)**
  * `[Input]` Tensor `Z_protein` và Ma trận `W_go`.
  * `[Operation]` Nhân ma trận (Dot Product).
  * `[Formula]` `Logits = DotProduct(Z_protein, W_go)`
  * `[Output]` Tensor `Logits` chứa điểm số thô cho **$C$** lớp GO (Kích thước: **$1 \times C$**).
* **Nhánh 3: Hàm Mất mát (Custom Loss Function)**
  * `[Input]` `Logits`, Nhãn thực tế (Ground Truth **$Y$**), và Ma trận phân cấp `DAG_Matrix`.
  * `[Component 1]`  **BCE Loss** : Đánh giá phân loại nhị phân độc lập cho từng nhãn.
  * `[Component 2]`  **DAG Loss** : Hàm phạt vi phạm phân cấp. Nếu xác suất lớp Con (**$P_{child}$**) dự đoán cao hơn lớp Cha (**$P_{parent}$**), áp dụng hình phạt.
    * `[Formula]` `L_DAG = max(0, P_child - P_parent)^2`
  * `[Final Output]` **`Total_Loss = BCE_Loss + (lambda * L_DAG)`** (Giá trị vô hướng dùng để Backpropagation cập nhật trọng số).

---

**💡 Mẹo sử dụng:** Bạn chỉ cần copy toàn bộ đoạn text Markdown này, đính kèm với "Master Prompt" ở phía trên và gửi cho AI. Với cấu trúc luồng (Input **$\rightarrow$** Operation **$\rightarrow$** Formula **$\rightarrow$** Output) rõ ràng như thế này, AI sẽ lập tức mapping nó thành các class `nn.Module` và hàm `forward()` trong PyTorch mà không bị nhầm lẫn chiều (dimension) của các Tensor.
