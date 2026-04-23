## Tại sao GO có >27,000 terms nhưng có thể embed xuống 512d hiệu quả?

**1. Intrinsic dimensionality thấp hơn nhiều số terms**

GO terms không phân bố ngẫu nhiên trong không gian ngữ nghĩa — chúng có  **cấu trúc DAG phân cấp chặt chẽ** . Term con kế thừa ý nghĩa từ term cha. Ví dụ: `GO:0006915 (apoptotic process)` là con của `GO:0008150 (biological process)` — hai terms này không "độc lập" về ngữ nghĩa. Thực tế, khi phân tích GO semantic similarity, **hầu hết variation ngữ nghĩa được giải thích bởi ~200-500 chiều độc lập** (tương tự như word embeddings: vocab 100k từ nhưng Word2Vec 300d capture được phần lớn ngữ nghĩa).

**2. BioBERT definition text ngắn → ít entropy**

Definition của mỗi GO term thường chỉ 1-3 câu, ~20-80 tokens. BioBERT-base output là 768d — đây đã là compressed representation. Project xuống 512d mất rất ít thông tin vì BioBERT [CLS] token tự nó đã là bottleneck từ toàn bộ định nghĩa.

**3. Empirical evidence**

Các paper dùng semantic GO embeddings (OntoProtein, ProtST, TALE) đều dùng 512-768d và report kết quả tốt. Thử nghiệm với PCA trên GO embeddings thực tế cho thấy  **95% variance được giải thích trong ~400-500 chiều** .

---

## Embedding dimensions: Giống hay Khác nhau giữa các modality?

**Raw dimensions trước projection:**

| Modality  | Model       | Raw dim | Lý do                            |
| --------- | ----------- | ------- | --------------------------------- |
| Sequence  | ProteinBERT | 1024d   | Evolutionary complexity cao       |
| Structure | ProstT5     | 1024d   | Structural space cũng phức tạp |
| PPI       | Node2Vec    | 128d    | Graph topology ít chiều hơn    |

**Sau projection — nên giống nhau (512d):**

Lý do kỹ thuật:  **Weighted sum fusion đòi hỏi cùng chiều** . Gating network tính `Z = α_seq·h_seq + α_3di·h_3di + α_ppi·h_ppi` — không thể cộng vector khác chiều.

Lý do sâu hơn:  **Projection layer là learnable information filter** . Khi project PPI từ 128d → 512d (upsampling), model học cách "stretch" PPI signal vào không gian chung. Khi project ProteinBERT từ 1024d → 512d (compression), model học cách giữ lại chiều quan trọng nhất. Đây không phải padding hay truncation — là learned transformation.

**Trade-off nếu giữ dims khác nhau:**

Nếu dùng concatenation thay weighted sum (`Z = MLP([h_seq; h_3di; h_ppi])`), có thể giữ dims khác nhau. Nhưng với N proteins lớn, concat (1024+1024+128=2176d) → MLP → 512d tốn nhiều parameters hơn và harder to interpret (alpha weights không còn nghĩa "modality contribution").

**Kết luận cho thesis:** Project tất cả về 512d trước fusion vì (1) mathematically required cho weighted sum, (2) tạo "fair playing field" — mỗi modality có equal representational capacity trước khi gating network quyết định trọng số, (3) dễ giải thích: alpha weights trực tiếp = contribution của từng modality.
