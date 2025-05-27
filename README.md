# Dự Án Phân Tích RFM và Phân Cụm Khách Hàng

## Giới thiệu
Dự án này thực hiện phân tích RFM (Recency, Frequency, Monetary) và phân cụm khách hàng dựa trên dữ liệu bán lẻ. Mục đích là phân nhóm khách hàng thành các phân khúc khác nhau để hỗ trợ chiến lược tiếp thị và chăm sóc khách hàng.

## Cấu trúc dự án
- `Online Retail.xlsx`: Dữ liệu gốc chứa thông tin giao dịch bán lẻ
- `rfm_extraction.py`: Script trích xuất các đặc trưng RFM từ dữ liệu gốc
- `rfm_extraction.ipynb`: Notebook tương tác để preview dữ liệu, trích xuất và phân tích RFM
- `rfm_clustering_kmeans.py`: Script phân cụm khách hàng sử dụng thuật toán K-means
- `evaluate_kmeans.ipynb`: Notebook đánh giá kết quả phân cụm
- `rfm_features.csv`: Dữ liệu RFM trích xuất
- `rfm_normalized.csv`: Dữ liệu RFM sau khi chuẩn hóa
- `rfm_clustering_results.csv`: Kết quả phân cụm khách hàng

## Quy trình xử lý

### 1. Trích xuất đặc trưng RFM
```powershell
python rfm_extraction.py
```

- Đọc dữ liệu từ file Excel
- Tiền xử lý: loại bỏ dữ liệu thiếu CustomerID và giá trị âm
- Tính toán các chỉ số RFM:
  - **Recency**: Số ngày kể từ lần mua hàng gần nhất
  - **Frequency**: Số lần mua hàng (số hóa đơn)
  - **Monetary**: Tổng chi tiêu
- Áp dụng log transformation để giảm độ lệch
- Chuẩn hóa dữ liệu sử dụng StandardScaler
- Lưu đặc trưng RFM và dữ liệu chuẩn hóa ra file CSV

### 2. Phân cụm khách hàng
```powershell
python rfm_clustering_kmeans.py
```

- Đọc dữ liệu RFM đã chuẩn hóa
- Áp dụng thuật toán K-means với K=3
- Phân tích đặc điểm của từng cụm khách hàng
- Trực quan hóa kết quả phân cụm
- Lưu kết quả phân cụm ra file CSV

### 3. Đánh giá kết quả
Chạy `evaluate_kmeans.ipynb` để đánh giá hiệu quả của phân cụm

## Cách sử dụng
1. Đảm bảo đã cài đặt các thư viện cần thiết:
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn
```

2. Đặt file "Online Retail.xlsx" vào thư mục dự án
3. Chạy script trích xuất RFM
4. Chạy script phân cụm K-means
5. Phân tích kết quả trong các notebook

## Kết quả
Sau khi chạy dự án, bạn sẽ có được phân khúc khách hàng dựa trên hành vi mua hàng (RFM), hỗ trợ cho việc:
- Xác định nhóm khách hàng tiềm năng
- Phát triển chiến lược tiếp thị phù hợp cho từng phân khúc
- Cải thiện chương trình chăm sóc khách hàng
