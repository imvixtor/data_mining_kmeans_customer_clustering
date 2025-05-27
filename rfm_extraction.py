import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Đọc dữ liệu
df = pd.read_excel("Online Retail.xlsx")

# Xóa dòng thiếu CustomerID hoặc dữ liệu âm
df.dropna(subset=["CustomerID"], inplace=True)
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Tính giá trị tiền
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Chuyển InvoiceDate về kiểu datetime
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Ngày tham chiếu để tính Recency
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# Tính RFM
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
    "InvoiceNo": "nunique",
    "TotalPrice": "sum"
}).reset_index()
rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]

# Ép kiểu integer
rfm['CustomerID'] = rfm['CustomerID'].astype(int)
rfm['Recency']    = rfm['Recency'].astype(int)
rfm['Frequency']  = rfm['Frequency'].astype(int)

# Làm tròn Monetary về 2 chữ số thập phân
rfm['Monetary']   = rfm['Monetary'].round(2)

# Lưu RFM ra file CSV
rfm.to_csv("rfm_features.csv", index=False)
print("Đã tạo file rfm_features.csv")

# Log transformation giúp giảm bớt độ lệch của RFM trước khi chuẩn hóa
rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
rfm['Monetary_log']  = np.log1p(rfm['Monetary'])
rfm['Recency_log']   = np.log1p(rfm['Recency'])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
rfm_scaled = pd.DataFrame()
rfm_scaled['CustomerID'] = rfm['CustomerID']
rfm_scaled[['Recency_z', 'Frequency_z', 'Monetary_z']] = scaler.fit_transform(rfm[['Recency_log','Frequency_log','Monetary_log']])

# Lưu ra file CSV
rfm_scaled.to_csv("rfm_normalized.csv", index=False)
print("Đã tạo file rfm_normalized.csv")