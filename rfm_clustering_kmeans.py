import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd

def kmeans_init_centers(X, k):
    # randomly pick k rows of X as initial centers
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers):
    # calculate pairwise distances btw data and centers
    D = cdist(X, centers)
    # return index of the closest center
    return np.argmin(D, axis = 1)

def kmeans_update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        # collect all points assigned to the k-th cluster 
        Xk = X[labels == k, :]
        # take average
        if len(Xk) > 0:  # Tránh lỗi khi một cụm không có điểm nào
            centers[k,:] = np.mean(Xk, axis = 0)
    return centers

def has_converged(centers, new_centers):
    # return True if two sets of centers are the same
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))

def kmeans(X, K, max_iter=100, tol=1e-4):
    """
    Thuật toán K-means clustering
    
    Tham số:
    - X: ma trận dữ liệu (mỗi hàng là một điểm dữ liệu)
    - K: số lượng cụm
    - max_iter: số lượng lần lặp tối đa
    - tol: ngưỡng dung sai để kiểm tra hội tụ
    
    Trả về:
    - centers: danh sách các trung tâm theo từng lần lặp
    - labels: danh sách các nhãn gán cho mỗi điểm theo từng lần lặp
    - it: số lần lặp thực tế
    """
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 
    while it < max_iter:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        
        if has_converged(centers[-1], new_centers):
            break
            
        # Kiểm tra hội tụ dựa trên sự thay đổi trung tâm
        if len(centers) > 1:
            center_shift = np.linalg.norm(new_centers - centers[-1])
            if center_shift < tol:
                break
                
        centers.append(new_centers)
        it += 1
        
    return (centers, labels, it)

def visualize_kmeans(X, centers, labels, title='K-means Clustering'):
    """
    Hiển thị kết quả thuật toán K-means
    
    Tham số:
    - X: dữ liệu
    - centers: trung tâm cuối cùng
    - labels: nhãn cuối cùng
    - title: tiêu đề
    """
    if X.shape[1] != 2:
        print("Chỉ có thể hiển thị dữ liệu 2D")
        return
        
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='x')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Ví dụ sử dụng
if __name__ == "__main__":
    # Đọc dữ liệu từ file rfm_normalized.csv
    print("Đang đọc dữ liệu từ file rfm_normalized.csv...")
    df = pd.read_csv('rfm_normalized.csv')
    
    # Kiểm tra dữ liệu
    print(f"Số lượng khách hàng: {df.shape[0]}")
    print(f"Các thuộc tính: {df.columns.tolist()}")
    
    # Chuẩn bị dữ liệu
    X = df[['Recency_z', 'Frequency_z', 'Monetary_z']].values
    
    # Chạy thuật toán K-means với K=3
    print("Đang chạy thuật toán K-means...")
    K = 3
    centers, labels, iterations = kmeans(X, K)
    
    print(f"Thuật toán hội tụ sau {iterations} lần lặp")
    
    # Thêm nhãn cụm vào dataframe
    df['Cluster'] = labels[-1]
    
    # Thống kê số lượng khách hàng trong mỗi cụm
    cluster_counts = df['Cluster'].value_counts().sort_index()
    print("\nSố lượng khách hàng trong mỗi cụm:")
    for cluster, count in cluster_counts.items():
        print(f"Cụm {cluster}: {count} khách hàng")
    
    # Tính giá trị trung bình của các thuộc tính trong mỗi cụm
    cluster_means = df.groupby('Cluster')[['Recency_z', 'Frequency_z', 'Monetary_z']].mean()
    print("\nGiá trị trung bình của các thuộc tính trong mỗi cụm:")
    print(cluster_means)
    
    # Hiển thị kết quả phân cụm trên biểu đồ 2D (sử dụng 2 trong 3 thuộc tính)
    plt.figure(figsize=(12, 8))
    
    # Biểu đồ Recency vs Frequency
    plt.subplot(1, 2, 1)
    for i in range(K):
        plt.scatter(X[labels[-1] == i, 0], X[labels[-1] == i, 1], label=f'Cụm {i}')
    plt.scatter(centers[-1][:, 0], centers[-1][:, 1], c='black', marker='x', s=200, label='Trung tâm')
    plt.title('Phân cụm khách hàng: Recency vs Frequency')
    plt.xlabel('Recency (chuẩn hóa)')
    plt.ylabel('Frequency (chuẩn hóa)')
    plt.legend()
    
    # Biểu đồ Recency vs Monetary
    plt.subplot(1, 2, 2)
    for i in range(K):
        plt.scatter(X[labels[-1] == i, 0], X[labels[-1] == i, 2], label=f'Cụm {i}')
    plt.scatter(centers[-1][:, 0], centers[-1][:, 2], c='black', marker='x', s=200, label='Trung tâm')
    plt.title('Phân cụm khách hàng: Recency vs Monetary')
    plt.xlabel('Recency (chuẩn hóa)')
    plt.ylabel('Monetary (chuẩn hóa)')
    plt.legend()

    # # Biểu đồ Frequency vs Monetary
    # plt.subplot(1, 2, 1)
    # for i in range(K):
    #     plt.scatter(X[labels[-1] == i, 1], X[labels[-1] == i, 2], label=f'Cụm {i}')
    # plt.scatter(centers[-1][:, 0], centers[-1][:, 1], c='black', marker='x', s=200, label='Trung tâm')
    # plt.title('Phân cụm khách hàng: Frequency vs Monetary')
    # plt.xlabel('Frequency (chuẩn hóa)')
    # plt.ylabel('Monetary (chuẩn hóa)')
    # plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Lưu kết quả
    output_file = 'rfm_clustering_results.csv'
    df.to_csv(output_file, index=False)
    print(f"\nĐã lưu kết quả phân cụm vào file {output_file}")