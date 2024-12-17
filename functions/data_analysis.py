# Thư viện cần thiết
import pandas as pd


# Tính số lượng của từng lớp trong một đặc trưng cùng với tần suất của nó
def value_cnt_norm_cal(df, feature):
    ftr_value_cnt = df[feature].value_counts()                                      # Số lượng xuất hiện từng giá trị trong đặc trưng          
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100             # Chuẩn hóa dữ liệu tần suất 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)   # Gộp kết quả
    
    # Đặt tên cột
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    return ftr_value_cnt_concat
