# Thư viện cần thiết
import pandas as pd
import numpy as np


# Tính số lượng của từng lớp trong một đặc trưng cùng với tần suất của nó
def value_cnt_norm_cal(df, feature):
    ftr_value_cnt = df[feature].value_counts()                                      # Số lượng xuất hiện từng giá trị trong đặc trưng          
    ftr_value_cnt_norm = df[feature].value_counts(normalize=True) * 100             # Chuẩn hóa dữ liệu tần suất 100
    ftr_value_cnt_concat = pd.concat([ftr_value_cnt, ftr_value_cnt_norm], axis=1)   # Gộp kết quả
    
    # Đặt tên cột
    ftr_value_cnt_concat.columns = ['Count', 'Frequency (%)']
    return ftr_value_cnt_concat


# Mô tả thông tin dữ liệu thống kê của từng đặc trưng thuộc tính
"""
+ Mô tả thống kê
+ Kiểu dữ liệu datatype
+ Số lượng giá trị và tần suất (đối với mỗi đặc trưng)
"""
def gen_info_feat(df, feature, cc_train_copy):
    # Điều kiện của thuộc tính Age
    if feature == 'Age':
        """
        Status: biểu diễn bằng số ngày (âm/dương)
        Phương pháp: age = |age|/365.25
        """
        print('Description:\n{}'.format((np.abs(df[feature])/365.25).describe()))
        
        print('*'*50)
        print('Object type:{}'.format(df[feature].dtype))

    # Điều kiện của thuộc tính Employment length
    if feature == 'Employment length':
        """
        Status: biểu thị thời gian làm việc (âm: thất nghiệp, dương: hiện tại)
        Phương pháp: lọc ra những người thất nghiệp => employment = |employment|/365.25
        """
        employment_len_no_ret = cc_train_copy['Employment length'][cc_train_copy['Employment length'] < 0]
        employment_len_no_ret_yrs = np.abs(employment_len_no_ret)/365.25        # Chuyển thất nghiệp thành có việc
        print('Description:\n{}'.format((employment_len_no_ret_yrs).describe()))

        print('*'*50)
        print('Object type:{}'.format(employment_len_no_ret.dtype))

    # Điều kiện của thuộc tính Account age và Income
    if feature == 'Account age' or feature == 'Income':
        """
        Status: biểu thị độ tuổi tài khoản và nguồn thu nhập
        """
        print('Description:\n{}'.format((np.abs(df[feature])).describe()))
        
        print('*'*50)
        print('Object type:{}'.format(df[feature].dtype))
    else:
        # Các đặc trưng còn lại
        print('Description:\n{}'.format(df[feature].describe()))        # Lấy dữ liệu thống kê
        print('*'*50)
        print('Object type:\n{}'.format(df[feature].dtype))
        print('*'*50)
        
        # Tính số lượng tần suất, số lượng
        value_cnt = value_cnt_norm_cal(df,feature)
        print('Value count:\n{}'.format(value_cnt))
