# Thư viện cần thiết
from functions.data_analysis import value_cnt_norm_cal
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Biểu đồ hình tròn
def create_pie_plot(df, feature):
    # Xử lý đặc trưng Dwelling và Education level
    if feature == 'Dwelling' or feature == 'Education level':
        # Tính số lượng, tần suất lớp của đặc trưng
        ratio_size = value_cnt_norm_cal(df, feature)     
        ratio_size_len = len(ratio_size.index)              
        ratio_list = []

        # Duyệt từng lớp để lấy tỷ lệ
        for i in range(ratio_size_len):
            ratio_list.append(ratio_size.iloc[i]['Frequency (%)'])

        # Tạo biểu đồ tròn
        fig, ax = plt.subplots(figsize=(8,8))
        plt.pie(ratio_list, startangle=90, wedgeprops={'edgecolor' :'black'})
        plt.title('Pie chart of {}'.format(feature))
        plt.legend(loc='best',labels=ratio_size.index)
        plt.axis('equal')

        return plt.show()

    # Các đặc trưng còn lại
    else:
        # Tính số lượng, tần suất lớp của đặc trưng
        ratio_size = value_cnt_norm_cal(df, feature)
        ratio_size_len = len(ratio_size.index)
        ratio_list = []

        # Duyệt từng lớp để lấy tỷ lệ
        for i in range(ratio_size_len):
            ratio_list.append(ratio_size.iloc[i]['Frequency (%)'])

        # Tạo biểu đồ tròn
        fig, ax = plt.subplots(figsize=(8,8))
        plt.pie(ratio_list, labels=ratio_size.index, autopct='%1.2f%%', startangle=90, wedgeprops={'edgecolor' :'black'})
        plt.title('Pie chart of {}'.format(feature))
        plt.legend(loc='best')
        plt.axis('equal')

        return plt.show()


# Biểu đồ hình cột
def create_bar_plot(df, feature):
    # Xử lý đặc trưng: Marital status, Dwelling, Job title, Employment status, Education level
    if feature == 'Marital status' or feature == 'Dwelling' or feature == 'Job title' or feature == 'Employment status' or feature == 'Education level':
        # Tạo biểu đồ cột
        fig, ax = plt.subplots(figsize=(6,10))
        sns.barplot(x=value_cnt_norm_cal(df, feature).index, y=value_cnt_norm_cal(df, feature).values[:,0])
        ax.set_xticklabels(labels=value_cnt_norm_cal(df, feature).index, rotation=45, ha='right')
        plt.xlabel('{}'.format(feature))
        plt.ylabel('Count')
        plt.title('{} count'.format(feature))
        
        return plt.show()
    
    # Xử lý các đặc trưng còn lại
    else:
        fig, ax = plt.subplots(figsize=(6,10))
        sns.barplot(x=value_cnt_norm_cal(df, feature).index, y=value_cnt_norm_cal(df, feature).values[:,0])
        plt.xlabel('{}'.format(feature))
        plt.ylabel('Count')
        plt.title('{} count'.format(feature))

        return plt.show()


# Biểu đồ hình hộp
def create_box_plot(df, feature, cc_train_copy):
    # Xử lý đặc trưng Age
    if feature == 'Age':
        fig, ax = plt.subplots(figsize=(2,8))
        sns.boxplot(y=np.abs(df[feature])/365.25)       # Chuẩn hóa đặc trưng tuổi
        plt.title('{} distribution(Boxplot)'.format(feature))
        
        return plt.show()
    
    # Xử lý đặc trưng Children count
    if feature == 'Children count':
        fig, ax = plt.subplots(figsize=(2,8))
        sns.boxplot(y=df[feature])
        plt.title('{} distribution(Boxplot)'.format(feature))
        plt.yticks(np.arange(0,df[feature].max(),1))
        return plt.show()
    
    # Xử lý đặc trưng Employment length
    if feature == 'Employment length':
        fig, ax = plt.subplots(figsize=(2,8))
        
        # Lấy dữ liệu khách hành thất nghiệp
        employment_len_no_ret = cc_train_copy['Employment length'][cc_train_copy['Employment length'] < 0]
        employment_len_no_ret_yrs = np.abs(employment_len_no_ret)/365.25
        
        # Tạo biểu đồ
        sns.boxplot(y=employment_len_no_ret_yrs)
        plt.title('{} distribution(Boxplot)'.format(feature))
        plt.yticks(np.arange(0,employment_len_no_ret_yrs.max(),2))
        return plt.show()
    
    # Xử lý đặc trưng Income
    if feature == 'Income':
        fig, ax = plt.subplots(figsize=(2,8))
        sns.boxplot(y=df[feature])
        plt.title('{} distribution(Boxplot)'.format(feature))
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        return plt.show()
    
    # Xử lý đặc trưng Account age
    if feature == 'Account age':
        fig, ax = plt.subplots(figsize=(2,8))
        sns.boxplot(y=np.abs(df[feature]))
        plt.title('{} distribution(Boxplot)'.format(feature))
        return plt.show()
    else:
        fig, ax = plt.subplots(figsize=(2,8))
        sns.boxplot(y=df[feature])
        plt.title('{} distribution(Boxplot)'.format(feature))
        return plt.show()


# Biểu đồ dạng histogram
def create_hist_plot(df, feature, cc_train_copy, the_bins=50):
    # Xử lý dữ liệu Age
    if feature == 'Age':
        fig, ax = plt.subplots(figsize=(18,10))
        sns.histplot(np.abs(df[feature])/365.25, bins=the_bins, kde=True)       # Chuẩn hóa dữ liệu
        plt.title('{} distribution'.format(feature))
        return plt.show()
    
    # Xử lý dữ liệu Income
    if feature == 'Income':
        fig, ax = plt.subplots(figsize=(18,10))
        sns.histplot(df[feature], bins=the_bins, kde=True)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.title('{} distribution'.format(feature))
        return plt.show()
    
    # Xử lý dữ liệu Employment length
    if feature == 'Employment length':
        # Lấy dữ liệu khách hàng thất nghiệp
        employment_len_no_ret = cc_train_copy['Employment length'][cc_train_copy['Employment length'] < 0]
        employment_len_no_ret_yrs = np.abs(employment_len_no_ret)/365.25        # Chuẩn hóa dữ liệu
        fig, ax = plt.subplots(figsize=(18,10))
        sns.histplot(employment_len_no_ret_yrs, bins=the_bins, kde=True)
        plt.title('{} distribution'.format(feature))
        return plt.show()
    
    # Xử lý dữ liệu Account age
    if feature == 'Account age':
        fig, ax = plt.subplots(figsize=(18,10))
        sns.histplot(np.abs(df[feature]), bins=the_bins, kde=True)
        plt.title('{} distribution'.format(feature))
        return plt.show()
    else:
        fig, ax = plt.subplots(figsize=(18,10))
        sns.histplot(df[feature], bins=the_bins, kde=True)
        plt.title('{} distribution'.format(feature))
        return plt.show()
    