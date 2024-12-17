# Thư viện cần thiết
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


# Biểu đồ hộp thể hiện tỷ lệ rủi ro
def low_high_risk_box_plot(df, feature, cc_train_copy):
    # Xử lý đặc trưng Age
    if feature == 'Age':
        print(np.abs(df.groupby('Is high risk')[feature].mean()/365.25))
        fig, ax = plt.subplots(figsize=(5,8))
        
        # Tạo biểu đồ cho IsHighRisk (is high risk: No and Yes)
        sns.boxplot(y=np.abs(df[feature])/365.25, x=df['Is high risk'])
        plt.xticks(ticks=[0,1], labels=['no','yes'])
        plt.title('High risk individuals grouped by age')
        return plt.show()
    
    # Xử lý dữ liệu Income
    if feature == 'Income':
        print(np.abs(df.groupby('Is high risk')[feature].mean()))
        fig, ax = plt.subplots(figsize=(5,8))
        sns.boxplot(y=np.abs(df[feature]),x=df['Is high risk'])
        plt.xticks(ticks=[0,1],labels=['no','yes'])
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.title('High risk individuals grouped by {}'.format(feature))
        return plt.show()
    
    if feature == 'Employment length':
        # Kiểm tra khách hàng có tỷ lệ rủi ro cao không
        employment_no_ret = cc_train_copy['Employment length'][cc_train_copy['Employment length'] < 0]
        employment_no_ret_idx = employment_no_ret.index
        employment_len_no_ret_yrs = np.abs(employment_no_ret)/365.25

        # Trích xuất dữ liệu khác hàng cùng với tỷ lệ high risk
        employment_no_ret_df = cc_train_copy.iloc[employment_no_ret_idx][['Employment length','Is high risk']]
        employment_no_ret_is_high_risk = employment_no_ret_df.groupby('Is high risk')['Employment length'].mean()
        print(np.abs(employment_no_ret_is_high_risk)/365.25)
        
        # Vẽ biểu đồ hộp
        fig, ax = plt.subplots(figsize=(5,8))
        sns.boxplot(y=employment_len_no_ret_yrs, x=df['Is high risk'])
        plt.xticks(ticks=[0,1], labels=['no','yes'])
        plt.title('High vs low risk individuals grouped by {}'.format(feature))
        return plt.show()
    else:
        print(np.abs(df.groupby('Is high risk')[feature].mean()))
        fig, ax = plt.subplots(figsize=(5,8))
        sns.boxplot(y=np.abs(df[feature]),x=df['Is high risk'])
        plt.xticks(ticks=[0,1],labels=['no','yes'])
        plt.title('High risk individuals grouped by {}'.format(feature))
        return plt.show()
    

# Biểu đồ cột thể hiện rủi ro
def low_high_risk_bar_plot(df, feature):
    # Khách hàng có rủi ro cao
    is_high_risk_grp = df.groupby(feature)['Is high risk'].sum()
    is_high_risk_grp_srt = is_high_risk_grp.sort_values(ascending=False)
    print(dict(is_high_risk_grp_srt))

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(6,10))
    sns.barplot(x=is_high_risk_grp_srt.index, y=is_high_risk_grp_srt.values)
    ax.set_xticklabels(labels=is_high_risk_grp_srt.index, rotation=45, ha='right')
    plt.ylabel('Count')
    plt.title('High risk applicants count grouped by {}'.format(feature))
    return plt.show()
