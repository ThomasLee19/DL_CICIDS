# CICIDS2017数据集预处理脚本

# 1. 导入必要的库
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# 设置更好的图表样式
plt.style.use('ggplot')
sns.set(style="whitegrid")

# 2. 定义预处理函数
def preprocess_cicids_file(file_path, add_day_info=True):
    """
    对CICIDS2017数据集中的单个CSV文件进行预处理
    
    参数:
        file_path: CSV文件路径
        add_day_info: 是否添加表示数据来源天数的列
        
    返回:
        预处理后的DataFrame
    """
    print(f"正在预处理文件: {os.path.basename(file_path)}")
    
    # 2.1 使用适当的编码加载文件
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except:
            continue
    
    if df is None:
        print(f"无法读取文件 {file_path}，请检查文件路径或格式")
        return None
    
    # 2.2 添加数据来源信息
    if add_day_info:
        file_name = os.path.basename(file_path)
        # 从文件名中提取星期几
        if "Monday" in file_name:
            day = "Monday"
        elif "Tuesday" in file_name:
            day = "Tuesday"  
        elif "Wednesday" in file_name:
            day = "Wednesday"
        elif "Thursday" in file_name:
            day = "Thursday"
        elif "Friday" in file_name:
            day = "Friday"
        else:
            day = "Unknown"
        
        # 添加更详细的数据来源信息
        df['Day'] = day
        
        # 进一步区分不同的攻击场景
        if "Morning-WebAttacks" in file_name:
            df['Scenario'] = "WebAttacks"
        elif "Afternoon-Infilteration" in file_name:
            df['Scenario'] = "Infilteration"
        elif "Morning" in file_name and "Friday" in file_name:
            df['Scenario'] = "Friday-Morning"
        elif "Afternoon-PortScan" in file_name:
            df['Scenario'] = "PortScan"
        elif "Afternoon-DDos" in file_name:
            df['Scenario'] = "DDoS"
        else:
            df['Scenario'] = "Normal"
    
    # 2.3 处理列名格式问题
    # 统一列名格式：删除列名开头的空格
    df.columns = df.columns.str.strip()
    
    # 2.4 处理缺失值
    # 计算并显示缺失值
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if len(missing_cols) > 0:
        print(f"缺失值列统计:")
        print(missing_cols)
        
        # 对于Flow Bytes/s和Flow Packets/s列，用中位数填充缺失值
        for col in missing_cols.index:
            if col in df.columns:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"列 '{col}' 的缺失值已用中位数 {median_value} 填充")
    else:
        print("数据集中没有缺失值")
    
    # 2.5 处理无穷值
    # 检查无穷值
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_cols.append((col, inf_count))
    
    if inf_cols:
        print("包含无穷值的列:")
        for col, count in inf_cols:
            print(f"- {col}: {count}个无穷值")
            
            # 用列的中位数替换无穷值
            median_value = df[col].replace([np.inf, -np.inf], np.nan).median()
            df[col].replace([np.inf, -np.inf], median_value, inplace=True)
            print(f"  已用中位数 {median_value} 替换无穷值")
    else:
        print("数据集中没有无穷值")
    
    # 统一Label列格式(如果存在)
    # 在CICIDS2017数据集中，标签列名可能是'Label'或' Label'
    if 'Label' in df.columns:
        label_col = 'Label'
    elif ' Label' in df.columns:
        label_col = ' Label'
        # 重命名为统一的'Label'
        df.rename(columns={' Label': 'Label'}, inplace=True)
        label_col = 'Label'
    else:
        print("警告: 未找到标准标签列")
        label_col = None
    
    # 2.7 检查并处理标签值
    if label_col:
        # 统一标签格式（大小写，空格等）
        df[label_col] = df[label_col].str.strip().str.upper()
        
        # 显示标签分布
        label_counts = df[label_col].value_counts()
        print(f"\n标签分布:")
        print(label_counts)
    
    # 2.8 处理异常值
    # 对于数值列，可以考虑处理极端异常值
    # 这里使用IQR方法识别异常值，但仅报告不删除
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # 定义异常值边界
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # 计算异常值数量
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            outlier_percent = (outliers / len(df)) * 100
            outlier_summary[col] = (outliers, outlier_percent)
    
    if outlier_summary:
        print("\n潜在异常值报告 (使用IQR方法):")
        for col, (count, percent) in outlier_summary.items():
            print(f"- {col}: {count}个异常值 ({percent:.2f}%)")
    
    # 2.9 特征规模化
    # 在所有文件合并后进行，这里不执行
    
    print(f"预处理完成: {os.path.basename(file_path)}")
    print(f"处理后数据形状: {df.shape}")
    
    return df

# 3. 主函数：处理文件夹中的所有CSV文件
def preprocess_all_cicids_files(data_dir, output_dir=None, save_intermediate=True):
    """
    预处理CICIDS2017数据集中的所有CSV文件
    
    参数:
        data_dir: 包含CSV文件的目录
        output_dir: 保存处理后文件的目录
        save_intermediate: 是否保存中间处理结果
        
    返回:
        包含所有预处理后数据的字典
    """
    # 确保输出目录存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 获取所有CSV文件
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"在目录 {data_dir} 中未找到CSV文件")
        return None
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 存储预处理后的数据帧
    processed_dfs = {}
    
    # 处理每个文件
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\n{'='*50}")
        print(f"开始处理: {file_name}")
        
        # 预处理文件
        processed_df = preprocess_cicids_file(file_path)
        
        if processed_df is not None:
            processed_dfs[file_name] = processed_df
            
            # 保存中间结果（可选）
            if save_intermediate and output_dir:
                output_path = os.path.join(output_dir, f"preprocessed_{file_name}")
                processed_df.to_csv(output_path, index=False)
                print(f"保存预处理后的文件到: {output_path}")
    
    print(f"\n{'='*50}")
    print(f"所有 {len(csv_files)} 个文件处理完成!")
    
    return processed_dfs

# 4. 合并预处理后的文件
def merge_preprocessed_files(processed_dfs, output_dir=None):
    """
    合并所有预处理后的数据帧
    
    参数:
        processed_dfs: 包含预处理后数据帧的字典
        output_dir: 保存合并结果的目录
        
    返回:
        合并后的DataFrame
    """
    if not processed_dfs:
        print("没有可合并的数据")
        return None
    
    print("\n合并预处理后的文件...")
    
    # 合并所有数据帧
    merged_df = pd.concat(processed_dfs.values(), ignore_index=True)
    
    print(f"合并后的数据形状: {merged_df.shape}")
    
    # 显示合并后的标签分布（如果存在）
    if 'Label' in merged_df.columns:
        label_counts = merged_df['Label'].value_counts()
        print("\n合并后的标签分布:")
        print(label_counts)
    
    # 保存合并后的结果
    if output_dir:
        output_path = os.path.join(output_dir, "CICIDS2017_merged_preprocessed.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"保存合并后的数据到: {output_path}")
    
    return merged_df

# 5. 进行预处理
# 设置数据路径
data_dir = '/root/autodl-tmp/projects/DL/dataset/extracted/MachineLearningCVE'
output_dir = '/root/autodl-tmp/projects/DL/dataset/preprocessed'

# 执行预处理
processed_dfs = preprocess_all_cicids_files(data_dir, output_dir)

# 合并预处理后的文件
merged_df = merge_preprocessed_files(processed_dfs, output_dir)