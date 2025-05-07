# CICIDS2017 Dataset Preprocessing

# 1. Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Set better chart style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# 2. Define preprocessing function
def preprocess_cicids_file(file_path, add_day_info=True):

    print(f"Preprocessing file: {os.path.basename(file_path)}")
    
    # 2.1 Load file with appropriate encoding
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully read file using {encoding} encoding")
            break
        except:
            continue
    
    if df is None:
        print(f"Unable to read file {file_path}, please check file path or format")
        return None
    
    # 2.2 Add data source information
    if add_day_info:
        file_name = os.path.basename(file_path)
        # Extract day of the week from filename
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
        
        # Add more detailed data source information
        df['Day'] = day
        
        # Further differentiate attack scenarios
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
    
    # 2.3 Handle column name formatting issues
    # Standardize column names: remove leading spaces
    df.columns = df.columns.str.strip()
    
    # 2.4 Handle missing values
    # Calculate and display missing values
    missing_values = df.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    if len(missing_cols) > 0:
        print(f"Missing values statistics:")
        print(missing_cols)
        
        # For Flow Bytes/s and Flow Packets/s columns, fill missing values with median
        for col in missing_cols.index:
            if col in df.columns:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                print(f"Missing values in column '{col}' filled with median {median_value}")
    else:
        print("No missing values in dataset")
    
    # 2.5 Handle infinity values
    # Check for infinity values
    inf_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_cols.append((col, inf_count))
    
    if inf_cols:
        print("Columns containing infinity values:")
        for col, count in inf_cols:
            print(f"- {col}: {count} infinity values")
            
            # Replace infinity values with column median
            median_value = df[col].replace([np.inf, -np.inf], np.nan).median()
            df[col].replace([np.inf, -np.inf], median_value, inplace=True)
            print(f"  Infinity values replaced with median {median_value}")
    else:
        print("No infinity values in dataset")
    
    # Standardize Label column format (if exists)
    # In CICIDS2017 dataset, label column name might be 'Label' or ' Label'
    if 'Label' in df.columns:
        label_col = 'Label'
    elif ' Label' in df.columns:
        label_col = ' Label'
        # Rename to standardized 'Label'
        df.rename(columns={' Label': 'Label'}, inplace=True)
        label_col = 'Label'
    else:
        print("Warning: Standard label column not found")
        label_col = None
    
    # 2.7 Check and process label values
    if label_col:
        # Standardize label format (case, spaces, etc.)
        df[label_col] = df[label_col].str.strip().str.upper()
        
        # Display label distribution
        label_counts = df[label_col].value_counts()
        print(f"\nLabel distribution:")
        print(label_counts)
    
    # 2.8 Handle outliers
    # For numeric columns, consider handling extreme outliers
    # Using IQR method to identify outliers, but only report without removing
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier boundaries
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Calculate number of outliers
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            outlier_percent = (outliers / len(df)) * 100
            outlier_summary[col] = (outliers, outlier_percent)
    
    if outlier_summary:
        print("\nPotential outlier report (using IQR method):")
        for col, (count, percent) in outlier_summary.items():
            print(f"- {col}: {count} outliers ({percent:.2f}%)")
    
    # 2.9 Feature scaling
    # Performed after merging all files, not executed here
    
    print(f"Preprocessing completed: {os.path.basename(file_path)}")
    print(f"Processed data shape: {df.shape}")
    
    return df

# 3. Main function: Process all CSV files in a directory
def preprocess_all_cicids_files(data_dir, output_dir=None, save_intermediate=True):

    # Ensure output directory exists
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get all CSV files
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in directory {data_dir}")
        return None
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Store preprocessed dataframes
    processed_dfs = {}
    
    # Process each file
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"\n{'='*50}")
        print(f"Starting processing: {file_name}")
        
        # Preprocess file
        processed_df = preprocess_cicids_file(file_path)
        
        if processed_df is not None:
            processed_dfs[file_name] = processed_df
            
            # Save intermediate results (optional)
            if save_intermediate and output_dir:
                output_path = os.path.join(output_dir, f"preprocessed_{file_name}")
                processed_df.to_csv(output_path, index=False)
                print(f"Saved preprocessed file to: {output_path}")
    
    print(f"\n{'='*50}")
    print(f"All {len(csv_files)} files processed!")
    
    return processed_dfs

# 4. Merge preprocessed files
def merge_preprocessed_files(processed_dfs, output_dir=None):

    if not processed_dfs:
        print("No data to merge")
        return None
    
    print("\nMerging preprocessed files...")
    
    # Merge all dataframes
    merged_df = pd.concat(processed_dfs.values(), ignore_index=True)
    
    print(f"Merged data shape: {merged_df.shape}")
    
    # Display merged label distribution (if exists)
    if 'Label' in merged_df.columns:
        label_counts = merged_df['Label'].value_counts()
        print("\nMerged label distribution:")
        print(label_counts)
    
    # Save merged results
    if output_dir:
        output_path = os.path.join(output_dir, "CICIDS2017_merged_preprocessed.csv")
        merged_df.to_csv(output_path, index=False)
        print(f"Saved merged data to: {output_path}")
    
    return merged_df

# 5. Perform preprocessing
# Set data paths
data_dir = '/root/autodl-tmp/projects/DL/dataset/extracted/MachineLearningCVE'
output_dir = '/root/autodl-tmp/projects/DL/dataset/preprocessed'

# Execute preprocessing
processed_dfs = preprocess_all_cicids_files(data_dir, output_dir)

# Merge preprocessed files
merged_df = merge_preprocessed_files(processed_dfs, output_dir)