import wandb

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
# from tueplots.bundles import neurips2024
import os.path as osp
import numpy as np
from scipy.interpolate import interp1d


DIV_LINE_WIDTH = 50

def preprocess_data(data_list, k):
    for data in data_list:
        if data.iloc[0]['Condition1'] == 'POWR (Ours)':
            for index, row in data.iterrows():
                if row['timestep'] > k:
                    # Get the value before modification
                    value_before = data.loc[data['timestep'] < k, 'train_mean_reward']
                    # Replace all values before k with the value at k
                    data.loc[data['timestep'] < k, 'train_mean_reward'] = -200 #row['train_mean_reward']
                    # Get the value after modification
                    value_after = data.loc[data['timestep'] < k, 'train_mean_reward']
                    print(f"Algorithm: {row['Condition1']}") # , Timestep: {row['timestep']}, Value before: {value_before}, Value after: {value_after}")
                    break
    return data_list

def read_csvs_from_directory(directory):
    """
    Read CSV files from directory and group them by algorithm name.
    Algorithm name is determined by splitting the filename by '___' and taking the first part.
    """
    grouped_dataframes = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            algorithm_name = filename.split("___")[0]
            df = pd.read_csv(os.path.join(directory, filename))
            
            if algorithm_name not in grouped_dataframes:
                grouped_dataframes[algorithm_name] = []
            
            grouped_dataframes[algorithm_name].append(df)
    
    return grouped_dataframes

def calculate_interquartile_mean(values):
    """
    Calculate the interquartile mean (IQM) - 25% trimmed mean.
    Discards the bottom and top 25% of values and calculates the mean of the remaining 50%.
    
    Parameters:
    - values: Array-like of numerical values
    
    Returns:
    - Interquartile mean value
    """
    if len(values) == 0:
        return np.nan
    
    values = np.array(values)
    values = values[~np.isnan(values)]  # Remove NaN values
    
    if len(values) == 0:
        return np.nan
    
    if len(values) < 4:
        # For very small samples, fall back to regular mean
        return np.mean(values)
    
    # Calculate Q1 and Q3
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    
    # Filter values within interquartile range
    iqr_values = values[(values >= q1) & (values <= q3)]
    
    if len(iqr_values) == 0:
        return np.mean(values)  # Fallback to regular mean
    
    return np.mean(iqr_values)

def aggregate_runs(dataframes, x_column, y_column, aggregator='mean'):
    """
    Aggregate multiple runs into a single dataframe.
    
    Parameters:
    - dataframes: List of dataframes to aggregate
    - x_column: Column name for x-axis
    - y_column: Column name for y-axis values to aggregate
    - aggregator: Function or string specifying how to aggregate
    
    Returns:
    - Aggregated dataframe with columns: x_column, y_mean, y_min, y_max, y_std, y_stderr, y_iqm
    """
    # Convert string aggregator to numpy function
    if isinstance(aggregator, str):
        if aggregator == 'mean':
            agg_func = np.mean
        elif aggregator == 'median':
            agg_func = np.median
        elif aggregator == 'min':
            agg_func = np.min
        elif aggregator == 'max':
            agg_func = np.max
        elif aggregator == 'iqm' or aggregator == 'interquartile_mean':
            agg_func = calculate_interquartile_mean
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
    else:
        agg_func = aggregator  # Use the provided function directly
    
    # Collect all x values across dataframes
    all_x_values = set()
    for df in dataframes:
        all_x_values.update(df[x_column].values)
    
    all_x_values = sorted(list(all_x_values))
    
    # Initialize arrays to store aggregated values
    y_values = []
    
    # For each x value, collect all corresponding y values from all dataframes
    for x_val in all_x_values:
        y_for_x = []
        for df in dataframes:
            # Get rows where x_column equals x_val
            rows = df[df[x_column] == x_val]
            if not rows.empty:
                y_for_x.extend(rows[y_column].values)
        
        if y_for_x:
            y_values.append(y_for_x)
        else:
            y_values.append([np.nan])
    
    # Create the result dataframe
    result = pd.DataFrame({
        x_column: all_x_values,
        'y_mean': [np.mean(y) for y in y_values],
        'y_median': [np.median(y) if len(y) > 0 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_min': [np.min(y) if len(y) > 0 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_max': [np.max(y) if len(y) > 0 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_std': [np.std(y) if len(y) > 1 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_stderr': [np.std(y) / np.sqrt(len(y)) if len(y) > 1 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_iqm': [calculate_interquartile_mean(y) if len(y) > 0 and not all(np.isnan(y)) else np.nan for y in y_values],
        'y_count': [len(y) for y in y_values],
        'y_aggregated': [agg_func(y) if len(y) > 0 and not all(np.isnan(y)) else np.nan for y in y_values]
    })
    
    return result

def stratified_bootstrap_aggregate(dataframes, x_column, y_column, aggregator='mean', n_bootstrap=1000):
    """
    Aggregate multiple runs using stratified bootstrap confidence intervals.
    Re-samples runs with replacement independently for each x value to construct
    bootstrap samples and calculate statistics.
    
    Parameters:
    - dataframes: List of dataframes to aggregate (each represents one run/task)
    - x_column: Column name for x-axis
    - y_column: Column name for y-axis values to aggregate
    - aggregator: Function or string specifying how to aggregate
    - n_bootstrap: Number of bootstrap samples to generate
    
    Returns:
    - Aggregated dataframe with bootstrap-based statistics
    """
    if isinstance(aggregator, str):
        if aggregator == 'mean':
            agg_func = np.mean
        elif aggregator == 'median':
            agg_func = np.median
        elif aggregator == 'min':
            agg_func = np.min
        elif aggregator == 'max':
            agg_func = np.max
        elif aggregator == 'iqm' or aggregator == 'interquartile_mean':
            agg_func = calculate_interquartile_mean
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
    else:
        agg_func = aggregator
    
    # Collect all x values across dataframes
    all_x_values = set()
    for df in dataframes:
        all_x_values.update(df[x_column].values)
    
    all_x_values = sorted(list(all_x_values))
    
    # For each x value, perform stratified bootstrap
    bootstrap_results = {}
    
    print(f"Performing stratified bootstrap with {n_bootstrap} samples...")
    
    for x_val in all_x_values:
        # Collect y values from each run/task for this x value
        task_values = []
        for df in dataframes:
            rows = df[df[x_column] == x_val]
            if not rows.empty:
                # Take all y values for this task at this x
                y_vals = rows[y_column].dropna().values
                if len(y_vals) > 0:
                    task_values.append(y_vals)
        
        if not task_values:
            continue
        
        # Perform bootstrap resampling
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample each task independently with replacement
            bootstrap_sample = []
            for task_vals in task_values:
                if len(task_vals) > 0:
                    # Sample with replacement from this task
                    resampled = np.random.choice(task_vals, size=len(task_vals), replace=True)
                    bootstrap_sample.extend(resampled)
            
            if bootstrap_sample:
                # Calculate statistic for this bootstrap sample
                bootstrap_stats.append(agg_func(bootstrap_sample))
        
        bootstrap_results[x_val] = np.array(bootstrap_stats)
    
    # Calculate final statistics from bootstrap distribution
    result_data = {
        x_column: [],
        'y_mean': [],
        'y_median': [],
        'y_min': [],
        'y_max': [],
        'y_std': [],
        'y_stderr': [],
        'y_iqm': [],
        'y_aggregated': [],
        'y_bootstrap_mean': [],
        'y_bootstrap_std': [],
        'y_bootstrap_ci_2_5': [],
        'y_bootstrap_ci_97_5': [],
        'y_bootstrap_ci_5': [],
        'y_bootstrap_ci_95': [],
        'y_count': []
    }
    
    for x_val in all_x_values:
        if x_val not in bootstrap_results:
            continue
            
        bootstrap_stats = bootstrap_results[x_val]
        
        # Original statistics (non-bootstrap)
        original_values = []
        for df in dataframes:
            rows = df[df[x_column] == x_val]
            if not rows.empty:
                original_values.extend(rows[y_column].dropna().values)
        
        result_data[x_column].append(x_val)
        result_data['y_mean'].append(np.mean(original_values) if original_values else np.nan)
        result_data['y_median'].append(np.median(original_values) if original_values else np.nan)
        result_data['y_min'].append(np.min(original_values) if original_values else np.nan)
        result_data['y_max'].append(np.max(original_values) if original_values else np.nan)
        result_data['y_std'].append(np.std(original_values) if len(original_values) > 1 else np.nan)
        result_data['y_stderr'].append(np.std(original_values) / np.sqrt(len(original_values)) if len(original_values) > 1 else np.nan)
        result_data['y_iqm'].append(calculate_interquartile_mean(original_values) if original_values else np.nan)
        result_data['y_aggregated'].append(agg_func(original_values) if original_values else np.nan)
        
        # Bootstrap statistics
        result_data['y_bootstrap_mean'].append(np.mean(bootstrap_stats))
        result_data['y_bootstrap_std'].append(np.std(bootstrap_stats))
        result_data['y_bootstrap_ci_2_5'].append(np.percentile(bootstrap_stats, 2.5))
        result_data['y_bootstrap_ci_97_5'].append(np.percentile(bootstrap_stats, 97.5))
        result_data['y_bootstrap_ci_5'].append(np.percentile(bootstrap_stats, 5))
        result_data['y_bootstrap_ci_95'].append(np.percentile(bootstrap_stats, 95))
        result_data['y_count'].append(len(original_values))
    
    return pd.DataFrame(result_data)

def stratified_bootstrap_aggregate_with_tasks(grouped_dataframes_with_tasks, x_column, y_column, aggregator='mean', n_bootstrap=1000):
    """
    Aggregate multiple runs using stratified bootstrap with proper task stratification.
    
    Parameters:
    - grouped_dataframes_with_tasks: dict with algorithm -> list of (dataframe, task_name)
    - x_column: Column name for x-axis
    - y_column: Column name for y-axis values to aggregate
    - aggregator: Function or string specifying how to aggregate
    - n_bootstrap: Number of bootstrap samples to generate
    
    Returns:
    - Aggregated dataframe with bootstrap-based statistics
    """
    if isinstance(aggregator, str):
        if aggregator == 'mean':
            agg_func = np.mean
        elif aggregator == 'median':
            agg_func = np.median
        elif aggregator == 'min':
            agg_func = np.min
        elif aggregator == 'max':
            agg_func = np.max
        elif aggregator == 'iqm' or aggregator == 'interquartile_mean':
            agg_func = calculate_interquartile_mean
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
    else:
        agg_func = aggregator
    
    # Collect all x values across all dataframes
    all_x_values = set()
    for algorithm, df_task_pairs in grouped_dataframes_with_tasks.items():
        for df, task_name in df_task_pairs:
            all_x_values.update(df[x_column].values)
    
    all_x_values = sorted(list(all_x_values))
    
    # For each x value, perform stratified bootstrap
    bootstrap_results = {}
    
    print(f"Performing stratified bootstrap with {n_bootstrap} samples...")
    
    for x_val in all_x_values:
        # Organize data by task for this x value
        task_algorithm_values = {}  # task_name -> {algorithm -> [values]}
        
        for algorithm, df_task_pairs in grouped_dataframes_with_tasks.items():
            for df, task_name in df_task_pairs:
                if task_name not in task_algorithm_values:
                    task_algorithm_values[task_name] = {}
                if algorithm not in task_algorithm_values[task_name]:
                    task_algorithm_values[task_name][algorithm] = []
                
                # Get values for this x_val
                rows = df[df[x_column] == x_val]
                if not rows.empty:
                    y_vals = rows[y_column].dropna().values
                    if len(y_vals) > 0:
                        task_algorithm_values[task_name][algorithm].extend(y_vals)
        
        if not task_algorithm_values:
            continue
        
        # Perform bootstrap resampling
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = []
            
            # For each task, resample runs independently
            for task_name, algorithm_values in task_algorithm_values.items():
                for algorithm, values in algorithm_values.items():
                    if len(values) > 0:
                        # Sample with replacement from this algorithm's runs in this task
                        resampled = np.random.choice(values, size=len(values), replace=True)
                        bootstrap_sample.extend(resampled)
            
            if bootstrap_sample:
                # Calculate statistic for this bootstrap sample
                bootstrap_stats.append(agg_func(bootstrap_sample))
        
        bootstrap_results[x_val] = np.array(bootstrap_stats)
        
        # Debug info
        total_runs = sum(len(alg_vals) for task_vals in task_algorithm_values.values() 
                        for alg_vals in task_vals.values())
        print(f"  x={x_val}: {len(task_algorithm_values)} tasks, {total_runs} total runs")
    
    # Calculate final statistics from bootstrap distribution
    result_data = {
        x_column: [],
        'y_mean': [],
        'y_median': [],
        'y_min': [],
        'y_max': [],
        'y_std': [],
        'y_stderr': [],
        'y_iqm': [],
        'y_aggregated': [],
        'y_bootstrap_mean': [],
        'y_bootstrap_std': [],
        'y_bootstrap_ci_2_5': [],
        'y_bootstrap_ci_97_5': [],
        'y_bootstrap_ci_5': [],
        'y_bootstrap_ci_95': [],
        'y_count': []
    }
    
    for x_val in all_x_values:
        if x_val not in bootstrap_results:
            continue
            
        bootstrap_stats = bootstrap_results[x_val]
        
        # Original statistics (non-bootstrap)
        original_values = []
        for algorithm, df_task_pairs in grouped_dataframes_with_tasks.items():
            for df, task_name in df_task_pairs:
                rows = df[df[x_column] == x_val]
                if not rows.empty:
                    original_values.extend(rows[y_column].dropna().values)
        
        result_data[x_column].append(x_val)
        result_data['y_mean'].append(np.mean(original_values) if original_values else np.nan)
        result_data['y_median'].append(np.median(original_values) if original_values else np.nan)
        result_data['y_min'].append(np.min(original_values) if original_values else np.nan)
        result_data['y_max'].append(np.max(original_values) if original_values else np.nan)
        result_data['y_std'].append(np.std(original_values) if len(original_values) > 1 else np.nan)
        result_data['y_stderr'].append(np.std(original_values) / np.sqrt(len(original_values)) if len(original_values) > 1 else np.nan)
        result_data['y_iqm'].append(calculate_interquartile_mean(original_values) if original_values else np.nan)
        result_data['y_aggregated'].append(agg_func(original_values) if original_values else np.nan)
        
        # Bootstrap statistics
        result_data['y_bootstrap_mean'].append(np.mean(bootstrap_stats))
        result_data['y_bootstrap_std'].append(np.std(bootstrap_stats))
        result_data['y_bootstrap_ci_2_5'].append(np.percentile(bootstrap_stats, 2.5))
        result_data['y_bootstrap_ci_97_5'].append(np.percentile(bootstrap_stats, 97.5))
        result_data['y_bootstrap_ci_5'].append(np.percentile(bootstrap_stats, 5))
        result_data['y_bootstrap_ci_95'].append(np.percentile(bootstrap_stats, 95))
        result_data['y_count'].append(len(original_values))
    
    return pd.DataFrame(result_data)

def detect_task_structure(csv_path):
    """
    Detect if the path contains single task or multiple tasks.
    
    Parameters:
    - csv_path: Path to analyze
    
    Returns:
    - tuple: (is_single_task, task_data)
      - is_single_task: True if single task, False if multi-task
      - task_data: dict with task organization
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"Path {csv_path} does not exist")
    
    # Check if there are CSV files directly in this directory
    csv_files = [f for f in os.listdir(csv_path) if f.endswith('.csv')]
    
    # Check if there are subdirectories
    subdirs = [d for d in os.listdir(csv_path) 
               if os.path.isdir(os.path.join(csv_path, d))]
    
    if csv_files and not subdirs:
        # Single task: CSV files directly in the directory
        return True, {'single_task': csv_path}
    elif subdirs and not csv_files:
        # Multi-task: subdirectories containing CSV files
        task_data = {}
        for subdir in subdirs:
            subdir_path = os.path.join(csv_path, subdir)
            subdir_csvs = [f for f in os.listdir(subdir_path) if f.endswith('.csv')]
            if subdir_csvs:
                task_data[subdir] = subdir_path
        return False, task_data
    else:
        raise ValueError(f"Ambiguous directory structure in {csv_path}. "
                        "Directory should contain either CSV files directly (single task) "
                        "or subdirectories with CSV files (multi-task), but not both.")

def read_csvs_with_task_awareness(csv_path):
    """
    Read CSV files with task awareness for stratified bootstrap.
    
    Returns:
    - tuple: (grouped_dataframes, task_structure)
      - grouped_dataframes: dict with algorithm -> list of (dataframe, task_name)
      - task_structure: dict with task organization info
    """
    is_single_task, task_data = detect_task_structure(csv_path)
    
    grouped_dataframes = {}
    
    if is_single_task:
        # Single task: read all CSVs from the directory
        print(f"Detected single task in: {csv_path}")
        task_name = os.path.basename(csv_path)
        
        for filename in os.listdir(csv_path):
            if filename.endswith(".csv"):
                algorithm_name = filename.split("___")[0]
                df = pd.read_csv(os.path.join(csv_path, filename))
                
                if algorithm_name not in grouped_dataframes:
                    grouped_dataframes[algorithm_name] = []
                
                # Store dataframe with task information
                grouped_dataframes[algorithm_name].append((df, task_name))
    
    else:
        # Multi-task: read CSVs from each subdirectory
        print(f"Detected multi-task structure with tasks: {list(task_data.keys())}")
        
        for task_name, task_path in task_data.items():
            print(f"  Reading task: {task_name}")
            
            for filename in os.listdir(task_path):
                if filename.endswith(".csv"):
                    algorithm_name = filename.split("___")[0]
                    df = pd.read_csv(os.path.join(task_path, filename))
                    
                    if algorithm_name not in grouped_dataframes:
                        grouped_dataframes[algorithm_name] = []
                    
                    # Store dataframe with task information
                    grouped_dataframes[algorithm_name].append((df, task_name))
    
    task_structure = {
        'is_single_task': is_single_task,
        'task_data': task_data
    }
    
    return grouped_dataframes, task_structure

def stratified_bootstrap_aggregate_with_tasks(grouped_dataframes_with_tasks, x_column, y_column, aggregator='mean', n_bootstrap=1000):
    """
    Aggregate multiple runs using stratified bootstrap with proper task stratification.
    
    Parameters:
    - grouped_dataframes_with_tasks: dict with algorithm -> list of (dataframe, task_name)
    - x_column: Column name for x-axis
    - y_column: Column name for y-axis values to aggregate
    - aggregator: Function or string specifying how to aggregate
    - n_bootstrap: Number of bootstrap samples to generate
    
    Returns:
    - Aggregated dataframe with bootstrap-based statistics
    """
    if isinstance(aggregator, str):
        if aggregator == 'mean':
            agg_func = np.mean
        elif aggregator == 'median':
            agg_func = np.median
        elif aggregator == 'min':
            agg_func = np.min
        elif aggregator == 'max':
            agg_func = np.max
        elif aggregator == 'iqm' or aggregator == 'interquartile_mean':
            agg_func = calculate_interquartile_mean
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")
    else:
        agg_func = aggregator
    
    # Collect all x values across all dataframes
    all_x_values = set()
    for algorithm, df_task_pairs in grouped_dataframes_with_tasks.items():
        for df, task_name in df_task_pairs:
            all_x_values.update(df[x_column].values)
    
    all_x_values = sorted(list(all_x_values))
    
    # For each x value, perform stratified bootstrap
    bootstrap_results = {}
    
    print(f"Performing stratified bootstrap with {n_bootstrap} samples...")
    
    for x_val in all_x_values:
        # Organize data by task for this x value
        task_algorithm_values = {}  # task_name -> {algorithm -> [values]}
        
        for algorithm, df_task_pairs in grouped_dataframes_with_tasks.items():
            for df, task_name in df_task_pairs:
                if task_name not in task_algorithm_values:
                    task_algorithm_values[task_name] = {}
                if algorithm not in task_algorithm_values[task_name]:
                    task_algorithm_values[task_name][algorithm] = []
                
                # Get values for this x_val
                rows = df[df[x_column] == x_val]
                if not rows.empty:
                    y_vals = rows[y_column].dropna().values
                    if len(y_vals) > 0:
                        task_algorithm_values[task_name][algorithm].extend(y_vals)
        
        if not task_algorithm_values:
            continue
        
        # Perform bootstrap resampling
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = []
            
            # For each task, resample runs independently
            for task_name, algorithm_values in task_algorithm_values.items():
                for algorithm, values in algorithm_values.items():
                    if len(values) > 0:
                        # Sample with replacement from this algorithm's runs in this task
                        resampled = np.random.choice(values, size=len(values), replace=True)
                        bootstrap_sample.extend(resampled)
            
            if bootstrap_sample:
                # Calculate statistic for this bootstrap sample
                bootstrap_stats.append(agg_func(bootstrap_sample))
        
        bootstrap_results[x_val] = np.array(bootstrap_stats)
        
        # Debug info
        total_runs = sum(len(alg_vals) for task_vals in task_algorithm_values.values() 
                        for alg_vals in task_vals.values())
        print(f"  x={x_val}: {len(task_algorithm_values)} tasks, {total_runs} total runs")
    
    # Calculate final statistics from bootstrap distribution
    result_data = {
        x_column: [],
        'y_mean': [],
        'y_median': [],
        'y_min': [],
        'y_max': [],
        'y_std': [],
        'y_stderr': [],
        'y_iqm': [],
        'y_aggregated': [],
        'y_bootstrap_mean': [],
        'y_bootstrap_std': [],
        'y_bootstrap_ci_2_5': [],
        'y_bootstrap_ci_97_5': [],
        'y_bootstrap_ci_5': [],
        'y_bootstrap_ci_95': [],
        'y_count': []
    }
    
    for x_val in all_x_values:
        if x_val not in bootstrap_results:
            continue
            
        bootstrap_stats = bootstrap_results[x_val]
        
        # Original statistics (non-bootstrap)
        original_values = []
        for algorithm, df_task_pairs in grouped_dataframes_with_tasks.items():
            for df, task_name in df_task_pairs:
                rows = df[df[x_column] == x_val]
                if not rows.empty:
                    original_values.extend(rows[y_column].dropna().values)
        
        result_data[x_column].append(x_val)
        result_data['y_mean'].append(np.mean(original_values) if original_values else np.nan)
        result_data['y_median'].append(np.median(original_values) if original_values else np.nan)
        result_data['y_min'].append(np.min(original_values) if original_values else np.nan)
        result_data['y_max'].append(np.max(original_values) if original_values else np.nan)
        result_data['y_std'].append(np.std(original_values) if len(original_values) > 1 else np.nan)
        result_data['y_stderr'].append(np.std(original_values) / np.sqrt(len(original_values)) if len(original_values) > 1 else np.nan)
        result_data['y_iqm'].append(calculate_interquartile_mean(original_values) if original_values else np.nan)
        result_data['y_aggregated'].append(agg_func(original_values) if original_values else np.nan)
        
        # Bootstrap statistics
        result_data['y_bootstrap_mean'].append(np.mean(bootstrap_stats))
        result_data['y_bootstrap_std'].append(np.std(bootstrap_stats))
        result_data['y_bootstrap_ci_2_5'].append(np.percentile(bootstrap_stats, 2.5))
        result_data['y_bootstrap_ci_97_5'].append(np.percentile(bootstrap_stats, 97.5))
        result_data['y_bootstrap_ci_5'].append(np.percentile(bootstrap_stats, 5))
        result_data['y_bootstrap_ci_95'].append(np.percentile(bootstrap_stats, 95))
        result_data['y_count'].append(len(original_values))
    
    return pd.DataFrame(result_data)

def extract_dataframes_only(grouped_dataframes_with_tasks):
    """
    Extract only dataframes from the task-aware structure for compatibility.
    """
    grouped_dataframes = {}
    for algorithm, df_task_pairs in grouped_dataframes_with_tasks.items():
        grouped_dataframes[algorithm] = [df for df, task_name in df_task_pairs]
    return grouped_dataframes

def create_value_formatter(scale_factor=None):
    """
    Create a formatter function for axis tick values.
    
    Parameters:
    - scale_factor: Value by which to divide the tick values (e.g., 1e5, 1e3)
    
    Returns:
    - A formatter function for matplotlib ticker
    """
    def format_func(value, ticker):
        if np.isnan(value) or value < 0:
            return value
        elif scale_factor is None:
            return value
        else:
            # Divide by scale factor
            scaled_value = value / scale_factor
            # Check if the value has no decimal part (is a whole number)
            if scaled_value == int(scaled_value):
                return "{:d}".format(int(scaled_value))
            else:
                return "{:.1f}".format(scaled_value)
    
    return format_func

def print_color_menu():
    """
    Stampa un menu colorato dei colori disponibili per seaborn.
    """
    # Colori base disponibili in seaborn/matplotlib
    colors = {
        'blue': '\033[94m',
        'green': '\033[92m', 
        'red': '\033[91m',
        'orange': '\033[93m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'pink': '\033[95m',
        'brown': '\033[33m',
        'gray': '\033[90m',
        'olive': '\033[93m'
    }
    
    reset_color = '\033[0m'
    
    print("\n=== MENU COLORI DISPONIBILI ===")
    print("Seleziona un colore digitando il nome corrispondente:")
    
    for color_name, color_code in colors.items():
        print(f"{color_code}● {color_name}{reset_color}")
    
    print("\nPuoi anche usare codici esadecimali (es: #FF5733) o nomi matplotlib standard.")
    print("====================================\n")

def load_or_create_color_mapping(csv_path, algorithm_names, force_color=False):
    """
    Carica o crea un mapping dei colori per gli algoritmi.
    
    Parameters:
    - csv_path: Percorso della cartella contenente i CSV
    - algorithm_names: Lista dei nomi degli algoritmi (dopo rinominazione)
    - force_color: Se True, ignora il file esistente e chiede nuovi colori
    
    Returns:
    - Dizionario con il mapping nome_algoritmo -> colore
    """
    mapping_file = os.path.join(csv_path, "algorithm_color_mapping.txt")
    mapping = {}
    
    # Se force_color è True, chiedi sempre all'utente i nuovi colori
    if force_color:
        print("Force color attivato: creazione di un nuovo mapping colori...")
        if os.path.exists(mapping_file):
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                response = input(f"Il file {mapping_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
                if response == 'y':
                    break
                elif response == 'n':
                    print("Operazione annullata.")
                    return {}  # Restituisci mapping vuoto per usare colori default
                else:
                    attempts += 1
                    print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
            
            if attempts == max_attempts:
                raise ValueError("Troppi tentativi falliti. Operazione annullata.")
        
        # Crea nuovo mapping da zero
        print(f"Algoritmi per cui assegnare colori: {algorithm_names}")
        for alg_name in algorithm_names:
            mapping[alg_name] = get_algorithm_color(alg_name)
        
        # Salva il mapping
        save_color_mapping(mapping_file, mapping, overwrite=True)
        return mapping
    
    # Prova a caricare il mapping esistente se force_color è False
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        algorithm, color = line.split(':', 1)
                        mapping[algorithm.strip()] = color.strip()
            print(f"Caricato mapping colori esistente da {mapping_file}")
            
            # Verifica se tutti gli algoritmi sono nel mapping
            missing_algorithms = [alg for alg in algorithm_names if alg not in mapping]
            if missing_algorithms:
                print(f"Algoritmi mancanti nel mapping colori: {missing_algorithms}")
                # Aggiungi gli algoritmi mancanti
                for alg in missing_algorithms:
                    mapping[alg] = get_algorithm_color(alg)
                save_color_mapping(mapping_file, mapping, overwrite=True)
            
            return mapping
        except Exception as e:
            print(f"Errore nel caricamento del mapping colori: {e}")
            print("Creazione di un nuovo mapping colori...")
    
    # Crea un nuovo mapping
    print("Creazione mapping colori algoritmi...")
    print(f"Algoritmi per cui assegnare colori: {algorithm_names}")
    
    for alg_name in algorithm_names:
        mapping[alg_name] = get_algorithm_color(alg_name)
    
    # Salva il mapping
    save_color_mapping(mapping_file, mapping, overwrite=False)
    
    return mapping

def get_algorithm_color(algorithm_name):
    """
    Chiede all'utente che colore assegnare a un algoritmo e conferma la scelta.
    
    Parameters:
    - algorithm_name: Nome dell'algoritmo
    
    Returns:
    - Colore confermato dall'utente
    """
    while True:
        print_color_menu()
        color = input(f"Che colore vuoi assegnare a '{algorithm_name}'? ").strip()
        if not color:
            print("Devi inserire un colore. Riprova...")
            continue
        
        # Chiedi conferma
        confirm = input(f"Confermi di assegnare il colore '{color}' a '{algorithm_name}'? [Y/n]: ").strip().lower()
        if confirm == '' or confirm == 'y':
            return color
        elif confirm == 'n':
            print("Riprova...")
            continue
        else:
            print("Risposta non valida. Riprova...")

def save_color_mapping(mapping_file, mapping, overwrite=False):
    """
    Salva il mapping dei colori in un file.
    
    Parameters:
    - mapping_file: Percorso del file di mapping
    - mapping: Dizionario con il mapping
    - overwrite: Se True, forza la sovrascrittura senza chiedere
    """
    should_save = True
    
    if os.path.exists(mapping_file) and not overwrite:
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            response = input(f"Il file {mapping_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
            if response == 'y':
                should_save = True
                break
            elif response == 'n':
                should_save = False
                break
            else:
                attempts += 1
                print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
        
        if attempts == max_attempts:
            raise ValueError("Troppi tentativi falliti. Operazione annullata.")
    
    if should_save:
        with open(mapping_file, 'w') as f:
            for algorithm, color in mapping.items():
                f.write(f"{algorithm}: {color}\n")
        print(f"Mapping colori salvato in {mapping_file}")
    else:
        print("Mapping colori non salvato.")

def plot_data_with_ci(aggregated_data, xaxis='buffer_size', yaxis='y_aggregated', 
                     ci_type='std_err', log_x=False, log_y=False, 
                     x_min=None, x_max=None, y_min=None, y_max=None,
                     x_scale_factor=None, y_scale_factor=None,
                     x_label=None, y_label=None, plot_title=None, 
                     color_mapping=None, **kwargs):
    """
    Plot aggregated data with confidence intervals.
    
    Parameters:
    - ci_type: Type of confidence interval to show ('min_max', 'std', 'std_err', 'samples', 
              'bootstrap_95', 'bootstrap_90', or None)
    """
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 6))
    
    # Use custom colors if provided, otherwise use default seaborn palette
    if color_mapping:
        colors = [color_mapping.get(alg, sns.color_palette("tab10")[i]) 
                 for i, alg in enumerate(aggregated_data.keys())]
    else:
        colors = sns.color_palette("tab10", len(aggregated_data))
    
    for i, (algorithm, data) in enumerate(aggregated_data.items()):
        color = colors[i]
        
        # Plot the main line
        plt.plot(data[xaxis], data[yaxis], label=algorithm, color=color, linewidth=2)
        
        # Add confidence interval
        if ci_type == 'min_max':
            plt.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                             color=color, alpha=0.2)
        elif ci_type == 'std':
            plt.fill_between(data[xaxis], data[yaxis] - data['y_std'], 
                             data[yaxis] + data['y_std'], color=color, alpha=0.2)
        elif ci_type == 'std_err':
            plt.fill_between(data[xaxis], data[yaxis] - data['y_stderr'], 
                             data[yaxis] + data['y_stderr'], color=color, alpha=0.2)
        elif ci_type == 'bootstrap_95' and 'y_bootstrap_ci_2_5' in data.columns:
            plt.fill_between(data[xaxis], data['y_bootstrap_ci_2_5'], 
                             data['y_bootstrap_ci_97_5'], color=color, alpha=0.2)
        elif ci_type == 'bootstrap_90' and 'y_bootstrap_ci_5' in data.columns:
            plt.fill_between(data[xaxis], data['y_bootstrap_ci_5'], 
                             data['y_bootstrap_ci_95'], color=color, alpha=0.2)
        elif ci_type == 'samples' and 'y_count' in data.columns:
            # Adjust alpha based on sample count
            plt.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                             color=color, alpha=0.1)
    
    ax = plt.gca()
    
    # Set x-axis properties
    if log_x:
        ax.set_xscale('log')
        x_axis_label = x_label or 'Timestep (logscale)'
        if x_label and x_scale_factor is not None:
            x_scale_notation = f"1e{int(np.log10(x_scale_factor))}"
            x_axis_label = f"{x_label} (logscale, {x_scale_notation})"
    else:
        # Format x-axis ticks if scale factor is provided
        if x_scale_factor is not None:
            x_formatter = create_value_formatter(x_scale_factor)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_formatter))
            x_scale_notation = f"1e{int(np.log10(x_scale_factor))}"
            
            # If custom label provided, append scale notation
            if x_label:
                x_axis_label = f"{x_label} ({x_scale_notation})"
            else:
                x_axis_label = f"{xaxis.capitalize()} ({x_scale_notation})"
        else:
            x_axis_label = x_label or xaxis.capitalize()
    
    plt.xlabel(x_axis_label)
    
    # Set y-axis properties
    if log_y:
        ax.set_yscale('log')
        y_axis_label = y_label or 'Reward (logscale)'
        if y_label and y_scale_factor is not None:
            y_scale_notation = f"1e{int(np.log10(y_scale_factor))}"
            y_axis_label = f"{y_label} (logscale, {y_scale_notation})"
    else:
        # Format y-axis ticks if scale factor is provided
        if y_scale_factor is not None:
            y_formatter = create_value_formatter(y_scale_factor)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
            y_scale_notation = f"1e{int(np.log10(y_scale_factor))}"
            
            # If custom label provided, append scale notation
            if y_label:
                y_axis_label = f"{y_label} ({y_scale_notation})"
            else:
                y_axis_label = f"Reward ({y_scale_notation})"
        else:
            y_axis_label = y_label or 'Reward'
    
    plt.ylabel(y_axis_label)
    
    # Set custom title or default
    if plot_title is not None:
        plt.title(plot_title)
    
    # Set axis limits if provided
    if x_min is not None:
        ax.set_xlim(left=x_min)
    if x_max is not None:
        ax.set_xlim(right=x_max)
    if y_min is not None:
        ax.set_ylim(bottom=y_min)
    if y_max is not None:
        ax.set_ylim(top=y_max)
    
    plt.legend()
    plt.tight_layout(pad=0.5)

def apply_name_mapping(grouped_dataframes, name_mapping):
    """
    Applica il mapping dei nomi ai dataframe raggruppati.
    Aggrega i dataframe per i nomi rinominati se ci sono duplicati.
    """
    renamed_dataframes = {}
    
    for original_name, dataframes in grouped_dataframes.items():
        new_name = name_mapping.get(original_name, original_name)
        
        # Se il nome rinominato esiste già, aggiungi i dataframe a quelli esistenti
        if new_name in renamed_dataframes:
            renamed_dataframes[new_name].extend(dataframes)
        else:
            renamed_dataframes[new_name] = dataframes.copy()
    
    # Stampa informazioni sull'aggregazione
    print("\n=== AGGREGAZIONE ALGORITMI ===")
    for renamed, dataframes in renamed_dataframes.items():
        original_names = [orig for orig, new in name_mapping.items() if new == renamed]
        if not original_names:  # Se non c'è mapping, usa il nome stesso
            original_names = [renamed]
        print(f"'{renamed}': {len(dataframes)} run da {len(original_names)} algoritmi originali")
        if len(original_names) > 1:
            print(f"  Algoritmi originali: {original_names}")
    print("===============================\n")
    
    return renamed_dataframes

def load_or_create_name_mapping(csv_path, algorithm_names, force_rename=False):
    """
    Carica o crea un mapping dei nomi degli algoritmi.
    """
    mapping_file = os.path.join(csv_path, "algorithm_name_mapping.txt")
    mapping = {}
    
    if force_rename:
        print("Force rename attivato: creazione di un nuovo mapping...")
        if os.path.exists(mapping_file):
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                response = input(f"Il file {mapping_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
                if response == 'y':
                    break
                elif response == 'n':
                    print("Operazione annullata.")
                    return {name: name for name in algorithm_names}
                else:
                    attempts += 1
                    print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
            
            if attempts == max_attempts:
                raise ValueError("Troppi tentativi falliti. Operazione annullata.")
        
        print(f"Algoritmi trovati: {algorithm_names}")
        for alg_name in algorithm_names:
            mapping[alg_name] = get_renamed_algorithm(alg_name)
        
        save_name_mapping(mapping_file, mapping, overwrite=True)
        return mapping
    
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ':' in line:
                        original, renamed = line.split(':', 1)
                        original = original.strip()
                        renamed = renamed.strip()
                        
                        # Solo aggiungi il mapping se l'algoritmo originale esiste
                        if original in algorithm_names:
                            mapping[original] = renamed
            
            print(f"Caricato mapping esistente da {mapping_file}")
            
            missing_algorithms = [alg for alg in algorithm_names if alg not in mapping]
            if missing_algorithms:
                print(f"Algoritmi mancanti nel mapping: {missing_algorithms}")
                for alg in missing_algorithms:
                    mapping[alg] = get_renamed_algorithm(alg)
                save_name_mapping(mapping_file, mapping, overwrite=True)
            
            return mapping
        except Exception as e:
            print(f"Errore nel caricamento del mapping: {e}")
            print("Creazione di un nuovo mapping...")
    
    print("Creazione mapping nomi algoritmi...")
    print(f"Algoritmi trovati: {algorithm_names}")
    
    for alg_name in algorithm_names:
        mapping[alg_name] = get_renamed_algorithm(alg_name)
    
    save_name_mapping(mapping_file, mapping, overwrite=False)
    return mapping

def get_renamed_algorithm(original_name):
    """
    Chiede all'utente come rinominare un algoritmo e conferma la scelta.
    
    Parameters:
    - original_name: Nome originale dell'algoritmo
    
    Returns:
    - Nome rinominato confermato dall'utente
    """
    while True:
        new_name = input(f"Come vuoi rinominare '{original_name}'? (premi invio per mantenere il nome originale): ").strip()
        if not new_name:
            new_name = original_name
        
        # Chiedi conferma
        confirm = input(f"Confermi di rinominare '{original_name}' in '{new_name}'? [Y/n]: ").strip().lower()
        if confirm == '' or confirm == 'y':
            return new_name
        elif confirm == 'n':
            print("Riprova...")
            continue
        else:
            print("Risposta non valida. Riprova...")

def save_name_mapping(mapping_file, mapping, overwrite=False):
    """
    Salva il mapping dei nomi in un file.
    
    Parameters:
    - mapping_file: Percorso del file di mapping
    - mapping: Dizionario con il mapping
    - overwrite: Se True, forza la sovrascrittura senza chiedere
    """
    should_save = True
    
    if os.path.exists(mapping_file) and not overwrite:
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts:
            response = input(f"Il file {mapping_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
            if response == 'y':
                should_save = True
                break
            elif response == 'n':
                should_save = False
                break
            else:
                attempts += 1
                print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
        
        if attempts == max_attempts:
            raise ValueError("Troppi tentativi falliti. Operazione annullata.")
    
    if should_save:
        with open(mapping_file, 'w') as f:
            for original, renamed in mapping.items():
                f.write(f"{original}: {renamed}\n")
        print(f"Mapping salvato in {mapping_file}")
    else:
        print("Mapping non salvato.")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--xaxis', '-x', default='buffer_size', help='Column name for x-axis data')
    parser.add_argument('--value', '-y', default='eval/episode_reward', help='Column name for y-axis data')
    parser.add_argument('--est', default='median', choices=['mean', 'median', 'min', 'max', 'iqm', 'interquartile_mean'], 
                        help='Aggregation function')
    parser.add_argument('--ci', default='std_err', choices=['min_max', 'std', 'std_err', 'samples', 'bootstrap_95', 'bootstrap_90', 'none'],
                        help='Confidence interval type')
    
    # Bootstrap options
    parser.add_argument('--bootstrap', action='store_true', default=False,
                        help='Use stratified bootstrap for confidence intervals')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                        help='Number of bootstrap samples (default: 1000)')
    
    # Axis scaling options
    parser.add_argument('--log_x', action='store_true', default=False, help='Use log scale for x-axis')
    parser.add_argument('--log_y', action='store_true', default=False, help='Use log scale for y-axis')
    
    # Axis limits
    parser.add_argument('--x_min', type=float, default=0, help='Minimum value for x-axis')
    parser.add_argument('--x_max', type=float, default=100_000, help='Maximum value for x-axis')
    parser.add_argument('--y_min', type=float, default=0, help='Minimum value for y-axis')
    parser.add_argument('--y_max', type=float, default=None, help='Maximum value for y-axis')
    
    # Axis formatting
    parser.add_argument('--x_scale', type=float, default=1e4, help='Scale factor for x-axis values (e.g., 1e5)')
    parser.add_argument('--y_scale', type=float, default=None, help='Scale factor for y-axis values (e.g., 1e3)')
    parser.add_argument('--x_label', type=str, default='Buffer Size', help='Custom x-axis label')
    parser.add_argument('--y_label', type=str, default=None, help='Custom y-axis label')
    
    # Plot title
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    
    parser.add_argument('--csv_path', type=str, default="data_plot/", help='csv folder')
    parser.add_argument('--output', type=str, default="plot.png", help='Output filename')
    
    # Rename options
    parser.add_argument('--rename', action='store_true', default=False, 
                        help='Enable algorithm name renaming with interactive input')
    parser.add_argument('--force-rename', action='store_true', default=False,
                        help='Force renaming even if mapping file exists')
    
    # Color mapping options
    parser.add_argument('--color', action='store_true', default=False,
                        help='Enable algorithm color mapping with interactive input')
    parser.add_argument('--force-color', action='store_true', default=False,
                        help='Force color mapping even if mapping file exists')
    
    args = parser.parse_args()

    # Always use task-aware reading to handle both single and multi-task structures
    grouped_dataframes_with_tasks, task_structure = read_csvs_with_task_awareness(args.csv_path)
    print(f"Task structure: {task_structure}")
    
    # Extract dataframes for compatibility with existing functions
    grouped_dataframes = extract_dataframes_only(grouped_dataframes_with_tasks)
    
    # Apply renaming if requested
    if args.rename:
        algorithm_names = list(grouped_dataframes.keys())
        name_mapping = load_or_create_name_mapping(
            args.csv_path, 
            algorithm_names, 
            force_rename=getattr(args, 'force_rename', False)
        )
        
        if args.bootstrap:
            # Apply renaming to task-aware structure
            renamed_dataframes_with_tasks = {}
            for original_name, df_task_pairs in grouped_dataframes_with_tasks.items():
                new_name = name_mapping.get(original_name, original_name)
                if new_name in renamed_dataframes_with_tasks:
                    renamed_dataframes_with_tasks[new_name].extend(df_task_pairs)
                else:
                    renamed_dataframes_with_tasks[new_name] = df_task_pairs.copy()
            grouped_dataframes_with_tasks = renamed_dataframes_with_tasks
        
        grouped_dataframes = apply_name_mapping(grouped_dataframes, name_mapping)
    
    # Apply color mapping if requested
    color_mapping = None
    if args.color:
        algorithm_names = list(grouped_dataframes.keys())  # Usa i nomi dopo la rinominazione
        color_mapping = load_or_create_color_mapping(
            args.csv_path,
            algorithm_names,
            force_color=getattr(args, 'force_color', False)
        )
    
    # Aggregate runs for each algorithm
    aggregated_data = {}
    for algorithm in grouped_dataframes.keys():
        if args.bootstrap:
            # Use the task-aware bootstrap function
            algorithm_df_task_pairs = grouped_dataframes_with_tasks[algorithm]
            single_algorithm_dict = {algorithm: algorithm_df_task_pairs}
            
            aggregated_data[algorithm] = stratified_bootstrap_aggregate_with_tasks(
                single_algorithm_dict,
                x_column=args.xaxis, 
                y_column=args.value, 
                aggregator=args.est,
                n_bootstrap=args.n_bootstrap
            )
        else:
            aggregated_data[algorithm] = aggregate_runs(
                grouped_dataframes[algorithm], 
                x_column=args.xaxis, 
                y_column=args.value, 
                aggregator=args.est
            )
    
    # Plot the aggregated data with confidence intervals
    print("Creating plot...")
    plot_data_with_ci(
        aggregated_data, 
        xaxis=args.xaxis, 
        yaxis='y_aggregated',
        ci_type=args.ci if args.ci.lower() != 'none' else None,
        log_x=args.log_x,
        log_y=args.log_y,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        x_scale_factor=args.x_scale,
        y_scale_factor=args.y_scale,
        x_label=args.x_label,
        y_label=args.y_label,
        plot_title=args.title,
        color_mapping=color_mapping
    )
    
    # Create output path in the same folder as csv_path
    output_path = os.path.join(args.csv_path, args.output)
    
    # Save the plot
    plt.savefig(output_path, format='png', dpi=300)
    print(f"Plot saved as {output_path}")

if __name__ == "__main__":
    main()