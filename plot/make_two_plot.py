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
    def format_func(value, tick_pos):
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
    def format_func(value, tick_pos):
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

def detect_subplot_structure(csv_path):
    """
    Rileva se csv_path contiene sottocartelle per subplot multipli.
    
    Returns:
    - tuple: (is_subplot_structure, subplot_data)
      - is_subplot_structure: True se ci sono sottocartelle con CSV
      - subplot_data: dict con subplot_name -> subplot_path
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"Path {csv_path} non esiste")
    
    # Cerca sottocartelle
    subdirs = [d for d in os.listdir(csv_path) 
               if os.path.isdir(os.path.join(csv_path, d))]
    
    if not subdirs:
        # Nessuna sottocartella, comportamento normale
        return False, {}
    
    # Verifica se le sottocartelle contengono CSV o altre sottocartelle (task structure)
    subplot_data = {}
    for subdir in subdirs:
        subdir_path = os.path.join(csv_path, subdir)
        
        try:
            items = os.listdir(subdir_path)
        except PermissionError:
            continue
            
        # Cerca CSV direttamente nella sottocartella o nelle sue sottocartelle (task structure)
        csv_files = [f for f in items if f.endswith('.csv')]
        has_task_subdirs = any(os.path.isdir(os.path.join(subdir_path, d)) for d in items)
        
        if csv_files or has_task_subdirs:
            subplot_data[subdir] = subdir_path
    
    if not subplot_data:
        # Nessuna sottocartella con CSV, comportamento normale
        return False, {}
    
    return True, subplot_data

def read_subplot_title(subplot_path):
    """
    Legge il titolo del subplot dal file title.txt nella cartella.
    Se non esiste, usa il nome della cartella.
    
    Parameters:
    - subplot_path: Percorso della sottocartella del subplot
    
    Returns:
    - Titolo del subplot
    """
    title_file = os.path.join(subplot_path, 'title.txt')
    
    if os.path.exists(title_file):
        try:
            with open(title_file, 'r') as f:
                title = f.read().strip()
                if title:
                    return title
        except Exception as e:
            print(f"Errore nella lettura del file titolo {title_file}: {e}")
    
    # Fallback al nome della cartella
    return os.path.basename(subplot_path)

def create_subplot_grid(n_subplots):
    """
    Calcola il layout ottimale della griglia di subplot.
    
    Parameters:
    - n_subplots: Numero di subplot da creare
    
    Returns:
    - tuple: (n_rows, n_cols)
    """
    if n_subplots <= 0:
        return 0, 0
    elif n_subplots == 1:
        return 1, 1
    elif n_subplots == 2:
        return 1, 2
    elif n_subplots <= 4:
        return 2, 2
    elif n_subplots <= 6:
        return 2, 3
    elif n_subplots <= 9:
        return 3, 3
    elif n_subplots <= 12:
        return 3, 4
    else:
        # Per più di 12 subplot, usa una griglia 4xN
        n_cols = 4
        n_rows = (n_subplots + n_cols - 1) // n_cols
        return n_rows, n_cols

def plot_multiple_subplots(subplot_aggregated_data, subplot_titles, 
                          xaxis='buffer_size', yaxis='y_aggregated',
                          ci_type='std_err', log_x=False, log_y=False,
                          x_min=None, x_max=None, y_min=None, y_max=None,
                          x_scale_factor=None, y_scale_factor=None,
                          x_label=None, y_label=None,
                          color_mapping=None, overall_title=None,
                          legend_order=None):
    """
    Crea subplot multipli con legenda condivisa.
    
    Parameters:
    - subplot_aggregated_data: dict con subplot_name -> aggregated_data
    - subplot_titles: dict con subplot_name -> titolo
    - legend_order: Lista ordinata dei nomi degli algoritmi per la legenda
    """
    sns.set_style("darkgrid")
    
    n_subplots = len(subplot_aggregated_data)
    n_rows, n_cols = create_subplot_grid(n_subplots)
    
    # Dimensioni della figura basate sul numero di subplot
    fig_width = 6 * n_cols
    fig_height = 5 * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Assicurati che axes sia sempre un array
    if n_subplots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Raccogli tutti gli algoritmi unici attraverso tutti i subplot
    all_algorithms = set()
    for aggregated_data in subplot_aggregated_data.values():
        all_algorithms.update(aggregated_data.keys())
    all_algorithms = sorted(list(all_algorithms))
    
    # Crea mapping colori consistente per tutti i subplot
    if color_mapping:
        colors = {alg: color_mapping.get(alg, sns.color_palette("tab10")[i % 10]) 
                 for i, alg in enumerate(all_algorithms)}
    else:
        palette = sns.color_palette("tab10", len(all_algorithms))
        colors = {alg: palette[i] for i, alg in enumerate(all_algorithms)}
    
    # Plotta ogni subplot
    subplot_names = sorted(subplot_aggregated_data.keys())
    for idx, subplot_name in enumerate(subplot_names):
        ax = axes[idx]
        aggregated_data = subplot_aggregated_data[subplot_name]
        
        for algorithm, data in aggregated_data.items():
            color = colors[algorithm]
            
            # Plotta la linea principale
            ax.plot(data[xaxis], data[yaxis], label=algorithm, color=color, linewidth=2)
            
            # Aggiungi intervallo di confidenza
            if ci_type == 'min_max':
                ax.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                               color=color, alpha=0.2)
            elif ci_type == 'std':
                ax.fill_between(data[xaxis], data[yaxis] - data['y_std'], 
                               data[yaxis] + data['y_std'], color=color, alpha=0.2)
            elif ci_type == 'std_err':
                ax.fill_between(data[xaxis], data[yaxis] - data['y_stderr'], 
                               data[yaxis] + data['y_stderr'], color=color, alpha=0.2)
            elif ci_type == 'bootstrap_95' and 'y_bootstrap_ci_2_5' in data.columns:
                ax.fill_between(data[xaxis], data['y_bootstrap_ci_2_5'], 
                               data['y_bootstrap_ci_97_5'], color=color, alpha=0.2)
            elif ci_type == 'bootstrap_90' and 'y_bootstrap_ci_5' in data.columns:
                ax.fill_between(data[xaxis], data['y_bootstrap_ci_5'], 
                               data['y_bootstrap_ci_95'], color=color, alpha=0.2)
            elif ci_type == 'samples' and 'y_count' in data.columns:
                ax.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                               color=color, alpha=0.1)
        
        # Configura assi
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        
        # Formattazione assi
        if x_scale_factor is not None and not log_x:
            x_formatter = create_value_formatter(x_scale_factor)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_formatter))
        
        if y_scale_factor is not None and not log_y:
            y_formatter = create_value_formatter(y_scale_factor)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
        
        # Limiti assi
        if x_min is not None:
            ax.set_xlim(left=x_min)
        if x_max is not None:
            ax.set_xlim(right=x_max)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if y_max is not None:
            ax.set_ylim(top=y_max)
        
        # Titolo del subplot
        ax.set_title(subplot_titles.get(subplot_name, subplot_name))
        
        # Etichette assi (solo sui bordi esterni)
        if idx >= (n_rows - 1) * n_cols:  # Ultima riga
            if x_label:
                label = x_label
                if x_scale_factor is not None and not log_x:
                    scale_notation = format_scale_notation(x_scale_factor)
                    label = f"{x_label} ({scale_notation})"
                ax.set_xlabel(label)
            else:
                ax.set_xlabel(xaxis.capitalize())
        
        if idx % n_cols == 0:  # Prima colonna
            if y_label:
                label = y_label
                if y_scale_factor is not None and not log_y:
                    scale_notation = format_scale_notation(y_scale_factor)
                    label = f"{y_label} ({scale_notation})"
                ax.set_ylabel(label)
            else:
                ax.set_ylabel('Reward')
    
    # Nascondi subplot extra
    for idx in range(len(subplot_names), len(axes)):
        axes[idx].set_visible(False)
    
    # Crea legenda unica con ordine specificato
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Applica l'ordine della legenda se fornito
    if legend_order:
        # Crea un dizionario handle per algoritmo
        label_to_handle = {label: handle for label, handle in zip(labels, handles)}
        
        # Riordina secondo legend_order
        ordered_handles = []
        ordered_labels = []
        for alg in legend_order:
            if alg in label_to_handle:
                ordered_handles.append(label_to_handle[alg])
                ordered_labels.append(alg)
        
        handles = ordered_handles
        labels = ordered_labels
    else:
        # Ordine alfabetico di default
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
        handles = [handles[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    
    # Posiziona legenda sotto i subplot
    legend_ncol = min(len(labels), 8)
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
              ncol=legend_ncol, frameon=True, fontsize=10)
    
    # Titolo generale se fornito
    if overall_title:
        fig.suptitle(overall_title, fontsize=16, y=0.98)
    
    # Adjust layout per fare spazio alla legenda sotto
    plt.tight_layout(rect=[0, 0.05, 1.0, 0.96 if overall_title else 0.98])

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

def detect_subplot_structure(csv_path):
    """
    Rileva se csv_path contiene sottocartelle per subplot multipli.
    
    Returns:
    - tuple: (is_subplot_structure, subplot_data)
      - is_subplot_structure: True se ci sono sottocartelle con CSV
      - subplot_data: dict con subplot_name -> subplot_path
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"Path {csv_path} non esiste")
    
    # Cerca sottocartelle
    subdirs = [d for d in os.listdir(csv_path) 
               if os.path.isdir(os.path.join(csv_path, d))]
    
    if not subdirs:
        # Nessuna sottocartella, comportamento normale
        return False, {}
    
    # Verifica se le sottocartelle contengono CSV o altre sottocartelle (task structure)
    subplot_data = {}
    for subdir in subdirs:
        subdir_path = os.path.join(csv_path, subdir)
        
        try:
            items = os.listdir(subdir_path)
        except PermissionError:
            continue
            
        # Cerca CSV direttamente nella sottocartella o nelle sue sottocartelle (task structure)
        csv_files = [f for f in items if f.endswith('.csv')]
        has_task_subdirs = any(os.path.isdir(os.path.join(subdir_path, d)) for d in items)
        
        if csv_files or has_task_subdirs:
            subplot_data[subdir] = subdir_path
    
    if not subplot_data:
        # Nessuna sottocartella con CSV, comportamento normale
        return False, {}
    
    return True, subplot_data

def read_subplot_title(subplot_path):
    """
    Legge il titolo del subplot dal file title.txt nella cartella.
    Se non esiste, usa il nome della cartella.
    
    Parameters:
    - subplot_path: Percorso della sottocartella del subplot
    
    Returns:
    - Titolo del subplot
    """
    title_file = os.path.join(subplot_path, 'title.txt')
    
    if os.path.exists(title_file):
        try:
            with open(title_file, 'r') as f:
                title = f.read().strip()
                if title:
                    return title
        except Exception as e:
            print(f"Errore nella lettura del file titolo {title_file}: {e}")
    
    # Fallback al nome della cartella
    return os.path.basename(subplot_path)

def create_subplot_grid(n_subplots):
    """
    Calcola il layout ottimale della griglia di subplot.
    
    Parameters:
    - n_subplots: Numero di subplot da creare
    
    Returns:
    - tuple: (n_rows, n_cols)
    """
    if n_subplots <= 0:
        return 0, 0
    elif n_subplots == 1:
        return 1, 1
    elif n_subplots == 2:
        return 1, 2
    elif n_subplots <= 4:
        return 2, 2
    elif n_subplots <= 6:
        return 2, 3
    elif n_subplots <= 9:
        return 3, 3
    elif n_subplots <= 12:
        return 3, 4
    else:
        # Per più di 12 subplot, usa una griglia 4xN
        n_cols = 4
        n_rows = (n_subplots + n_cols - 1) // n_cols
        return n_rows, n_cols

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
            plt.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                             color=color, alpha=0.1)
    
    ax = plt.gca()
    
    # Set x-axis properties
    if log_x:
        ax.set_xscale('log')
        x_axis_label = x_label or 'Timestep (logscale)'
        if x_label and x_scale_factor is not None:
            scale_notation = format_scale_notation(x_scale_factor)
            x_axis_label = f"{x_label} (logscale, {scale_notation})"
    else:
        # Format x-axis ticks if scale factor is provided
        if x_scale_factor is not None:
            x_formatter = create_value_formatter(x_scale_factor)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_formatter))
            scale_notation = format_scale_notation(x_scale_factor)
            
            # If custom label provided, append scale notation
            if x_label:
                x_axis_label = f"{x_label} ({scale_notation})"
            else:
                x_axis_label = f"{xaxis.capitalize()} ({scale_notation})"
        else:
            x_axis_label = x_label or xaxis.capitalize()
    
    plt.xlabel(x_axis_label)
    
    # Set y-axis properties
    if log_y:
        ax.set_yscale('log')
        y_axis_label = y_label or 'Reward (logscale)'
        if y_label and y_scale_factor is not None:
            scale_notation = format_scale_notation(y_scale_factor)
            y_axis_label = f"{y_label} (logscale, {scale_notation})"
    else:
        # Format y-axis ticks if scale factor is provided
        if y_scale_factor is not None:
            y_formatter = create_value_formatter(y_scale_factor)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
            scale_notation = format_scale_notation(y_scale_factor)
            
            # If custom label provided, append scale notation
            if y_label:
                y_axis_label = f"{y_label} ({scale_notation})"
            else:
                y_axis_label = f"Reward ({scale_notation})"
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

def load_or_create_legend_order(csv_path, algorithm_names, force_legend=False):
    """
    Carica o crea l'ordine della legenda per gli algoritmi.
    L'utente può specificare l'ordine digitando i numeri (es: 2,1,3).
    
    Parameters:
    - csv_path: Percorso della cartella contenente i CSV
    - algorithm_names: Lista dei nomi degli algoritmi ordinati
    - force_legend: Se True, ignora il file esistente e chiede nuovo ordine
    
    Returns:
    - Lista ordinata dei nomi degli algoritmi
    """
    legend_file = os.path.join(csv_path, "algorithm_legend_order.txt")
    
    if force_legend:
        print("Force legend order attivato: creazione di un nuovo ordine...")
        if os.path.exists(legend_file):
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                response = input(f"Il file {legend_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
                if response == 'y':
                    break
                elif response == 'n':
                    print("Operazione annullata.")
                    return sorted(algorithm_names)
                else:
                    attempts += 1
                    print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
            
            if attempts == max_attempts:
                raise ValueError("Troppi tentativi falliti. Operazione annullata.")
        
        return get_legend_order(algorithm_names, legend_file)
    
    if os.path.exists(legend_file):
        try:
            with open(legend_file, 'r') as f:
                content = f.read().strip()
                if content:
                    ordered_names = content.split('\n')
                    # Verifica che tutti gli algoritmi siano presenti
                    if set(ordered_names) == set(algorithm_names):
                        print(f"Caricato ordine legenda esistente da {legend_file}")
                        return ordered_names
                    else:
                        print("Ordine legenda non valido. Creazione di un nuovo ordine...")
        except Exception as e:
            print(f"Errore nel caricamento dell'ordine legenda: {e}")
            print("Creazione di un nuovo ordine...")
    
    return get_legend_order(algorithm_names, legend_file)

def get_legend_order(algorithm_names, legend_file):
    """
    Chiede all'utente di specificare l'ordine della legenda.
    L'utente digita i numeri separati da virgole (es: 2,1,3).
    
    Parameters:
    - algorithm_names: Lista dei nomi degli algoritmi
    - legend_file: Percorso del file dove salvare l'ordine
    
    Returns:
    - Lista ordinata dei nomi degli algoritmi
    """
    print("\n=== ORDINAMENTO LEGENDA ===")
    print("Algoritmi disponibili:")
    for i, alg in enumerate(algorithm_names, 1):
        print(f"  {i}. {alg}")
    
    print("\nSpecifica l'ordine digitando i numeri separati da virgole.")
    print("Esempio: '2,1,3' per mettere il secondo algoritmo per primo.")
    
    while True:
        order_input = input("Ordine desiderato (o premi invio per ordine alfabetico): ").strip()
        
        if not order_input:
            # Ordine alfabetico di default
            ordered_names = sorted(algorithm_names)
            save_legend_order(legend_file, ordered_names)
            return ordered_names
        
        try:
            # Parse i numeri
            indices = [int(x.strip()) for x in order_input.split(',')]
            
            # Verifica validità
            if len(indices) != len(algorithm_names):
                print(f"Errore: devi specificare {len(algorithm_names)} numeri, hai fornito {len(indices)}")
                continue
            
            if not all(1 <= idx <= len(algorithm_names) for idx in indices):
                print(f"Errore: i numeri devono essere tra 1 e {len(algorithm_names)}")
                continue
            
            if len(set(indices)) != len(indices):
                print("Errore: non puoi ripetere gli stessi numeri")
                continue
            
            # Crea la lista ordinata
            ordered_names = [algorithm_names[idx - 1] for idx in indices]
            
            # Mostra anteprima
            print("\nAnteprima ordine legenda:")
            for i, alg in enumerate(ordered_names, 1):
                print(f"  {i}. {alg}")
            
            # Chiedi conferma
            confirm = input("\nConfermi questo ordine? [Y/n]: ").strip().lower()
            if confirm == '' or confirm == 'y':
                save_legend_order(legend_file, ordered_names)
                return ordered_names
            else:
                print("Riprova...")
                continue
        
        except ValueError:
            print("Errore: inserisci numeri separati da virgole (es: 1,2,3)")
            continue

def save_legend_order(legend_file, ordered_names):
    """
    Salva l'ordine della legenda in un file.
    
    Parameters:
    - legend_file: Percorso del file
    - ordered_names: Lista ordinata dei nomi degli algoritmi
    """
    with open(legend_file, 'w') as f:
        for name in ordered_names:
            f.write(f"{name}\n")
    print(f"Ordine legenda salvato in {legend_file}")

def format_scale_notation(scale_factor):
    """
    Converte un fattore di scala in notazione LaTeX.
    
    Parameters:
    - scale_factor: Fattore di scala (es: 1e4, 1e3)
    
    Returns:
    - Stringa LaTeX formattata (es: "$\\times 10^4$")
    """
    if scale_factor is None:
        return None
    
    # Calcola l'esponente
    exponent = int(np.log10(scale_factor))
    
    # Ritorna in formato LaTeX
    return f"$\\times 10^{{{exponent}}}$"

def create_value_formatter(scale_factor=None):
    """
    Create a formatter function for axis tick values.
    
    Parameters:
    - scale_factor: Value by which to divide the tick values (e.g., 1e5, 1e3)
    
    Returns:
    - A formatter function for matplotlib ticker
    """
    def format_func(value, tick_pos):
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
    def format_func(value, tick_pos):
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

def detect_subplot_structure(csv_path):
    """
    Rileva se csv_path contiene sottocartelle per subplot multipli.
    
    Returns:
    - tuple: (is_subplot_structure, subplot_data)
      - is_subplot_structure: True se ci sono sottocartelle con CSV
      - subplot_data: dict con subplot_name -> subplot_path
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"Path {csv_path} non esiste")
    
    # Cerca sottocartelle
    subdirs = [d for d in os.listdir(csv_path) 
               if os.path.isdir(os.path.join(csv_path, d))]
    
    if not subdirs:
        # Nessuna sottocartella, comportamento normale
        return False, {}
    
    # Verifica se le sottocartelle contengono CSV o altre sottocartelle (task structure)
    subplot_data = {}
    for subdir in subdirs:
        subdir_path = os.path.join(csv_path, subdir)
        
        try:
            items = os.listdir(subdir_path)
        except PermissionError:
            continue
            
        # Cerca CSV direttamente nella sottocartella o nelle sue sottocartelle (task structure)
        csv_files = [f for f in items if f.endswith('.csv')]
        has_task_subdirs = any(os.path.isdir(os.path.join(subdir_path, d)) for d in items)
        
        if csv_files or has_task_subdirs:
            subplot_data[subdir] = subdir_path
    
    if not subplot_data:
        # Nessuna sottocartella con CSV, comportamento normale
        return False, {}
    
    return True, subplot_data

def read_subplot_title(subplot_path):
    """
    Legge il titolo del subplot dal file title.txt nella cartella.
    Se non esiste, usa il nome della cartella.
    
    Parameters:
    - subplot_path: Percorso della sottocartella del subplot
    
    Returns:
    - Titolo del subplot
    """
    title_file = os.path.join(subplot_path, 'title.txt')
    
    if os.path.exists(title_file):
        try:
            with open(title_file, 'r') as f:
                title = f.read().strip()
                if title:
                    return title
        except Exception as e:
            print(f"Errore nella lettura del file titolo {title_file}: {e}")
    
    # Fallback al nome della cartella
    return os.path.basename(subplot_path)

def create_subplot_grid(n_subplots):
    """
    Calcola il layout ottimale della griglia di subplot.
    
    Parameters:
    - n_subplots: Numero di subplot da creare
    
    Returns:
    - tuple: (n_rows, n_cols)
    """
    if n_subplots <= 0:
        return 0, 0
    elif n_subplots == 1:
        return 1, 1
    elif n_subplots == 2:
        return 1, 2
    elif n_subplots <= 4:
        return 2, 2
    elif n_subplots <= 6:
        return 2, 3
    elif n_subplots <= 9:
        return 3, 3
    elif n_subplots <= 12:
        return 3, 4
    else:
        # Per più di 12 subplot, usa una griglia 4xN
        n_cols = 4
        n_rows = (n_subplots + n_cols - 1) // n_cols
        return n_rows, n_cols

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
            plt.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                             color=color, alpha=0.1)
    
    ax = plt.gca()
    
    # Set x-axis properties
    if log_x:
        ax.set_xscale('log')
        x_axis_label = x_label or 'Timestep (logscale)'
        if x_label and x_scale_factor is not None:
            scale_notation = format_scale_notation(x_scale_factor)
            x_axis_label = f"{x_label} (logscale, {scale_notation})"
    else:
        # Format x-axis ticks if scale factor is provided
        if x_scale_factor is not None:
            x_formatter = create_value_formatter(x_scale_factor)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_formatter))
            scale_notation = format_scale_notation(x_scale_factor)
            
            # If custom label provided, append scale notation
            if x_label:
                x_axis_label = f"{x_label} ({scale_notation})"
            else:
                x_axis_label = f"{xaxis.capitalize()} ({scale_notation})"
        else:
            x_axis_label = x_label or xaxis.capitalize()
    
    plt.xlabel(x_axis_label)
    
    # Set y-axis properties
    if log_y:
        ax.set_yscale('log')
        y_axis_label = y_label or 'Reward (logscale)'
        if y_label and y_scale_factor is not None:
            scale_notation = format_scale_notation(y_scale_factor)
            y_axis_label = f"{y_label} (logscale, {scale_notation})"
    else:
        # Format y-axis ticks if scale factor is provided
        if y_scale_factor is not None:
            y_formatter = create_value_formatter(y_scale_factor)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
            scale_notation = format_scale_notation(y_scale_factor)
            
            # If custom label provided, append scale notation
            if y_label:
                y_axis_label = f"{y_label} ({scale_notation})"
            else:
                y_axis_label = f"Reward ({scale_notation})"
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

def load_or_create_legend_order(csv_path, algorithm_names, force_legend=False):
    """
    Carica o crea l'ordine della legenda per gli algoritmi.
    L'utente può specificare l'ordine digitando i numeri (es: 2,1,3).
    
    Parameters:
    - csv_path: Percorso della cartella contenente i CSV
    - algorithm_names: Lista dei nomi degli algoritmi ordinati
    - force_legend: Se True, ignora il file esistente e chiede nuovo ordine
    
    Returns:
    - Lista ordinata dei nomi degli algoritmi
    """
    legend_file = os.path.join(csv_path, "algorithm_legend_order.txt")
    
    if force_legend:
        print("Force legend order attivato: creazione di un nuovo ordine...")
        if os.path.exists(legend_file):
            attempts = 0
            max_attempts = 3
            
            while attempts < max_attempts:
                response = input(f"Il file {legend_file} esiste già. Vuoi sovrascriverlo? [y/n]: ").strip().lower()
                if response == 'y':
                    break
                elif response == 'n':
                    print("Operazione annullata.")
                    return sorted(algorithm_names)
                else:
                    attempts += 1
                    print(f"Risposta non valida. Tentativi rimanenti: {max_attempts - attempts}")
            
            if attempts == max_attempts:
                raise ValueError("Troppi tentativi falliti. Operazione annullata.")
        
        return get_legend_order(algorithm_names, legend_file)
    
    if os.path.exists(legend_file):
        try:
            with open(legend_file, 'r') as f:
                content = f.read().strip()
                if content:
                    ordered_names = content.split('\n')
                    # Verifica che tutti gli algoritmi siano presenti
                    if set(ordered_names) == set(algorithm_names):
                        print(f"Caricato ordine legenda esistente da {legend_file}")
                        return ordered_names
                    else:
                        print("Ordine legenda non valido. Creazione di un nuovo ordine...")
        except Exception as e:
            print(f"Errore nel caricamento dell'ordine legenda: {e}")
            print("Creazione di un nuovo ordine...")
    
    return get_legend_order(algorithm_names, legend_file)

def get_legend_order(algorithm_names, legend_file):
    """
    Chiede all'utente di specificare l'ordine della legenda.
    L'utente digita i numeri separati da virgole (es: 2,1,3).
    
    Parameters:
    - algorithm_names: Lista dei nomi degli algoritmi
    - legend_file: Percorso del file dove salvare l'ordine
    
    Returns:
    - Lista ordinata dei nomi degli algoritmi
    """
    print("\n=== ORDINAMENTO LEGENDA ===")
    print("Algoritmi disponibili:")
    for i, alg in enumerate(algorithm_names, 1):
        print(f"  {i}. {alg}")
    
    print("\nSpecifica l'ordine digitando i numeri separati da virgole.")
    print("Esempio: '2,1,3' per mettere il secondo algoritmo per primo.")
    
    while True:
        order_input = input("Ordine desiderato (o premi invio per ordine alfabetico): ").strip()
        
        if not order_input:
            # Ordine alfabetico di default
            ordered_names = sorted(algorithm_names)
            save_legend_order(legend_file, ordered_names)
            return ordered_names
        
        try:
            # Parse i numeri
            indices = [int(x.strip()) for x in order_input.split(',')]
            
            # Verifica validità
            if len(indices) != len(algorithm_names):
                print(f"Errore: devi specificare {len(algorithm_names)} numeri, hai fornito {len(indices)}")
                continue
            
            if not all(1 <= idx <= len(algorithm_names) for idx in indices):
                print(f"Errore: i numeri devono essere tra 1 e {len(algorithm_names)}")
                continue
            
            if len(set(indices)) != len(indices):
                print("Errore: non puoi ripetere gli stessi numeri")
                continue
            
            # Crea la lista ordinata
            ordered_names = [algorithm_names[idx - 1] for idx in indices]
            
            # Mostra anteprima
            print("\nAnteprima ordine legenda:")
            for i, alg in enumerate(ordered_names, 1):
                print(f"  {i}. {alg}")
            
            # Chiedi conferma
            confirm = input("\nConfermi questo ordine? [Y/n]: ").strip().lower()
            if confirm == '' or confirm == 'y':
                save_legend_order(legend_file, ordered_names)
                return ordered_names
            else:
                print("Riprova...")
                continue
        
        except ValueError:
            print("Errore: inserisci numeri separati da virgole (es: 1,2,3)")
            continue

def save_legend_order(legend_file, ordered_names):
    """
    Salva l'ordine della legenda in un file.
    
    Parameters:
    - legend_file: Percorso del file
    - ordered_names: Lista ordinata dei nomi degli algoritmi
    """
    with open(legend_file, 'w') as f:
        for name in ordered_names:
            f.write(f"{name}\n")
    print(f"Ordine legenda salvato in {legend_file}")

def format_scale_notation(scale_factor):
    """
    Converte un fattore di scala in notazione LaTeX.
    
    Parameters:
    - scale_factor: Fattore di scala (es: 1e4, 1e3)
    
    Returns:
    - Stringa LaTeX formattata (es: "$\\times 10^4$")
    """
    if scale_factor is None:
        return None
    
    # Calcola l'esponente
    exponent = int(np.log10(scale_factor))
    
    # Ritorna in formato LaTeX
    return f"$\\times 10^{{{exponent}}}$"

def create_value_formatter(scale_factor=None):
    """
    Create a formatter function for axis tick values.
    
    Parameters:
    - scale_factor: Value by which to divide the tick values (e.g., 1e5, 1e3)
    
    Returns:
    - A formatter function for matplotlib ticker
    """
    def format_func(value, tick_pos):
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

def detect_subplot_structure(csv_path):
    """
    Rileva se csv_path contiene sottocartelle per subplot multipli.
    
    Returns:
    - tuple: (is_subplot_structure, subplot_data)
      - is_subplot_structure: True se ci sono sottocartelle con CSV
      - subplot_data: dict con subplot_name -> subplot_path
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"Path {csv_path} non esiste")
    
    # Cerca sottocartelle
    subdirs = [d for d in os.listdir(csv_path) 
               if os.path.isdir(os.path.join(csv_path, d))]
    
    if not subdirs:
        # Nessuna sottocartella, comportamento normale
        return False, {}
    
    # Verifica se le sottocartelle contengono CSV o altre sottocartelle (task structure)
    subplot_data = {}
    for subdir in subdirs:
        subdir_path = os.path.join(csv_path, subdir)
        
        try:
            items = os.listdir(subdir_path)
        except PermissionError:
            continue
            
        # Cerca CSV direttamente nella sottocartella o nelle sue sottocartelle (task structure)
        csv_files = [f for f in items if f.endswith('.csv')]
        has_task_subdirs = any(os.path.isdir(os.path.join(subdir_path, d)) for d in items)
        
        if csv_files or has_task_subdirs:
            subplot_data[subdir] = subdir_path
    
    if not subplot_data:
        # Nessuna sottocartella con CSV, comportamento normale
        return False, {}
    
    return True, subplot_data

def read_subplot_title(subplot_path):
    """
    Legge il titolo del subplot dal file title.txt nella cartella.
    Se non esiste, usa il nome della cartella.
    
    Parameters:
    - subplot_path: Percorso della sottocartella del subplot
    
    Returns:
    - Titolo del subplot
    """
    title_file = os.path.join(subplot_path, 'title.txt')
    
    if os.path.exists(title_file):
        try:
            with open(title_file, 'r') as f:
                title = f.read().strip()
                if title:
                    return title
        except Exception as e:
            print(f"Errore nella lettura del file titolo {title_file}: {e}")
    
    # Fallback al nome della cartella
    return os.path.basename(subplot_path)

def create_subplot_grid(n_subplots):
    """
    Calcola il layout ottimale della griglia di subplot.
    
    Parameters:
    - n_subplots: Numero di subplot da creare
    
    Returns:
    - tuple: (n_rows, n_cols)
    """
    if n_subplots <= 0:
        return 0, 0
    elif n_subplots == 1:
        return 1, 1
    elif n_subplots == 2:
        return 1, 2
    elif n_subplots <= 4:
        return 2, 2
    elif n_subplots <= 6:
        return 2, 3
    elif n_subplots <= 9:
        return 3, 3
    elif n_subplots <= 12:
        return 3, 4
    else:
        # Per più di 12 subplot, usa una griglia 4xN
        n_cols = 4
        n_rows = (n_subplots + n_cols - 1) // n_cols
        return n_rows, n_cols

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
            plt.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                             color=color, alpha=0.1)
    
    ax = plt.gca()
    
    # Set x-axis properties
    if log_x:
        ax.set_xscale('log')
        x_axis_label = x_label or 'Timestep (logscale)'
        if x_label and x_scale_factor is not None:
            scale_notation = format_scale_notation(x_scale_factor)
            x_axis_label = f"{x_label} (logscale, {scale_notation})"
    else:
        # Format x-axis ticks if scale factor is provided
        if x_scale_factor is not None:
            x_formatter = create_value_formatter(x_scale_factor)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_formatter))
            scale_notation = format_scale_notation(x_scale_factor)
            
            # If custom label provided, append scale notation
            if x_label:
                x_axis_label = f"{x_label} ({scale_notation})"
            else:
                x_axis_label = f"{xaxis.capitalize()} ({scale_notation})"
        else:
            x_axis_label = x_label or xaxis.capitalize()
    
    plt.xlabel(x_axis_label)
    
    # Set y-axis properties
    if log_y:
        ax.set_yscale('log')
        y_axis_label = y_label or 'Reward (logscale)'
        if y_label and y_scale_factor is not None:
            scale_notation = format_scale_notation(y_scale_factor)
            y_axis_label = f"{y_label} (logscale, {scale_notation})"
    else:
        # Format y-axis ticks if scale factor is provided
        if y_scale_factor is not None:
            y_formatter = create_value_formatter(y_scale_factor)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
            scale_notation = format_scale_notation(y_scale_factor)
            
            # If custom label provided, append scale notation
            if y_label:
                y_axis_label = f"{y_label} ({scale_notation})"
            else:
                y_axis_label = f"Reward ({scale_notation})"
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

def plot_multiple_subplots(subplot_aggregated_data, subplot_titles, 
                          xaxis='buffer_size', yaxis='y_aggregated',
                          ci_type='std_err', log_x=False, log_y=False,
                          x_min=None, x_max=None, y_min=None, y_max=None,
                          x_scale_factor=None, y_scale_factor=None,
                          x_label=None, y_label=None,
                          color_mapping=None, overall_title=None,
                          legend_order=None):
    """
    Crea subplot multipli con legenda condivisa.
    
    Parameters:
    - subplot_aggregated_data: dict con subplot_name -> aggregated_data
    - subplot_titles: dict con subplot_name -> titolo
    - legend_order: Lista ordinata dei nomi degli algoritmi per la legenda
    """
    sns.set_style("darkgrid")
    
    n_subplots = len(subplot_aggregated_data)
    n_rows, n_cols = create_subplot_grid(n_subplots)
    
    # Dimensioni della figura basate sul numero di subplot
    fig_width = 6 * n_cols
    fig_height = 5 * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    
    # Assicurati che axes sia sempre un array
    if n_subplots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Raccogli tutti gli algoritmi unici attraverso tutti i subplot
    all_algorithms = set()
    for aggregated_data in subplot_aggregated_data.values():
        all_algorithms.update(aggregated_data.keys())
    all_algorithms = sorted(list(all_algorithms))
    
    # Crea mapping colori consistente per tutti i subplot
    if color_mapping:
        colors = {alg: color_mapping.get(alg, sns.color_palette("tab10")[i % 10]) 
                 for i, alg in enumerate(all_algorithms)}
    else:
        palette = sns.color_palette("tab10", len(all_algorithms))
        colors = {alg: palette[i] for i, alg in enumerate(all_algorithms)}
    
    # Plotta ogni subplot
    subplot_names = sorted(subplot_aggregated_data.keys())
    for idx, subplot_name in enumerate(subplot_names):
        ax = axes[idx]
        aggregated_data = subplot_aggregated_data[subplot_name]
        
        for algorithm, data in aggregated_data.items():
            color = colors[algorithm]
            
            # Plotta la linea principale
            ax.plot(data[xaxis], data[yaxis], label=algorithm, color=color, linewidth=2)
            
            # Aggiungi intervallo di confidenza
            if ci_type == 'min_max':
                ax.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                               color=color, alpha=0.2)
            elif ci_type == 'std':
                ax.fill_between(data[xaxis], data[yaxis] - data['y_std'], 
                               data[yaxis] + data['y_std'], color=color, alpha=0.2)
            elif ci_type == 'std_err':
                ax.fill_between(data[xaxis], data[yaxis] - data['y_stderr'], 
                               data[yaxis] + data['y_stderr'], color=color, alpha=0.2)
            elif ci_type == 'bootstrap_95' and 'y_bootstrap_ci_2_5' in data.columns:
                ax.fill_between(data[xaxis], data['y_bootstrap_ci_2_5'], 
                               data['y_bootstrap_ci_97_5'], color=color, alpha=0.2)
            elif ci_type == 'bootstrap_90' and 'y_bootstrap_ci_5' in data.columns:
                ax.fill_between(data[xaxis], data['y_bootstrap_ci_5'], 
                               data['y_bootstrap_ci_95'], color=color, alpha=0.2)
            elif ci_type == 'samples' and 'y_count' in data.columns:
                ax.fill_between(data[xaxis], data['y_min'], data['y_max'], 
                               color=color, alpha=0.1)
        
        # Configura assi
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        
        # Formattazione assi
        if x_scale_factor is not None and not log_x:
            x_formatter = create_value_formatter(x_scale_factor)
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_formatter))
        
        if y_scale_factor is not None and not log_y:
            y_formatter = create_value_formatter(y_scale_factor)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
        
        # Limiti assi
        if x_min is not None:
            ax.set_xlim(left=x_min)
        if x_max is not None:
            ax.set_xlim(right=x_max)
        if y_min is not None:
            ax.set_ylim(bottom=y_min)
        if y_max is not None:
            ax.set_ylim(top=y_max)
        
        # Titolo del subplot
        ax.set_title(subplot_titles.get(subplot_name, subplot_name))
        
        # Etichette assi (solo sui bordi esterni)
        if idx >= (n_rows - 1) * n_cols:  # Ultima riga
            if x_label:
                label = x_label
                if x_scale_factor is not None and not log_x:
                    scale_notation = format_scale_notation(x_scale_factor)
                    label = f"{x_label} ({scale_notation})"
                ax.set_xlabel(label)
            else:
                ax.set_xlabel(xaxis.capitalize())
        
        if idx % n_cols == 0:  # Prima colonna
            if y_label:
                label = y_label
                if y_scale_factor is not None and not log_y:
                    scale_notation = format_scale_notation(y_scale_factor)
                    label = f"{y_label} ({scale_notation})"
                ax.set_ylabel(label)
            else:
                ax.set_ylabel('Reward')
    
    # Nascondi subplot extra
    for idx in range(len(subplot_names), len(axes)):
        axes[idx].set_visible(False)
    
    # Crea legenda unica con ordine specificato
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Applica l'ordine della legenda se fornito
    if legend_order:
        # Crea un dizionario handle per algoritmo
        label_to_handle = {label: handle for label, handle in zip(labels, handles)}
        
        # Riordina secondo legend_order
        ordered_handles = []
        ordered_labels = []
        for alg in legend_order:
            if alg in label_to_handle:
                ordered_handles.append(label_to_handle[alg])
                ordered_labels.append(alg)
        
        handles = ordered_handles
        labels = ordered_labels
    else:
        # Ordine alfabetico di default
        sorted_indices = sorted(range(len(labels)), key=lambda i: labels[i])
        handles = [handles[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
    
    # Posiziona legenda sotto i subplot
    legend_ncol = min(len(labels), 8)
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
              ncol=legend_ncol, frameon=True, fontsize=10)
    
    # Titolo generale se fornito
    if overall_title:
        fig.suptitle(overall_title, fontsize=16, y=0.98)
    
    # Adjust layout per fare spazio alla legenda sotto
    plt.tight_layout(rect=[0, 0.05, 1.0, 0.96 if overall_title else 0.98])

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--xaxis', '-x', default='train/frame', help='Column name for x-axis data')
    parser.add_argument('--value', '-y', default='train/episode_reward', help='Column name for y-axis data')
    parser.add_argument('--est', default='mean', choices=['mean', 'median', 'min', 'max', 'iqm', 'interquartile_mean'], 
                        help='Aggregation function')
    parser.add_argument('--ci', default='std', choices=['min_max', 'std', 'std_err', 'samples', 'bootstrap_95', 'bootstrap_90', 'none'],
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
    parser.add_argument('--x_max', type=float, default=50_000, help='Maximum value for x-axis')
    parser.add_argument('--y_min', type=float, default=None, help='Minimum value for y-axis')
    parser.add_argument('--y_max', type=float, default=None, help='Maximum value for y-axis')
    
    # Axis formatting
    parser.add_argument('--x_scale', type=float, default=1e4, help='Scale factor for x-axis values (e.g., 1e5)')
    parser.add_argument('--y_scale', type=float, default=None, help='Scale factor for y-axis values (e.g., 1e3)')
    parser.add_argument('--x_label', type=str, default='Timestep', help='Custom x-axis label')
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
    
    # Subplot structure option
    parser.add_argument('--subplot', action='store_true', default=False,
                        help='Interpreta le sottocartelle come subplot invece che come task')
    
    # Legend order option
    parser.add_argument('--legend', action='store_true', default=False,
                        help='Enable legend order customization with interactive input')
    parser.add_argument('--force-legend', action='store_true', default=False,
                        help='Force legend order even if mapping file exists')
    
    args = parser.parse_args()

    # Rileva struttura subplot se flag --subplot è attiva
    if args.subplot:
        is_subplot_structure, subplot_data = detect_subplot_structure(args.csv_path)
    else:
        is_subplot_structure = False
        subplot_data = {}
    
    if is_subplot_structure:
        print(f"Rilevata struttura subplot con {len(subplot_data)} subplot")
        print(f"Subplot: {list(subplot_data.keys())}")
        
        # Elabora ogni subplot
        subplot_aggregated_data = {}
        subplot_titles = {}
        all_algorithm_names = set()
        
        for subplot_name, subplot_path in subplot_data.items():
            print(f"\n{'='*50}")
            print(f"Elaborazione subplot: {subplot_name}")
            print(f"{'='*50}")
            
            # Leggi il titolo del subplot
            subplot_titles[subplot_name] = read_subplot_title(subplot_path)
            print(f"Titolo: {subplot_titles[subplot_name]}")
            
            # Leggi i CSV per questo subplot
            grouped_dataframes_with_tasks, task_structure = read_csvs_with_task_awareness(subplot_path)
            grouped_dataframes = extract_dataframes_only(grouped_dataframes_with_tasks)
            
            # Raccogli nomi algoritmi
            all_algorithm_names.update(grouped_dataframes.keys())
            
            # Aggrega i dati
            aggregated_data = {}
            for algorithm in grouped_dataframes.keys():
                if args.bootstrap:
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
            
            subplot_aggregated_data[subplot_name] = aggregated_data
        
        # Applica rinominazione globale se richiesta
        if args.rename:
            all_algorithm_names = sorted(list(all_algorithm_names))
            name_mapping = load_or_create_name_mapping(
                args.csv_path,
                all_algorithm_names,
                force_rename=getattr(args, 'force_rename', False)
            )
            
            # Applica mapping a tutti i subplot
            for subplot_name in subplot_aggregated_data.keys():
                renamed_data = {}
                for original_name, data in subplot_aggregated_data[subplot_name].items():
                    new_name = name_mapping.get(original_name, original_name)
                    if new_name in renamed_data:
                        # Merge se necessario (non dovrebbe accadere con aggregati)
                        print(f"Warning: nome duplicato {new_name} in subplot {subplot_name}")
                    renamed_data[new_name] = data
                subplot_aggregated_data[subplot_name] = renamed_data
            
            all_algorithm_names = sorted(list(set(name_mapping.values())))
        
        # Applica mapping colori se richiesto
        color_mapping = None
        if args.color:
            color_mapping = load_or_create_color_mapping(
                args.csv_path,
                all_algorithm_names,
                force_color=getattr(args, 'force_color', False)
            )
        
        if args.legend:
            legend_order = load_or_create_legend_order(
                args.csv_path,
                all_algorithm_names,
                force_legend=getattr(args, 'force_legend', False)
            )
        else:
            legend_order = None
        # Crea i subplot multipli
        print("\n" + "="*50)
        print("Creazione plot multipli...")
        print("="*50)
        
        plot_multiple_subplots(
            subplot_aggregated_data,
            subplot_titles,
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
            color_mapping=color_mapping,
            overall_title=args.title,
            legend_order=legend_order
        )
        
    else:
        # Comportamento originale per singolo plot (con o senza task structure)
        print("Nessuna struttura subplot rilevata, creazione plot singolo")
        
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
        
        if args.legend:
            algorithm_names = list(grouped_dataframes.keys())
            legend_order = load_or_create_legend_order(
                args.csv_path,
                algorithm_names,
                force_legend=getattr(args, 'force_legend', False)
            )
        else:
            legend_order = None
            
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
    
    # Crea output path nella stessa cartella di csv_path
    output_path = os.path.join(args.csv_path, args.output)
    
    # Salva il plot
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot salvato come {output_path}")

if __name__ == "__main__":
    main()