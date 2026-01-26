"""
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/basketball --filter_by_config "agent/no_taco!=true&env_name=basketball-v2&agent/pretrained_path!=/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_Basketball_0.99_lr=0.0005_ts=409600_curl_rew_best.pt" --filter_by_tags     "(MT50&BEST)|baseline" --group_by_config agent/pretrained_path --project taco_metaworld_batch1 
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/basketball --filter_by_config agent/no_taco!=true,env_name=basketball-v2 --filter_by_tags     MT50 --group_by_config agent/pretrained_path --project taco_metaworld_debug 
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/bin-picking --filter_by_config "agent/no_taco!=true&env_name=bin-picking-v2&agent/pretrained_path!=/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_BinPicking_0.99_lr=0.0005_ts=36300800_curl_rew_best.pt" --filter_by_tags     "(MT50&BEST)|baseline" --group_by_config agent/pretrained_path --project taco_metaworld_batch1 
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/bin-picking --filter_by_config "agent/no_taco!=true&env_name=bin-picking-v2" --filter_by_tags     MT50 --group_by_config agent/pretrained_path --project taco_metaworld_debug 
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/button-press --filter_by_config "agent/no_taco!=true&env_name=button-press-v2&agent/pretrained_path!=/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_ButtonPress_0.99_lr=0.0005_ts=16435200_curl_rew_best.pt" --filter_by_tags     "(MT50&BEST)|baseline" --group_by_config agent/pretrained_path --project taco_metaworld_batch1 
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/button-press --filter_by_config "agent/no_taco!=true&env_name=button-press-v2" --filter_by_tags     MT50 --group_by_config agent/pretrained_path --project taco_metaworld_debug 
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/push --filter_by_config "agent/no_taco!=true&env_name=push-v2&agent/pretrained_path!=/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_0.99_lr=0.0005_ts=50000896_curl_rew.pt" --filter_by_tags     "(MT50)|baseline" --group_by_config agent/pretrained_path --project taco_metaworld_batch1 
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/push --filter_by_config "agent/no_taco!=true&env_name=push-v2" --filter_by_tags     MT50 --group_by_config agent/pretrained_path --project taco_metaworld_debug 
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/shelf-place --filter_by_config "agent/no_taco!=true&env_name=shelf-place-v2&agent/pretrained_path!=/home/mprattico/Pretrain-TACO/models/taco_MT_MT50_OOD_ShelfPlace_0.99_lr=0.0005_ts=43315200_curl_rew_best.pt" --filter_by_tags     "(MT50&BEST)|baseline" --group_by_config agent/pretrained_path --project taco_metaworld_batch1 
python /home/mprattico/Pretrain-TACO/plot/download_data_from_wandb.py --csv_path data_plot/exp/shelf-place --filter_by_config "agent/no_taco!=true&env_name=shelf-place-v2" --filter_by_tags     MT50 --group_by_config agent/pretrained_path --project taco_metaworld_debug 

python plot/data_downloader.py --csv_path data_plot/states/multirooms --filter_by_config "env/name=MultipleRooms-v0&obs_type=pixels" --group_by_config agent/_target_ --project finetune_gym --download --processing --n_points 80


"""
import argparse
import pandas as pd
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import os
import os.path as osp
import numpy as np
from scipy.interpolate import interp1d

def read_csvs_from_directory(directory):
    dataframes = []
    csv_filenames = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            dataframes.append(df)
            csv_filenames.append(filename)
    
    return dataframes, csv_filenames

def flatten_dict(d):
    """Flatten a nested dictionary."""
    items = []
    for key, value in d.items():
        if isinstance(value, dict):
            # The name of the nested keys are <parent_key>/<child_key>
            for sub_key, sub_value in value.items():
                new_key = f"{key}/{sub_key}"
                if isinstance(sub_value, dict):
                    items.extend(flatten_dict(sub_value).items())
                else:
                    items.append((new_key, sub_value))
        else:
            items.append((key, value))
    return dict(items)

def check_config_match(run_config, config_filters):
    """Check if a run's config matches the specified filters."""
    flattened_config = flatten_dict(run_config)
    for key, filter_info in config_filters.items():
        operator, value = filter_info
        
        # Check if the key exists in the flattened config
        if key not in flattened_config:
            # If using != and the key doesn't exist, that's actually a match
            if operator == "!=":
                continue
            return False
        
        # Convert the config value to string for comparison
        config_value = str(flattened_config[key]).lower()
        filter_value = str(value).lower()
        
        # Apply the appropriate comparison based on operator
        if operator == "==" and config_value != filter_value:
            return False
        elif operator == "!=" and config_value == filter_value:
            return False
    
    return True

def evaluate_config_expression(run_config, expression):
    """Evaluate a config expression with logical operators against a run's config."""
    if not expression:
        return True
    
    flattened_config = flatten_dict(run_config)
    expr = expression
    
    print(f"DEBUG: Original expression: {expression}")
    print(f"DEBUG: Available config keys: {list(flattened_config.keys())}")
    
    
    # Find all config filter patterns in the expression
    import re
    # Improved pattern to handle file paths with special characters
    # Use non-greedy matching and handle paths properly
    config_patterns = []
    
    # Split by & and | first, then parse each part
    parts = re.split(r'([&|()])', expression)
    for part in parts:
        part = part.strip()
        if not part or part in ['&', '|', '(', ')']:
            continue
        
        # Now parse this individual filter
        if "!=" in part:
            key, value = part.split("!=", 1)
            config_patterns.append((part.strip(), key.strip(), "!=", value.strip()))
        elif "==" in part:
            key, value = part.split("==", 1)
            config_patterns.append((part.strip(), key.strip(), "==", value.strip()))
        elif "=" in part:
            key, value = part.split("=", 1)
            config_patterns.append((part.strip(), key.strip(), "==", value.strip()))
    
    print(f"DEBUG: Found config patterns: {config_patterns}")
    
    # Replace each config filter with True or False
    for pattern, key, operator, value in config_patterns:
        # Convert value to appropriate type
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
            value = float(value)
        
        # Check if the config matches
        config_match = False
        if key not in flattened_config:
            # If using != and the key doesn't exist, that's actually a match
            if operator == "!=":
                config_match = True
        else:
            # For string comparison, be more careful
            config_value = str(flattened_config[key])
            filter_value = str(value)
            
            print(f"DEBUG: Comparing '{key}': config='{config_value}' vs filter='{filter_value}' (operator: {operator})")
            
            # Apply the appropriate comparison based on operator
            if operator == "==" and config_value == filter_value:
                config_match = True
            elif operator == "!=" and config_value != filter_value:
                config_match = True
        
        print(f"DEBUG: Pattern '{pattern}' -> {config_match}")
        
        # Replace the pattern with the result
        expr = expr.replace(pattern, str(config_match))
    
    # Replace operators with Python equivalents
    expr = expr.replace('&', ' and ').replace('|', ' or ')
    
    print(f"DEBUG: Final expression: {expr}")
    
    try:
        # Safely evaluate the boolean expression
        result = eval(expr)
        print(f"DEBUG: Expression result: {result}")
        return result
    except Exception as e:
        print(f"Error evaluating config expression '{expression}': {e}")
        # If evaluation fails, fall back to original logic
        return True

def parse_config_filters(config_filter_str):
    """Parse the config filter string into a complex expression or dictionary."""
    if not config_filter_str:
        return None, {}
    
    # Check if the string contains logical operators
    if any(op in config_filter_str for op in ['&', '|', '(', ')']):
        # Return the expression as-is for complex evaluation
        return config_filter_str, {}
    
    # Fall back to the original simple parsing
    filters = {}
    filter_pairs = config_filter_str.split(',')
    for pair in filter_pairs:
        # Check for inequality operator
        if "!=" in pair:
            key, value = pair.split("!=", 1)
            operator = "!="
        # Check for equality operator (explicit or implicit)
        elif "==" in pair:
            key, value = pair.split("==", 1)
            operator = "=="
        elif "=" in pair:
            key, value = pair.split("=", 1)
            operator = "=="
        else:
            continue
            
        # Try to convert value to appropriate type
        if value.lower() == 'true':
            value = True
        elif value.lower() == 'false':
            value = False
        elif value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit() and value.count('.') <= 1:
            value = float(value)
        
        # Store both the operator and the value
        filters[key.strip()] = (operator, value)
    
    return None, filters

def evaluate_tag_expression(tags, expression):
    """Evaluate a tag expression with logical operators against a list of tags."""
    if not expression:
        return True
    
    # Replace tag names with True/False based on presence in tags
    expr = expression
    
    # Find all unique tag names in the expression
    import re
    tag_names = re.findall(r'\b[a-zA-Z0-9_-]+\b', expression)
    
    # Replace each tag name with True or False
    for tag_name in set(tag_names):
        if tag_name.lower() in ['and', 'or', 'true', 'false']:
            continue
        expr = expr.replace(tag_name, str(tag_name in tags))
    
    # Replace operators with Python equivalents
    expr = expr.replace('&', ' and ').replace('|', ' or ')
    
    try:
        # Safely evaluate the boolean expression
        return eval(expr)
    except:
        # If evaluation fails, fall back to checking if any tag matches
        return any(tag in tags for tag in tag_names)

def parse_tag_filters(tags_filter_str):
    """Parse tag filter string into a complex expression or simple lists."""
    if not tags_filter_str:
        return None, [], []
    
    # Check if the string contains logical operators
    if any(op in tags_filter_str for op in ['&', '|', '(', ')']):
        # Return the expression as-is for complex evaluation
        return tags_filter_str, [], []
    
    # Fall back to the original simple parsing
    include_tags = []
    exclude_tags = []
    
    tag_filters = tags_filter_str.split(',')
    for tag_filter in tag_filters:
        if tag_filter.endswith('!='):
            # Remove the != operator and add to exclude list
            exclude_tags.append(tag_filter[:-2].strip())
        elif '!=' in tag_filter:
            # Format: "tag!=value"
            exclude_tags.append(tag_filter.split('!=')[1].strip())
        elif tag_filter.endswith('='):
            # Format: "tag="
            include_tags.append(tag_filter[:-1].strip())
        elif '=' in tag_filter:
            # Format: "tag=value"
            include_tags.append(tag_filter.split('=')[1].strip())
        else:
            # Simple tag name
            include_tags.append(tag_filter.strip())
    
    return None, include_tags, exclude_tags

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default="data_plot/", help='csv folder') 
    parser.add_argument('--keys', type=str, default="train/episode_reward", help='Data to be saved')
    parser.add_argument('--x-key', type=str, default="train/frame", help='X axis key')
    parser.add_argument('--filter_by_tags', type=str, default="", 
                      help='Filter by tags. Use simple comma-separated list or complex expressions with & (AND), | (OR), and parentheses. Example: "(MT50&baseline)|BEST"')
    parser.add_argument('--filter_by_config', type=str, default="", 
                        help='Filter by config parameters. Use simple format (key1=value1,key2=value2) or complex expressions with & (AND), | (OR), and parentheses. Example: "(agent/no_taco!=true&env_name=basketball-v2)|batch_size=1024"')
    parser.add_argument('--download', action='store_true', default=False, help='Download data from wandb')
    parser.add_argument('--processing', action='store_true', default=False)
    parser.add_argument('--project', type=str, default='taco_metaworld', help='Project name') 
    parser.add_argument('--entity', type=str, default=None, help='WandB entity/team name (optional)')
    parser.add_argument('--n_points', type=int, default=1000, help='Number of points to plot')
    parser.add_argument('--max_x', type=int, default=10_000, help='maximum x axis value of points to plot')
    parser.add_argument('--min_x', type=int, default=0, help='minimum x axis value of points to plot')
    parser.add_argument('--group_by_config', type=str, default="pretrained_path", help='Config parameter to use for grouping and naming saved files')
    parser.add_argument('--max_runs_per_group', type=int, default=15, 
                        help='Maximum number of runs to download per group (based on group_by_config). If None, download all runs')

    args = parser.parse_args()
    keys = args.keys.split(",")
    tag_expression, include_tags, exclude_tags = parse_tag_filters(args.filter_by_tags)
    config_expression, config_filters = parse_config_filters(args.filter_by_config)

    print("Start Downloading")
    if args.download:
        
        os.makedirs(args.csv_path, exist_ok=True)
        
        # Initialize WandB API with debugging
        try:
            api = wandb.Api(timeout=120)
            print(f"WandB API inizializzata correttamente")
            print(f"Current user: {api.viewer}")
        except Exception as e:
            print(f"Errore nell'inizializzazione dell'API WandB: {e}")
            print("Prova a fare login con: wandb login")
            return
        
        # Construct project path - if no entity specified, try with user's entity
        if args.entity:
            project_path = f"{args.entity}/{args.project}"
        else:
            # Try to get user's default entity
            try:
                user_entity = api.viewer['entity']
                project_path = f"{user_entity}/{args.project}"
                print(f"Nessuna entità specificata, usando l'entità dell'utente: {user_entity}")
            except:
                project_path = args.project
        
        print(f"Tentativo di accesso al progetto: {project_path}")
        
        try:
            # Get project info first
            project = api.project(project_path)
            print(f"Progetto trovato: {project.name}")
            print(f"Entità del progetto: {project.entity}")
        except Exception as e:
            print(f"Errore nell'accesso al progetto {project_path}: {e}")
            print("Progetti disponibili:")
            try:
                for proj in api.projects():
                    print(f"  - {proj.entity}/{proj.name}")
            except Exception as proj_e:
                print(f"Errore nel listare i progetti: {proj_e}")
            return
        
        # Get runs with detailed debugging
        try:
            print(f"Recupero runs dal progetto {project_path}...")
            runs = api.runs(project_path)
            runs_list = list(runs)
            print(f"Numero totale di runs trovati: {len(runs_list)}")
            
            if len(runs_list) == 0:
                print("NESSUN RUN TROVATO!")
                print("Possibili cause:")
                print("1. Il progetto non contiene run")
                print("2. Non hai i permessi per vedere i run")
                print("3. Il nome del progetto o dell'entità è errato")
                return
            else:
                print(f"Primi 5 run trovati:")
                for i, run in enumerate(runs_list[:5]):
                    print(f"  {i+1}. ID: {run.id}, Nome: {run.name}, Tags: {run.tags}")
            
        except Exception as e:
            print(f"Errore nel recupero dei runs: {e}")
            return
        
        runs = runs_list
        
        # Filter runs by tags and config
        filtered_runs = []
        for run in runs:
            # Check tags using new expression evaluator or old logic
            if tag_expression:
                tags_match = evaluate_tag_expression(run.tags, tag_expression)
            else:
                # Use original simple logic
                tags_include_match = not include_tags or any(tag in run.tags for tag in include_tags)
                tags_exclude_match = not exclude_tags or not any(tag in run.tags for tag in exclude_tags)
                tags_match = tags_include_match and tags_exclude_match
            
            # Check config using new expression evaluator or old logic
            if config_expression:
                print(f"\nDEBUG: Evaluating config for run {run.id}")
                config_match = evaluate_config_expression(run.config, config_expression)
            else:
                # Use original simple logic
                config_match = check_config_match(run.config, config_filters)
            
            print(f"DEBUG: Run {run.id} - tags_match: {tags_match}, config_match: {config_match}")
            if config_match:
                print(f"OOOoooooooooooooooooooooooooooooooooooooooo\n\nn\oo\ooo\oooooooooo")
            
            if tags_match and config_match:
                filtered_runs.append(run)

        print(f"Runs dopo filtrazione: {len(filtered_runs)}")
        runs = filtered_runs
        
        # Group runs by the specified config parameter if max_runs_per_group is set
        if args.max_runs_per_group is not None:
            print(f"Limiting to {args.max_runs_per_group} runs per group (grouped by {args.group_by_config})")
            
            # Group runs by the config parameter
            grouped_runs = {}
            for run in runs:
                flattened_config = flatten_dict(run.config)
                if args.group_by_config in flattened_config:
                    param_value = str(flattened_config[args.group_by_config]).split("/")[-1]
                else:
                    param_value = "unknown"
                
                if param_value not in grouped_runs:
                    grouped_runs[param_value] = []
                grouped_runs[param_value].append(run)
            
            # Limit the number of runs per group
            limited_runs = []
            for group_name, group_runs in grouped_runs.items():
                selected_runs = group_runs[:args.max_runs_per_group]
                limited_runs.extend(selected_runs)
                print(f"Group '{group_name}': selected {len(selected_runs)} out of {len(group_runs)} runs")
            
            runs = limited_runs
            print(f"Total runs after limiting: {len(runs)}")
        
        all_keys = keys.copy()
        all_keys.append(args.x_key)
        for run in runs:
            flattened_config = flatten_dict(run.config)
            print(f"\n=== SAVING RUN {run.id} ===")
            print(f"Run name: {run.name}")
            print(f"Tags: {run.tags}")
            
            
            history = run.history(keys=all_keys)  

            if args.group_by_config in flattened_config:
                param_value = str(flattened_config[args.group_by_config]).split("/")[-1]
                print(f"param_value: {param_value}")
            else:
                # Fallback if the parameter doesn't exist
                param_value = "unknown"
                
            run_name = f'{param_value}___{run.id}'
            history.to_csv(f"{args.csv_path}/{run_name}.csv")
            print(f"saved {run_name}.csv")
            print(f"=== END RUN {run.id} ===\n")
            
        print("saving DONE")
    
    print("Start Processing")
    if args.processing:
        
        directory = args.csv_path
        print("reading...")
        runs, filenames = read_csvs_from_directory(directory)
        print("reading DONE")
        for run, filename in zip(runs, filenames):
            
            print("processing: ", filename)
            history = run
           
            # x_axis = np.linspace(int(history.iloc[0][args.x_key]), int(history.iloc[-1][args.x_key]) if args.max_x is None else args.max_x, args.n_points, dtype = int)
            
            x_axis = np.linspace(int(history.iloc[0][args.x_key]) if args.min_x is None else args.min_x, int(history.iloc[-1][args.x_key]) if args.max_x is None else args.max_x, args.n_points, dtype = int)
            new_history = pd.DataFrame({
                    str(args.x_key): x_axis,
                })
            
            # Convert the 'buffer_size' column to numeric
            for key in keys:
                
                f = interp1d(history[args.x_key], history[key], fill_value="extrapolate")
                new_history[key] = f(x_axis)

            print("saving...")
            new_history.to_csv(f"{directory}/{filename}")
            
            print("saving DONE")
        
        # all_datasets = preprocess_data(all_datasets, args.k)
  

if __name__ == "__main__":
    main()