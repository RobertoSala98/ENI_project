from functions import *

if __name__ == '__main__':
    
    with open("config_analysis_log.yaml", "r") as file:
        config = yaml.safe_load(file)

    log_folder = config['log_folder']
    experimental_campaigns = [_ for _ in os.listdir(log_folder) if os.path.isdir(log_folder + _) and _ not in config['skip_folders']]
    threshold = config['threshold']
    test_campaigns = config['test_campaigns']

    # 1. Parse the logs
    parse_all_logs(log_folder, experimental_campaigns)

    # 2. Extract average duration
    extract_average_duration(log_folder, experimental_campaigns, threshold)

    # 3. Prepare datasets for ML models
    prepare_csv_ML(log_folder, experimental_campaigns, threshold, test_campaigns)

    # 4. Train ML models
    train_ML_models()

    import pdb; pdb.set_trace()

