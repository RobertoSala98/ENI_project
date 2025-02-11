from utils import *

def parse_all_logs(log_folder, experiments):

    print("\n\n************************************************************\n")
    print("1. Analysing all provided logs...")
    print("\n************************************************************\n\n")

    for experiment in experiments:

        print(f"Experimental campaign: {experiment} \n")

        found = False
        root = f"{log_folder}{experiment}"

        file_list = []
        cores_number = cpu_count()

        for path, subdirs, files in os.walk(root):
            for name in files:
                filename = os.path.join(path,name)
                
                if ".log" in filename[-4:]:
                    file_list.append(filename)

        print(f"\tNumber of logs found: {len(file_list)} \n")
        
        setting_parallel = split_list(file_list, cores_number)
        with Pool() as pool:
            partial_gp = functools.partial(process_batch)        
            batch_results_parallel = pool.map(partial_gp, setting_parallel)

        data = {}

        for cc in range(cores_number):
            data.update(batch_results_parallel[cc])
            
        for value in data.values():
            for shot in value.values():

                if not found:
                    first_time = datetime.datetime.strptime(shot['start_time'], datetime_format)
                    last_time = datetime.datetime.strptime(shot['end_time'], datetime_format)
                    found = True
                else:
                    first_time = min(first_time, datetime.datetime.strptime(shot['start_time'], datetime_format))
                    last_time = max(last_time, datetime.datetime.strptime(shot['end_time'], datetime_format))

        print("\tFirst time:", first_time)
        print("\tLast time: ", last_time, "\n")

        with open(f'{root}/times.json', "w") as json_file:
            json.dump(data, json_file, indent=4)

        print("\n")


def extract_average_duration(log_folder, experimental_campaigns, threshold):

    print("\n\n************************************************************\n")
    print("2. Extracting durations from logs...")
    print("\n************************************************************\n\n")

    for experiment in experimental_campaigns:

        print(f"Experimental campaign: {experiment} \n")

        with open(f'{log_folder}{experiment}/times.json', 'r') as jobs_file:
            jobs = json.load(jobs_file)

        durations = {}

        xx_ = []
        yy_ = []
        shot_names = []
        counter = 0

        for job in jobs:

            shots = []
            for key in jobs[job].keys():
                if "shot_" in key:
                    shots.append(key)

            for shot in shots:
                if jobs[job][shot]["success"]:

                    counter += 1

                    if int(jobs[job][shot]["number_of_cells"]) in durations:
                        durations[int(jobs[job][shot]["number_of_cells"])].append(jobs[job][shot]['duration'])
                    else:
                        durations[int(jobs[job][shot]["number_of_cells"])] = [jobs[job][shot]['duration']]

                    xx_.append(jobs[job][shot]["number_of_cells"])
                    yy_.append(jobs[job][shot]['duration'])
                    
                    shot_names.append(shot)

        xx__ = np.array(xx_).copy()
        xx = xx__.reshape(-1, 1)
        yy = np.array(yy_).reshape(-1, 1)
        model = LinearRegression()
        model.fit(xx, yy)
        yy_pred = model.predict(xx)
        sqrt_mse = np.sqrt(np.mean((yy - yy_pred) ** 2))

        index_min_xx = np.argmin(xx__)
        index_max_xx = np.argmax(xx__)

        for ii in range(len(xx_)):
            if ii == 0:
                plt.plot(xx_[ii], yy_[ii], 'x', markersize=5, color='lightgrey', label='data')
            else:
                plt.plot(xx_[ii], yy_[ii], 'x', markersize=5, color='lightgrey')

        plt.plot([xx__[index_min_xx],xx__[index_max_xx]], [yy_pred[index_min_xx],yy_pred[index_max_xx]], color='green', label="Regression model") 
        plt.plot([xx__[index_min_xx],xx__[index_max_xx]], [yy_pred[index_min_xx]+3*sqrt_mse,yy_pred[index_max_xx]+3*sqrt_mse], '--', color='blue', label = '$\mu ± 3\sigma$')
        plt.plot([xx__[index_min_xx],xx__[index_max_xx]], [yy_pred[index_min_xx]-3*sqrt_mse,yy_pred[index_max_xx]-3*sqrt_mse], '--', color='blue')

        found_90 = False
        found_95 = False
        idx = 0

        idx_ = np.arange(model.intercept_+sqrt_mse, model.intercept_+3*sqrt_mse + 0.05, 0.05)

        over_ = [ii for ii in range(len(xx_))]
        under = 0
        over = len(over_)

        pbar = tqdm()
        pbar.total = len(idx_)
        pbar.refresh()

        for inter_new in idx_:

            pbar.n = idx
            pbar.refresh()

            for ii in over_:
                if yy_[ii] < inter_new + model.coef_ * xx_[ii]:
                    under += 1
                    over -= 1
                    over_.remove(ii)

            if not found_90 and under / (under + over) >= 0.9:
                print("intercept, 0.9:", inter_new)
                inter_90 = inter_new
                found_90 = True
                plt.plot([xx__[index_min_xx], xx__[index_max_xx]], [float(inter_90 + xx__[index_min_xx]*model.coef_), float(inter_90 + xx__[index_max_xx]*model.coef_)], '--', color='purple', label='90 %')

            if not found_95 and under / (under + over) >= 0.95:
                print("intercept, 0.95:", inter_new)
                inter_95 = inter_new
                found_95 = True
                plt.plot([xx__[index_min_xx], xx__[index_max_xx]], [float(inter_95 + xx__[index_min_xx]*model.coef_), float(inter_95 + xx__[index_max_xx]*model.coef_)], '--', color='red', label='95 %')

            if under / (under + over) >= threshold:
                print(f"intercept, {threshold}:", inter_new)
                threshold_value = inter_new
                plt.plot([xx__[index_min_xx], xx__[index_max_xx]], [float(threshold + xx__[index_min_xx]*model.coef_), float(threshold + xx__[index_max_xx]*model.coef_)], '--', color='orange', label=f'{threshold} %')
                break

            idx += 1

        pbar.refresh()
        pbar.close()
                
        plt.ylim([yy_pred[index_min_xx]-3.2*sqrt_mse, yy_pred[index_max_xx]+3.2*sqrt_mse])

        plt.legend(loc='best')
        plt.title(f"Experiment: {experiment}")
        plt.savefig(f'{log_folder}{experiment}/data_plot.png', dpi=300)
        plt.close()

        lines_data = {
            'coefficient': float(model.coef_),
            'intercept avg': float(model.intercept_),
            'intercept90': float(inter_90),
            'intercept95': float(inter_95),
            f'intercept{threshold}': float(threshold_value)
        }

        with open(f'{log_folder}{experiment}/lines_data.json', 'w') as file:
            json.dump(lines_data, file, indent=4)

        print(f"\n\tSuccesfull jobs: {counter}")

        print("\n")


def prepare_csv_ML(log_folder, experimental_campaigns, threshold, test_campaigns):

    print("\n\n************************************************************\n")
    print("3. Preparing datasets for ML models...")
    print("\n************************************************************\n\n")

    data = [["number_of_cells", "number_of_timestep", "nodes_number", "gathers", "wave_front_tracking", "hilbert_filter", "isotropic_kernel", "exec_time", "over"]]
    upper_bound = [["number_of_cells", "number_of_timestep", "nodes_number", "gathers", "wave_front_tracking", "hilbert_filter", "isotropic_kernel", "exec_time"]]

    all_data = []
    candidate_train = []
    train_data = []
    test_data = []

    actual_data = 0
    previous_data = 0

    for experiment in experimental_campaigns:

        with open(f'{log_folder}{experiment}/times.json', 'r') as jobs_file:
            jobs = json.load(jobs_file)

        with open(f'{log_folder}{experiment}/lines_data.json', 'r') as experiment_file:
            data_experiment = json.load(experiment_file)

        for job in jobs:

            shots = []
            for key in jobs[job].keys():
                if "shot_" in key:
                    shots.append(key)

            for shot in shots:
                if jobs[job][shot]["success"]:

                    if jobs[job][shot]['duration'] > data_experiment[f'intercept{threshold}'] + data_experiment['coefficient']*jobs[job][shot]["number_of_cells"]:
                        over = 1
                    else:
                        over = 0

                    data.append([jobs[job][shot]["number_of_cells"],
                                jobs[job][shot]["number_of_timestep"],
                                jobs[job][shot]["nodes_number"],
                                jobs[job][shot]["gathers"],
                                jobs[job][shot]["wave_front_tracking"],
                                jobs[job][shot]["hilbert_filter"],
                                jobs[job][shot]["isotropic_kernel"],
                                jobs[job][shot]['duration'],
                                over
                                ])
                    
                    upper_bound.append(tuple([jobs[job][shot]["number_of_cells"],
                                jobs[job][shot]["number_of_timestep"],
                                jobs[job][shot]["nodes_number"],
                                jobs[job][shot]["gathers"],
                                jobs[job][shot]["wave_front_tracking"],
                                jobs[job][shot]["hilbert_filter"],
                                jobs[job][shot]["isotropic_kernel"],
                                data_experiment[f'intercept{threshold}'] + data_experiment['coefficient']*jobs[job][shot]["number_of_cells"]
                                ]))
                    
                    all_data.append(tuple([jobs[job][shot]["number_of_cells"],
                                jobs[job][shot]["number_of_timestep"],
                                jobs[job][shot]["nodes_number"],
                                jobs[job][shot]["gathers"],
                                jobs[job][shot]["wave_front_tracking"],
                                jobs[job][shot]["hilbert_filter"],
                                jobs[job][shot]["isotropic_kernel"]
                                ]))
            
        actual_data = len(data)-1 - previous_data

        for idx in range(len(data)-1-actual_data,len(data)-1):
            if experiment not in test_campaigns:
                train_data.append(data[idx+1])
            else:
                test_data.append(data[idx+1])

        previous_data = len(data)-1
        
    unique_list = []
    for item in upper_bound:
        if item not in unique_list:
            unique_list.append(item)

    with open('upper_bound_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(unique_list)

    with open('training_set.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["number_of_cells", "number_of_timestep", "nodes_number", "gathers", "wave_front_tracking", "hilbert_filter", "isotropic_kernel", "exec_time", "over"])
        writer.writerows(train_data)

    with open('test_set.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["number_of_cells", "number_of_timestep", "nodes_number", "gathers", "wave_front_tracking", "hilbert_filter", "isotropic_kernel", "exec_time", "over"])
        writer.writerows(test_data)


def train_ML_models():




    return