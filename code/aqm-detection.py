import pandas as pd
import numpy as np

from scipy.signal import find_peaks
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score




# Data preparation, feature engineering

def max_subsequence_sum(arr):
    max_so_far = arr[0]
    max_ending_here = arr[0]

    for x in arr[1:]:
        max_ending_here = max(x, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

def min_subsequence_sum(arr):
    min_so_far = arr[0]
    min_ending_here = arr[0]

    for x in arr[1:]:
        min_ending_here = min(x, min_ending_here + x)
        min_so_far = min(min_so_far, min_ending_here)

    return min_so_far


def total_variation(arr):
    diffs = np.diff(arr)
    return np.sum(diffs ** 2)


# Load the data from CSV files
cwnd_data = pd.read_csv("/Users/fatihberkay/Desktop/Courses/AI : ML Network Traffic Analysis/Project/Data : Code/cwnd_main.csv")
srtt_data = pd.read_csv("/Users/fatihberkay/Desktop/Courses/AI : ML Network Traffic Analysis/Project/Data : Code/srtt_main.csv")
time_data = pd.read_csv("/Users/fatihberkay/Desktop/Courses/AI : ML Network Traffic Analysis/Project/Data : Code/time_main.csv") 

cwnd_data_test = pd.read_csv("/Users/fatihberkay/Desktop/Courses/AI : ML Network Traffic Analysis/Project/Data : Code/latest_topology_cwnd_main.csv")
srtt_data_test = pd.read_csv("/Users/fatihberkay/Desktop/Courses/AI : ML Network Traffic Analysis/Project/Data : Code/latest_topology_srtt_main.csv")
time_data_test = pd.read_csv("/Users/fatihberkay/Desktop/Courses/AI : ML Network Traffic Analysis/Project/Data : Code/latest_topology_time_main.csv") 

time_data = time_data - time_data.min()

time_data_test = time_data_test - time_data_test.min()

experiments = cwnd_data.columns.tolist()

train_experiments = experiments

test_experiments = cwnd_data_test.columns.tolist()


# Define the durations to test (in seconds)

durations = [1, 5, 10, 15, 20]  # Add more durations as needed

# Dictionary to store test sets for different durations
test_sets = {}

# Prepare a list to store feature dictionaries and labels
features_list = []
labels = []

# Loop through each experiment (column) in the CWND and RTT data
for col in train_experiments:
    
    df = pd.DataFrame({
    'time': time_data[col],
    'cwnd': cwnd_data[col],
    'srtt': srtt_data[col]})
    
    df = df.dropna()
    
    cwnd_series = df['cwnd'].values
    srtt_series = df['srtt'].values
    
    # # Extract the CWND and RTT data series, dropping any NaN values
    # cwnd_series = cwnd_data[col].dropna().values
    # srtt_series = srtt_data[col].dropna().values


    # Extract and convert label to numeric format
    experiment_name = col  # Column name is the experiment identifier
    label = 0 if "FIFO" in experiment_name else 1 if "pie_drop" in experiment_name else None

    # Proceed only if label is valid
    if label is not None:
        # Initialize a dictionary to hold features for this experiment
        features_dict = {}

        ### CWND Features Calculation ###

        # Compute gradients (first differences) and second gradients (second differences)
        cwnd_grad = np.gradient(cwnd_series)
        cwnd_second_grad = np.gradient(cwnd_grad)

        # Maximum and minimum gradients
        features_dict['CWND-max-grad'] = np.max(cwnd_grad)
        
        
        features_dict['CWND-max-subgrad'] = max_subsequence_sum(cwnd_grad)
        features_dict['CWND-min-subgrad'] = min_subsequence_sum(cwnd_grad)

        # Maximum and minimum second gradients
        features_dict['CWND-max-secondGrad'] = np.max(cwnd_second_grad)
        
        features_dict['CWND-max-subsecondGrad'] = max_subsequence_sum(cwnd_second_grad)
        features_dict['CWND-min-subsecondGrad'] = min_subsequence_sum(cwnd_second_grad)

        # Statistical features of CWND values
        features_dict['CWND-mean-value'] = np.mean(cwnd_series)
        features_dict['CWND-std-value'] = np.std(cwnd_series)
        features_dict['CWND-totalVar-value'] = total_variation(cwnd_series)
        features_dict['CWND-10th-p-value'] = np.percentile(cwnd_series, 10)
        features_dict['CWND-90th-p-value'] = np.percentile(cwnd_series, 90)

        # Statistical features of CWND gradients
        features_dict['CWND-mean-grad'] = np.mean(cwnd_grad)
        features_dict['CWND-std-grad'] = np.std(cwnd_grad)
        features_dict['CWND-totalVar-grad'] = total_variation(cwnd_grad)
        features_dict['CWND-sum-grad'] = np.sum(cwnd_grad)
        features_dict['CWND-10th-p-grad'] = np.percentile(cwnd_grad, 10)
        features_dict['CWND-90th-p-grad'] = np.percentile(cwnd_grad, 90)

        # Statistical features of CWND second gradients
        features_dict['CWND-mean-secondGrad'] = np.mean(cwnd_second_grad)
        features_dict['CWND-std-secondGrad'] = np.std(cwnd_second_grad)
        features_dict['CWND-totalVar-secondGrad'] = total_variation(cwnd_second_grad)
        features_dict['CWND-sum-secondGrad'] = np.sum(cwnd_second_grad)
        features_dict['CWND-10th-p-secondGrad'] = np.percentile(cwnd_second_grad, 10)
        features_dict['CWND-90th-p-secondGrad'] = np.percentile(cwnd_second_grad, 90)

        # Find local minima and maxima in CWND values
        cwnd_local_max_indices, _ = find_peaks(cwnd_series)
        cwnd_local_min_indices, _ = find_peaks(-cwnd_series)
        cwnd_local_max_values = cwnd_series[cwnd_local_max_indices]
        cwnd_local_min_values = cwnd_series[cwnd_local_min_indices]

        # Number of local minima and maxima in CWND values
        features_dict['CWND-num-localMin'] = len(cwnd_local_min_indices)

        # Mean and standard deviation of local minima and maxima in CWND values
        features_dict['CWND-mean-localMin'] = np.mean(cwnd_local_min_values) if len(cwnd_local_min_values) > 0 else 0 #np.nan
        features_dict['CWND-std-localMin'] = np.std(cwnd_local_min_values) if len(cwnd_local_min_values) > 0 else 0 #np.nan
        features_dict['CWND-mean-localMax'] = np.mean(cwnd_local_max_values) if len(cwnd_local_max_values) > 0 else 0 #np.nan
        features_dict['CWND-std-localMax'] = np.std(cwnd_local_max_values) if len(cwnd_local_max_values) > 0 else 0 #np.nan

        features_dict['CWND-max-subvalue'] = max_subsequence_sum(cwnd_series)
        features_dict['CWND-min-subvalue'] = min_subsequence_sum(cwnd_series)

        # Find local minima and maxima in CWND gradients
        cwnd_grad_local_max_indices, _ = find_peaks(cwnd_grad)
        cwnd_grad_local_min_indices, _ = find_peaks(-cwnd_grad)
        cwnd_grad_local_max_values = cwnd_grad[cwnd_grad_local_max_indices]
        cwnd_grad_local_min_values = cwnd_grad[cwnd_grad_local_min_indices]

        # Number of local minima and maxima in CWND gradients
        features_dict['CWND-num-localMinGrad'] = len(cwnd_grad_local_min_indices)

        # Mean and standard deviation of local minima and maxima in CWND gradients
        features_dict['CWND-mean-localMinGrad'] = np.mean(cwnd_grad_local_min_values) if len(cwnd_grad_local_min_values) > 0 else 0 #np.nan
        features_dict['CWND-std-localMinGrad'] = np.std(cwnd_grad_local_min_values) if len(cwnd_grad_local_min_values) > 0 else 0 #np.nan
        features_dict['CWND-mean-localMaxGrad'] = np.mean(cwnd_grad_local_max_values) if len(cwnd_grad_local_max_values) > 0 else 0 #np.nan
        features_dict['CWND-std-localMaxGrad'] = np.std(cwnd_grad_local_max_values) if len(cwnd_grad_local_max_values) > 0 else 0 #np.nan

        # Find local minima in CWND second gradients
        cwnd_second_grad_local_min_indices, _ = find_peaks(-cwnd_second_grad)
        features_dict['CWND-num-localMinSecondGrad'] = len(cwnd_second_grad_local_min_indices)

        ### RTT Features Calculation ###

        # Compute gradients (first differences) and second gradients (second differences)
        srtt_grad = np.diff(srtt_series)
        srtt_second_grad = np.diff(srtt_grad)

        # Maximum and minimum gradients
        features_dict['RTT-max-grad'] = np.max(srtt_grad)
        
        features_dict['RTT-max-subgrad'] = max_subsequence_sum(srtt_grad)
        features_dict['RTT-min-subgrad'] = min_subsequence_sum(srtt_grad)
        

        # Maximum and minimum second gradients
        features_dict['RTT-max-secondGrad'] = np.max(srtt_second_grad)
        
        
        # Compute 'max-subsecondGrad' and 'min-subsecondGrad'
        features_dict['RTT-max-subsecondGrad'] = max_subsequence_sum(srtt_second_grad)
        features_dict['RTT-min-subsecondGrad'] = min_subsequence_sum(srtt_second_grad)

        # Statistical features of RTT values
        features_dict['RTT-mean-value'] = np.mean(srtt_series)
        features_dict['RTT-std-value'] = np.std(srtt_series)
        features_dict['RTT-totalVar-value'] = total_variation(srtt_series)
        features_dict['RTT-10th-p-value'] = np.percentile(srtt_series, 10)
        features_dict['RTT-90th-p-value'] = np.percentile(srtt_series, 90)

        # Statistical features of RTT gradients
        features_dict['RTT-mean-grad'] = np.mean(srtt_grad)
        features_dict['RTT-std-grad'] = np.std(srtt_grad)
        features_dict['RTT-totalVar-grad'] = total_variation(srtt_grad)
        features_dict['RTT-sum-grad'] = np.sum(srtt_grad)
        features_dict['RTT-10th-p-grad'] = np.percentile(srtt_grad, 10)
        features_dict['RTT-90th-p-grad'] = np.percentile(srtt_grad, 90)

        # Statistical features of RTT second gradients
        features_dict['RTT-mean-secondGrad'] = np.mean(srtt_second_grad)
        features_dict['RTT-std-secondGrad'] = np.std(srtt_second_grad)
        features_dict['RTT-totalVar-secondGrad'] = total_variation(srtt_second_grad)
        features_dict['RTT-sum-secondGrad'] = np.sum(srtt_second_grad)
        features_dict['RTT-10th-p-secondGrad'] = np.percentile(srtt_second_grad, 10)
        features_dict['RTT-90th-p-secondGrad'] = np.percentile(srtt_second_grad, 90)

        # Find local minima and maxima in RTT values
        srtt_local_max_indices, _ = find_peaks(srtt_series)
        srtt_local_min_indices, _ = find_peaks(-srtt_series)
        srtt_local_max_values = srtt_series[srtt_local_max_indices]
        srtt_local_min_values = srtt_series[srtt_local_min_indices]

        # Number of local minima and maxima in RTT values
        features_dict['RTT-num-localMin'] = len(srtt_local_min_indices)

        # Mean and standard deviation of local minima and maxima in RTT values
        features_dict['RTT-mean-localMin'] = np.mean(srtt_local_min_values) if len(srtt_local_min_values) > 0 else 0 #np.nan
        features_dict['RTT-std-localMin'] = np.std(srtt_local_min_values) if len(srtt_local_min_values) > 0 else 0 #np.nan
        features_dict['RTT-mean-localMax'] = np.mean(srtt_local_max_values) if len(srtt_local_max_values) > 0 else 0 #np.nan
        features_dict['RTT-std-localMax'] = np.std(srtt_local_max_values) if len(srtt_local_max_values) > 0 else 0 #np.nan

        # Compute 'max-subvalue' and 'min-subvalue'
        features_dict['RTT-max-subvalue'] = max_subsequence_sum(srtt_series)
        features_dict['RTT-min-subvalue'] = min_subsequence_sum(srtt_series)

        # Find local minima and maxima in RTT gradients
        srtt_grad_local_max_indices, _ = find_peaks(srtt_grad)
        srtt_grad_local_min_indices, _ = find_peaks(-srtt_grad)
        srtt_grad_local_max_values = srtt_grad[srtt_grad_local_max_indices]
        srtt_grad_local_min_values = srtt_grad[srtt_grad_local_min_indices]

        # Number of local minima and maxima in RTT gradients
        features_dict['RTT-num-localMinGrad'] = len(srtt_grad_local_min_indices)

        # Mean and standard deviation of local minima and maxima in RTT gradients
        features_dict['RTT-mean-localMinGrad'] = np.mean(srtt_grad_local_min_values) if len(srtt_grad_local_min_values) > 0 else 0 #np.nan
        features_dict['RTT-std-localMinGrad'] = np.std(srtt_grad_local_min_values) if len(srtt_grad_local_min_values) > 0 else 0 #np.nan
        features_dict['RTT-mean-localMaxGrad'] = np.mean(srtt_grad_local_max_values) if len(srtt_grad_local_max_values) > 0 else 0 #np.nan
        features_dict['RTT-std-localMaxGrad'] = np.std(srtt_grad_local_max_values) if len(srtt_grad_local_max_values) > 0 else 0 #np.nan

        # Find local minima in RTT second gradients
        srtt_second_grad_local_min_indices, _ = find_peaks(-srtt_second_grad)
        features_dict['RTT-num-localMinSecondGrad'] = len(srtt_second_grad_local_min_indices)

        # Append the features dictionary and label to the lists
        features_list.append(features_dict)
        labels.append(label)

# Create a DataFrame to store features and labels
data= pd.DataFrame(features_list)
data['Label'] = labels


# Prepare a list to store feature dictionaries and labels
features_list_test = []
labels_test = []

for duration in durations:
    # Create a list to store features for the current duration
    features_list_duration = []
    labels_duration = []

    # Loop through each experiment (column) in the CWND and RTT data
    for col in test_experiments:
        # Extract the CWND and RTT data series, dropping any NaN values
        
        df = pd.DataFrame({
        'time': time_data_test[col],
        'cwnd': cwnd_data_test[col],
        'srtt': srtt_data_test[col]})
        
        df = df.dropna()
        
        
        df = df[df['time'] <= duration]
        
        
        cwnd_series = df['cwnd'].values
        srtt_series = df['srtt'].values
        time_series = df['time'].values


        # Extract and convert label to numeric format
        experiment_name = col  # Column name is the experiment identifier

        # Filter data for the current duration
        indices = np.where(time_series <= duration)[0]
        cwnd_series = cwnd_series[indices]
        srtt_series = srtt_series[indices]

        label = 0 if "FIFO" in experiment_name else 1 if "pie_drop" in experiment_name else None

        # Proceed only if label is valid
        if label is not None:
            # Initialize a dictionary to hold features for this experiment
            features_dict = {}

            ### CWND Features Calculation ###

            # Compute gradients (first differences) and second gradients (second differences)
            cwnd_grad = np.gradient(cwnd_series)
            cwnd_second_grad = np.gradient(cwnd_grad)

            # Maximum and minimum gradients
            features_dict['CWND-max-grad'] = np.max(cwnd_grad)
            
            
            features_dict['CWND-max-subgrad'] = max_subsequence_sum(cwnd_grad)
            features_dict['CWND-min-subgrad'] = min_subsequence_sum(cwnd_grad)

            # Maximum and minimum second gradients
            features_dict['CWND-max-secondGrad'] = np.max(cwnd_second_grad)
            
            features_dict['CWND-max-subsecondGrad'] = max_subsequence_sum(cwnd_second_grad)
            features_dict['CWND-min-subsecondGrad'] = min_subsequence_sum(cwnd_second_grad)

            # Statistical features of CWND values
            features_dict['CWND-mean-value'] = np.mean(cwnd_series)
            features_dict['CWND-std-value'] = np.std(cwnd_series)
            features_dict['CWND-totalVar-value'] = total_variation(cwnd_series)
            features_dict['CWND-10th-p-value'] = np.percentile(cwnd_series, 10)
            features_dict['CWND-90th-p-value'] = np.percentile(cwnd_series, 90)

            # Statistical features of CWND gradients
            features_dict['CWND-mean-grad'] = np.mean(cwnd_grad)
            features_dict['CWND-std-grad'] = np.std(cwnd_grad)
            features_dict['CWND-totalVar-grad'] = total_variation(cwnd_grad)
            features_dict['CWND-sum-grad'] = np.sum(cwnd_grad)
            features_dict['CWND-10th-p-grad'] = np.percentile(cwnd_grad, 10)
            features_dict['CWND-90th-p-grad'] = np.percentile(cwnd_grad, 90)

            # Statistical features of CWND second gradients
            features_dict['CWND-mean-secondGrad'] = np.mean(cwnd_second_grad)
            features_dict['CWND-std-secondGrad'] = np.std(cwnd_second_grad)
            features_dict['CWND-totalVar-secondGrad'] = total_variation(cwnd_second_grad)
            features_dict['CWND-sum-secondGrad'] = np.sum(cwnd_second_grad)
            features_dict['CWND-10th-p-secondGrad'] = np.percentile(cwnd_second_grad, 10)
            features_dict['CWND-90th-p-secondGrad'] = np.percentile(cwnd_second_grad, 90)

            # Find local minima and maxima in CWND values
            cwnd_local_max_indices, _ = find_peaks(cwnd_series)
            cwnd_local_min_indices, _ = find_peaks(-cwnd_series)
            cwnd_local_max_values = cwnd_series[cwnd_local_max_indices]
            cwnd_local_min_values = cwnd_series[cwnd_local_min_indices]

            # Number of local minima and maxima in CWND values
            features_dict['CWND-num-localMin'] = len(cwnd_local_min_indices)

            # Mean and standard deviation of local minima and maxima in CWND values
            features_dict['CWND-mean-localMin'] = np.mean(cwnd_local_min_values) if len(cwnd_local_min_values) > 0 else 0 #np.nan
            features_dict['CWND-std-localMin'] = np.std(cwnd_local_min_values) if len(cwnd_local_min_values) > 0 else 0 #np.nan
            features_dict['CWND-mean-localMax'] = np.mean(cwnd_local_max_values) if len(cwnd_local_max_values) > 0 else 0 #np.nan
            features_dict['CWND-std-localMax'] = np.std(cwnd_local_max_values) if len(cwnd_local_max_values) > 0 else 0 #np.nan

            features_dict['CWND-max-subvalue'] = max_subsequence_sum(cwnd_series)
            features_dict['CWND-min-subvalue'] = min_subsequence_sum(cwnd_series)

            # Find local minima and maxima in CWND gradients
            cwnd_grad_local_max_indices, _ = find_peaks(cwnd_grad)
            cwnd_grad_local_min_indices, _ = find_peaks(-cwnd_grad)
            cwnd_grad_local_max_values = cwnd_grad[cwnd_grad_local_max_indices]
            cwnd_grad_local_min_values = cwnd_grad[cwnd_grad_local_min_indices]

            # Number of local minima and maxima in CWND gradients
            features_dict['CWND-num-localMinGrad'] = len(cwnd_grad_local_min_indices)

            # Mean and standard deviation of local minima and maxima in CWND gradients
            features_dict['CWND-mean-localMinGrad'] = np.mean(cwnd_grad_local_min_values) if len(cwnd_grad_local_min_values) > 0 else 0 #np.nan
            features_dict['CWND-std-localMinGrad'] = np.std(cwnd_grad_local_min_values) if len(cwnd_grad_local_min_values) > 0 else 0 #np.nan
            features_dict['CWND-mean-localMaxGrad'] = np.mean(cwnd_grad_local_max_values) if len(cwnd_grad_local_max_values) > 0 else 0 #np.nan
            features_dict['CWND-std-localMaxGrad'] = np.std(cwnd_grad_local_max_values) if len(cwnd_grad_local_max_values) > 0 else 0 #np.nan

            # Find local minima in CWND second gradients
            cwnd_second_grad_local_min_indices, _ = find_peaks(-cwnd_second_grad)
            features_dict['CWND-num-localMinSecondGrad'] = len(cwnd_second_grad_local_min_indices)

            ### RTT Features Calculation ###

            # Compute gradients (first differences) and second gradients (second differences)
            srtt_grad = np.diff(srtt_series)
            srtt_second_grad = np.diff(srtt_grad)

            # Maximum and minimum gradients
            features_dict['RTT-max-grad'] = np.max(srtt_grad)
            
            features_dict['RTT-max-subgrad'] = max_subsequence_sum(srtt_grad)
            features_dict['RTT-min-subgrad'] = min_subsequence_sum(srtt_grad)
            

            # Maximum and minimum second gradients
            features_dict['RTT-max-secondGrad'] = np.max(srtt_second_grad)
            
            
            # Compute 'max-subsecondGrad' and 'min-subsecondGrad'
            features_dict['RTT-max-subsecondGrad'] = max_subsequence_sum(srtt_second_grad)
            features_dict['RTT-min-subsecondGrad'] = min_subsequence_sum(srtt_second_grad)

            # Statistical features of RTT values
            features_dict['RTT-mean-value'] = np.mean(srtt_series)
            features_dict['RTT-std-value'] = np.std(srtt_series)
            features_dict['RTT-totalVar-value'] = total_variation(srtt_series)
            features_dict['RTT-10th-p-value'] = np.percentile(srtt_series, 10)
            features_dict['RTT-90th-p-value'] = np.percentile(srtt_series, 90)

            # Statistical features of RTT gradients
            features_dict['RTT-mean-grad'] = np.mean(srtt_grad)
            features_dict['RTT-std-grad'] = np.std(srtt_grad)
            features_dict['RTT-totalVar-grad'] = total_variation(srtt_grad)
            features_dict['RTT-sum-grad'] = np.sum(srtt_grad)
            features_dict['RTT-10th-p-grad'] = np.percentile(srtt_grad, 10)
            features_dict['RTT-90th-p-grad'] = np.percentile(srtt_grad, 90)

            # Statistical features of RTT second gradients
            features_dict['RTT-mean-secondGrad'] = np.mean(srtt_second_grad)
            features_dict['RTT-std-secondGrad'] = np.std(srtt_second_grad)
            features_dict['RTT-totalVar-secondGrad'] = total_variation(srtt_second_grad)
            features_dict['RTT-sum-secondGrad'] = np.sum(srtt_second_grad)
            features_dict['RTT-10th-p-secondGrad'] = np.percentile(srtt_second_grad, 10)
            features_dict['RTT-90th-p-secondGrad'] = np.percentile(srtt_second_grad, 90)

            # Find local minima and maxima in RTT values
            srtt_local_max_indices, _ = find_peaks(srtt_series)
            srtt_local_min_indices, _ = find_peaks(-srtt_series)
            srtt_local_max_values = srtt_series[srtt_local_max_indices]
            srtt_local_min_values = srtt_series[srtt_local_min_indices]

            # Number of local minima and maxima in RTT values
            features_dict['RTT-num-localMin'] = len(srtt_local_min_indices)

            # Mean and standard deviation of local minima and maxima in RTT values
            features_dict['RTT-mean-localMin'] = np.mean(srtt_local_min_values) if len(srtt_local_min_values) > 0 else 0 #np.nan
            features_dict['RTT-std-localMin'] = np.std(srtt_local_min_values) if len(srtt_local_min_values) > 0 else 0 #np.nan
            features_dict['RTT-mean-localMax'] = np.mean(srtt_local_max_values) if len(srtt_local_max_values) > 0 else 0 #np.nan
            features_dict['RTT-std-localMax'] = np.std(srtt_local_max_values) if len(srtt_local_max_values) > 0 else 0 #np.nan

            # Compute 'max-subvalue' and 'min-subvalue'
            features_dict['RTT-max-subvalue'] = max_subsequence_sum(srtt_series)
            features_dict['RTT-min-subvalue'] = min_subsequence_sum(srtt_series)

            # Find local minima and maxima in RTT gradients
            srtt_grad_local_max_indices, _ = find_peaks(srtt_grad)
            srtt_grad_local_min_indices, _ = find_peaks(-srtt_grad)
            srtt_grad_local_max_values = srtt_grad[srtt_grad_local_max_indices]
            srtt_grad_local_min_values = srtt_grad[srtt_grad_local_min_indices]

            # Number of local minima and maxima in RTT gradients
            features_dict['RTT-num-localMinGrad'] = len(srtt_grad_local_min_indices)

            # Mean and standard deviation of local minima and maxima in RTT gradients
            features_dict['RTT-mean-localMinGrad'] = np.mean(srtt_grad_local_min_values) if len(srtt_grad_local_min_values) > 0 else 0 #np.nan
            features_dict['RTT-std-localMinGrad'] = np.std(srtt_grad_local_min_values) if len(srtt_grad_local_min_values) > 0 else 0 #np.nan
            features_dict['RTT-mean-localMaxGrad'] = np.mean(srtt_grad_local_max_values) if len(srtt_grad_local_max_values) > 0 else 0 #np.nan
            features_dict['RTT-std-localMaxGrad'] = np.std(srtt_grad_local_max_values) if len(srtt_grad_local_max_values) > 0 else 0 #np.nan

            # Find local minima in RTT second gradients
            srtt_second_grad_local_min_indices, _ = find_peaks(-srtt_second_grad)
            features_dict['RTT-num-localMinSecondGrad'] = len(srtt_second_grad_local_min_indices)

            # Append the features dictionary and label to the lists
            features_list_duration.append(features_dict)
            labels_duration.append(label)


    # Convert to DataFrame for the current duration
    test_set = pd.DataFrame(features_list_duration)
    test_set['Label'] = labels_duration
    test_sets[duration] = test_set

# %%
features_list_test = []
labels_test = []


# Loop through each experiment (column) in the CWND and RTT data
for col in test_experiments:
    # Extract the CWND and RTT data series, dropping any NaN values
    
    df = pd.DataFrame({
    'time': time_data_test[col],
    'cwnd': cwnd_data_test[col],
    'srtt': srtt_data_test[col]})
    
    df = df.dropna()
    
    
    cwnd_series = df['cwnd'].values
    srtt_series = df['srtt'].values
    time_series = df['time'].values

    # Extract and convert label to numeric format
    experiment_name = col  # Column name is the experiment identifier

    label = 0 if "FIFO" in experiment_name else 1 if "pie_drop" in experiment_name else None

    # Proceed only if label is valid
    if label is not None:
        # Initialize a dictionary to hold features for this experiment
        features_dict = {}

        ### CWND Features Calculation ###

        # Compute gradients (first differences) and second gradients (second differences)
        cwnd_grad = np.gradient(cwnd_series)
        cwnd_second_grad = np.gradient(cwnd_grad)

        # Maximum and minimum gradients
        features_dict['CWND-max-grad'] = np.max(cwnd_grad)
        
        
        features_dict['CWND-max-subgrad'] = max_subsequence_sum(cwnd_grad)
        features_dict['CWND-min-subgrad'] = min_subsequence_sum(cwnd_grad)

        # Maximum and minimum second gradients
        features_dict['CWND-max-secondGrad'] = np.max(cwnd_second_grad)
        
        features_dict['CWND-max-subsecondGrad'] = max_subsequence_sum(cwnd_second_grad)
        features_dict['CWND-min-subsecondGrad'] = min_subsequence_sum(cwnd_second_grad)

        # Statistical features of CWND values
        features_dict['CWND-mean-value'] = np.mean(cwnd_series)
        features_dict['CWND-std-value'] = np.std(cwnd_series)
        features_dict['CWND-totalVar-value'] = total_variation(cwnd_series)
        features_dict['CWND-10th-p-value'] = np.percentile(cwnd_series, 10)
        features_dict['CWND-90th-p-value'] = np.percentile(cwnd_series, 90)

        # Statistical features of CWND gradients
        features_dict['CWND-mean-grad'] = np.mean(cwnd_grad)
        features_dict['CWND-std-grad'] = np.std(cwnd_grad)
        features_dict['CWND-totalVar-grad'] = total_variation(cwnd_grad)
        features_dict['CWND-sum-grad'] = np.sum(cwnd_grad)
        features_dict['CWND-10th-p-grad'] = np.percentile(cwnd_grad, 10)
        features_dict['CWND-90th-p-grad'] = np.percentile(cwnd_grad, 90)

        # Statistical features of CWND second gradients
        features_dict['CWND-mean-secondGrad'] = np.mean(cwnd_second_grad)
        features_dict['CWND-std-secondGrad'] = np.std(cwnd_second_grad)
        features_dict['CWND-totalVar-secondGrad'] = total_variation(cwnd_second_grad)
        features_dict['CWND-sum-secondGrad'] = np.sum(cwnd_second_grad)
        features_dict['CWND-10th-p-secondGrad'] = np.percentile(cwnd_second_grad, 10)
        features_dict['CWND-90th-p-secondGrad'] = np.percentile(cwnd_second_grad, 90)

        # Find local minima and maxima in CWND values
        cwnd_local_max_indices, _ = find_peaks(cwnd_series)
        cwnd_local_min_indices, _ = find_peaks(-cwnd_series)
        cwnd_local_max_values = cwnd_series[cwnd_local_max_indices]
        cwnd_local_min_values = cwnd_series[cwnd_local_min_indices]

        # Number of local minima and maxima in CWND values
        features_dict['CWND-num-localMin'] = len(cwnd_local_min_indices)

        # Mean and standard deviation of local minima and maxima in CWND values
        features_dict['CWND-mean-localMin'] = np.mean(cwnd_local_min_values) if len(cwnd_local_min_values) > 0 else 0 #np.nan
        features_dict['CWND-std-localMin'] = np.std(cwnd_local_min_values) if len(cwnd_local_min_values) > 0 else 0 #np.nan
        features_dict['CWND-mean-localMax'] = np.mean(cwnd_local_max_values) if len(cwnd_local_max_values) > 0 else 0 #np.nan
        features_dict['CWND-std-localMax'] = np.std(cwnd_local_max_values) if len(cwnd_local_max_values) > 0 else 0 #np.nan

        features_dict['CWND-max-subvalue'] = max_subsequence_sum(cwnd_series)
        features_dict['CWND-min-subvalue'] = min_subsequence_sum(cwnd_series)

        # Find local minima and maxima in CWND gradients
        cwnd_grad_local_max_indices, _ = find_peaks(cwnd_grad)
        cwnd_grad_local_min_indices, _ = find_peaks(-cwnd_grad)
        cwnd_grad_local_max_values = cwnd_grad[cwnd_grad_local_max_indices]
        cwnd_grad_local_min_values = cwnd_grad[cwnd_grad_local_min_indices]

        # Number of local minima and maxima in CWND gradients
        features_dict['CWND-num-localMinGrad'] = len(cwnd_grad_local_min_indices)

        # Mean and standard deviation of local minima and maxima in CWND gradients
        features_dict['CWND-mean-localMinGrad'] = np.mean(cwnd_grad_local_min_values) if len(cwnd_grad_local_min_values) > 0 else 0 #np.nan
        features_dict['CWND-std-localMinGrad'] = np.std(cwnd_grad_local_min_values) if len(cwnd_grad_local_min_values) > 0 else 0 #np.nan
        features_dict['CWND-mean-localMaxGrad'] = np.mean(cwnd_grad_local_max_values) if len(cwnd_grad_local_max_values) > 0 else 0 #np.nan
        features_dict['CWND-std-localMaxGrad'] = np.std(cwnd_grad_local_max_values) if len(cwnd_grad_local_max_values) > 0 else 0 #np.nan

        # Find local minima in CWND second gradients
        cwnd_second_grad_local_min_indices, _ = find_peaks(-cwnd_second_grad)
        features_dict['CWND-num-localMinSecondGrad'] = len(cwnd_second_grad_local_min_indices)

        ### RTT Features Calculation ###

        # Compute gradients (first differences) and second gradients (second differences)
        srtt_grad = np.diff(srtt_series)
        srtt_second_grad = np.diff(srtt_grad)

        # Maximum and minimum gradients
        features_dict['RTT-max-grad'] = np.max(srtt_grad)
        
        
        features_dict['RTT-max-subgrad'] = max_subsequence_sum(srtt_grad)
        features_dict['RTT-min-subgrad'] = min_subsequence_sum(srtt_grad)
        

        # Maximum and minimum second gradients
        features_dict['RTT-max-secondGrad'] = np.max(srtt_second_grad)
        
        
        # Compute 'max-subsecondGrad' and 'min-subsecondGrad'
        features_dict['RTT-max-subsecondGrad'] = max_subsequence_sum(srtt_second_grad)
        features_dict['RTT-min-subsecondGrad'] = min_subsequence_sum(srtt_second_grad)

        # Statistical features of RTT values
        features_dict['RTT-mean-value'] = np.mean(srtt_series)
        features_dict['RTT-std-value'] = np.std(srtt_series)
        features_dict['RTT-totalVar-value'] = total_variation(srtt_series)
        features_dict['RTT-10th-p-value'] = np.percentile(srtt_series, 10)
        features_dict['RTT-90th-p-value'] = np.percentile(srtt_series, 90)

        # Statistical features of RTT gradients
        features_dict['RTT-mean-grad'] = np.mean(srtt_grad)
        features_dict['RTT-std-grad'] = np.std(srtt_grad)
        features_dict['RTT-totalVar-grad'] = total_variation(srtt_grad)
        features_dict['RTT-sum-grad'] = np.sum(srtt_grad)
        features_dict['RTT-10th-p-grad'] = np.percentile(srtt_grad, 10)
        features_dict['RTT-90th-p-grad'] = np.percentile(srtt_grad, 90)

        # Statistical features of RTT second gradients
        features_dict['RTT-mean-secondGrad'] = np.mean(srtt_second_grad)
        features_dict['RTT-std-secondGrad'] = np.std(srtt_second_grad)
        features_dict['RTT-totalVar-secondGrad'] = total_variation(srtt_second_grad)
        features_dict['RTT-sum-secondGrad'] = np.sum(srtt_second_grad)
        features_dict['RTT-10th-p-secondGrad'] = np.percentile(srtt_second_grad, 10)
        features_dict['RTT-90th-p-secondGrad'] = np.percentile(srtt_second_grad, 90)

        # Find local minima and maxima in RTT values
        srtt_local_max_indices, _ = find_peaks(srtt_series)
        srtt_local_min_indices, _ = find_peaks(-srtt_series)
        srtt_local_max_values = srtt_series[srtt_local_max_indices]
        srtt_local_min_values = srtt_series[srtt_local_min_indices]

        # Number of local minima and maxima in RTT values
        features_dict['RTT-num-localMin'] = len(srtt_local_min_indices)

        # Mean and standard deviation of local minima and maxima in RTT values
        features_dict['RTT-mean-localMin'] = np.mean(srtt_local_min_values) if len(srtt_local_min_values) > 0 else np.nan
        features_dict['RTT-std-localMin'] = np.std(srtt_local_min_values) if len(srtt_local_min_values) > 0 else np.nan
        features_dict['RTT-mean-localMax'] = np.mean(srtt_local_max_values) if len(srtt_local_max_values) > 0 else np.nan
        features_dict['RTT-std-localMax'] = np.std(srtt_local_max_values) if len(srtt_local_max_values) > 0 else np.nan

        # Compute 'max-subvalue' and 'min-subvalue'
        features_dict['RTT-max-subvalue'] = max_subsequence_sum(srtt_series)
        features_dict['RTT-min-subvalue'] = min_subsequence_sum(srtt_series)

        # Find local minima and maxima in RTT gradients
        srtt_grad_local_max_indices, _ = find_peaks(srtt_grad)
        srtt_grad_local_min_indices, _ = find_peaks(-srtt_grad)
        srtt_grad_local_max_values = srtt_grad[srtt_grad_local_max_indices]
        srtt_grad_local_min_values = srtt_grad[srtt_grad_local_min_indices]

        # Number of local minima and maxima in RTT gradients
        features_dict['RTT-num-localMinGrad'] = len(srtt_grad_local_min_indices)

        # Mean and standard deviation of local minima and maxima in RTT gradients
        features_dict['RTT-mean-localMinGrad'] = np.mean(srtt_grad_local_min_values) if len(srtt_grad_local_min_values) > 0 else np.nan
        features_dict['RTT-std-localMinGrad'] = np.std(srtt_grad_local_min_values) if len(srtt_grad_local_min_values) > 0 else np.nan
        features_dict['RTT-mean-localMaxGrad'] = np.mean(srtt_grad_local_max_values) if len(srtt_grad_local_max_values) > 0 else np.nan
        features_dict['RTT-std-localMaxGrad'] = np.std(srtt_grad_local_max_values) if len(srtt_grad_local_max_values) > 0 else np.nan

        # Find local minima in RTT second gradients
        srtt_second_grad_local_min_indices, _ = find_peaks(-srtt_second_grad)
        features_dict['RTT-num-localMinSecondGrad'] = len(srtt_second_grad_local_min_indices)

        # Append the features dictionary and label to the lists
        features_list_test.append(features_dict)
        labels_test.append(label)


# Create a DataFrame to store features and labels
data_test= pd.DataFrame(features_list_test)
data_test['Label'] = labels_test

#%%

## Split the data

# Extract features and labels from your DataFrame
X = data.iloc[:, :-1].values  # First 72 columns as features
y = data.iloc[:, -1].values   # Last column as labels


# # Extract features and labels from your DataFrame

X_test_orj = data_test.iloc[:, :-1].values  # First 72 columns as features
y_test_orj = data_test.iloc[:, -1].values   # Last column as labels


X_tests={}
y_tests={}

# Assuming test_sets is a dictionary where each key is a duration and each value is a DataFrame
for duration, test_set in test_sets.items():
    # Extract features (excluding the 'Label' column)
    
    X_test=test_set.iloc[:, :-1].values
    
    y_test=test_set.iloc[:, -1].values
        
    X_tests[duration] = X_test
    y_tests[duration] = y_test

X_initial_train = X
y_initial_train = y

# %% Random Forest


X_rf_train, X_rf_val, y_rf_train, y_rf_val = train_test_split(X_initial_train, y_initial_train , test_size=0.1, random_state=42)  # 75% train, 25% temp

# Define the parameter grid for n_estimators
param_grid = {
    'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Values to test
    'random_state': [20],  # Keep reproducibility
}



rf_model = RandomForestClassifier()


# Use GridSearchCV for cross-validation
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='accuracy',  # Metric to optimize
    cv=10,               # Number of cross-validation folds
    n_jobs=-1,           # Use all processors
    verbose=1            # Show progress
)

# Perform Grid Search on the training set
grid_search.fit(X_rf_train, y_rf_train)



# Get the best estimator and its parameters
best_rf_model = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"Best Parameters: {best_params}")

# Evaluate on the validation set
y_val_pred = best_rf_model.predict(X_rf_val)
val_accuracy = accuracy_score(y_rf_val, y_val_pred)
print(f"Validation Accuracy with Best Model: {val_accuracy:.4f}")

# Evaluate on the test set
y_test_pred = best_rf_model.predict(X_test_orj)
test_accuracy = accuracy_score(y_test_orj, y_test_pred)
print(f"Test Accuracy with Best Model: {test_accuracy:.4f}")


# Classification Report
print("Classification Report:")
print(classification_report(y_test_orj, y_test_pred))


# Metrics
accuracy = accuracy_score(y_test_orj, y_test_pred)
f1 = f1_score(y_test_orj, y_test_pred, average='weighted')  # Weighted F1-score

# Confusion Matrix
cm = confusion_matrix(y_test_orj, y_test_pred)

# Normalize confusion matrix
cm_normalized = cm / cm.sum(axis=1, keepdims=True) * 100

# Confusion Matrix Plot
plt.figure(figsize=(6, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False,
            xticklabels=['Tail-drop', 'PIE'], yticklabels=['Tail-drop', 'PIE'])

# Add Title with Metrics
plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2f}, F1 Score: {f1:.2f})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Save the plot
plt.savefig("confusion_matrix_with_metrics_RF_2.png", dpi=300, bbox_inches="tight", format="png")

plt.show()


# Get probabilities for the positive class
y_test_proba = best_rf_model.predict_proba(X_test_orj)[:, 1]

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test_orj, y_test_proba)

# Calculate AUC score
auc_score = roc_auc_score(y_test_orj, y_test_proba)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})", color="blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.savefig("roc_curve_random_forest.png", dpi=300, bbox_inches="tight", format="png")
plt.show()

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test_orj, y_test_proba)

# Calculate Average Precision (AP) score
ap_score = average_precision_score(y_test_orj, y_test_proba)

# Plot PR curve
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, label=f"PR Curve (AP = {ap_score:.2f})", color="green")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.savefig("pr_curve_random_forest.png", dpi=300, bbox_inches="tight", format="png")
plt.show()


# Feature Importances
feature_importances = best_rf_model.feature_importances_
feature_names = data.columns[:-1]  # Exclude the label column
importances = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
print("\nTop 10 Feature Importances:")
for feature, importance in importances[:10]:
    print(f"{feature}: {importance:.4f}")

print("\nBottom 10 Feature Importances:")
for feature, importance in importances[-10:]:
    print(f"{feature}: {importance:.4f}")


# Optional: Plot feature importances

plt.figure(figsize=(10, 6))
plt.barh([x[0] for x in importances[:10]], [x[1] for x in importances[:10]])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.savefig("feature-importance.png", dpi=300, bbox_inches="tight", format="png")

plt.show()

plt.figure(figsize=(10, 6))
plt.barh([x[0] for x in importances[-10:]], [x[1] for x in importances[-10:]])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Bottom 10 Feature Importances')
plt.gca().invert_yaxis()
plt.savefig("feature-importance-bottom-10.png", dpi=300, bbox_inches="tight", format="png")

plt.show()




# %% Neural Network Model

class SimpleNN(nn.Module):
    def __init__(self, nin, nh, nout):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(nin, nh)     # Hidden layer 
        self.output = nn.Linear(nh, nout)    # Output layer 

    def forward(self, x):
        x = torch.relu(self.hidden(x))       # ReLU activation for hidden layer
        x = torch.sigmoid(self.output(x))    # Sigmoid activation for binary classification
        return x

def train_one_epoch(data_loader):
    
    model.train(True)

    running_loss = 0
    running_correct = 0
    running_samples = 0



    for i, data in enumerate(data_loader):
        # Every data instance is an X, y pair
        X, y = data
        y = y.unsqueeze(1) # make it the same shape as predictions

        # Zero gradients for every batch!
        optimizer.zero_grad()

        # Forward pass makes predictions for this batch
        y_pred = model(X)

        # Compute the loss and its gradients
        loss = loss_fn(y_pred, y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
   
        # Update running loss, accuracy, and number of samples
        running_correct += ( (y_pred >= 0.5) == y).sum().item()
        running_samples += y.size(0)
        running_loss    += loss*y.size(0)


    return float(running_loss / running_samples), float(running_correct / running_samples)



def eval_model(data_loader):

    running_loss    = 0
    running_correct = 0
    running_samples = 0

    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation for faster computation/reduced memory
    with torch.no_grad():

      for i, data in enumerate(data_loader):
          # Every data instance is an X, y pair
          X, y = data
          y = y.unsqueeze(1) # make it the same shape as predictions

          # Forward pass makes predictions for this batch
          y_pred = model(X)



          # Inside eval_model, just before computing the loss
          #epsilon = 1e-7
          #y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
          #print(y_pred)
          # Compute the loss
          loss = loss_fn(y_pred, y)

          # Update running loss, accuracy, and number of samples
          running_correct += ( (y_pred.data >= 0.5) == y).sum().item()
          running_samples += y.size(0)
          running_loss    += loss*y.size(0)

    # return average loss, average accuracy
    return float(running_loss/running_samples), float(running_correct/running_samples)

# %% Hyper-parameter tuning

hidden_layer_nn = [4, 8, 16]  # Focus on a small range of neurons per layer
alpha_options = [1e-5, 1e-4, 1e-3, 1e-2]  # Remove extreme values
batch_size_options = [12, 24, 36]  # Commonly used batch sizes
epoch_options = [50, 100, 200]  # Limit to 3 options



# Define dimensions for your model
nin = 72    # Number of input features 
nout = 1  

# Number of folds
nfold = 10
# Create a k-fold object

best_hyperparams = None
best_val_accuracy = 0.0

results = []

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


for nh in hidden_layer_nn:
    for batch_size in batch_size_options:
        for epoch_number in epoch_options:
            for alpha in alpha_options:
                
                print(f"\nEvaluating combination: Neurons={nh}, Batch_size={batch_size}, epoch={epoch_number}, alpha={alpha}")
                
                val_accuracies = []
                val_losses = []
                
                train_accuracies = []
                train_losses = []
                
                kf = KFold(n_splits=nfold, shuffle=True, random_state=42)


                fold = 1

                for i, idx_split in enumerate(kf.split(X_initial_train)):
                    
                                    
                    train_index, val_index = idx_split
                    
                
                    # Split the data for this fold
                    X_train_fold, X_val_fold = X_initial_train[train_index], X_initial_train[val_index]
                    y_train_fold, y_val_fold = y_initial_train[train_index], y_initial_train[val_index]
                    
                    
                    # Normalize the features
                    scaler = StandardScaler()
                    X_train_fold = scaler.fit_transform(X_train_fold)
                    X_val_fold = scaler.transform(X_val_fold)
                
                
                    train_loader = DataLoader(
                        TensorDataset(torch.Tensor(X_train_fold), torch.Tensor(y_train_fold)),
                        batch_size = batch_size, shuffle=True)
                
                
                    val_loader = DataLoader(
                        TensorDataset(torch.Tensor(X_val_fold), torch.Tensor(y_val_fold)),
                        batch_size = batch_size, shuffle=False)
                
                
                
                    model = SimpleNN(nin, nh, nout)
                    loss_fn = nn.BCELoss()

                    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=alpha)
                
                    
                    # Train and validate for n_epochs
                    fold_metrics = {'train_losses': [], 'train_accuracies': [], 'val_losses': [], 'val_accuracies': []}
                
                
                    for epoch in range(epoch_number):
                
                        train_loss, train_accuracy = train_one_epoch(train_loader)
                        fold_metrics['train_losses'].append(train_loss)
                        fold_metrics['train_accuracies'].append(train_accuracy)
                
                        # Evaluate on the validation data
                        val_loss, val_accuracy = eval_model(val_loader)
                        fold_metrics['val_losses'].append(val_loss)
                        fold_metrics['val_accuracies'].append(val_accuracy)
                
                
                    train_accuracies.append(train_accuracy)
                    train_losses.append(train_loss)
                    val_accuracies.append(val_accuracy)
                    val_losses.append(val_loss)

                    #print(f"Fold {fold}, Epoch {epoch+1}/{epoch_number} - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f} - Val_Loss: {val_loss:.4f} - Val_Accuracy: {val_accuracy:.4f}")
                
                    fold += 1
                
                
                # Compute average metrics across folds
                avg_train_accuracy = np.mean(train_accuracies)
                avg_train_loss = np.mean(train_losses)
                avg_val_accuracy = np.mean(val_accuracies)
                avg_val_loss = np.mean(val_losses)
                
                print(f"Average Train Accuracy: {avg_train_accuracy:.4f}, Average Train Loss: {avg_train_loss:.4f}")
                print(f"Average Val Accuracy: {avg_val_accuracy:.4f}, Average Val Loss: {avg_val_loss:.4f}")    
                
                
                # Store results for this parameter combination
                results.append({
                    'neurons': nh,
                    'batch_size': batch_size,
                    'epochs': epoch_number,
                    'alpha': alpha,
                    'avg_train_accuracy': avg_train_accuracy,
                    'avg_train_loss': avg_train_loss,
                    'avg_val_accuracy': avg_val_accuracy,
                    'avg_val_loss': avg_val_loss
                })
# %%
import pandas as pd

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

best_result = results_df.sort_values(by='avg_val_accuracy', ascending=False).iloc[0]

print("Best Hyperparameter Combination:")
print(best_result)



# %%  Final model training


# Define dimensions for your model
nin = 72    # Number of input features 
nh = int(best_result['neurons'])   # Number of neurons in the hidden layer
nout = 1    # Output (1 output unit for binary classification)


# Instantiate the model
model = SimpleNN(nin, nh, nout)
print(model)


# Initialize model, loss function, and optimizer with LBFGS and weight decay
loss_fn = nn.BCELoss()
alpha = best_result['alpha']  # Regularization parameter for L2 weight decay

#optimizer = optim.LBFGS(model.parameters(), lr=0.001, line_search_fn="strong_wolfe")
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=alpha)


batch_size = int(best_result['batch_size'])
n_epochs = int(best_result['epochs'])

metrics = {'train_losses': [], 'train_accuracies': [], 'val_losses': [], 'val_accuracies': []}


# Normalize the features
scaler = StandardScaler()
X_initial_train = scaler.fit_transform(X_initial_train)
X_test_orj=scaler.transform(X_test_orj)


for key in X_tests:
    X_tests[key] =  scaler.transform(X_tests[key]) # Reshape to 2D
    
    
train_loader = DataLoader(
    TensorDataset(torch.Tensor(X_initial_train), torch.Tensor(y_initial_train)),
    batch_size = batch_size, shuffle=True)
    
metrics = {'train_losses': [], 'train_accuracies': [], 'val_losses': [], 'val_accuracies': []}

for epoch in range(n_epochs):
    # Train on the full training dataset
    #train_loss, train_accuracy = train_one_epoch(X_train_tensor, y_train_tensor)
    train_loss, train_accuracy = train_one_epoch(train_loader)
    metrics['train_losses'].append(train_loss)
    metrics['train_accuracies'].append(train_accuracy)

    print(f'Epoch {epoch+1}/{n_epochs} - Loss: {train_loss:.4f} - Accuracy: {train_accuracy:.4f}')   

    
    
    
# %% Testing


test_loader = DataLoader(
    TensorDataset(torch.Tensor(X_test_orj), torch.Tensor(y_test_orj)),
    batch_size = batch_size, shuffle=False)

# Assuming test_loss, test_accuracy, y_pred_all, y_all are defined as in your code
test_loss, test_accuracy = eval_model(test_loader)
print(f'Accuracy on the test set: {test_accuracy:.4f}')

y_pred_all = []
y_all = []
with torch.no_grad():
    for X, y in test_loader:
        y_pred = model(X)
        y_pred_label = (y_pred.data > 0.5).float()
        y_pred_all.extend(y_pred_label.numpy())
        y_all.extend(y.numpy())

# Compute normalized confusion matrix and convert to percentage
cm = confusion_matrix(y_all, y_pred_all, normalize='true') * 100

# Format each cell as a string with a '%' symbol for annotations
cm_percentage = np.array([["{:.2f}%".format(value) for value in row] for row in cm])

# Metrics
f1 = f1_score(y_all, y_pred_all, average='weighted')  # Weighted F1-score

# Plot the normalized confusion matrix
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=cm_percentage, fmt="", cmap="Blues", cbar=False,
            xticklabels=['Tail-drop', 'PIE'], yticklabels=['Tail-drop', 'PIE'])

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title(f"Confusion Matrix (Accuracy: {test_accuracy:.2f}, F1 Score: {f1:.2f})")

plt.savefig("confusion_matrix_high_quality_NN.png", dpi=300, bbox_inches="tight", format="png")


plt.show()
# %% Extension

# Initialize lists to store metrics
durations_list = []
accuracies = []
f1_scores = []

# Loop through each duration and its corresponding test set
for duration in durations:
    print(f"Evaluating test set for duration: {duration} seconds")


    # Extract features and labels for this test set
    X_test_duration = X_tests[duration]
    y_test_duration = y_tests[duration]

    # Create a DataLoader for the test set
    test_loader_duration = DataLoader(
        TensorDataset(torch.Tensor(X_test_duration), torch.Tensor(y_test_duration)),
        batch_size=batch_size, shuffle=False
    )

    # Evaluate the model on this test set
    test_loss, test_accuracy = eval_model(test_loader_duration)
    print(f'Accuracy on the {duration}s test set: {test_accuracy:.4f}')

    # Compute predictions and true labels
    y_pred_all = []
    y_all = []
    with torch.no_grad():
        for X, y in test_loader_duration:
            y_pred = model(X)
            y_pred_label = (y_pred.data > 0.5).float()
            y_pred_all.extend(y_pred_label.numpy())
            y_all.extend(y.numpy())

    # Compute normalized confusion matrix
    cm = confusion_matrix(y_all, y_pred_all, normalize='true') * 100

    # Format each cell as a string with a '%' symbol for annotations
    cm_percentage = np.array([["{:.2f}%".format(value) for value in row] for row in cm])
    
    f1 = f1_score(y_all, y_pred_all, average='weighted')  # Weighted F1-score

    durations_list.append(duration)
    accuracies.append(test_accuracy)
    f1_scores.append(f1)
    
    # # Plot the normalized confusion matrix
    # plt.figure(figsize=(5, 5))
    # sns.heatmap(cm, annot=cm_percentage, fmt="", cmap="Blues", cbar=False,
    #             xticklabels=['Tail-drop', 'PIE'], yticklabels=['Tail-drop', 'PIE'])

    # plt.xlabel("Predicted label")
    # plt.ylabel("True label")
    # plt.title(f"Confusion Matrix ({duration}s - (Accuracy: {test_accuracy:.2f}, F1 Score: {f1:.2f})")

    # # Save the confusion matrix for the current duration
    # #plt.savefig(f"confusion_matrix_{duration}s.png", dpi=300, bbox_inches="tight", format="png")
    # plt.show()

# Plot Accuracy and F1 Score over Duration
plt.figure(figsize=(8, 5))

# Accuracy Plot
plt.plot(durations_list, accuracies, label='Accuracy', marker='o')

# Optional: Add F1 Score Plot
plt.plot(durations_list, f1_scores, label='F1 Score', marker='s', linestyle='--')

# Add labels, legend, and title
plt.xlabel("Duration (seconds)")
plt.ylabel("Metric Value")
plt.title("Model Performance Over Different Durations")
plt.legend()
plt.grid()

# Save the plot
plt.savefig("accuracy-vs-duration.png", dpi=300, bbox_inches="tight", format="png")

plt.show()
















