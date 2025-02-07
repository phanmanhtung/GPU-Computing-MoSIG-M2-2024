'''
Comparison test:
thread_per_block: 64
Execution time comparison: (cuda, C, sklearn)
Each test: Avg of 5 times
'''


import subprocess
import csv
import re
import os

# CUDA executable
cuda_executable = "./cuda"

# Dataset files
datasets = [
    "dataset_2000_3_5.txt",
    "dataset_5000_50_10.txt",
    "dataset_10000_20_10.txt",
    "dataset_20000_80_15.txt",
    "dataset_50000_5_15.txt"
]

# Path to dataset files
dataset_path = "/kaggle/input/my-data/"

# Number of runs per dataset
num_runs = 5

# Number of threads per block
threads_per_block = 64

# CSV output file
output_csv = "cuda_test_results2.csv"

def parse_output(output):
    """Extract iteration and execution time from CUDA output."""
    iter_match = re.search(r'Converged at iteration (\d+)', output)
    time_match = re.search(r'Total execution time: ([\d\.]+) ms', output)
    if iter_match and time_match:
        return int(iter_match.group(1)), float(time_match.group(1))
    return None, None

# Run tests and collect results
results = []
for dataset in datasets:
    file_path = os.path.join(dataset_path, dataset)
    match = re.match(r'dataset_(\d+)_(\d+)_(\d+)\.txt', dataset)
    if not match:
        continue
    num_points, dimensions, clusters = map(int, match.groups())
    
    total_iterations = 0
    total_time = 0.0
    
    for _ in range(num_runs):
        result = subprocess.run([cuda_executable, str(threads_per_block), file_path], capture_output=True, text=True)
        iterations, exec_time = parse_output(result.stdout)
        if iterations is not None and exec_time is not None:
            total_iterations += iterations
            total_time += exec_time
    
    avg_iterations = round(total_iterations / num_runs)
    avg_time = round(total_time / num_runs)

    # Append the results with avg_time before avg_iterations
    results.append([dataset, num_points, dimensions, clusters, avg_time, avg_iterations])

# Write results to CSV
with open(output_csv, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File_name", "num_point", "dimension", "cluster", "avg_time_ms", "avg_iterations"])
    writer.writerows(results)

print(f"Results saved to {output_csv}")
