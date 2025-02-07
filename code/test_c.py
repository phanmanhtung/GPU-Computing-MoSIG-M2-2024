import subprocess
import csv
import re
import os

# C executable
c_executable = "./kmeans"

# Dataset files
datasets = [
    "/content/dataset_2000_3_5.txt",
    "/content/dataset_5000_50_10.txt",
    "/content/dataset_10000_20_10.txt",
    "/content/dataset_20000_80_15.txt",
    "/content/dataset_50000_5_15.txt"
]

# Number of runs per dataset
num_runs = 5

# CSV output file
output_csv = "kmeans_test_results.csv"

def parse_output(output):
    """Extract execution time and iteration count from C output."""
    time_match = re.search(r'Execution Time: ([\d\.]+) seconds', output)
    iter_match = re.search(r'Iterations till convergence: (\d+)', output)
    
    exec_time = float(time_match.group(1)) if time_match else None
    iterations = int(iter_match.group(1)) if iter_match else None

    return exec_time, iterations

# Run tests and collect results
results = []
for dataset in datasets:
    match = re.match(r'.*/dataset_(\d+)_(\d+)_(\d+)\.txt', dataset)
    if not match:
        continue
    num_points, dimensions, clusters = map(int, match.groups())
    
    total_time = 0.0
    total_iterations = 0
    
    for _ in range(num_runs):
        result = subprocess.run([c_executable, dataset], capture_output=True, text=True)
        exec_time, iterations = parse_output(result.stdout)

        if exec_time is not None and iterations is not None:
            total_time += exec_time
            total_iterations += iterations
    
    avg_time = total_time / num_runs
    avg_iterations = total_iterations / num_runs

    results.append([os.path.basename(dataset), num_points, dimensions, clusters, avg_time, avg_iterations])

# Write results to CSV
with open(output_csv, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File_name", "num_point", "dimension", "cluster", "avg_time_sec", "avg_iter"])
    writer.writerows(results)

print(f"Results saved to {output_csv}")
