import numpy as np
import csv
import time
import os
from sklearn.cluster import KMeans

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
output_csv = "sklearn_kmeans_test_results.csv"

def load_dataset(file_path):
    """Loads dataset from file. Returns (data, num_points, dimensions, num_clusters)."""
    with open(file_path, "r") as f:
        first_line = f.readline().strip().split()
        num_points, dimensions, num_clusters = map(int, first_line)
        
        # Load the dataset
        data = np.loadtxt(f, dtype=np.float32)
        
    return data, num_points, dimensions, num_clusters

# Run tests and collect results
results = []
for dataset in datasets:
    dataset_name = os.path.basename(dataset)
    
    data, num_points, dimensions, num_clusters = load_dataset(dataset)

    total_time = 0.0
    total_iterations = 0

    for _ in range(num_runs):
        start_time = time.time()
        
        kmeans = KMeans(n_clusters=num_clusters, max_iter=10000, tol=0.1, n_init=10, random_state=42)
        kmeans.fit(data)
        
        end_time = time.time()
        
        exec_time = end_time - start_time
        iterations = kmeans.n_iter_

        total_time += exec_time
        total_iterations += iterations

    # Convert time to milliseconds and round to the nearest integer
    avg_time_ms = int(round((total_time / num_runs) * 1000))
    avg_iterations = int(round(total_iterations / num_runs))

    results.append([dataset_name, num_points, dimensions, num_clusters, avg_time_ms, avg_iterations])

# Write results to CSV
with open(output_csv, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["File_name", "num_point", "dimension", "cluster", "avg_time_ms", "avg_iter"])
    writer.writerows(results)

print(f"Results saved to {output_csv}")