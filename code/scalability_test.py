'''
Optimum thread_per_block config:
Run dataset_10000_20_10 10 times each config
Plot
'''


import subprocess
import re
import os
import plotly.graph_objects as go

# CUDA executable
cuda_executable = "./cuda"

# Dataset file
dataset = "dataset_10000_20_10.txt"

# Path to dataset file
dataset_path = "/kaggle/input/my-data/"
file_path = os.path.join(dataset_path, dataset)

# Number of trials per thread configuration
num_trials = 10

# Explicit thread configurations
thread_configs = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

def parse_output(output):
    """Extract execution time from CUDA output."""
    time_match = re.search(r'Total execution time: ([\d\.]+) ms', output)
    if time_match:
        return float(time_match.group(1))
    return None

# Run tests and collect results
avg_times = []
for threads_per_block in thread_configs:
    total_time = 0.0
    for _ in range(num_trials):
        result = subprocess.run([cuda_executable, str(threads_per_block), file_path], capture_output=True, text=True)
        exec_time = parse_output(result.stdout)
        if exec_time is not None:
            total_time += exec_time
    avg_time = total_time / num_trials
    avg_times.append(avg_time)

# Create a Plotly figure
fig = go.Figure()

# Add a trace (line plot)
fig.add_trace(go.Scatter(
    x=thread_configs,
    y=avg_times,
    mode='lines+markers',
    name='Execution Time',
    text=[f"Threads: {x}<br>Execution Time: {y:.2f} ms" for x, y in zip(thread_configs, avg_times)],  # Tooltip text
    hoverinfo='text',  # Show custom text in the tooltip
))

# Customize layout
fig.update_layout(
    title="CUDA Performance with Increasing Threads per Block",
    xaxis_title="Threads per Block",
    yaxis_title="Average Execution Time (ms)",
    template="plotly",  # White background
    plot_bgcolor='white',  # Set plot area background to white
    paper_bgcolor='white',  # Set overall figure background to white
)

# Save plot to HTML file
fig.write_html("cuda_performance_interactive.html")

# Show plot
fig.show()

print("Visualization saved and displayed successfully.")
