# GPU-Computing-MoSIG-M2-2024

# GPU Project: Performance Comparison of CUDA, C, and Sklearn Implementations

This project focuses on benchmarking the performance of different implementations—**CUDA**, **C**, and **Sklearn**—for kmean algorithm. The goal is to analyze execution efficiency, scalability, and the impact of GPU acceleration using CUDA compared to traditional CPU-based approaches.

## 📂 **Project Structure**

```
GPU_PROJECT/
├── code/
│   ├── efficiency_test.py        # Script to evaluate efficiency of implementations
│   ├── final.c                   # ***Core CUDA implementation***
│   ├── kmeans.c                  # K-means clustering algorithm in C
│   ├── scalability_test.py       # Script for testing scalability (different threads/block)
│   ├── sklearn.py                # Kmeans using sklearn
│   └── test_c.py                 # Test to execute kmeans.c
├── data/
│   ├── dataset_2000_3_5.txt      # Sample datasets with varying sizes and dimensions
│   ├── dataset_5000_50_10.txt
│   ├── dataset_10000_20_10.txt
│   ├── dataset_20000_80_15.txt
│   └── dataset_50000_5_15.txt
├── result/
│   ├── c_results.csv             # Performance results from C implementation
│   ├── cuda_results.csv          # Performance results from CUDA implementation
│   ├── cuda_threads_block.html   # Visualization of CUDA threads/block performance
│   └── sklearn_results.csv       # Performance results from sklearn implementation
└── README.md
```

## 🚀 **How to Run**

I run these code on Google Colab and Kaggle Notebook environment

1. **Compile C and CUDA Code:**
   ```bash
   gcc code/final.c -o final
   nvcc code/kmeans.c -o kmeans
   ```

2. **Run Python Scripts:**
   ```bash
   python code/efficiency_test.py
   python code/scalability_test.py
   ```

3. **Analyze Results:**
   - Results are stored in the `result/` folder as CSV files.
   - For CUDA thread/block analysis, open `cuda_threads_block.html` in a browser.

## 📊 **Datasets**

The `data/` folder contains datasets with varying sizes and dimensions to test scalability and efficiency. File naming convention:  
`dataset_<num_points>_<num_dimensions>_<num_clusters>.txt`

## ⚡ **Key Features**

- **CUDA vs. C vs. Sklearn:** Benchmarking GPU and CPU performance
- **Testing:** Datasets with variety in size and dimensions
- **Performance Metrics:** Execution time, iteration counts

## 🛠️ **Dependencies**

- **Python 3.x**  
  - NumPy, Scikit-learn, Plotly  
- **CUDA Toolkit** (for GPU-based execution)  
- **GCC Compiler** (for C programs)  

