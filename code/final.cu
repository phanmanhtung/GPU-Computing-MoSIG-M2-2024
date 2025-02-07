#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cfloat>

using namespace std;

// Global parameters
int MAX_ITER = 100000;
float THRESHOLD = 0.1f;

__device__ float calculateDistance(const float* point, const float* centroid, int dimensions) {
    float dist = 0;
    for (int i = 0; i < dimensions; ++i) {
        dist += (point[i] - centroid[i]) * (point[i] - centroid[i]);
    }
    return sqrtf(dist);
}

__global__ void assignCluster(const float* data, float* centroids, int* clusterAssignments, int numPoints, int dimensions, int clusters) {
    extern __shared__ float sharedCentroids[];
    int tid = threadIdx.x;

    for (int d = tid; d < clusters * dimensions; d += blockDim.x) {
        sharedCentroids[d] = centroids[d];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPoints) {
        float minDist = FLT_MAX;
        int bestCluster = -1;

        for (int c = 0; c < clusters; ++c) {
            float dist = calculateDistance(&data[idx * dimensions], &sharedCentroids[c * dimensions], dimensions);
            if (dist < minDist) {
                minDist = dist;
                bestCluster = c;
            }
        }
        clusterAssignments[idx] = bestCluster;
    }
}

__global__ void updateCentroids(const float* data, const int* clusterAssignments, float* centroids, int* clusterCounts, int numPoints, int dimensions, int clusters) {
    extern __shared__ float sharedCentroids[];
    int tid = threadIdx.x;
    int clusterIdx = blockIdx.x;

    if (tid < dimensions) {
        sharedCentroids[tid] = 0.0f;
    }
    __syncthreads();

    for (int i = tid; i < numPoints; i += blockDim.x) {
        if (clusterAssignments[i] == clusterIdx) {
            for (int d = 0; d < dimensions; d++) {
                atomicAdd(&sharedCentroids[d], data[i * dimensions + d]);
            }
            atomicAdd(&clusterCounts[clusterIdx], 1);
        }
    }
    __syncthreads();

    if (tid < dimensions && clusterCounts[clusterIdx] > 0) {
        centroids[clusterIdx * dimensions + tid] = sharedCentroids[tid] / clusterCounts[clusterIdx];
    }
}

bool loadData(const string& filename, vector<float>& data, int& numPoints, int& dimensions, int& clusters) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return false;
    }
    file >> numPoints >> dimensions >> clusters;
    float value;
    while (file >> value) {
        data.push_back(value);
    }
    file.close();
    return data.size() == numPoints * dimensions;
}

void initializeCentroids(const float* data, float* centroids, int numPoints, int dimensions, int clusters) {
    srand(42);
    for (int c = 0; c < clusters; ++c) {
        int idx = rand() % numPoints;
        for (int d = 0; d < dimensions; ++d) {
            centroids[c * dimensions + d] = data[idx * dimensions + d];
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <threads_per_block>  <datasetFile>" << endl;
        return -1;
    }
    int threadsPerBlock = atoi(argv[1]);
    string datasetFile = argv[2];
    
    int numPoints, dimensions, clusters;
    vector<float> hostData;
    if (!loadData(datasetFile, hostData, numPoints, dimensions, clusters)) {
        return -1;
    }

    float *deviceData, *deviceCentroids;
    int *deviceAssignments, *deviceClusterCounts;
    cudaMalloc(&deviceData, numPoints * dimensions * sizeof(float));
    cudaMalloc(&deviceCentroids, clusters * dimensions * sizeof(float));
    cudaMalloc(&deviceAssignments, numPoints * sizeof(int));
    cudaMalloc(&deviceClusterCounts, clusters * sizeof(int));

    cudaMemcpy(deviceData, hostData.data(), numPoints * dimensions * sizeof(float), cudaMemcpyHostToDevice);

    vector<float> centroids(clusters * dimensions);
    initializeCentroids(hostData.data(), centroids.data(), numPoints, dimensions, clusters);
    cudaMemcpy(deviceCentroids, centroids.data(), clusters * dimensions * sizeof(float), cudaMemcpyHostToDevice);

    bool converged = false;
    int iteration = 0;

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    while (!converged && iteration < MAX_ITER) {
        iteration++;
        assignCluster<<<(numPoints + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock, clusters * dimensions * sizeof(float)>>>(deviceData, deviceCentroids, deviceAssignments, numPoints, dimensions, clusters);
        cudaDeviceSynchronize();
        cudaMemset(deviceClusterCounts, 0, clusters * sizeof(int));
        updateCentroids<<<clusters, threadsPerBlock, dimensions * sizeof(float)>>>(deviceData, deviceAssignments, deviceCentroids, deviceClusterCounts, numPoints, dimensions, clusters);
        cudaDeviceSynchronize();

        vector<float> prevCentroids(centroids);
        cudaMemcpy(centroids.data(), deviceCentroids, clusters * dimensions * sizeof(float), cudaMemcpyDeviceToHost);
        float maxChange = 0.0f;
        for (int i = 0; i < clusters * dimensions; i++) {
            maxChange = max(maxChange, abs(centroids[i] - prevCentroids[i]));
        }
        converged = (maxChange < THRESHOLD);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cout << "Converged at iteration " << iteration << "." << endl;
    cout << "Total execution time: " << elapsedTime << " ms" << endl;

    cudaFree(deviceData);
    cudaFree(deviceCentroids);
    cudaFree(deviceAssignments);
    cudaFree(deviceClusterCounts);

    return 0;
}
