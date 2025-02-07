#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Convergence threshold
#define THRESHOLD 0.1f

// Function to load the dataset from the file
int load_data(const char *filename, float **data, int *num_points, int *dimensions, int *num_clusters) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        return -1;
    }

    // Read first line to get number of points, dimensions, and clusters
    fscanf(file, "%d %d %d", num_points, dimensions, num_clusters);

    // Allocate memory for the data
    *data = (float *)malloc((*num_points) * (*dimensions) * sizeof(float));
    if (*data == NULL) {
        printf("Memory allocation failed.\n");
        fclose(file);
        return -1;
    }

    // Read the dataset into the data array
    for (int i = 0; i < *num_points; i++) {
        for (int j = 0; j < *dimensions; j++) {
            fscanf(file, "%f", &(*data)[i * (*dimensions) + j]);
        }
    }

    fclose(file);
    return 0;
}

// Function to calculate the Euclidean distance
float calculate_distance(float *point, float *centroid, int dimensions) {
    float dist = 0.0;
    for (int i = 0; i < dimensions; i++) {
        dist += (point[i] - centroid[i]) * (point[i] - centroid[i]);
    }
    return sqrtf(dist);
}

// Function to initialize centroids randomly
void initialize_centroids(float *data, float *centroids, int num_points, int num_clusters, int dimensions) {
    for (int i = 0; i < num_clusters; i++) {
        int idx = rand() % num_points;
        for (int j = 0; j < dimensions; j++) {
            centroids[i * dimensions + j] = data[idx * dimensions + j];
        }
    }
}

// K-means algorithm with convergence tracking
int kmeans(float *data, float *centroids, int *cluster_assignments, int num_points, int num_clusters, int dimensions, int max_iter) {
    int *new_assignments = (int *)malloc(num_points * sizeof(int));
    float *previous_centroids = (float *)malloc(num_clusters * dimensions * sizeof(float));

    int iteration_count = 0;

    // Initialize cluster assignments
    for (int i = 0; i < num_points; i++) {
        cluster_assignments[i] = -1;
    }

    for (iteration_count = 0; iteration_count < max_iter; iteration_count++) {
        int converged = 1;

        // Step 1: Assign points to the nearest centroid
        for (int i = 0; i < num_points; i++) {
            float min_dist = INFINITY;
            int best_cluster = -1;

            for (int j = 0; j < num_clusters; j++) {
                float dist = calculate_distance(&data[i * dimensions], &centroids[j * dimensions], dimensions);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }

            new_assignments[i] = best_cluster;

            if (cluster_assignments[i] != new_assignments[i]) {
                converged = 0;
            }
        }

        if (converged) {
            printf("Convergence reached at iteration %d.\n", iteration_count + 1);
            break;
        }

        // Step 2: Update centroids
        memset(previous_centroids, 0, num_clusters * dimensions * sizeof(float));
        int *cluster_counts = (int *)calloc(num_clusters, sizeof(int));

        for (int i = 0; i < num_points; i++) {
            int cluster = new_assignments[i];
            cluster_counts[cluster]++;
            for (int j = 0; j < dimensions; j++) {
                previous_centroids[cluster * dimensions + j] += data[i * dimensions + j];
            }
        }

        // Compute new centroids
        for (int j = 0; j < num_clusters; j++) {
            if (cluster_counts[j] > 0) {
                for (int k = 0; k < dimensions; k++) {
                    previous_centroids[j * dimensions + k] /= cluster_counts[j];
                }
            }
        }

        // Step 3: Check convergence based on centroid movement
        float max_diff = 0.0;
        for (int j = 0; j < num_clusters; j++) {
            float diff = 0.0;
            for (int k = 0; k < dimensions; k++) {
                float change = previous_centroids[j * dimensions + k] - centroids[j * dimensions + k];
                diff += change * change;
            }
            max_diff = fmaxf(max_diff, sqrtf(diff));
        }

        free(cluster_counts);

        // Update centroids
        memcpy(centroids, previous_centroids, num_clusters * dimensions * sizeof(float));

        if (max_diff < THRESHOLD) {
            printf("Centroids stabilized at iteration %d.\n", iteration_count + 1);
            break;
        }

        // Update assignments
        memcpy(cluster_assignments, new_assignments, num_points * sizeof(int));
    }

    free(new_assignments);
    free(previous_centroids);

    return iteration_count + 1;  // Return total iterations till convergence
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <dataset_file>\n", argv[0]);
        return -1;
    }

    const char *input_file = argv[1];  // Dataset file from argument
    float *data, *centroids;
    int *cluster_assignments;
    int num_points, dimensions, num_clusters, max_iter = 10000;

    // Load the dataset
    if (load_data(input_file, &data, &num_points, &dimensions, &num_clusters) != 0) {
        return -1;
    }

    // Allocate memory for centroids and cluster assignments
    centroids = (float *)malloc(num_clusters * dimensions * sizeof(float));
    cluster_assignments = (int *)malloc(num_points * sizeof(int));

    // Initialize random number generator
    srand(time(NULL));

    // Initialize centroids
    initialize_centroids(data, centroids, num_points, num_clusters, dimensions);

    // Measure the execution time
    clock_t start_time = clock();

    // Perform K-means clustering
    int iterations = kmeans(data, centroids, cluster_assignments, num_points, num_clusters, dimensions, max_iter);

    clock_t end_time = clock();
    double execution_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Print execution time and iterations
    printf("\nExecution Time: %f seconds\n", execution_time);
    printf("Iterations till convergence: %d\n", iterations);

    // Free allocated memory
    free(data);
    free(centroids);
    free(cluster_assignments);

    return 0;
}
