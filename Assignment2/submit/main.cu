#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK_SIZE 32  

__global__ void dkernel(const long int *matrix, const long int *filter, long int *result, int h, int w, int c, int r, int s, int k)
{
    extern __shared__ long int s_flt[];
    unsigned id_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned id_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned id_z = blockIdx.z;

    if (id_x >= w || id_y >= h)
        return;

    // Load filter into shared memory
    for (int i = threadIdx.y; i < r; i += blockDim.y) {
        for (int j = threadIdx.x; j < s; j += blockDim.x) {
            for (int ch = 0; ch < c; ch++) {
                long int flt_idx = ((id_z * c + ch) * r + i) * s + j;
                s_flt[(ch * r + i) * s + j] = filter[flt_idx];
            }
        }
    }

    __syncthreads();

    long long sum = 0;
    for (int ch = 0; ch < c; ch++) {
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < s; j++) {
                int row = id_x + j - (s / 2);
                int col = id_y + i - (r / 2);
                if (row >= 0 && row < w && col >= 0 && col < h) {
                    long int image_idx = (ch * h + col) * w + row;
                    long int filter_idx = (ch * r + i) * s + j;
                    sum += matrix[image_idx] * s_flt[filter_idx];
                }
            }
        }
    }
    result[(id_z * h + id_y) * w + id_x] = sum;
}

int main(int argc, char **argv)
{
    int h, w, c;
    cin >> h >> w >> c;
    long int *h_mat = new long int[h * w * c];
    for (long int i = 0; i < h * w * c; i++) {
        cin >> h_mat[i];
    }

    int cf, r, s, k;
    cin >> cf >> r >> s >> k;

    long int *h_filter = new long int[r * s * c * k];
    for (long int i = 0; i < r * s * c * k; i++) {
        cin >> h_filter[i];
    }
    long int *h_ans = new long int[h * w * k];

    auto start = std::chrono::high_resolution_clock::now(); // Start timing

    long int *gpu_mat;
    cudaMalloc(&gpu_mat, sizeof(long int) * h * w * c);
    long int *gpu_filter;
    cudaMalloc(&gpu_filter, sizeof(long int) * r * s * c * k);
    long int *gpu_ans;
    cudaMalloc(&gpu_ans, sizeof(long int) * h * w * k);
    cudaMemcpy(gpu_mat, h_mat, sizeof(long int) * h * w * c, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_filter, h_filter, sizeof(long int) * r * s * c * k, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((w + BLOCK_SIZE - 1) / BLOCK_SIZE, (h + BLOCK_SIZE - 1) / BLOCK_SIZE, k);
    size_t dm_size = r * s * c * sizeof(long int);

    dkernel<<<gridDim, blockDim, dm_size>>>(gpu_mat, gpu_filter, gpu_ans, h, w, c, r, s, k);

    cudaMemcpy(h_ans, gpu_ans, sizeof(long int) * h * w * k, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now(); // End timing
    std::chrono::duration<double> elapsed1 = end - start;

    cudaFree(gpu_mat);
    cudaFree(gpu_filter);
    cudaFree(gpu_ans);

    // Write output to file
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < h * k; i++) {
            for (long int j = 0; j < w; j++) {
                file << h_ans[i * w + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if (file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    delete[] h_mat;
    delete[] h_filter;
    delete[] h_ans;

    return 0;
}
