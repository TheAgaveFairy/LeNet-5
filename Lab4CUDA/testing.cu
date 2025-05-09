#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// Paulie D., stole this from StackOverflow to handle errors
#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
__global__ void SoftmaxWithoutLoss(double *input, double *output, int count) {
  __shared__ double shm[32]; // only needs to be 16, but, warps, y'know

  size_t idx = threadIdx.x; // call as 1d <<<1, 32>>>
  double max_val = -INFINITY;
  if (idx < count)
    max_val = input[idx];

  // reduce - trying my hand at the warp shuffling <3
  // stride needs to be 2 ^ ceiling of log(count) (aka 10 classes needs 4 bits)
  for (int stride = 16; stride > 0; stride >>= 1) {
    printf("here %u - ", stride);
    if (idx < stride && idx + stride < count) {
      // printf("%lf yea ", max_val);
      max_val = fmax(max_val, __shfl_down_sync(0xffffffff, max_val, stride));
    }
  }
  __syncthreads();

  printf("%u max_val %lf\n", idx, max_val);
  // broadcast operation from idx 0 to all pos
  max_val = __shfl_sync(0xffffffff, max_val, 0);
  printf("%u shfl ", idx);

  double thread_exp = 0.0;
  if (idx < count) {
    thread_exp = exp(input[idx] - max_val);
    output[idx] = thread_exp;
  }

  shm[idx] = idx < count ? thread_exp : 0.0;
  __syncthreads();
  printf("%u shmd ", idx);

  for (int stride = 16; stride > 0; stride >>= 1) {
    if (idx < stride) {
      shm[idx] += shm[idx + stride];
    }
    __syncthreads();
  }

  printf("%u here ", idx);

  double sum_exp = shm[0];

  if (idx < count) {
    output[idx] = thread_exp / sum_exp;
  }
}
__global__ void softmax_kernel(double *input, double *output, int count) {
  // For small output size (count=10), use a single thread block approach
  __shared__ double shared_max[32]; // For finding max
  __shared__ double shared_sum[32]; // For summing exponentials

  // Initialize shared memory
  if (threadIdx.x < count) {
    shared_max[threadIdx.x] = input[threadIdx.x];
  } else {
    shared_max[threadIdx.x] = -INFINITY;
  }
  __syncthreads();

  // Find maximum value - simple reduction approach without shuffle operations
  for (int stride = 16; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] =
          fmax(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  // At this point, shared_max[0] contains the maximum value
  double max_val = shared_max[0];

  // Compute exp(x_i - max)
  double thread_exp = 0.0;
  if (threadIdx.x < count) {
    thread_exp = exp(input[threadIdx.x] - max_val);
    output[threadIdx.x] = thread_exp; // Store temporarily
  }

  // Load into shared memory for reduction
  shared_sum[threadIdx.x] = (threadIdx.x < count) ? thread_exp : 0.0;
  __syncthreads();

  // Sum up all exp values
  for (int stride = 16; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  // Normalize by the sum
  double sum_exp = shared_sum[0];
  if (threadIdx.x < count) {
    output[threadIdx.x] = thread_exp / sum_exp;
  }
}
// CPU reference implementation of softmax for verification
void softmax_cpu(double *input, double *output, int count) {
  // Find maximum for numerical stability
  double max_val = -INFINITY;
  for (int i = 0; i < count; i++) {
    if (input[i] > max_val) {
      max_val = input[i];
    }
  }

  // Compute exp(x_i - max) and sum
  double sum_exp = 0.0;
  for (int i = 0; i < count; i++) {
    output[i] = expf(input[i] - max_val);
    sum_exp += output[i];
  }

  // Normalize
  for (int i = 0; i < count; i++) {
    output[i] /= sum_exp;
  }
}

// Function to check if two arrays are approximately equal
bool check_results(double *a, double *b, int count, double epsilon) {
  for (int i = 0; i < count; i++) {
    if (fabs(a[i] - b[i]) > epsilon) {
      printf("Mismatch at index %d: CPU=%f, GPU=%f\n", i, a[i], b[i]);
      return false;
    }
  }
  return true;
}

int main() {
  const int count = 10; // Output size for LeNet-5

  // Allocate host memory
  double *h_input = (double *)malloc(count * sizeof(double));
  double *h_output_gpu = (double *)malloc(count * sizeof(double));
  double *h_output_cpu = (double *)malloc(count * sizeof(double));

  // Initialize input with sample data
  fprintf(stderr, "Input values:\n");
  for (int i = 0; i < count; i++) {
    // Use some varied values to test
    h_input[i] = (i % 3 == 0) ? 10.0f : ((i % 3 == 1) ? -5.0f : 0.0f);
    fprintf(stderr, "%lf ", h_input[i]);
  }
  fprintf(stderr, "\n\n");

  // Allocate device memory
  double *d_input;
  double *d_output;
  gpuErrchk(cudaMalloc((void **)&d_input, count * sizeof(double)));
  gpuErrchk(cudaMalloc((void **)&d_output, count * sizeof(double)));

  // Copy input to device
  cudaMemcpy(d_input, h_input, count * sizeof(double), cudaMemcpyHostToDevice);

  // Launch kernel
  fprintf(stderr, "cuda softmax\n");
  softmax_kernel<<<1, 32>>>(d_input, d_output, count);
  cudaDeviceSynchronize();

  // Copy output back to host
  cudaMemcpy(h_output_gpu, d_output, count * sizeof(double),
             cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    return 1;
  }

  fprintf(stderr, "starting cpu softmax\n");
  // Compute softmax on CPU for reference
  softmax_cpu(h_input, h_output_cpu, count);

  // Print and compare results
  fprintf(stderr, "GPU Softmax output:\n");
  double gpu_sum = 0.0;
  for (int i = 0; i < count; i++) {
    fprintf(stderr, "%f ", h_output_gpu[i]);
    gpu_sum += h_output_gpu[i];
  }
  fprintf(stderr, "\nSum: %lf (should be very close to 1.0)\n\n", gpu_sum);

  fprintf(stderr, "CPU Softmax output:\n");
  double cpu_sum = 0.0;
  for (int i = 0; i < count; i++) {
    fprintf(stderr, "%f ", h_output_cpu[i]);
    cpu_sum += h_output_cpu[i];
  }
  fprintf(stderr, "\nSum: %lf (should be very close to 1.0)\n\n", cpu_sum);

  // Verify results
  const double epsilon = 0.00001;
  bool match = check_results(h_output_cpu, h_output_gpu, count, epsilon);

  if (match) {
    fprintf(stderr,
            "TEST PASSED: CPU and GPU results match within tolerance.\n");
  } else {
    fprintf(stderr, "TEST FAILED: CPU and GPU results don't match.\n");
  }

  // Free memory
  free(h_input);
  free(h_output_gpu);
  free(h_output_cpu);
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
