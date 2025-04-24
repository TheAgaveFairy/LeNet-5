#include "lenet.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define GETLENGTH(array) (sizeof(array) / sizeof(*(array)))

#define GETCOUNT(array) (sizeof(array) / sizeof(double))

#define FOREACH(i, count) for (int i = 0; i < count; ++i)

// Paulie D., stole this from StackOverflow to handle errors
#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }

#define CONVOLUTE_VALID(input, output, weight)                                 \
  {                                                                            \
    FOREACH(o0, GETLENGTH(output))                                             \
    FOREACH(o1, GETLENGTH(*(output)))                                          \
    FOREACH(w0, GETLENGTH(weight))                                             \
    FOREACH(w1, GETLENGTH(*(weight)))                                          \
    (output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];          \
  }

#define CONVOLUTE_FULL(input, output, weight)                                  \
  {                                                                            \
    FOREACH(i0, GETLENGTH(input))                                              \
    FOREACH(i1, GETLENGTH(*(input)))                                           \
    FOREACH(w0, GETLENGTH(weight))                                             \
    FOREACH(w1, GETLENGTH(*(weight)))                                          \
    (output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];          \
  }

#define CONVOLUTION_FORWARD(input, output, weight, bias, action)               \
  {                                                                            \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                \
      for (int y = 0; y < GETLENGTH(*weight); ++y)                             \
        CONVOLUTE_VALID(input[x], output[y], weight[x][y]);                    \
    FOREACH(j, GETLENGTH(output))                                              \
    FOREACH(i, GETCOUNT(output[j]))                                            \
    ((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);     \
  }

#define CONVOLUTION_BACKWARD(input, inerror, outerror, weight, wd, bd,         \
                             actiongrad)                                       \
  {                                                                            \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                \
      for (int y = 0; y < GETLENGTH(*weight); ++y)                             \
        CONVOLUTE_FULL(outerror[y], inerror[x], weight[x][y]);                 \
    FOREACH(i, GETCOUNT(inerror))                                              \
    ((double *)inerror)[i] *= actiongrad(((double *)input)[i]);                \
    FOREACH(j, GETLENGTH(outerror))                                            \
    FOREACH(i, GETCOUNT(outerror[j]))                                          \
    bd[j] += ((double *)outerror[j])[i];                                       \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                \
      for (int y = 0; y < GETLENGTH(*weight); ++y)                             \
        CONVOLUTE_VALID(input[x], wd[x][y], outerror[y]);                      \
  }

#define SUBSAMP_MAX_FORWARD(input, output)                                     \
  {                                                                            \
    const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));               \
    const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));             \
    FOREACH(i, GETLENGTH(output))                                              \
    FOREACH(o0, GETLENGTH(*(output)))                                          \
    FOREACH(o1, GETLENGTH(**(output))) {                                       \
      int x0 = 0, x1 = 0, ismax;                                               \
      FOREACH(l0, len0)                                                        \
      FOREACH(l1, len1) {                                                      \
        ismax = input[i][o0 * len0 + l0][o1 * len1 + l1] >                     \
                input[i][o0 * len0 + x0][o1 * len1 + x1];                      \
        x0 += ismax * (l0 - x0);                                               \
        x1 += ismax * (l1 - x1);                                               \
      }                                                                        \
      output[i][o0][o1] = input[i][o0 * len0 + x0][o1 * len1 + x1];            \
    }                                                                          \
  }

#define SUBSAMP_MAX_BACKWARD(input, inerror, outerror)                         \
  {                                                                            \
    const int len0 = GETLENGTH(*(inerror)) / GETLENGTH(*(outerror));           \
    const int len1 = GETLENGTH(**(inerror)) / GETLENGTH(**(outerror));         \
    FOREACH(i, GETLENGTH(outerror))                                            \
    FOREACH(o0, GETLENGTH(*(outerror)))                                        \
    FOREACH(o1, GETLENGTH(**(outerror))) {                                     \
      int x0 = 0, x1 = 0, ismax;                                               \
      FOREACH(l0, len0)                                                        \
      FOREACH(l1, len1) {                                                      \
        ismax = input[i][o0 * len0 + l0][o1 * len1 + l1] >                     \
                input[i][o0 * len0 + x0][o1 * len1 + x1];                      \
        x0 += ismax * (l0 - x0);                                               \
        x1 += ismax * (l1 - x1);                                               \
      }                                                                        \
      inerror[i][o0 * len0 + x0][o1 * len1 + x1] = outerror[i][o0][o1];        \
    }                                                                          \
  }

#define DOT_PRODUCT_FORWARD(input, output, weight, bias, action)               \
  {                                                                            \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                \
      for (int y = 0; y < GETLENGTH(*weight); ++y)                             \
        ((double *)output)[y] += ((double *)input)[x] * weight[x][y];          \
    FOREACH(j, GETLENGTH(bias))                                                \
    ((double *)output)[j] = action(((double *)output)[j] + bias[j]);           \
  }

#define DOT_PRODUCT_BACKWARD(input, inerror, outerror, weight, wd, bd,         \
                             actiongrad)                                       \
  {                                                                            \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                \
      for (int y = 0; y < GETLENGTH(*weight); ++y)                             \
        ((double *)inerror)[x] += ((double *)outerror)[y] * weight[x][y];      \
    FOREACH(i, GETCOUNT(inerror))                                              \
    ((double *)inerror)[i] *= actiongrad(((double *)input)[i]);                \
    FOREACH(j, GETLENGTH(outerror))                                            \
    bd[j] += ((double *)outerror)[j];                                          \
    for (int x = 0; x < GETLENGTH(weight); ++x)                                \
      for (int y = 0; y < GETLENGTH(*weight); ++y)                             \
        wd[x][y] += ((double *)input)[x] * ((double *)outerror)[y];            \
  }

// Paulie D. CUDA functions to replace FORWARD macros above
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
// no padding "convolution" aka cross correlation
__global__ void ConvoluteValid(const double *d_in, const double *d_weight,
                               double *d_out, size_t in_height,
                               size_t in_width) {

  size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;

  size_t out_height = in_height - LENGTH_KERNEL + 1;
  size_t out_width = in_width - LENGTH_KERNEL + 1;

  //__shared__ double shm[LENGTH_FEATURE0][LENGTH_FEATURE0]; // input will be
  // this size or smaller (32 x 32)
  if (row < out_height && col < out_width) {
    // size_t half_k_width = LENGTH_KERNEL / 2;
    // size_t half_k_height = LENGTH_KERNEL / 2;

    double result = 0.0;

    for (int i = 0; i < LENGTH_KERNEL; ++i) {
      for (int j = 0; j < LENGTH_KERNEL; ++j) {
        size_t in_row = row + i; //- half_k_height + i;
        size_t in_col = col + j; //- half_k_width + j;

        // if (i_row > -1 && i_row < in_height && i_col >= 0 && i_col <
        // in_width) {
        result +=
            d_in[in_row * in_width + in_col] * d_weight[i * LENGTH_KERNEL + j];
        //}
      }
    }
    d_out[row * out_width + col] = result;
  }
}

// c = a.dot(b) matrix multiplication, naive / nonoptimal
// a => m x l, b => l x n, c = m x n dimensions
__global__ void naiveOneDimKernel(double *a, double *b, double *c, int m, int l,
                                  int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    double sum = 0;
    for (int k = 0; k < l; k++) {
      sum += a[row * n + k] * b[k * n + col];
    }
    c[row * m + col] = sum;
  }
}

// performs addition of bias (1 per channel) and ReLU (currently hard coded
// "action")
__global__ void ActionAndBias(double *d_feature, size_t f_height,
                              size_t f_width, double bias) {
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;

  // action is hardcoded ReLU (temp * (temp > 0))

  if (row < f_height && col < f_width) {
    size_t temp_idx = row * f_width + col;
    double temp = d_feature[temp_idx] + bias;
    d_feature[temp_idx] = temp * (temp > 0);
  }
}
__global__ void SoftmaxWithoutLoss(double *input, double *output, int count) {
  __shared__ double shared_max[32];
  __shared__ double shared_sum[32];

  if (threadIdx.x < count) {
    shared_max[threadIdx.x] = input[threadIdx.x];
  } else {
    shared_max[threadIdx.x] = -INFINITY;
  }
  __syncthreads();

  // reduce
  for (int stride = 16; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_max[threadIdx.x] =
          fmax(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  double max_val = shared_max[0]; // numerical stability

  double thread_exp = 0.0;
  if (threadIdx.x < count) {
    thread_exp = exp(input[threadIdx.x] - max_val);
    output[threadIdx.x] = thread_exp; // Store temporarily
  }

  shared_sum[threadIdx.x] = (threadIdx.x < count) ? thread_exp : 0.0;
  __syncthreads();

  for (int stride = 16; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  double sum_exp = shared_sum[0];
  if (threadIdx.x < count) {
    output[threadIdx.x] = thread_exp / sum_exp;
  }
}

// adds device_b to device_a
__global__ void AddArrays(double *d_a, double *d_b, int height, int width) {
  size_t row = blockDim.y * blockIdx.y + threadIdx.y;
  size_t col = blockDim.x * blockIdx.x + threadIdx.x;

  if (row < height && col < width) {
    int idx = row * width + col;
    d_a[idx] += d_b[idx];
  }
}

__global__ void Maxpool2D(double *input, double *output, int in_h, int in_w) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  int out_h = in_h / 2;
  int out_w = in_w / 2;

  if (row < out_h && col < out_w) {
    int t_r = row * 2;
    int t_c = col * 2;

    int t_idx = t_r * in_w + t_c;

    // A B
    // C D
    double temp_max = fmax(input[t_idx], input[t_idx + 1]); // A and B
    temp_max = fmax(temp_max, input[t_idx + in_w]);         // temp and C
    temp_max = fmax(temp_max, input[t_idx + in_w + 1]);     // temp and D

    output[row * out_w + col] = temp_max;
  }
}
// this assumes everything is already on device...
void ConvoluteForward(double *d_in, double *d_out, double *d_weight,
                      double *d_bias, int in_channels, int out_channels,
                      int in_height, int in_width) {
  int h_out = in_height - LENGTH_KERNEL + 1;
  int w_out = in_width - LENGTH_KERNEL + 1;

  int TPD = 16; // threads per dimension, not sure what this really needs to be
  dim3 blockDim(TPD, TPD);
  dim3 gridDim((w_out + blockDim.x - 1) / blockDim.x,
               (h_out + blockDim.y - 1) / blockDim.y);

  double *d_temp;
  cudaMalloc(&d_temp, h_out * w_out * sizeof(double));
  cudaMemset(d_out, 0, out_channels * h_out * w_out * sizeof(double));

  int k_sq =
      LENGTH_KERNEL * LENGTH_KERNEL; // we'll need this for accessing memory

  for (int y = 0; y < out_channels; ++y) {
    cudaMemset(d_temp, 0, h_out * w_out * sizeof(double));

    for (int x = 0; x < in_channels; ++x) {
      const double *d_in_channel = d_in + (x * in_height * in_width);
      const double *d_weight_xy =
          d_weight + (x * out_channels * k_sq + y * k_sq);

      ConvoluteValid<<<gridDim, blockDim>>>(d_in_channel, d_weight_xy, d_temp,
                                            in_height, in_width);
      gpuErrchk(cudaPeekAtLastError());
      cudaDeviceSynchronize();

      double *d_out_chan = d_out + y * h_out * w_out;
      AddArrays<<<gridDim, blockDim>>>(d_out_chan, d_temp, h_out, w_out);
    }
    double *d_out_channel = d_out + y * h_out * w_out;
    ActionAndBias<<<gridDim, blockDim>>>(d_out_channel, h_out, w_out,
                                         d_bias[y]);
  }
}

// void ConvolutePreparation(input, output, weight, bias) { return; }
/*
void flatten_2d(double *dest, double src[][], int rows, int cols) {
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < rows; c++)
      dest[r * rows + c] = src[r][c];
}

void flatten_3d(double *dest, double src[][H][W], int C, int H, int W) {
  for (int c = 0; c < C; ++c)
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w)
        dest[c * H * W + h * W + w] = src[c][h][w];
}
void flatten_4d(double *dest, double src[][C_OUT][K][K], int C_IN, int C_OUT) {
  int K = LENGTH_KERNEL;
  for (int x = 0; x < C_IN; ++x)
    for (int y = 0; y < C_OUT; ++y)
      for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j)
          dest[x * C_OUT * K * K + y * K * K + i * K + j] = src[x][y][i][j];
}
*/
void PrepareLeNet5Device(LeNet5 *host_model, LeNet5Device *dev_model) {
  // int k_sq = LENGTH_KERNEL * LENGTH_KERNEL;

  /*
  int size_w01_temp = INPUT * LAYER1 * k_sq * sizeof(double);
  printf("Size of w01 calc: %d, sizeof() call: %d\n", size_w01_temp,
         sizeof(host_model->weight0_1));
  */
  // WEIGHTS

  size_t size_w01 = sizeof(host_model->weight0_1);
  gpuErrchk(cudaMalloc(&dev_model->weight0_1, size_w01));
  gpuErrchk(cudaMemcpy(dev_model->weight0_1, host_model->weight0_1, size_w01,
                       cudaMemcpyHostToDevice));

  int size_w23 = sizeof(host_model->weight2_3);
  gpuErrchk(cudaMalloc(&dev_model->weight2_3, size_w23));
  gpuErrchk(cudaMemcpy(dev_model->weight2_3, host_model->weight2_3, size_w23,
                       cudaMemcpyHostToDevice));

  int size_w45 = sizeof(host_model->weight4_5);
  gpuErrchk(cudaMalloc(&dev_model->weight4_5, size_w45));
  gpuErrchk(cudaMemcpy(dev_model->weight4_5, host_model->weight4_5, size_w45,
                       cudaMemcpyHostToDevice));

  int size_w56 = sizeof(host_model->weight5_6);
  gpuErrchk(cudaMalloc(&dev_model->weight5_6, size_w56));
  gpuErrchk(cudaMemcpy(dev_model->weight5_6, host_model->weight5_6, size_w56,
                       cudaMemcpyHostToDevice));

  // BIASES
  int size_b01 = sizeof(host_model->bias0_1);
  gpuErrchk(cudaMalloc(&dev_model->bias0_1, size_b01));
  gpuErrchk(cudaMemcpy(dev_model->bias0_1, host_model->bias0_1, size_b01,
                       cudaMemcpyHostToDevice));

  int size_b23 = sizeof(host_model->bias2_3);
  gpuErrchk(cudaMalloc(&dev_model->bias2_3, size_b23));
  gpuErrchk(cudaMemcpy(dev_model->bias2_3, host_model->bias2_3, size_b23,
                       cudaMemcpyHostToDevice));

  int size_b45 = sizeof(host_model->bias4_5);
  gpuErrchk(cudaMalloc(&dev_model->bias4_5, size_b45));
  gpuErrchk(cudaMemcpy(dev_model->bias4_5, host_model->bias4_5, size_b45,
                       cudaMemcpyHostToDevice));

  int size_b56 = sizeof(host_model->bias5_6);
  gpuErrchk(cudaMalloc(&dev_model->bias5_6, size_b56));
  gpuErrchk(cudaMemcpy(dev_model->bias5_6, host_model->bias5_6, size_b56,
                       cudaMemcpyHostToDevice));

  printf("LeNet5Device successfully allocated and moved to GPU.\n");
}

void FreeLeNet5Device(LeNet5Device *model) {
  gpuErrchk(cudaFree(model->weight0_1));
  gpuErrchk(cudaFree(model->weight2_3));
  gpuErrchk(cudaFree(model->weight4_5));
  gpuErrchk(cudaFree(model->weight5_6));

  gpuErrchk(cudaFree(model->bias0_1));
  gpuErrchk(cudaFree(model->bias2_3));
  gpuErrchk(cudaFree(model->bias4_5));
  gpuErrchk(cudaFree(model->bias5_6));
}

void PrepareFeatureDevice(Feature *host_feat, FeatureDevice *dev_feat) {
  size_t size_input = sizeof(host_feat->input);
  gpuErrchk(cudaMalloc(&dev_feat->input, size_input));
  gpuErrchk(cudaMemcpy(dev_feat->input, host_feat->input, size_input,
                       cudaMemcpyHostToDevice));

  size_t size_layer1 = sizeof(host_feat->layer1);
  gpuErrchk(cudaMalloc(&dev_feat->layer1, size_layer1));
  gpuErrchk(cudaMemcpy(dev_feat->layer1, host_feat->layer1, size_layer1,
                       cudaMemcpyHostToDevice));

  size_t size_layer2 = sizeof(host_feat->layer2);
  gpuErrchk(cudaMalloc(&dev_feat->layer2, size_layer2));
  gpuErrchk(cudaMemcpy(dev_feat->layer2, host_feat->layer2, size_layer2,
                       cudaMemcpyHostToDevice));

  size_t size_layer3 = sizeof(host_feat->layer3);
  gpuErrchk(cudaMalloc(&dev_feat->layer3, size_layer3));
  gpuErrchk(cudaMemcpy(dev_feat->layer3, host_feat->layer3, size_layer3,
                       cudaMemcpyHostToDevice));

  size_t size_layer4 = sizeof(host_feat->layer4);
  gpuErrchk(cudaMalloc(&dev_feat->layer4, size_layer4));
  gpuErrchk(cudaMemcpy(dev_feat->layer4, host_feat->layer4, size_layer4,
                       cudaMemcpyHostToDevice));

  size_t size_layer5 = sizeof(host_feat->layer5);
  gpuErrchk(cudaMalloc(&dev_feat->layer5, size_layer5));
  gpuErrchk(cudaMemcpy(dev_feat->layer5, host_feat->layer5, size_layer5,
                       cudaMemcpyHostToDevice));

  size_t size_output = sizeof(host_feat->output);
  gpuErrchk(cudaMalloc(&dev_feat->output, size_output));
  gpuErrchk(cudaMemcpy(dev_feat->output, host_feat->output, size_output,
                       cudaMemcpyHostToDevice));

  printf("FeatureDevice successfully allocated and moved to GPU.\n");
}

void FreeFeatureDevice(FeatureDevice *feat) {
  gpuErrchk(cudaFree(feat->input));
  gpuErrchk(cudaFree(feat->layer1));
  gpuErrchk(cudaFree(feat->layer2));
  gpuErrchk(cudaFree(feat->layer3));
  gpuErrchk(cudaFree(feat->layer4));
  gpuErrchk(cudaFree(feat->layer5));
  gpuErrchk(cudaFree(feat->output));
}
// end Paulie D.

double relu(double x) { return x * (x > 0); }

double relugrad(double y) { return y > 0; }

static void forward(LeNet5 *lenet, Feature *features,
                    double (*action)(double)) {
  CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1,
                      lenet->bias0_1, action);
  SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
  CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3,
                      lenet->bias2_3, action);
  SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
  CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5,
                      lenet->bias4_5, action);
  DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6,
                      lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors,
                     Feature *features, double (*actiongrad)(double)) {
  DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output,
                       lenet->weight5_6, deltas->weight5_6, deltas->bias5_6,
                       actiongrad);
  CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5,
                       lenet->weight4_5, deltas->weight4_5, deltas->bias4_5,
                       actiongrad);
  SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
  CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3,
                       lenet->weight2_3, deltas->weight2_3, deltas->bias2_3,
                       actiongrad);
  SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
  CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1,
                       lenet->weight0_1, deltas->weight0_1, deltas->bias0_1,
                       actiongrad);
}

static inline void load_input(Feature *features, image input) {
  double(*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
  const long sz = sizeof(image) / sizeof(**input);
  double mean = 0, std = 0;
  FOREACH(j, sizeof(image) / sizeof(*input))
  FOREACH(k, sizeof(*input) / sizeof(**input)) {
    mean += input[j][k];
    std += input[j][k] * input[j][k];
  }
  mean /= sz;
  std = sqrt(std / sz - mean * mean);
  FOREACH(j, sizeof(image) / sizeof(*input))
  FOREACH(k, sizeof(*input) / sizeof(**input)) {
    layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
  }
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label,
                           int count) {
  double inner = 0;
  for (int i = 0; i < count; ++i) {
    double res = 0;
    for (int j = 0; j < count; ++j) {
      res += exp(input[j] - input[i]);
    }
    loss[i] = 1. / res;
    inner -= loss[i] * loss[i];
  }
  inner += loss[label];
  for (int i = 0; i < count; ++i) {
    loss[i] *= (i == label) - loss[i] - inner;
  }
}

static void load_target(Feature *features, Feature *errors, int label) {
  double *output = (double *)features->output;
  double *error = (double *)errors->output;
  softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count) {
  double *output = (double *)features->output;
  const int outlen = GETCOUNT(features->output);
  uint8 result = 0;
  double maxvalue = *output;
  for (uint8 i = 1; i < count; ++i) {
    if (output[i] > maxvalue) {
      maxvalue = output[i];
      result = i;
    }
  }
  return result;
}

static double f64rand() {
  static int randbit = 0;
  if (!randbit) {
    srand((unsigned)time(0));
    for (int i = RAND_MAX; i; i >>= 1, ++randbit)
      ;
  }
  unsigned long long lvalue = 0x4000000000000000L;
  int i = 52 - randbit;
  for (; i > 0; i -= randbit)
    lvalue |= (unsigned long long)rand() << i;
  lvalue |= (unsigned long long)rand() >> -i;
  return *(double *)&lvalue - 3;
}

void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize) {
  double buffer[GETCOUNT(LeNet5)] = {0};
  int i = 0;
#pragma omp parallel for
  for (i = 0; i < batchSize; ++i) {
    Feature features = {0};
    Feature errors = {0};
    LeNet5 deltas = {0};
    load_input(&features, inputs[i]);
    forward(lenet, &features, relu);
    load_target(&features, &errors, labels[i]);
    backward(lenet, &deltas, &errors, &features, relugrad);
#pragma omp critical
    {
      FOREACH(j, GETCOUNT(LeNet5))
      buffer[j] += ((double *)&deltas)[j];
    }
  }
  double k = ALPHA / batchSize;
  FOREACH(i, GETCOUNT(LeNet5))
  ((double *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 label) {
  Feature features = {0};
  Feature errors = {0};
  LeNet5 deltas = {0};
  load_input(&features, input);
  forward(lenet, &features, relu);
  load_target(&features, &errors, label);
  backward(lenet, &deltas, &errors, &features, relugrad);
  FOREACH(i, GETCOUNT(LeNet5))
  ((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(LeNet5 *lenet, image input, uint8 count) {
  Feature features = {0};
  load_input(&features, input);
  forward(lenet, &features, relu);
  return get_result(&features, count);
}

void Initial(LeNet5 *lenet) {
  for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1;
       *pos++ = f64rand())
    ;
  for (double *pos = (double *)lenet->weight0_1;
       pos < (double *)lenet->weight2_3;
       *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))))
    ;
  for (double *pos = (double *)lenet->weight2_3;
       pos < (double *)lenet->weight4_5;
       *pos++ *=
       sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))))
    ;
  for (double *pos = (double *)lenet->weight4_5;
       pos < (double *)lenet->weight5_6;
       *pos++ *=
       sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))))
    ;
  for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1;
       *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)))
    ;
  for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0)
    ;
}
