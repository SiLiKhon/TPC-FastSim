// Borrowed and modified from:
// https://gist.github.com/carlthome/6ae8a570e21069c60708017e3f96c9fd

#define EIGEN_USE_THREADS
#define EIGEN_USE_CUSTOM_THREAD_POOL

#include "graph.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "model.h"

Eigen::ThreadPool *tp;
Eigen::ThreadPoolDevice *device;
GRAPH_CLASS *graph;

extern "C" int get_batch_size() {
  return BATCH_SIZE;
}

extern "C" void model_init() {
  if (!tp) tp = new Eigen::ThreadPool(std::thread::hardware_concurrency());
  if (!device) device = new Eigen::ThreadPoolDevice(tp, tp->NumThreads());
  if (!graph) {
    graph = new GRAPH_CLASS;
    graph->set_thread_pool(device);
  }
}

// TODO: let the user access input/output memory directly to avoid extra copying
extern "C" int model_run(float *input, float *output, int input_size, int output_size) {
  if (!tp || !device || !graph) return -2;
  std::copy(input, input + input_size, graph->arg0_data());
  auto ok = graph->Run();
  if (!ok) return -1;
  std::copy(graph->result0_data(), graph->result0_data() + output_size, output);
  return 0;
}

extern "C" void model_free() {
  delete graph;
  graph = nullptr;

  delete device;
  device = nullptr;

  delete tp;
  tp = nullptr;
}
