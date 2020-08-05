#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "model.h"


int main(int argc, char **argv) {
  model_init();

  int batch_size = get_batch_size();

  std::vector<float> input (batch_size * 4);
  for (int ib = 0; ib < batch_size; ++ib) {
      input[4 * ib + 0] = 45.f;
      input[4 * ib + 1] = 15.f;
      input[4 * ib + 2] = 150.f;
      input[4 * ib + 3] = 40.f;
  }
  std::vector<float> output(batch_size * 8 * 16, 0.f);

  std::vector<float> duration_values;
  int num_operations = std::max(10000 / batch_size, 1);
  for (int i_exp = 0; i_exp < 5; ++i_exp) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i_op = 0; i_op < num_operations; ++i_op) {
      model_run(input.data(), output.data(), batch_size * 4, batch_size * 8 * 16);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    duration_values.push_back(((float)duration) / (num_operations * batch_size));
  }

  float sum = 0, sum2 = 0;
  for (auto duration : duration_values) {
    sum += duration;
    sum2 += duration * duration;
  }

  float mean = sum / duration_values.size();
  float mean2 = sum2 / duration_values.size();
  float std = sqrt((mean2 - mean * mean) / (duration_values.size() - 1));

  std::cout << "With batch_size = " << batch_size
            << ", duration per 1 generation is: "
            << mean << " +\\- " << std << "ms" << std::endl;

//  std::cout << "Example output:" << std::endl;
//  for (int ix = 0; ix < 8; ix++) {
//    for (int iy = 0; iy < 16; iy++) {
//      std::cout << output[ix * 16 + iy] << " ";
//    }
//    std::cout << std::endl;
//  }

  model_free();
  return 0;
}
