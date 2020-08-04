#include <iostream>
#include <vector>

#include "model.h"


int main(int argc, char **argv) {
  model_init();
  std::vector<float> input ({45.f, 15.f, 150.f, 40.f});
  std::vector<float> output(8 * 16, 0.f);

  model_run(input.data(), output.data(), 4, 8 * 16);

  for (int ix = 0; ix < 8; ix++) {
    for (int iy = 0; iy < 16; iy++) {
      std::cout << output[ix * 16 + iy] << " ";
    }
    std::cout << std::endl;
  }

  model_free();
  return 0;
}
