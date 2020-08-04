extern "C" void model_init();
extern "C" int model_run(float *input, float *output, int input_size, int output_size);
extern "C" void model_free();
