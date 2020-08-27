/**
 * Get the batch size this model was compiled with.
 *
 * @return The batch size.
 */
extern "C" int get_batch_size();

/**
 * Initialize the model.
 *
 * This function instantiates the model graph, as well as
 * Eigen::ThreadPool and Eigen::ThreadPoolDevice to run the graph.
 *
 * @param num_threads Number of threads for Eigen::ThreadPool. If smaller than 1,
 *                    will deduce automatically with std::thread::hardware_concurrency().
 *                    Defaults to 0.
 */
extern "C" void model_init(int num_threads = 0);

/**
 * Run the model.
 *
 * @param input Pointer to input data. This will be copied to the model's
 *              memory prior to actually running the model.
 * @param output Pointer to the output data. The result will be copied here
 *              from the model's memory after running the model.
 * @param input_size Size of the input buffer. Must be batch_size * 4.
 * @param output_size Size of the output buffer. Must be batch_size * 8 * 16.
 *
 * @return Status.
 *         -2 = model was not initialized
 *         -1 = running the graph failed
 *         0 = success
 */
extern "C" int model_run(float *input, float *output, int input_size, int output_size);

/**
 * De-initialize the model.
 */
extern "C" void model_free();
