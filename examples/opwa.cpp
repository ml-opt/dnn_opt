#include <iostream>
#include <vector>
#include <chrono>

#include <common.h>
#include <dnn_opt.h>

using namespace std;
using namespace std::chrono;

#ifdef ENABLE_CUDA
using namespace dnn_opt::cuda;
#elif ENABLE_COPT
using namespace dnn_opt::copt;
#elif ENABLE_CORE
using namespace dnn_opt::core;
#endif

int main(int argc, char** argv)
{
  /* command line argument collection */

  int n = input("-n", 100, argc, argv);
  int p = input("-p", 40, argc, argv);
  int eta = input("-eta", 40000, argc, argv);
  int solution_type = input("-s", 8, argc, argv);
  int algorithm_type = input("-a", 0, argc, argv);
  int output_type = input("-o", 3, argc, argv);

  /* generator that defines the search space */
  auto* generator = generators::uniform::make(-10.0f, 10.0f);

  /* set that contains the individuals of the population */
  auto* solutions = set<>::make(p);

  for (int i = 0; i < p; ++i)
  {
    solutions->add(create_solution(solution_type, n, generator));
  }

  /* random generation of initial population according the generator */
  solutions->generate();

  /* creating algorithm */
  auto* opwa = algorithms::opwa::make(5, solutions, [algorithm_type, argc, argv](set<>* partition)
  {
    auto* algorithm = create_algorithm(algorithm_type, partition);

    /* hyper-parameters, see @ref dnn_opt::core::algorithm::set_params() */
    set_hyper(algorithm_type, algorithm, argc, argv);

    return algorithm;
  });

  /* optimize for eta iterations */

  auto start = high_resolution_clock::now();
  opwa->optimize_eval(eta, []()
  {
    return true;
  });
  auto end = high_resolution_clock::now();

  /* collect statics */

  float time = duration_cast<milliseconds>(end - start).count();
  float fitness = opwa->get_best()->fitness();

  example_out(output_type, time, fitness);

  /* delete allocated memory */
  /* dnn_opt::core::set::clean() is a helper to delete solutions */

  delete solutions->clean();
  delete opwa;
  delete generator;

  return 0;
}
