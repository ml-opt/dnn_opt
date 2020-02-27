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
  std::string db_train = input_s("-db", "", argc, argv);
  std::string db_test = input_s("-dbt", "", argc, argv);
  int h_n = input("-h_n", 45, argc, argv);
  int p = input("-p", 40, argc, argv);
  int eta = input("-eta", 4000, argc, argv);
  int part = input("-part", 5, argc, argv);
  int algorithm_type = input("-a", 0, argc, argv);
  int output_type = input("-o", 3, argc, argv);

  auto* generator = generators::uniform::make(-1.0f, 1.0f);
  auto* train = readers::csv_reader::make(db_train, 20, 1, ' ', true);
  auto* test = readers::csv_reader::make(db_test, 20, 1, ' ', true);
  auto* act = activations::sigmoid::make();
  auto* solutions = set<solutions::network>::make(p);

  for (int i = 0; i < p; i++)
  {
    auto* nn = solutions::network::make(generator, train, errors::mse::make());

    nn->add_layer(
    {
      layers::fc::make(train->get_in_dim(), h_n, act),
      layers::fc::make(h_n, 1, act)
    });

    solutions->add(nn);
  }

  solutions->generate();

  auto* algorithm = algorithms::opwa::make(part, solutions, [&](set<>* solutions)
  {
    auto* algorithm = create_algorithm(algorithm_type, solutions);

    set_hyper(algorithm_type, algorithm, argc, argv);

    return algorithm;
  });

  float fitness = 0;
  float time = 0;

  auto start = high_resolution_clock::now();
  algorithm->optimize_eval(eta, []()
  {
    return true;
  });
  auto end = high_resolution_clock::now();

  time = duration_cast<milliseconds>(end - start).count();
  fitness = dynamic_cast<solutions::network*>(algorithm->get_best())->test(test);

  example_out(output_type, time, fitness);

  delete solutions->clean();
  delete act;
  delete test, train;
  delete algorithm;
  delete generator;

  return 0;
}
