#include <iostream>
#include <string>
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

  std::string db_train = input_s("-db", "", argc, argv);
  std::string db_test = input_s("-dbt", "", argc, argv);

  int expected_layers = input("-hidden-layers", 1, argc, argv);
  vector<int> hidden_units;

  for(int i = 0; i < expected_layers; i++)
  {
    int layer_size = input("-layer-" + to_string(i), 10, argc, argv);

    if(layer_size > 0)
    {
      hidden_units.push_back(layer_size);
    }
  }

  int p = input("-p", 40, argc, argv);
  int eta = input("-eta", 4000, argc, argv);
  int algorithm_type = input("-a", 0, argc, argv);
  
  int in_dim = input("-in-dim", 20, argc, argv);
  int out_dim = input("-out-dim", 20, argc, argv);

  /* generator that defines the search space */
  auto* generator = generators::uniform::make(-1.0f, 1.0f);
  auto* train = readers::csv_reader::make(db_train, in_dim, out_dim, ' ', true);
  auto* test = readers::csv_reader::make(db_test, in_dim, out_dim, ' ', true);
  auto* act = activations::sigmoid::make();

  /* set that contains the individuals of the population */
  auto* solutions = set<solutions::network>::make(p);

  for (int i = 0; i < p; i++)
  {
    auto* nn = solutions::network::make(generator, train, errors::mse::make());

    /* create perceptron model */ 
    nn->add_layer(layers::fc::make(train->get_in_dim(), hidden_units[0], act));
    for(int j = 1; j < hidden_units.size() - 1; j++)
    {
      nn->add_layer(layers::fc::make(hidden_units[j - 1], hidden_units[j], act));
    }
    nn->add_layer(layers::fc::make(hidden_units.size() - 1, 1, act));

    solutions->add(nn);
  }

  /* random generation of initial population according the generator */
  solutions->generate();

  /* creating algorithm */
  auto* algorithm = create_algorithm(algorithm_type, solutions);

  /* hyper-parameters, see @ref dnn_opt::core::algorithm::set_params() */
  set_hyper(algorithm_type, algorithm, argc, argv);

  /* optimize for eta iterations */

  float terror = 0;
  float gerror = 0;
  float time = 0;


  auto start = high_resolution_clock::now();
  algorithm->optimize_eval(eta, []()
  {
    return true;
  });
  auto end = high_resolution_clock::now();

  /* collect statics */

  time = duration_cast<milliseconds>(end - start).count();
  terror = dynamic_cast<solutions::network*>(algorithm->get_best())->test(train);
  gerror = dynamic_cast<solutions::network*>(algorithm->get_best())->test(test);

  cout << time << " " << terror << " " << gerror << endl;

  /* delete allocated memory */

  delete solutions->clean();
  delete act;
  delete test, train;
  delete algorithm;
  delete generator;

  return 0;
}
