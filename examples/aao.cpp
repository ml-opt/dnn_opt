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

  std::string db = input_s("-db", "", argc, argv);
  int k = input("-k", 4, argc, argv);
  int p = input("-p", 40, argc, argv);
  int eta = input("-eta", 40000, argc, argv);
  int algorithm_type = input("-a", 0, argc, argv);
  int output_type = input("-o", 3, argc, argv);

  /* generator that defines the search space */
  auto* generator = generators::uniform::make(-5.0f, 5.0f);
  auto* reader = readers::file_reader::make(db);
  auto** train = proxy_sampler::make_fold_prop(reader, p, 0.2);
  auto* act = activations::sigmoid::make();

  /* set that contains the individuals of the population */
  auto* solutions = set<>::make(p);

  for (int i = 0; i < p; i++)
  {
    auto* nn = solutions::network::make(generator, train[i], errors::mse::make());

    nn->add_layer(
    {
      layers::fc::make(reader->get_in_dim(), 15, act),
      layers::fc::make(15, 1, act)
    });

    solutions->add(nn);
  }

  /* random generation of initial population according the generator */
  solutions->generate();

  /* creating algorithm */
  auto* algorithm = create_algorithm(algorithm_type, solutions);

  /* hyper-parameters, see @ref dnn_opt::core::algorithm::set_params() */
  set_hyper(algorithm_type, algorithm, argc, argv);

  /* optimize for eta iterations */

  float fitness = 0;
  float time = 0;

  auto start = high_resolution_clock::now();

  algorithm->optimize_eval(eta, [](){return true;});

  auto end = high_resolution_clock::now();

  /* collect statics */

  time = duration_cast<milliseconds>(end - start).count();

  int out_n = reader->size() * reader->get_out_dim();

  float* prediction = new float[out_n];
  std::fill_n(prediction, out_n, 0.0f);

  solutions->sort(false);

  for(int i = 0; i < k; i++)
  {
      auto* nn = dynamic_cast<solutions::network*>(solutions->get(i));
      float* i_pred = nn->predict(reader);

      for (int j = 0; j < out_n; j++)
      {
        prediction[j] += i_pred[j];
      }

      delete[] i_pred;
  }

  for (int j = 0; j < out_n; ++j)
  {
    prediction[j] /= k;
  }

  auto* mse = errors::mse::make();

  mse->ff(reader->size(), reader->get_out_dim(), prediction, reader->out_data());

  fitness = mse->f();

  example_out(output_type, time, fitness);

  /* delete allocated memory */
  /* dnn_opt::core::set::clean() is a helper to delete solutions */

  for (int i = 0; i < p; i++)
  {
    delete train[i];
  }

  delete[] train;
  delete[] prediction;
  delete mse;

  delete solutions->clean();
  delete act;
  delete reader;
  delete algorithm;
  delete generator;

  return 0;
}
