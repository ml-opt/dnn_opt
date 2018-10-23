/*
Copyright (c) 2018, Jairo Rojas-Delgado <jrdelgado@uci.cu>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

/**
 * This example deals with optimizing hyper-parameters of meta-heuristics with
 * meta-heuristics. Hyper-parameter optimization (HPO) is a task of high complexity
 * that involves two nested cycles of optimization, that includes training
 * in machine learning.
 *
 * Here we refer to the regular optimization algoritm simply as algorithm. The
 * optimization algorithm that deals with hyper-parameter optimization of the
 * regular algorithm meta-algorithm.
 *
 * The hyper-parameters for the meta-algorithm are choosen from literature
 * recommendation. This is based in the idea that as the regular optimization
 * problem may have hundreds of dimensions and standard literature recomendation
 * may not be applicable, hyper-parameter optimization usually involves less
 * less than 10 parameters.
 *
 * Additionally, while some automatic hyper-parameter optimization  algorithms
 * such as SMAC and TPE are availiable, the practitioner needs to know details
 * about the regular optimization algoritm plus the hyper-parameter optimization
 * algorithm. With this approach, if the practitioner is using PSO can use PSO
 * for HPO.
 *
 * See benchark example for a description of the basic workflow of the library.
 *
 * From the command line, you can pass the following arguments:
 *
 * -n dimension of the target solution, default: 256.
 * -meta_n number of hyper-parameters, default: 4.
 * -p size of the population, default: 40.
 * -meta_p size of the meta-population
 * -e number of iterations of the meta-heuristic, dafault: 1000.
 * -s solution type, default: dnn_opt::core::solutions::de_jung.
 * -a meta-heuristic type, default: dnn_opt::core::algorithms::pso.
 * -o output (0 - None, 1-Simple, 2-HPOLib), default: 1.
 *
 * Depending on the optimization algorithm, you can specify hyper-parameters. See
 * the documentation for each meta-heuristic, specifically set_params() method.
 * You can specify the hyper-parameters in the following order:
 *
 * -ha first hyper-parameter
 * -hb second hyper-parameter
 * -hc third hyper-parameter
 * -hd fourth hyper-parameter
 * -he fifth hyper-parameter
 * -hf sixth hyper-parameter
 *
 * The method set_hyper() used in this example will pass the hyper-parameters
 * to the meta-heuristic, or use default hyper-parameters in case that those
 * are not specified.
 *
 * For default values, hyper-parameters have been configured using automatic
 * hyper-parameter optimization. See corresponding example, @ref hpo.cpp.
 *
 * Solutions that can be created are listed below:
 *
 * 0 - @ref dnn_opt::core::solutions::de_jung.
 *
 * Algorithms that can be created with this function are listed below:
 *
 * 0 - dnn_opt::core::algorithms::pso.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date October, 2019
 */

#include <iostream>
#include <vector>
#include <chrono>

#include <common.h>
#include <dnn_opt.h>

using namespace std;
using namespace std::chrono;
using namespace dnn_opt::core;

int main(int argc, char** argv)
{
  /* command line argument collection */

  int n = input("-n", 256, argc, argv);
  int meta_n = input("-n", 4, argc, argv);
  int p = input("-p", 40, argc, argv);
  int meta_p = input("-m_p", 10, argc, argv);
  int eta = input("-eta", 1000, argc, argv);
  int meta_eta = input("-m_eta", 50, argc, argv);
  int solution_type = input("-s", 0, argc, argv);
  int algorithm_type = input("-a", 0, argc, argv);
  int output_type = input("-o", 1, argc, argv);

  /* generator that defines the search space */
  auto* generator = generators::uniform::make(-10.0f, 10.0f);

  /* set that contains the individuals of the population */
  auto* solutions = solution_set<>::make(p);

  /* creating a population of size p */

  for(int i = 0; i < p; i++)
  {
    auto* solution = create_solution(solution_type, n, generator);

    solutions->add(solution);
  }

  solutions->generate();

  /* creating algorithm */
  auto* algorithm = create_algorithm(algorithm_type, solutions);

  auto* meta_generator = generators::uniform::make(0.0f, 3.0f);
  auto* meta_solutions = solution_set<>::make(meta_p);

  for(int i = 0; i < meta_p; i++)
  {
    auto* solution = solutions::hyper::make(meta_generator, algorithm, meta_n);

    solution->set_iteration_count(eta);

    meta_solutions->add(solution);
  }

  meta_solutions->generate();

  auto* meta_algorithm = create_algorithm(algorithm_type, meta_solutions);

  /* hyper-parameters, see @ref dnn_opt::core::algorithm::set_params() */
  set_hyper(algorithm_type, meta_algorithm, argc, argv);

  /* optimize for eta iterations */

  auto start = high_resolution_clock::now();

  int i = 0;

  meta_algorithm->optimize(meta_eta, [&i]()
  {
    cout << "Meta iteration: " << i + 1 << endl;
  });

  auto end = high_resolution_clock::now();

  /* collect statics */

  float time = duration_cast<milliseconds>(end - start).count();
  float fitness = meta_algorithm->get_best()->fitness();

  /* show params in standard output */

  float* params = meta_algorithm->get_best()->get_params();

  for(int i = 0; i < meta_solutions->get_dim(); i++)
  {
    cout << params[i] << " ";
  }

  cout << endl;

  /* quality of hyper-parameters */

  example_out(output_type, time, fitness);

  /* delete allocated memory */
  /* dnn_opt::core::solution_set::clean() is a helper to delete solutions */

  delete meta_solutions->clean();
  delete meta_algorithm;
  delete meta_generator;

  delete solutions->clean();
  delete algorithm;
  delete generator;  

  return 0;
}
