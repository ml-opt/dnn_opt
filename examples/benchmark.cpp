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
 * This example deals with optimizing common benchmark functions by using
 * meta-heuristic algorithms.
 *
 * The general workflow is straight forward:
 *
 * 1. Create the @ref dnn_opt::core::generator that defines the search space.
 * 2. Create and generate the initial population of solutions.
 * 3. Create the optimization meta-heuristic and set the hyper-parameters.
 * 4. Start optimization.
 * 5. Collect statistics.
 * 6. Free memory.
 *
 * In this library, the @ref dnn_opt::core::solution class represents the
 * fitness funcion and contains the parameters of individuals. Actually,
 * @ref dnn_opt::core::solution should be considered a generic abstract class
 * and derived classes are the ones that define custom fitness functions.
 *
 * From the command line, you can pass the following arguments:
 *
 * -n dimension of the target solution, default: 256.
 * -p size of the population, default: 40.
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
  int p = input("-p", 40, argc, argv);
  int eta = input("-eta", 1000, argc, argv);
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

  /* random generation of initial population according the generator */
  solutions->generate();

  /* creating algorithm */
  auto* algorithm = create_algorithm(algorithm_type, solutions);

  /* hyper-parameters, see @ref dnn_opt::core::algorithm::set_params() */
  set_hyper(algorithm_type, algorithm, argc, argv);

  /* optimize for eta iterations */

  auto start = high_resolution_clock::now();
  algorithm->optimize(eta);
  auto end = high_resolution_clock::now();

  /* collect statics */

  float time = duration_cast<milliseconds>(end - start).count();
  float fitness = algorithm->get_best()->fitness();

  example_out(output_type, time, fitness);

  /* delete allocated memory */
  /* dnn_opt::core::solution_set::clean() is a helper to delete solutions */

  delete solutions->clean();
  delete algorithm;
  delete generator;

  return 0;
}
