/*
  Copyright (c) 2016, Jairo Rojas-Delgado
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

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef DNN_OPT_CUDA_ALGORITHMS_FIREFLY
#define DNN_OPT_CUDA_ALGORITHMS_FIREFLY

#include <cuda/base/set.h>
#include <cuda/base/algorithm.h>
#include <cuda/generators/uniform.h>
#include <core/algorithms/firefly.h>

namespace dnn_opt
{
namespace cuda
{
namespace algorithms
{

/**
 * @copydoc core::algorithms::firefly
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date July, 2016
 * @version 1.0
 */
class firefly : public algorithm,
                public core::algorithms::firefly
{
public:

  /**
   * @brief Create a new instance of the firefly class.
   *
   * @param solutions the solution set representing the population to
   * optimize.
   */
  template<class t_solution>
  static firefly* make(const set<t_solution>* solutions);

  /**
   * The basic destructor of the firefly class.
   */
  virtual ~firefly() override;

protected:

  /**
   * @brief Create a new instance of the firefly class.
   *
   * @param solutions the solution set representing the population to
   * optimize.
   */
  template<class t_solution>
  firefly(const set<t_solution>* solutions);

  virtual void init() override;

  /**
   * @copydoc core::algorithm::firefly::move
   *
   * Performs a parallel transform to perform the movement of the @ref s
   * firefly in parallel.
   */
  void move(int s, int t) override;

  /**
   * @copydoc core::algorithm::firefly::distance
   *
   * Performs a parallel transform followed by a parallel reduce to calculate
   * the euclidean distance in device.
   */
  float distance(int s, int t) override;

};

template<class t_solution>
firefly* firefly::make(const set<t_solution>* solutions)
{
  auto* result = new firefly(solutions);

  result->init();

  return result;
}

template<class t_solution>
firefly::firefly(const set<t_solution>* solutions)
: core::algorithms::firefly(solutions),
  core::algorithm(solutions),
  algorithm(solutions)
{

}

} // namespace algorithms
} // namespace cuda
} // namespace dnn_opt

#endif
