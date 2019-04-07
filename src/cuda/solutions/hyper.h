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

#ifndef DNN_OPT_CUDA_SOLUTIONS_HYPER
#define DNN_OPT_CUDA_SOLUTIONS_HYPER

#include <functional>
#include <cuda/base/generator.h>
#include <cuda/base/solution.h>
#include <cuda/base/algorithm.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

/**
 * @brief The hyper class represents an optimization algorithm hyper-parameter
 * solution.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date November, 2017
 */
class hyper : public virtual solution
{
public:
// TODO: Fix this
  static hyper* make(generator* generator, algorithm* base, unsigned int size);

  virtual algorithm* get_algorithm() const;

  void set_do_optimize(std::function<void(algorithm*)> do_optimize);

  virtual hyper* clone() override;

  virtual bool assignable(const core::solution* s) const override;

  virtual void assign(core::solution* s) override;

  virtual ~hyper();

protected:

   /**
    * @copydoc solution::calculate_fitness()
    *
    * Performs @ref get_iteration_count() optimization steeps of the provided
    * @ref get_algorithm() and returns its fitness.
    *
    * @return the fitness of this solution.
    */
  virtual float calculate_fitness() override;

  /**
   * @brief The basic contructor for this class.
   *
   * @param generator an instance of a generator class.
   * The generator is used to initialize the parameters of this
   * solution.
   *
   * @param size is the number of parameters for this solution. Default is 10.
   */
  hyper(generator* generator, algorithm* base, unsigned int size = 10 );

  /** The elementary optimization algorithm */
  algorithm* _base;

  std::function<void(algorithm*)> _do_optimize;

};

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt

#endif
