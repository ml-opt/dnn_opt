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

#ifndef DNN_OPT_CUDA_SOLUTIONS_ALPINE
#define DNN_OPT_CUDA_SOLUTIONS_ALPINE

#include <cuda/base/generator.h>
#include <cuda/base/solution.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

/**
 * @brief The alpine class represents an optimization solution which fitness
 * cost is calculated via alpine function.
 *
 * The equation for this function is given by:
 * 
 * f(x) = \sum_{i=0}^n{|x_i\sin(x_i) + 0.1\x_i|}
 *
 * Alpine function have a global minima in {0,..., 0} with a value of 0.
 * A commonly used search domain for testing is [-10, 10]. Alpine is 
 * continuous, non-differentiable, separable, non-scalable, multimodal. See
 * the following reference [f_6] in:
 * 
 * MOMIN, JAMIL; YANG, Xin-She. A literature survey of benchmark functions for 
 * global optimization problems. Journal of Mathematical Modelling and Numerical 
 * Optimisation, 2013, vol. 4, no 2, p. 150-194.
 *
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date November, 2018
 */
class alpine : public virtual solution
{
public:

  /**
   * @brief Returns an instance of the alpine class.
   *
   * @param generator an instance of a generator class. The
   * generator is used to initialize the parameters of this solution.
   *
   * @param size is the number of parameters for this solution. Default is 10.
   *
   * @return an instance of alpine class.
   */
  static alpine* make(generator* generator, unsigned int size = 10);

  virtual ~alpine();

protected:

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
  alpine(generator* generator, unsigned int size = 10 );

};

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt

#endif
