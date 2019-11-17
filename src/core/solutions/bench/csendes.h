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
#ifndef DNN_OPT_CORE_SOLUTIONS_BENCH_CSENDES
#define DNN_OPT_CORE_SOLUTIONS_BENCH_CSENDES

#include <core/base/generator.h>
#include <core/base/solution.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
namespace bench
{

/**
 * @brief The csendes class represents an optimization solutions which
 * fitness cost is calculated via Csendes function.
 *
 * The equation for this function is given by:
 *
 * f(x) = -0.1 * {\sum_{i=0}^{n}{{x_i}^{6}*(2+sin(1/{x_i}))}
 *
 * Csendes function have a global minima in {0,..., 0} with a value of 0.
 * A commonly used search domain for testing is [-1, 1]. Csendes 
 * is continuous, differentiable, separable, scalable and multimodal. 
 * See the following reference [f_40] in:
 * 
 * MOMIN, JAMIL; YANG, Xin-She. A literature survey of benchmark functions for 
 * global optimization problems. Journal of Mathematical Modelling and Numerical 
 * Optimisation, 2013, vol. 4, no 2, p. 150-194.
 *
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date November, 2016
 */
class csendes : public virtual solution
{
public:

  /**
   * @brief Returns an instance of this object. This method
   * is an implementation of the factory pattern.
   *
   * @param generator an instance of a generator class. The
   * generator is used to initialize the parameters of this solution.
   *
   * @param size is the number of parameters for this solution. Default is 2.
   *
   * @return a pointer to an instance of the csendes class.
   */
  static csendes* make(generator* generator, unsigned int size = 2);

  virtual ~csendes();

protected:

  virtual float calculate_fitness();

  /**
   * @brief This is the basic contructor for this class.
   * @param generator an instance of a generator class.
   *
   * @param size is the number of parameters for this solution. Default is 2.
   */
  csendes(generator* generator, unsigned int size = 2);

};

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt

#endif
