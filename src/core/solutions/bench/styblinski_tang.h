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

#ifndef DNN_OPT_CORE_SOLUTIONS_BENCH_STYBLINSKI_TANG
#define DNN_OPT_CORE_SOLUTIONS_BENCH_STYBLINSKI_TANG

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
 * @brief The styblinski_tang class represents an optimization solution 
 * which fitness cost is calculated via Styblinski-Tang function.
 * 
 * The equation for this function is given by:
 *
 * f(x) = 0.5 * \sum_{i=0}^n{x_i^4 + 16x_i^2 + 5x_i}
 *
 * Styblinski-Tang function have a global minima in {-2.093,..., 2.9053} with 
 * a value of -78.332. A commonly used search domain for testing is [-5, 5].
 * Styblinski-Tang is continuous, differentiable, non-separable, non-scalable
 * and multimodal.  See the following reference[f_144] in:
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
class styblinski_tang : public virtual solution
{
public:
  
  /**
   * @brief Returns an instance the styblinski_tang class
   *
   * @param generator an instance of a generator class. 
   *
   * @param size is the number of parameters for this solution. Default is 10.
   *
   * @return an instance of styblinski_tang class.
   */
  static styblinski_tang* make(generator* generator, unsigned int size = 10);

  virtual ~styblinski_tang();

protected:

  virtual float calculate_fitness() override;

  /**
   * @brief The basic contructor for this class.
   *
   * @param generator an instance of a generator class.
   *
   * @param size is the number of parameters for this solution. Default is 10.
   */
  styblinski_tang(generator* generator, unsigned int size );

};

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt

#endif
