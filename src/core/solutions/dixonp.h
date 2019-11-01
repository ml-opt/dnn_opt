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
#ifndef DNN_OPT_CORE_SOLUTIONS_DIXONP
#define DNN_OPT_CORE_SOLUTIONS_DIXONP

#include <core/base/generator.h>
#include <core/base/solution.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

/**
 * @brief The dixonp class represents an optimization solutions which
 * fitness cost is calculated via Dixon & Price function.
 *
 * The equation for this function is given by:
 *
 * f(x) = {({x_1} - 1)}^2 + \sum_{i = 2}^{n}{i{(2{x_i}^2 - x_{i-1})}}
 *
 * Dixon & Price function have a global minima in 2(\frac{{2^n} - 2}{2^n}) with 
 * a value of 0, where n represents the dimension of the problem.
 * A commonly used search domain for testing is [-10, 10]. Dixon & Price 
 * is continuous, differentiable, non-separable, scalable and unimodal. 
 * See the following reference [f_48] in:
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
class dixonp : public virtual solution
{
public:

  /**
   * @brief Returns an instance of this object. This method
   * is an implementation of the factory pattern.
   *
   * @param generator an instance of a generator class. The
   * generator is used to initialize the parameters of this solution.
   *
   * @param size is the number of parameters for this solution. Default is 20.
   *
   * @return a pointer to an instance of the dixonp class.
   */
  static dixonp* make(generator* generator, unsigned int size = 20);

  virtual ~dixonp();

protected:

  virtual float calculate_fitness();

  /**
   * @brief This is the basic contructor for this class.
   * @param generator an instance of a generator class.
   *
   * @param size is the number of parameters for this solution. Default is 20.
   */
  dixonp(generator* generator, unsigned int size = 20);

};

} // namespace solutions
} // namespace core
} // namespace dnn_opt

#endif
