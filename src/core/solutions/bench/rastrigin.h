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

#ifndef DNN_OPT_CORE_SOLUTIONS_BENCH_RASTRIGIN
#define DNN_OPT_CORE_SOLUTIONS_BENCH_RASTRIGIN

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
 * @brief The rastrigin class represents an optimization solutions which 
 * fitness cost is calculated via Rastrigin function.
 *
 * The equation for this function is given by:
 *
 * f(x) = 10n + \sum_{i=0}^n{x_i^2-10\cos(2\pi x_i)}
 *
 * Rastrigin function have a global minima in {0,..., 0} with a value of 0. 
 * A commonly used search domain for testing is [-5.12, 5.12]. Rastrigin is
 * continuous, differentiable, separable, scalable and multimodal. See the 
 * following reference:
 *
 * LIANG, Jane-Jing; SUGANTHAN, Ponnuthurai Nagaratnam; DEB, Kalyanmoy. Novel 
 * composition test functions for numerical global optimization. En Swarm 
 * Intelligence Symposium, 2005. SIS 2005. Proceedings 2005 IEEE. IEEE, 2005. 
 * p. 68-75.
 *
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date November, 2016
 */
class rastrigin : public virtual solution
{
public:

  /**
   * @brief Returns an instance of the rastrigin class.
   *
   * @param generator an instance of a generator class.
   *
   * @param size is the number of parameters for this solution. Default is 10.
   *
   * @return a pointer to an instance of the rastrigin class.
   */
  static rastrigin* make(generator* generator, unsigned int size = 10);
  
  virtual solution* clone() override;

  virtual ~rastrigin();

protected:

  virtual float calculate_fitness() override;

  /**
   * @brief The basic contructor for the ratrigin class.
   *
   * @param generator an instance of a generator class.
   *
   * @param size is the number of parameters for this solution. Default is 10.
   */
  rastrigin(generator* generator, unsigned int size );

};

} // namespace bench
} // namespace solutions
} // namespace core
} // namespace dnn_opt

#endif
