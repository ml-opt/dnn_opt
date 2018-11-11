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

#ifndef DNN_OPT_COPT_SOLUTION
#define DNN_OPT_COPT_SOLUTION

#include <core/base/solution.h>
#include <copt/base/generator.h>

namespace dnn_opt
{
namespace copt
{

/**
 * @brief The solution class is intended as an interface to provide custom
 * solutions to be optimized. The most important feature of a solution is
 * its fitness value that measures its quality.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date June, 2016
 * @version 1.0
 */
class solution : public virtual core::solution
{
public:

  static solution* make(generator* generator, unsigned int size);

  /**
   * @brief Creates an exact replica of this solution.
   *
   * @return a pointer to a copy of this object.
   */
  virtual solution* clone();

  /**
   * @brief Check if the given object instance is assignable to this solution.
   *
   * @param a solution to check if it is assignable to this solution.
   *
   * @return true if the given solution is the given solution is assignable,
   * false otherwise.
   */
  virtual bool assignable(const solution* s) const;

  virtual void assign(solution* s);

  virtual generator* get_generator() const override;

protected:

  /**
   * @brief The basic constructor of the solution class.
   */
  solution(generator* generator, unsigned int size);

  /** a pointer to _generator that do not degrade to core::generator */
  generator* _copt_generator;
};

} // namespace copt
} // namespace dnn_opt

#endif
