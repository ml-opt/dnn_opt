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

#ifndef DNN_OPT_COPT_SOLUTIONS_HYPER
#define DNN_OPT_COPT_SOLUTIONS_HYPER

#include <core/solutions/hyper.h>
#include <copt/base/generator.h>
#include <copt/base/solution.h>
#include <copt/base/algorithm.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{

/**
 * @copydoc core::solutions::hyper
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date November, 2017
 */
class hyper : public virtual solution,
              public virtual core::solutions::hyper
{
public:

  static hyper* make(generator* generator, algorithm* base, unsigned int size);

  virtual hyper* clone() override;

  virtual bool assignable(const solution* s) const override;

  virtual void assign(solution* s) override;

  virtual algorithm* get_algorithm() const;

  virtual ~hyper();

protected:

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

  /** A pointer to _algorithm that do not degrade to core::algorithm */
  algorithm* _copt_algorithm;

};

} // namespace solutions
} // namespace copt
} // namespace dnn_opt

#endif
