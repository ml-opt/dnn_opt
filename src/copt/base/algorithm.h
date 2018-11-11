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

#ifndef DNN_OPT_COPT_ALGORITHM
#define DNN_OPT_COPT_ALGORITHM

#include <vector>
#include <functional>
#include <core/base/algorithm.h>
#include <copt/base/solution_set.h>

namespace dnn_opt
{
namespace copt
{

/**
 * @copydoc core::algorithm
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date June, 2018
 * @version 1.0
 */
class algorithm : public virtual core::algorithm
{
protected:

  /**
   * @brief The basic contructor for an optimization algorithm.
   */
  template<class t_solution>
  algorithm(const solution_set<t_solution>* solutions);

private:

  /** The optimization operation performed by this algorithm */
  bool _maximization;

  solution_set<>* _solutions;

};

template<class t_solution>
algorithm::algorithm(const solution_set<t_solution>* solutions)
: core::algorithm(solutions)
{

}

} // namespace copt
} // namespace dnn_opt

#endif
