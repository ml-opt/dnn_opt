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

#ifndef DNN_OPT_COPT_SOLUTION_SET
#define DNN_OPT_COPT_SOLUTION_SET

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <core/base/solution_set.h>
#include <copt/base/solution.h>

using namespace std;

namespace dnn_opt
{
namespace copt
{

/**
 * @brief The solution_set class is intended to manage a set of optimization
 * solutions for a determined optimization problem.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date June, 2016
 * @version 1.0
 */
template<class t_solution = solution>
class solution_set : public core::solution_set<t_solution>
{

static_assert(true, "t_solution must derive from copt::solution");

public:

  static solution_set<t_solution>* make(unsigned int size = 10);

protected:

  /**
   * @brief The basic constructor for a solution_set.
   *
   * @param size the number of solutions of this container.
   */
  solution_set(unsigned int size = 10);

};

template<class t_solution>
solution_set<t_solution>* solution_set<t_solution>::make(unsigned int size)
{
  return new solution_set<t_solution>(size);
}

template<class t_solution>
solution_set<t_solution>::solution_set(unsigned int size)
: core::solution_set<t_solution>(size)
{

}

} // namespace copt
} // namespace dnn_opt

#endif
