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

#ifndef DNN_OPT_COPT_ALGORITHMS_OPWA
#define DNN_OPT_COPT_ALGORITHMS_OPWA

#include <vector>
#include <functional>
#include <core/algorithms/opwa.h>
#include <copt/base/algorithm.h>
#include <copt/base/set.h>
#include <copt/solutions/wrapper.h>
#include <copt/generators/uniform.h>

namespace dnn_opt
{
namespace copt
{
namespace algorithms
{

class opwa : public virtual algorithm,
             public virtual core::algorithms::opwa
{
public:

  template<class t_solution>
  static opwa* make(int count, const set<t_solution>* solutions, wa builder);

protected:

  template<class t_solution>
  opwa(int count, const set<t_solution>* solutions, wa builder);

  template<class t_solution>
  opwa(int count, const set<t_solution>* solutions);

};

template<class t_solution>
opwa* opwa::make(int count, const set<t_solution>* solutions, wa builder)
{
  auto* result = new opwa(count, solutions, builder);

  result->init();

  return result;
}

template<class t_solution>
opwa::opwa(int count, const set<t_solution>* solutions, wa builder)
: algorithm(solutions),
  core::algorithm(solutions),
  core::algorithms::opwa(count, solutions, builder)
{

}

template<class t_solution>
opwa::opwa(int count, const set<t_solution>* solutions)
: algorithm(solutions),
  core::algorithm(solutions),
  core::algorithms::opwa(count, solutions)
{

}

} // namespace algorithms
} // namespace copt
} // namespace dnn_opt

#endif
