/*
Copyright (c) 2017, Jairo Rojas-Delgado <jrdelgado@uci.cu>
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

#ifndef DNN_OPT_CUDA_ALGORITHMS_OPWA
#define DNN_OPT_CUDA_ALGORITHMS_OPWA

#include <vector>
#include <memory>
#include <core/algorithms/opwa.h>
#include <cuda/base/algorithm.h>
#include <cuda/base/solution_set.h>
#include <cuda/solutions/wrapper.h>

namespace dnn_opt
{
namespace cuda
{
namespace algorithms
{

class opwa : public virtual core::algorithms::opwa,
             public virtual algorithm
{
public:

  /**
   * Shorthand for the lambda function which creates wrapped algorithms that
   * operates in partitions.
   */
  typedef std::function<algorithm* (solution_set<>*)> wa;

  static opwa* make(int count, const solution_set<>* solutions, wa builder)
  {
    auto* result = new opwa(count, solutions, builder);

    result->init();

    return result;
  }

  virtual void init() override
  {
    for( int i = 0; i < _count; i++ )
    {
      auto* part = solution_set<>::make(_dev_solutions->size());

      for( int j = 0; j < get_solutions()->size(); j++ )
      {
        part->add(solutions::wrapper::make(i, _count, _dev_solutions->get(j)));
      }

      auto wa = std::unique_ptr<algorithm>(_dev_builder(part));
      _algorithms.push_back(std::move(wa));
    }
  }

  virtual ~opwa() override
  {
    delete _dev_solutions;
  }

protected:

  template<class t_solution>
  opwa(int count, const solution_set<t_solution>* solutions, wa builder)
  : core::algorithms::opwa(count, solutions),
    core::algorithm(solutions),
    algorithm(solutions)
  {
    _dev_solutions = solutions->cast_copy();
    _dev_builder = builder;
  }

protected:

  /** Pointer to _solutions that do not degrade to core::solution_set */
  solution_set<>* _dev_solutions;

  /** Lambda function that creates wrapped algorithms for GPU execution */
  wa _dev_builder;

};

} // namespace algorithms
} // namespace cuda
} // namespace dnn_opt

#endif
