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

#ifndef DNN_OPT_CUDA_ALGORITHMS_CUCKOO
#define DNN_OPT_CUDA_ALGORITHMS_CUCKOO

#include <core/algorithms/cuckoo.h>
#include <cuda/generators/normal.h>
#include <cuda/generators/uniform.h>
#include <cuda/base/algorithm.h>

namespace dnn_opt
{
namespace cuda
{
namespace algorithms
{

/**
 * @brief The cuckoo class implements an optimization metaheuristic algorithm
 * called Cuckoo Search (CS).
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date July, 2016
 * @version 1.0
 */
class cuckoo : public virtual algorithm,
               public virtual core::algorithms::cuckoo
{
public:

  template<class t_solution>
  static cuckoo* make(set<t_solution>* solutions);

  virtual ~cuckoo();

protected:

  /**
   * @copydoc core::algorithm::init
   */
  virtual void init() override;

  /**
   * @brief Generates a new candidate solution by performing a levy flight from
   * a given solution.
   *
   * @param index solution from which the levy flight will be performed.
   */
  virtual void generate_new_cuckoo(int cuckoo_idx) override;

  template<class t_solution>
  cuckoo(set<t_solution>* solutions);

};

template<class t_solution>
cuckoo* cuckoo::make(set<t_solution>* solutions)
{
  auto result = new cuckoo(solutions);

  result->init();

  return result;
}

template<class t_solution>
cuckoo::cuckoo(set<t_solution>* solutions)
: algorithm(solutions),
  core::algorithm(solutions),
  core::algorithms::cuckoo(solutions)
{
  
}

} // namespace algorithms
} // namespace cuda
} // namespace dnn_opt

#endif
