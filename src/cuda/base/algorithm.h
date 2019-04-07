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

#ifndef DNN_OPT_CUDA_ALGORITHM
#define DNN_OPT_CUDA_ALGORITHM

#include <cuda/base/set.h>
#include <core/base/algorithm.h>

namespace dnn_opt
{
namespace cuda
{

/**
 * This class represents an abstract optimization algorithm capable of
 * define the basic functionalities of any meta-heuristic. In order to extend
 * the library, new algorithms shuold derive from this class.
 * The CUDA wrapper for this class is intended for the execution of such optimization
 * algorithms in a CUDA capable device.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date June, 2017
 * @version 1.0
 */
class algorithm : public virtual core::algorithm
{
protected:

  template<class t_solution>
  algorithm(const set<t_solution>* solutions);

};

template<class t_solution>
algorithm::algorithm(const set<t_solution>* solutions)
: core::algorithm(solutions)
{

}

} // namespace cuda
} // namespace dnn_opt

#endif
