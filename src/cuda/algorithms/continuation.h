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

#ifndef DNN_OPT_CUDA_ALGORITHMS_CONTINUATION
#define DNN_OPT_CUDA_ALGORITHMS_CONTINUATION

#include <cuda/base/algorithm.h>
#include <cuda/base/reader.h>
#include <core/algorithms/continuation.h>

namespace dnn_opt
{
namespace cuda
{
namespace algorithms
{

class continuation : public virtual algorithm,
                     public virtual core::algorithms::continuation
{
public:

  /** Forward declaration of the builder seq of subsets of training patterns */
  class seq;

  /** Forward declaration of a descent builder */
  class descent;

  class fixed;

  static continuation* make(algorithm* base, seq* builder);

  virtual ~continuation() override;

protected:

  continuation(algorithm* base, seq* builder);

};

/**
 * @copydoc core::algorithms::continuation::seq
 * @author Jairo Rojas-Delgado<jrdelgado@uci.cu>
 * @date january, 2018
 * @version 1.0
 */
class continuation::seq : public virtual core::algorithms::continuation::seq
{

};

class continuation::descent : public virtual continuation::seq,
                              public virtual core::algorithms::continuation::descent
{
public :

  static descent* make(reader* dataset, int k, float beta);

  virtual void build() override;

  virtual reader* get(int idx) override;

protected:

  descent(reader* dataset, int k, float beta);

private:

  reader* _cuda_dataset;

  std::vector<reader*> _cuda_sequence;

};


class continuation::fixed : public virtual continuation::seq,
                            public virtual core::algorithms::continuation::fixed
{
public :

  static fixed* make(reader* dataset, int k, float beta);

  virtual void build() override;

  virtual reader* get(int idx) override;

  virtual ~fixed();

protected:

  fixed(reader* dataset, int k, float beta);

private:

  reader* _cuda_dataset;

  std::vector<reader*> _cuda_sequence;

};

} // namespace algorithms
} // namespace cuda
} // namespace dnn_opt

#endif
