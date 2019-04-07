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

#ifndef DNN_OPT_CUDA_SOLUTIONS_NETWORK
#define DNN_OPT_CUDA_SOLUTIONS_NETWORK

#include <cuda/base/error.h>
#include <cuda/base/reader.h>
#include <cuda/base/generator.h>
#include <cuda/base/solution.h>
#include <cuda/base/layer.h>
#include <core/solutions/network.h>

namespace dnn_opt
{
namespace cuda
{
namespace solutions
{

class network : public virtual solution,
                public virtual core::solutions::network
{
public:

  static network* make(generator* generator, reader* reader, error* error);

  virtual network* clone() override;

  bool assignable(const core::solution* s) const override;

  virtual reader* get_reader() const override;

  virtual void set_reader(core::reader* reader) override;

  virtual error* get_error() const override;

  virtual float* predict(core::reader* validation_set) override;

  virtual void init() override;

  virtual ~network() override;

protected:

  /** Forward declaration of linked network class  */
  class linked;

  network(generator* generator, reader* reader, error* error);

  network(generator* generator);

};

/* TODO: implement all methods respect _source! */
class network::linked : public virtual network,
                        public virtual core::solutions::network::linked
{
friend class network;

public:

  virtual reader* get_reader() const override;

  virtual void set_reader(core::reader* reader) override;

  virtual error* get_error() const override;

protected:

  linked(network* base);

  /** The linked network solution that is being tracked */
  network* _cuda_base;

};

} // namespace solutions
} // namespace cuda
} // namespace dnn_opt

#endif
