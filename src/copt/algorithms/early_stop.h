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

#ifndef DNN_OPT_COPT_ALGORITHMS_EARLY_STOP
#define DNN_OPT_COPT_ALGORITHMS_EARLY_STOP

#include <vector>
#include <functional>
#include <core/algorithms/early_stop.h>
#include <copt/base/algorithm.h>
#include <copt/base/sampler.h>
#include <copt/base/reader.h>
#include <copt/base/shufler.h>
#include <copt/base/set.h>
#include <copt/solutions/network.h>
#include <copt/generators/uniform.h>

namespace dnn_opt
{
namespace copt
{
namespace algorithms
{

class early_stop : public virtual algorithm,
                   public virtual core::algorithms::early_stop
{
public:

  class stopper;

  class test_increase;

  static early_stop* make(algorithm* base, stopper* stopper, reader* reader);

  virtual void init() override;

  using core::algorithms::early_stop::set_params;

  virtual void set_p(float p);

  virtual ~early_stop() override;

protected:

  virtual void set_reader();

  early_stop(algorithm* base, stopper *stopper, reader* reader);

  algorithm* _copt_base;

  float _p;

  reader* _copt_reader;
  stopper* _copt_stopper;
  reader* _copt_train_set;
  reader* _copt_test_set;
  shufler* _copt_shufler;

};


class early_stop::stopper : public virtual core::algorithms::early_stop::stopper
{
public:

  virtual bool stop(float train, float test) = 0;

};

class early_stop::test_increase : public virtual early_stop::stopper,
                                  public virtual core::algorithms::early_stop::test_increase
{
public :

  static test_increase* make(int count, bool is_maximization);

  virtual bool stop(float train, float test) override;

protected:

  test_increase(int count, bool is_maximization);

};

} // namespace algorithms
} // namespace copt
} // namespace dnn_opt

#endif
