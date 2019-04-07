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

#ifndef DNN_OPT_CORE_ALGORITHMS_EARLY_STOP
#define DNN_OPT_CORE_ALGORITHMS_EARLY_STOP

#include <vector>
#include <functional>
#include <core/base/algorithm.h>
#include <core/base/sampler.h>
#include <core/base/reader.h>
#include <core/base/shufler.h>
#include <core/base/set.h>
#include <core/solutions/network.h>
#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

class early_stop : public virtual algorithm
{
public:

  class stopper;

  class test_increase;

  static early_stop* make(algorithm* base, stopper* stopper, reader* reader);

  virtual void reset() override;

  virtual void optimize() override;

  virtual void optimize(int eta, std::function<bool()> on) override;

  virtual void optimize_idev(int count, float dev, std::function<bool()> on) override;

  virtual void optimize_dev(float dev, std::function<bool()> on) override;

  virtual void optimize_eval(int count, std::function<bool()> on) override;

  virtual solution* get_best() override;

  virtual void init() override;

  virtual void set_params(std::vector<float> &params) override;

  using algorithm::set_params;

  virtual float get_p();

  virtual void set_p(float p);

  virtual ~early_stop() override;

protected:

  virtual void set_reader();

  early_stop(algorithm* base, dnn_opt::core::algorithms::early_stop::stopper *stopper, reader* reader);

  /** The base algorithm that performs optimization */
  algorithm* _base;

  /** A pointer of @ref get_solutions() that do not degrade to core::solution */
  set<solutions::network>* _network_solutions;

  float _p;

  reader* _reader;
  stopper* _stopper;
  reader* _train_set;
  reader* _test_set;
  shufler* _shufler;

};


class early_stop::stopper
{
public:

  virtual bool stop(float train, float test) = 0;

};

class early_stop::test_increase : public virtual early_stop::stopper
{
public :

  static test_increase* make(int count, bool is_maximization);

  virtual bool stop(float train, float test) override;

protected:

  test_increase(int count, bool is_maximization);

  int _current;
  int _count;
  bool _is_maximization;
  float _prior_test;

};

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

#endif
