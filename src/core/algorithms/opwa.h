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

#ifndef DNN_OPT_CORE_ALGORITHMS_OPWA
#define DNN_OPT_CORE_ALGORITHMS_OPWA

#include <vector>
#include <memory>
#include <stdexcept>
#include <functional>
#include <core/base/algorithm.h>
#include <core/base/solution_set.h>
#include <core/solutions/wrapper.h>
#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
{
namespace algorithms
{

class opwa : public virtual algorithm
{
public:

  /**
   * Shorthand for the lambda function which creates wrapped algorithms that
   * operates in partitions.
   */
  typedef std::function<algorithm* (solution_set<>*)> wa;

  template<class t_solution>
  static opwa* make(int count, const solution_set<t_solution>* solutions, wa builder)
  {
    auto* result = new opwa(count, solutions, builder);

    result->init();

    return result;
  }

  virtual void reset() override
  {

  }

  virtual void optimize() override
  {
    for(auto& algorithm : _algorithms)
    {
      if(_generator->generate() <= get_density())
      {
        algorithm->optimize();
      }
    }

    for(int i = 0; i < get_solutions()->size(); i++)
    {
      get_solutions()->get(i)->set_modified(true);
    }
  }

  using algorithm::optimize;

  virtual solution* get_best() override
  {
    auto best = get_solutions()->get_best(is_maximization());

    return best;
  }

  virtual void init() override
  {
    for( int i = 0; i < _count; i++ )
    {
      // TODO: Re-use the container <part> multiple times instead delete it

      auto* part = solution_set<>::make(get_solutions()->size());

      for( int j = 0; j < get_solutions()->size(); j++ )
      {
        part->add(solutions::wrapper::make(i, _count, get_solutions()->get(j)));
      }

      auto wa = std::unique_ptr<algorithm>(_builder(part));
      _algorithms.push_back(std::move(wa));

      delete part;
    }
  }

  virtual void set_params(std::vector<float> &params) override
  {
    if(params.size() != 1)
    {
      std::invalid_argument("algorithms::opwa set_params expect 1 value");
    }

    set_density(params.at(0));
  }

  float get_density()
  {
    return _density;
  }

  void set_density(float density)
  {
    if(density < 0 || density > 1)
    {
      throw out_of_range("density is a probability factor in [0,1]");
    }

    _density = density;
  }

  virtual ~opwa() override
  {
    for(auto& algorithm : _algorithms)
    {
      algorithm->get_solutions()->clean();
    }
    delete _generator;
  }

protected:

  template<class t_solution>
  opwa(int count, const solution_set<t_solution>* solutions, wa builder)
  : algorithm(solutions)
  {
    _builder = builder;
    _count = count;
    _density = 1.0f;
    _generator = generators::uniform::make(0.0f, 1.0f);
  }

  template<class t_solution>
  opwa(int count, const solution_set<t_solution>* solutions)
  : algorithm(solutions)
  {
    _count = count;
    _density = 1.0f;
    _generator = generators::uniform::make(0.0f, 1.0f);
  }

protected:

  /** The amount of partitions to create */
  int _count;

  /** Lambda function that creates wrapped algorithms */
  wa _builder;

  /** A list of wrapped algorithms */
  std::vector<std::unique_ptr<algorithm>> _algorithms;

  /** The probablity for a partition to gets optimized */
  float _density;

  generators::uniform* _generator;

};

} // namespace algorithms
} // namespace core
} // namespace dnn_opt

#endif
