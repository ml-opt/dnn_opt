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

#ifndef DNN_OPT_CORE_SOLUTIONS_WRAPPER_SOLUTION
#define DNN_OPT_CORE_SOLUTIONS_WRAPPER_SOLUTION

#include <cassert>
#include <core/base/solution.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{

class wrapper : public virtual solution
{
public:

  static wrapper* make(int index, int count, solution* base)
  {
    auto* result = new wrapper(index, count, base);

    result->init();

    return result;
  }

  float fitness() override
  {
    return calculate_fitness();
  }

  void set(unsigned int index, float value) override
  {
    assert(index < _size);

    float* params = _base->get_params();
    params[index + _padding] = value;

    set_modified(true);
  }

  float get(unsigned int index) const override
  {
    assert(index < _size);

    float* params = _base->get_params();

    return params[index + _padding];
  }

  unsigned int size() const override
  {
    return _size;
  }

  float* get_params( ) const override
  {
    return _base->get_params() + _padding;
  }

  solution* clone() override
  {
    auto result = new wrapper(_index, _count, _base);

    return result;
  }

  bool assignable(const solution* s) const override
  {
    return true;

    /* TODO: Check this to include _index and _count comprobation */
  }

  virtual void set_modified(bool modified) override
  {
    _modified = modified && _modifiable;
  }

  virtual bool is_modifiable()
  {
    return _modifiable;
  }

  virtual void set_modifiable(bool modifiable)
  {
    _modifiable = modifiable;
  }

  virtual ~wrapper()
  {

  }

protected:

  wrapper(int index, int count, solution* base)
  : solution(base->get_generator(), 0)
  {
    assert(index >= 0 && index < count);
    assert(count <= base->size());

    _base = base;
    _index = index;
    _count = count;
    _modifiable = true;

    if(index + 1 == count)
    {
      _size =  base->size() / count + base->size() % count;
    } else
    {
      _size = base->size() / count;
    }

    _padding = index * (base->size() / count);
  }

  float calculate_fitness() override
  {
    return _base->fitness();
  }

protected:

  int _index;
  int _count;

  int _size;
  int  _padding;
  bool _modifiable;

  solution*  _base;

};

}
}
}

#endif
