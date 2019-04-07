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

#ifndef DNN_OPT_COPT_STATICS_CV
#define DNN_OPT_COPT_STATICS_CV

#include <vector>
#include <functional>
#include <copt/solutions/network.h>
#include <copt/base/algorithm.h>
#include <copt/base/proxy_sampler.h>
#include <copt/base/shufler.h>
#include <core/statics/cv.h>

namespace dnn_opt
{
namespace copt
{
namespace statics
{

/**
 * @brief The algorithm class provides basic functionalities to define and
 * implement custom meta-heuristic algorithms used for optimization. In order
 * to extend the library, new algorithms shuold derive from this class.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date June, 2016
 * @version 1.0
 */
class cv : public virtual algorithm,
           public virtual core::statics::cv
{
public:

  static cv* make(int k, algorithm* base, reader* reader);

  virtual void init() override;

  virtual algorithm* get_base() const override;

  virtual reader* get_reader() const override;

  virtual reader* get_fold(int idx) const override;

virtual reader* get_train_data() const;

  /**
   * The basic destructor of this class.
   */
  virtual ~cv();

protected:

  virtual shufler* get_shufler() const override;

  cv(int k, algorithm* base, reader* reader);

private:

  algorithm* _copt_base;
  reader* _copt_reader;
  shufler* _copt_shufler;
  reader* _copt_train_data;

  proxy_sampler** _copt_fold_containers;

};

} // namespace statics
} // namespace copt
} // namespace dnn_opt

#endif
