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

#ifndef DNN_OPT_COPT_SOLUTIONS_NETWORK
#define DNN_OPT_COPT_SOLUTIONS_NETWORK

#include <vector>
#include <initializer_list>
#include <core/solutions/network.h>
#include <copt/base/error.h>
#include <copt/base/reader.h>
#include <copt/base/generator.h>
#include <copt/base/solution.h>
#include <copt/base/layer.h>

namespace dnn_opt
{
namespace copt
{
namespace solutions
{

class network : public virtual solution,
                public virtual core::solutions::network
{
public:

  static network* make(generator* generator, reader* reader, error* error);

  virtual network* clone() override;

  virtual bool assignable(const dnn_opt::core::solution* s) const override;

  virtual reader* get_reader() const override;

  virtual error* get_error() const override;

  /**
   * @brief The basic destructor of the network class.
   */
  virtual ~network();

protected:

  /** Forward declaration of linked network class  */
  class linked;

  network(generator* generator, reader* reader, error* error);

};

class network::linked : public virtual network,
                        public virtual core::solutions::network::linked
{
friend class network;

  virtual reader* get_reader() const override;

protected:

  linked(network* source);

};

} // namespace solutions
} // namespace copt
} // namespace network

#endif
