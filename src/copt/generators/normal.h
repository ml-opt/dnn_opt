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

#ifndef DNN_OPT_COPT_GENERATORS_NORMAL
#define DNN_OPT_COPT_GENERATORS_NORMAL

#include <random>
#include <core/generators/normal.h>
#include <copt/base/generator.h>

namespace dnn_opt
{
namespace copt
{
namespace generators
{

/**
 * @copydoc core::generators::normal
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date September, 2018
 * @version 1.0
 */
class normal : public virtual generator,
               public virtual core::generators::normal
{
public:

  /**
   * Create a new instance of the normal class.
   *
   * @param mean the mean of the normally distributed generator.
   *
   * @param dev the standard deviation from the mean.
   *
   * @return a pointer to an instance of normal class.
   */
  static normal* make(float mean, float dev);

  void generate(int count, float* params) override;

  virtual float generate() override;

  virtual ~normal();

protected:

  /**
   * The basic constructor for the normal class.
   *
   * @param mean the mean of the normally distributed generator.
   *
   * @param dev the standard deviation from the mean.
   */
  normal(float mean, float dev);

};

} // namespace generators
} // namespace copt
} // namespace dnn_opt

#endif
