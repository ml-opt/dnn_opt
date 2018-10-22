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

#ifndef DNN_OPT_CORE_ERROR
#define DNN_OPT_CORE_ERROR

namespace dnn_opt
{
namespace core
{

/**
 * @brief The error class is intended as an interface for custom
 * error functions that can be used by an artificial neural network.
 *
 * @author Jairo Rojas-Delgado
 * @version 1.0
 * @date June, 2016
 */
class error
{
public:

  /**
   * @brief Accumulate the error value between the output of an artificial
   * neural network and the expected output of the current and previous batches
   * of training patters.
   *
   * @param size the amount of training patterns.
   *
   * @param dim the dimension of the network output.
   *
   * @param out a flatten array of dimension [size, dim] that contains
   * the network output in a row by row fashion.
   *
   * @param exp a flatten array of dimension [size, dim] that contains
   * the expected output in a row by row fashion.
   */
  virtual void ff(int size, int dim, const float* out, const float* exp) = 0;

  /**
   * @brief The value of the accumulated  error. Remove all accumulated error. 
   *
   * @return the accumulated error value.
   */
  virtual float f() = 0;

  /**
   * The basic destructor of this class.
   */
  virtual ~error();

};

} // namespace core
} // namespace dnn_opt

#endif

