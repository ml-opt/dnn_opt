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

#ifndef DNN_OPT_CORE_READER_H
#define DNN_OPT_CORE_READER_H

namespace dnn_opt
{
namespace core
{

/**
 * @brief The reader class is intended to  provide an interface to feed
 * training data into dnn_opt.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date June, 2016
 */
class reader
{
public:

  /**
   * @brief The input patterns.
   *
   * @return a flatten array of dimension [@ref size() x @ref in_get_dim()]
   * containing the input patterns in a column by column fashion.
   */
  virtual float* in_data() = 0;

  /**
   * @brief The output patterns.
   *
   * @return a flatten array of dimension [@ref size(), @ref out_get_dim()]
   * containing the output patterns in a column by column fashion.
   */
  virtual float* out_data() = 0;

  /**
   * @brief The number of dimensions of the input patterns.
   *
   * @return an integer with the number of dimensions of the input patterns.
   */
  virtual int get_in_dim() const = 0;

  /**
   * @brief The number of dimensions of the output patterns.
   *
   * @return an integer with the number of dimensions of the output patterns.
   */
  virtual int get_out_dim() const = 0;

  /**
   * @brief Returns the number of training patterns loaded by this reader.
   *
   * @return an integer with the number of training patterns.
   */
  virtual int size() const = 0;

  /**
   * @brief Swap the training patterns of this reader with the @ref other
   * reader.
   *
   * This reader and the other reader must have the same @ref size() and
   * @ref get_in_dim() and @ref get_out_dim().
   *
   * @param other the reader to swap for this.
   *
   * @throws std::invalid_argument if the other reader do not have the same
   * @ref size() or the same @ref get_in_dim() or the same @ref get_out_dim().
   */
  virtual void swap(reader* other);

  virtual ~reader();

};

} // namespace core
} // namespace dnn_opt

#endif
