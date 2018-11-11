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

#ifndef DNN_OPT_CORE_SHUFLER
#define DNN_OPT_CORE_SHUFLER

#include <core/base/reader.h>
#include <core/generators/uniform.h>

namespace dnn_opt
{
namespace core
{

class shufler : public virtual reader
{
public:

  static shufler* make(reader* reader, float sample_proportion);

  /**
   * @brief Create a specified number of samplers of the same size dividing
   * the training patterns contained in @ref reader equally.
   *
   * Is responsabilty of the user to de-allocate properly the returned samplers
   * and the array.
   *
   * @param reader the reader containing the original set of training patterns.
   *
   * @param folds the amount of equally divided samples of training patterns.
   *
   * @return an array of size @folds containing pointers to the created
   * samplers.
   */
  static shufler* make(reader* reader, int samples);

  virtual float* in_data() override;

  virtual float* out_data() override;

  virtual int get_in_dim() const override;

  virtual int get_out_dim() const override ;

  virtual int size() const override;

  virtual ~shufler();

protected:

  /**
   * @brief Fisher-Yates shuffle.
   */
  virtual void shufle();

  void swap(int i, int j);

  shufler(reader* reader, int samples);

  float* _in_data;
  float* _out_data;

  int _count;

  reader* _reader;
  int _samples;

  generators::uniform* _generator;

};

} // namespace core
} // namespace dnn_opt

#endif
