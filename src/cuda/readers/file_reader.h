/*
Copyright (c) 2017, Jairo Rojas-Delgado <jrdelgado@uci.cu>
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

#ifndef DNN_OPT_CUDA_READERS_FILE_READER
#define DNN_OPT_CUDA_READERS_FILE_READER

#include <core/readers/file_reader.h>
#include <cuda/base/reader.h>

namespace dnn_opt
{
namespace cuda
{
namespace readers
{

/**
 * @copydoc core::readers::file_reader
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date June, 2016
 */
class file_reader : public virtual reader,
                    public virtual core::readers::file_reader
{
public:

  /**
   * @brief Create a new instance of the file_reader class.
   *
   * @param file_name the location of the file containing training patterns.
   *
   * @return an instance of this class.
   */
  static file_reader* make(std::string file_name, int batches = 1);

  virtual float* in_data() override;

  virtual float* out_data() override;

  /**
   * @brief Destroys each loaded training pattern from memory.
   */
  virtual ~file_reader();

protected:

  /**
   * @brief The basic contructor for file_reader class.
   *
   * @param file_name the file location of the training database file.
   *
   * @throws assertion if the file_name provided is incorrect.
   */
  file_reader(std::string file_name, int batches = 1);

  /** The loaded in training data from file */
  float*  _dev_in_data;

  /** The loaded out training data from file */
  float*  _dev_out_data;

};

} // namespace readers
} // namespace core
} // namespace dnn_opt

#endif
