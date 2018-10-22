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

#ifndef DNN_OPT_CORE_READERS_FILE_READER
#define DNN_OPT_CORE_READERS_FILE_READER

#include <fstream>
#include <core/base/reader.h>

namespace dnn_opt
{
namespace core
{
namespace readers
{

/**
 * @brief
 * This class is intended to fetch training patterns from a file.
 * The file must have the following structure:
 *
 *      - In the first line three integers separated by a space: number 
 *      of training patterns p, in dimension n and out dimension m.
 *      - In the following p lines, each line represents a pattern 
 *      containing n floats separated by a space as in, followed by m 
 *      floats as expected out.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date June, 2016
 */
class file_reader : public virtual reader
{
public:

  /**
   * @brief Create a new instance of the file_reader class.
   * 
   * @param file_name the location of the file containing training patterns.
   *
   * @return an instance of this class.
   */
  static file_reader* make(std::string file_name);

  virtual float* const in_data() override;

  virtual float* const out_data() override;

  virtual int get_in_dim() const override;

  virtual int get_out_dim() const override;

  virtual int size() const override;

  /**
   * @brief Destroys each loaded training pattern from memory.
   */
  ~file_reader();

protected:

  /**
   * @brief Load from file.
   */
  void load();

  /**
   * @brief The basic contructor for file_reader class.
   *
   * @param file_name the file location of the training database file.
   *
   * @throws assertion if the file_name provided is incorrect.
   */
  file_reader(std::string file_name);

  /** The number of dimensions in the in training signal */
  int _in_dim;

  /** The number of dimensions in the out training signal */
  int _out_dim;

  /** The amount of training patterns */
  int _size;

  /** The loaded in training data from file */
  float*  _in_data;

  /** The loaded out training data from file */
  float*  _out_data;

  /** File input stream */
  std::ifstream _file;

};

} // namespace readers
} // namespace core
} // namespace dnn_opt

#endif
