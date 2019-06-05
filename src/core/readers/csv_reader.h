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

#ifndef DNN_OPT_CORE_READERS_CSV_READER
#define DNN_OPT_CORE_READERS_CSV_READER

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
 * This class is intended to fetch training patterns from a CSV file.
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date June, 2016
 */
class csv_reader : public virtual reader
{
public:

  /**
   * @brief Create a new instance of the csv_reader class.
   * 
   * @param file_name the location of the file containing training patterns.
   * @param in_dim number of input dimensions.
   * @param out_dim number of output dimensions.
   * @param sep separator character, default is a colon.
   * @param header if the csv file contains a header with column's name or not.
   * @param batches number of training patterns to load at once, use when the
   *        number of training patterns is too large to fit in memory.
   *
   * @return an instance of this class.
   */
  static csv_reader* make(std::string file_name, int in_dim, int out_dim, char sep = ',', bool header = false);

  virtual float* in_data() override;

  virtual float* out_data() override;

  virtual int get_in_dim() const override;

  virtual int get_out_dim() const override;

  virtual int size() const override;

  /**
   * @brief Destroys each loaded training pattern from memory.
   */
  ~csv_reader();

protected:

  /**
   * @brief Load the next batch from file.
   */
  void load();

  int get_line_count();

  /**
   * @brief The basic contructor for csv_reader class.
   *
   * @param file_name the location of the file containing training patterns.
   * @param in_dim number of input dimensions.
   * @param out_dim number of output dimensions.
   * @param sep separator character, default is a colon.
   * @param header if the csv file contains a header with column's name or not.
   * @param batches number of training patterns to load at once, use when the
   *        number of training patterns is too large to fit in memory.
   *
   * @throws assertion if the file_name provided is incorrect.
   */
  csv_reader(std::string file_name, int in_dim, int out_dim, char sep = ',', bool header = false);

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

  char _sep;

  bool _header;

  /** File input stream */
  std::ifstream _file;

};

} // namespace readers
} // namespace core
} // namespace dnn_opt

#endif
