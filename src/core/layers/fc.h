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

#ifndef DNN_OPT_CORE_LAYERS_fc
#define DNN_OPT_CORE_LAYERS_fc

#include <core/base/layer.h>
#include <core/base/activation.h>

namespace dnn_opt
{
namespace core
{
namespace layers
{

/**
 * @brief The fc_layer class represents a layer of processing
 * units of an artificial neural network where each unit is fully connected
 * to the out of the previous layer.
 *
 * @author: Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @date September, 2016
 * @version 1.0
 */
class fc : public virtual layer
{
public:

  /**
   * @brief Create a new instance of the fc class.
   *
   * @param in_dim the number of in values.
   *
   * @param out_dim the number of neurons of the layer.
   *
   * @param activation the activation function used by each neuron.
   *
   * @return a pointer to a new instance of the fc class.
   */
  static fc* make(int in_dim, int out_dim, activation* activation);

  /**
   * @copydoc dnn_opt::core::layer::prop()
   *
   * The first weight_size() parameters are considered weights and the other
   * biases_size() parameters are considered biases.
   */
  virtual void prop(int size, const float* in, const float* params, float* out) override;

  virtual int size() const override;

  virtual int weight_size() const override;

  virtual int bias_size() const override;

  virtual layer* clone() override;

protected:

  /**
   * Calculate the weighted sumatory of the inputs with the parameters for
   * each neuron in the layer and stores the results in the out array for each
   * training pattern.
   *
   * @param size the amount of training patterns to propagate.
   *
   * @param in a flatten array containing [size, in_get_dim()] elements
   * representing the in signal to be propagated in a row by row fashion.
   *
   * @param params an array containing at least size() elements
   * representing the parameters to be used by this layer.
   *
   * @param[out] out flatten array of containing @ref size x @ref out_get_dim()
   * elements representing the out signal in a column by column fashion.
   */
  virtual void ws(int size, const float* in, const float* params, float* out);

  /**
   * @brief fc_layer creates a fc_layer instance.
   *
   * @param in_dimension the number of in dimensions.
   *
   * @param out_dimension the number of out dimensions.
   */
  fc(int in_dim, int out_dim, activation* activation);

  /** The number of parameters required by this layer */
  int _size;

  /** The number of parameters considered as weights */
  int _weight_size;

};

} // namespace layers
} // namespace fc
} // namespace core

#endif
