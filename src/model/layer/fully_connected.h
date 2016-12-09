/*
    Copyright (c) 2016, Jairo Rojas-Delgado
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

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef DNN_OPT_CORE_LAYERS_FULLY_CONNECTED
#define DNN_OPT_CORE_LAYERS_FULLY_CONNECTED

#include <vector>
#include <memory>

#include <src/core/layer.h>

using namespace std;
using namespace dnn_opt::core;

namespace dnn_opt
{
namespace core
{
namespace layers
{
    /**
     * @brief The fully_connected_layer class represents a layer of processing units of an artificial
     * neural network where each unit is fully connected to the output of the previous layer. When considering
     * the layer parameters, those are arranged in a consecutive way such as: a unit weights and bias term are
     * followed by the next unit's weights and bias. Layer's parameters are provided externally, hence this class
     * is intended to provide only an add-hoc feature for ann_pt::solution derived classes.
     *
     * @author: Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date September, 2016
     * @version 1.0
     */

    class fully_connected : public layer
    {
    public:

        static shared_ptr< fully_connected > make( int in_dim, int out_dim, shared_ptr< activation_function > AF )
        {
            return shared_ptr< fully_connected >( new fully_connected( in_dim, out_dim, move( AF ) ) );
        }

        /**
         * @brief propagate a given input signal through the layer by calculating the multiplication of
         * each input signal by the unit's parameters.
         *
         * @param input, a vector containing the input signal to be propagated.
         * @param param_begin, an iterator pointing to the begining of a container with the
         * layer parameters.
         * @param param_end, an iterator pointing to the end of this layer parameters in a container.
         *
         * @return a vector with the ouput values of this layer.
         */
        void propagate( const vector< float >& input, const vector< float >& params, int start, int end ) override
        {
            assert( input_dimension() == ( int ) input.size() );
            assert( start <= end );
            assert( end <= ( int ) params.size() );
            assert( end - start == count() );

            /* weighted summatory of layer units */

            fill( _w_summ.begin(), _w_summ.end(), 0.0f );

            for( int i = start; i < end; i++ )
            {
                int unit   = ( i - start ) / ( input_dimension() + 1 );
                int signal = ( i - start ) % ( input_dimension() + 1 );

                _w_summ[ unit ] += params[ i ] * ( signal == input_dimension() ? 1 : input[ signal ] );
            }

            /* activation function */

            for( int i = 0; i < output_dimension(); i++ )
            {
                _output[ i ] = _AF->activation( _w_summ, i );
            }
        }

        /**
         * @brief count returns the number of parameters that is required by this layer.
         * This is input_dimension() * output_dimension() since this is a fully connected layer.
         *
         * @return the number of parameters.
         */
        int count() const override
        {
            return ( input_dimension() + 1) * ( output_dimension() );
        }

        const vector< float >& output() const override
        {
            return _output;
        }

        layer* clone()
        {
            return new fully_connected( _in_dim, _out_dim, _AF );
        }

    protected:

        /**
         * @brief fully_connected_layer creates a fully_connected_layer instance.
         *
         * @param input_dimension the number of input dimensions.
         * @param output_dimension the number of output dimensions.
         */
        fully_connected( int in_dim, int out_dim, shared_ptr < activation_function > AF )
        : layer( in_dim, out_dim, move( AF ) ),
          _output( out_dim ),
          _w_summ( out_dim )
        {

        }

        vector< float >             _output;
        vector< float >             _w_summ;
    };
}
}
}

#endif // FULLY_CONNECTED
