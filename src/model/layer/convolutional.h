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

#ifndef DNN_OPT_CORE_LAYERS_CONVOLUTIONAL
#define DNN_OPT_CORE_LAYERS_CONVOLUTIONAL

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
    class convolutional : public layer
    {
    public:

        static shared_ptr< convolutional > make (   int                         in_height,
                                                    int                         in_width,
                                                    int                         in_depth,
                                                    int                         w_height,
                                                    int                         w_width,
                                                    int                         kernel_count,
                                                    int                         padding,
                                                    int                         stride,
                                                    vector< vector< bool > >    connection_table,
                                                    shared_ptr < activation_function >   AF
                                                 )
        {
            return shared_ptr< convolutional >( new convolutional( in_height, in_width, in_depth, w_height, w_width, kernel_count, padding, stride, connection_table, AF ) );
        }

        static shared_ptr< convolutional > make (   int                         in_height,
                                                    int                         in_width,
                                                    int                         in_depth,
                                                    int                         w_height,
                                                    int                         w_width,
                                                    int                         kernel_count,
                                                    int                         padding,
                                                    int                         stride,
                                                    shared_ptr < activation_function >   AF
                                                 )
        {
            vector< vector< bool > > connection_table( in_depth );

            for( int i = 0; i < in_depth; i++ )
            {
                connection_table[ i ] = vector< bool >( kernel_count );

                fill( connection_table[i].begin(), connection_table[ i ].end(), true );
            }

            return shared_ptr< convolutional >( new convolutional( in_height, in_width, in_depth, w_height, w_width, kernel_count, padding, stride, connection_table, AF ) );
        }

        void propagate( const vector< float >& input, const vector< float > &params,  int start,  int end ) override
        {
            assert( input_dimension() == ( int ) input.size() );
            assert( start <= end );
            assert( end <= ( int ) params.size() );
            assert( end - start == count() );

            int  output_index = 0;

            for( int i = ( -1 ) * _padding; i + _w_height <= _in_height + _padding; i += _stride )
            {
                for( int j = ( -1 ) * _padding; j + _w_width <= _in_width + _padding; j += _stride )
                {
                    int parameter_index = start;

                    for(  int kernel = 0; kernel < _kernel_count; kernel++ )
                    {
                        float summatory = 0;

                        for( int w_i = i; w_i < i + _w_height; w_i++ )
                        {
                            for( int w_j = j; w_j < j + _w_width; w_j++ )
                            {
                                /* This is a cero padding z-column so result is cero*/

                                if( w_i < 0 || w_j < 0 || w_i >= _in_height || w_j >= _in_width )
                                {
                                    parameter_index += 1;
                                    continue;
                                }

                                for( int w_k = 0; w_k < _in_depth; w_k++ )
                                {
                                    if( _connection_table[ w_k ][ kernel ] )
                                    {
                                        summatory += input[ get_lineal_index( w_i, w_j, w_k ) ] * params[ parameter_index ];
                                    }

                                    parameter_index += 1;
                                }
                            }
                        }

                        _w_summ[ output_index ]  = summatory + params[ parameter_index ];
                        parameter_index         += 1;
                        output_index            += 1;
                    }
                }
            }

            /* Process activation function */

            for( unsigned int i = 0; i < _w_summ.size(); i++ )
            {
                _output[ i ] = _AF->activation( _w_summ, i );
            }
        }

        int count() const override
        {
            return ( _w_height * _w_width * _in_depth + 1 ) * _kernel_count;
        }

        const vector< float >& output() const override
        {
            return _output;
        }

        layer* clone()
        {
            return new convolutional( _in_height, _in_width, _in_depth, _w_height, _w_width, _kernel_count, _padding, _stride, _connection_table, _AF );
        }

    protected:

        /**
         * @brief convolutional creates a convolutional layer with the specified parameters and parameter
         * sharing. A convolutional layer expects to be provided with 3D inputs @see propagate in the
         * following way:
         *
         *          (1) depth column of (1,1) followed by,
         *          (2) depth column of (1,2) until,
         *          (3) depth column of (1, in_width) and then,
         *          (4) depth column of (in_height, in_width)
         *
         * @param in_height the height of the input volume.
         * @param in_width the width of the input volume.
         * @param in_depth the depth of the input volume.
         * @param w_height the height of a kernel.
         * @param w_width the width of a kernel.
         * @param kernel_count the number of kernels.
         * @param padding the number of ceros added to the border of the input volume.
         * @param stride the number of steeps used for kerners to slide the convolution window.
         * @param AF the activation function of this layer.
         *
         * @throws assertion, if the kernel window can not be stridded in the width/height dimension
         * of the input volume.
         */
        convolutional(  int                         in_height,
                        int                         in_width,
                        int                         in_depth,
                        int                         w_height,
                        int                         w_width,
                        int                         kernel_count,
                        int                         padding,
                        int                         stride,
                        vector< vector< bool > >    connection_table,
                        shared_ptr < activation_function >   AF
                     )
        : layer( in_height * in_width * in_depth,
                 kernel_count * ( ( in_width - w_width + 2 * padding ) / stride + 1 ) * ( ( in_height - w_height + 2 * padding ) / stride + 1 ),
                 move( AF )
               ),
          _output( kernel_count * ( ( in_width - w_width + 2 * padding ) / stride + 1 ) * ( ( in_height - w_height + 2 * padding ) / stride + 1 ) ),
          _w_summ( kernel_count * ( ( in_width - w_width + 2 * padding ) / stride + 1 ) * ( ( in_height - w_height + 2 * padding ) / stride + 1 ) ),
          _connection_table( connection_table )
        {
            assert( ( in_width - w_width + 2 * padding ) % stride == 0 );
            assert( ( in_height - w_height + 2 * padding ) % stride == 0 );

            assert( connection_table.size() == in_depth );

            /* Extend this check for every vector in the connection table */
            assert( connection_table[0].size() ==  kernel_count);

            _in_height      = in_height;
            _in_width       = in_width;
            _in_depth       = in_depth;
            _w_height       = w_height;
            _w_width        = w_width;
            _kernel_count   = kernel_count;
            _padding        = padding;
            _stride         = stride;
        }

    private:

        int get_lineal_index( int x, int y, int z )
        {
            return _in_depth * ( y * _in_width + x ) + z;
        }

         int                         _in_height;
         int                         _in_width;
         int                         _in_depth;
         int                         _w_height;
         int                         _w_width;
         int                         _kernel_count;
         int                         _padding;
         int                         _stride;

         vector< float >             _output;
         vector< float >             _w_summ;
         vector< vector< bool > >    _connection_table;
    };
}
}
}

#endif

