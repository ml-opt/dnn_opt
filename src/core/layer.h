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

#ifndef DNN_OPT_CORE_LAYER
#define DNN_OPT_CORE_LAYER

#include <vector>
#include <cassert>
#include <memory>

#include <src/core/solution.h>
#include <src/core/activation_function.h>

namespace dnn_opt
{
namespace core
{

    /**
     * @brief The layer class is a superclass for all types of layers that could be implemented for an
     * artificial neural network. Deep learning is a stack of layers in cascade that transform an input
     * signal into a high level representation of such input signal.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @version 1.0
     * @date June, 2016
     */
    class layer
    {
    public:

        /**
         * @brief propagate is an interface to propagate a given input signal to an output. The output value
         * is managed by this class, hence to obtain the current output value resulting of calling this method
         * you should call @see output(). Derived classes are encouraged to call this method for validation.
         *
         * **Warning:** the number of values in the input vector should ve equal to @see input_dimension(). At
         * the same time start and end must be valid indexes in the params vector, start <= end. Failing these requirements
         * produces assertion error.
         *
         * @param input is the input signal to be propagated.
         * @param params is a vector containing a parameter list to be used for this layer.
         * @param start is where the parameters of this layer starts in the params vector.
         * @param end is where the paramters of this layer ends in the params vector.
         *
         * @throws assertion if any of the previously mentioned conditions fails.
         */
        virtual void propagate( const vector< float >& input, const vector< float >& params, int start, int end )
        {
            assert( input.size() == input_dimension() );
            assert( start <= end );
            assert( start >= 0 );
            assert( end <= params.size() );
        }

        /**
         * @brief input_dimension specifies the number of values this layer accepts as input.
         *
         * @return the number of input dimensions.
         */
        int input_dimension() const
        {
            return _in_dim;
        }

        /**
         * @brief output_dimension specifies the number of values this layer produces as output.
         *
         * @return the number of output dimensions.
         */
        int output_dimension() const
        {
            return _out_dim;
        }

        /**
         * @brief count returns the number of parameters that is required by this layer.
         * @return the number of parameters.
         */
        virtual int count() const = 0;

        /**
         * @brief output returns a vector vith the output values of this layer after @see propagate() was called.
         * Calling this method before @see propagate() produces inconsistent results.
         *
         * @return a vector with @see output_dimension() values corresponging to the  output of this layer.
         */
        virtual const vector< float >& output() const = 0;

        /**
         * @brief clone returns an exact copy of this layer.
         * @return a pointer to the copyy of this layer.
         */
        virtual layer* clone() = 0;

    protected:

        /**
         * @brief layer is the basic contructor for this class. Is intended to be used by derived classes that
         * implements the factory pattern.
         *
         * @param in_dim the number of values that this layers expects as input.
         * @param out_dim the number of values that this layer will produce as output.
         * @param AF the activation function that will be used by the units of this layer.
         */
        layer( int in_dim, int out_dim, shared_ptr< activation_function > AF )
        {
            _in_dim  = in_dim;
            _out_dim = out_dim;
            _AF      = AF;
        }

        int                                 _in_dim;
        int                                 _out_dim;

        shared_ptr< activation_function >   _AF;
    };
}
}

#endif

