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

#ifndef DNN_OPT_CORE_ACTIVATION_FUNCTION
#define DNN_OPT_CORE_ACTIVATION_FUNCTION

#include <vector>

using namespace std;

namespace dnn_opt
{
namespace core
{

    /**
     * @brief The activation_function class is intended to provide an interface for custom activation
     * functions that can be used by an artificial neural network. Derived classes shuld expect to receive
     * weighted summatory outputs of units in a layer. In case of layers outputs that have some specific
     * spatial arrangement, like convolutional layers, specialiced activation functions should consider such
     * implementation details about its output.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @date September, 2016
     * @version 1.0
     */
    class activation_function
    {
    public:

        /**
         * @brief activation is a virtual method that have to be implemented by derived classes.
         * Given the weighted summatory of an artificial neural network layer and the index of a
         * specific unit it must return it's current activation value.
         *
         * @param output, a vector containing a layer output in terms of weighted summatory.
         * @param index, the position of the corresponding unit in the layer which output is going
         * to be calculated.
         *
         * @return the activation value of the processing unit.
         */
        virtual float activation( const vector< float > &output, int index ) = 0;
    };
}
}

#endif

