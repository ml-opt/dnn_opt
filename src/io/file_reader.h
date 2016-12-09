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

#ifndef DNN_OPT_CORE_IO_FILEREADER_H
#define DNN_OPT_CORE_IO_FILEREADER_H

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cassert>

#include <src/core/reader.h>

using namespace std;
using namespace dnn_opt::core;

namespace dnn_opt
{
namespace core
{
namespace io
{
    /**
     * @brief
     * This class is intended to fetch training patterns from file.
     * The file must have the following structure:
     *      - In the first line two integers separated by a space, the input dimension `n`
     *      and the output dimension `m`.
     *      - In the following lines, each line represents a pattern containing `n` doubles
     *      for the input followed by `m` doubles representing the expected output.
     *
     * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
     * @version 1.0
     * @date June, 2016
     */
    class file_reader : public reader
    {

    public:

        /**
         * @brief
         * Implements the factory pattern for this class. Returns an instance of the file_reader
         * class.
         * 
         * @param file_name the location of the file containing training patterns.
         *
         * @return a `shared_ptr< file_reader >` pointer to the `file_reader` class that has been created.
         */
        static shared_ptr< file_reader > make( string file_name )
        {
            return shared_ptr< file_reader >( new file_reader( file_name ) );
        }

        /**
         * @brief Returns a vector containing all input patterns of the file.
         * @return a vector with the input patterns.
         */
        vector< vector< float > >& get_input_data() override
        {
            return _input_data;
        }

        /**
         * @brief Returns a vector containing all output patterns of the file.
         * @return a vector of output patterns.
         */
        vector< vector< float > >& get_output_data() override
        {
            return _output_data;
        }

    protected:

        /**
         * @brief The basic contructor for file_reader class.
         * @param  file_name the file location of the training database file.
         *
         * @throws assertion if the file_name provided is incorrect
         */
        file_reader( string file_name )
        {
            /* TODO. Check correct file structure. */

            ifstream file( file_name );

            assert( file );

            int input_dimension;
            int output_dimension;
            float value;

            file >> input_dimension;
            file >> output_dimension;

            while( !file.eof() )
            {
                vector< float > input_pattern;
                vector< float > output_pattern;

                for( int i = 0; i< input_dimension; i++)
                {
                    file >> value;
                    input_pattern.push_back(value);
                }

                for( int i = 0; i< output_dimension; i++)
                {
                    file >> value;
                    output_pattern.push_back(value);
                }

                _input_data.push_back(input_pattern);
                _output_data.push_back(output_pattern);
            }

            file.close();
        }

    private:

        /** This vector contains the input patterns. */
        vector< vector< float > > _input_data;

        /** This vector contains the output patterns. */
        vector< vector< float > > _output_data;
    };
}
}
}

#endif
