/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   brown_function.h
 * Author: alejandrom247 amadruga@estudiantes.uci.cu
 *
 * Created on 28 de septiembre de 2019, 21:33
 */
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


#ifndef BROWN_FUNCTION_H
#define BROWN_FUNCTION_H

#include <core/base/generator.h>
#include <core/base/solution.h>

namespace dnn_opt
{
namespace core
{
namespace solutions
{
    /**
 * @brief The brown_function class represents an optimization solutions which
 * fitness cost is calculated via Brown Function.
 *
 * The equation for this function is given by:
 *
 * f(x) = \sum_{i=1}^{n-1}{{{{x_i}^2}^{{{x_i+1}^2}+1}}+{x_i+1}^{{{x_i}^2}+1}}
 *
 * Brown function have a global minima in x* = f(0,..., 0) with a value of 0.
 * A commonly used search domain for testing is [-1, 4]. Brown Function 
 * is continuous, differentiable, non-separable, scalable and unimodal. 
 * See the following reference [f_25] in:
 * 
 * MOMIN, JAMIL; YANG, Xin-She. A literature survey of benchmark functions for 
 * global optimization problems. Journal of Mathematical Modelling and Numerical 
 * Optimisation, 2013, vol. 4, no 2, p. 150-194.
 *
 *
 * @author Jairo Rojas-Delgado <jrdelgado@uci.cu>
 * @version 1.0
 * @date November, 2016
 */
    class brown_function : public virtual solution
    {
    public:
          /**
   * @brief Returns an instance of this object. This method
   * is an implementation of the factory pattern.
   *
   * @param generator an instance of a generator class. The
   * generator is used to initialize the parameters of this solution.
   *
   * @param size is the number of parameters for this solution. Default is 5.
   *
   * @return a pointer to an instance of the brown_function class.
   */
        
        static brown_function* make(generator* generator, unsigned int size=5);
        
        virtual ~brown_function();
        
    protected:
        
        virtual float calculate_fitness();
        
        brown_function(generator* generator,unsigned int size=5);
    };
}//namespace solutions
}//namespace core
}//namespace dnn_opt

#endif /* BROWN_FUNCTION_H */

