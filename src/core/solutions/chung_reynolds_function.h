/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   chung_reynolds_function.h
 * Author: Alejandro Ruiz Madruga <amadruga at estudiantes.uci.cu>
 *
 * Created on 29 de septiembre de 2019, 19:47
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

#ifndef CHUNG_REYNOLDS_FUNCTION_H
#define CHUNG_REYNOLDS_FUNCTION_H

#include <core/base/generator.h>
#include <core/base/solution.h>
namespace dnn_opt
{
namespace core
{
namespace solution
{
    /**
 * @brief The chung_reynolds_function class represents 
 * an optimization solutions which fitness cost is calculated 
 * via Chung-Reynolds Function.
 *
 * The equation for this function is given by:
 *
 * f(x) = (\sum_{i=0}^{n}{{x_i}^2})^2
 *
 * Chung-Reynolds function have a global minima in x* = f(0,..., 0) with a value
 * of 0.
 * A commonly used search domain for testing is [-100, 100]. Chung Reynolds 
 * Function is continuous, differentiable, partially-separable, scalable and 
 * unimodal. 
 * See the following reference [f_34] in:
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
    class chung_reynolds : public virtual solution
    {
    public:
          /**
   * @brief Returns an instance of this object. This method
   * is an implementation of the factory pattern.
   *
   * @param generator an instance of a generator class. The
   * generator is used to initialize the parameters of this solution.
   *
   * @param size is the number of parameters for this solution. Default is 200.
   *
   * @return a pointer to an instance of the chung_reynolds_function class.
   */
        static chung_reynolds* make(generator* generator, 
        unsigned int size = 200);
        
        virtual ~chung_reynolds();
        
    protected:
        virtual float calculate_fitness();
        
        chung_reynolds(generator* generator, unsigned int size = 200);
    };
}//namespace solution
}//namespace core
}//namespace dnn_opt

#endif /* CHUNG_REYNOLDS_FUNCTION_H */

