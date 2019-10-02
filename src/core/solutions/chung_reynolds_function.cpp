/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   chung_reynolds_function.cpp
 * Author: Alejandro Ruiz Madruga <amadruga at estudiantes.uci.cu>
 *
 * Created on 29 de septiembre de 2019, 20:03
 */

#include <math.h>
#include <core/solutions/chung_reynolds_function.h>

namespace dnn_opt
{
namespace core
{
namespace solution
{
    chung_reynolds_function* chung_reynolds_function::make(generator* generator,
    unsigned int size)
    {
        auto* result= new chung_reynolds_function(generator, size);
        result->init();
        return result;
    }
    float chung_reynolds_function::calculate_fitness()
    {
        float result=0;
        float* params=get_params();
        
        solution::calculate_fitness();
        
        for(int i=0;i<size();i++)
        {
            result+=pow(params[i],2);
        }
        return pow(result,2);
    }
    chung_reynolds_function::chung_reynolds_function(generator* generator, 
    unsigned int size):solution(generator,size)
    {
    
    }
    chung_reynolds_function::~chung_reynold_function()
    {
    
    }
}//namespace solution
}//namespace core
}//namespace dnn_opt
