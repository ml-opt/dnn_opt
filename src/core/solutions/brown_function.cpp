/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   brown_function.cpp
 * Author: Alejandro Ruiz Madruga <amadruga at estudiantes.uci.cu>
 *
 * Created on 28 de septiembre de 2019, 21:56
 */

#include <math.h>
#include <core/solutions/brown_function.h>


namespace dnn_opt
{
namespace core
{
namespace solutions
{
    brown_function* brown_function::make(generator* generator, unsigned int size)
    {
        auto* result = new brown_function(generator, size);
        result->init();
        return result;
    }
    float brown_function::calculate_fitness()
    {
        float result=0;
        float* params=get_params();
        
        solution::calculate_fitness();
        
        for(int i=0;i<size()-1;i++)
        {
           result+=(pow(params[i],2*pow(params[i+1],2)+2)+pow(pow(params[i+1],2),pow(params[i],2)+1));
        }
        return result;
    }
    brown_function::brown_function(generator* generator, unsigned int size)
    :solution(generator,size)
    {
        
    }
    brown_function::~brown_function()
    {
    
    }
    
}//namespace solutions
}//namespace core
}//namespace dnn_opt