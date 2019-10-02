/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   cosine_mixture.cpp
 * Author: Alejandro Ruiz Madruga <amadruga at estudiantes.uci.cu>
 *
 * Created on 29 de septiembre de 2019, 20:55
 */

#include <math.h>
#include <core/solutions/cosine_mixture.h>

namespace dnn_opt
{
namespace core
{
namespace solution
{
    cosine_mixture* cosine_mixture::make(generator* generator, 
    unsigned int size)
    {
        auto* result= new cosine_mixture(generator,size);
        result->init();
        return result;
    }
    float cosine_mixture::calculate_fitness()
    {
        float result1=0;
        float result2=0;
        float resultf=0;
        float* params=get_params();
        
        solution::calculate_fitness();
        
        for(int i=0;i<size();i++){
            result1+=cos(5*3.14*params[i]);
        }
        result1=result1*(-0.1);
        for(int j=0;j<size();j++){
            result2+=pow(params[j],2);
        }
        resultf=result1-result2;
        return resultf;
    }
    cosine_mixture::cosine_mixture(generator* generator, unsigned int size)
    : solution(generator,size)
    {
    
    }
    cosine_mixture::~cosine_mixture()
    {
    
    }
}//namespace solution
}//namespace core
}//namespace dnn_opt
