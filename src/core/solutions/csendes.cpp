/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   csendes.cpp
 * Author: Alejandro Ruiz Madruga <amadruga at estudiantes.uci.cu>
 *
 * Created on 2 de octubre de 2019, 10:19
 */

#include <math.h>
#include <core/solutions/csendes.h>

namespace dnn_opt
{
namespace core
{
namespace solution
{
    csendes* csendes::make(generator* generator, unsigned int size)
    {
        auto* result= new csendes(generator, size);
        result->init();
        return result;
    }
    float csendes::calculate_fitness()
    {
        float result=0;
        float* params=get_params();
        
        solution::calculate_fitness();
        
        for(int i=0;i<size();i++)
        {
            result+=(pow(params[i],6)*(2+sin(1/params[i])));
        }
        return result;
    }
    csendes::csendes(generator* generator, unsigned int size)
    :solution(generator, size)
    {
    
    }
    csendes::~csendes()
    {
    
    }
}
}
}