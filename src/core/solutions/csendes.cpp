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
        float result = 0;
        float* params = get_params();
        
        solution::calculate_fitness();
        
        for(int i = 0;i < size(); i++)
        {
            result+=(pow(params[i],6) * (2+sin(1 / params[i])));
        }
        return result;
    }
    csendes::csendes(generator* generator, unsigned int size):
    solution(generator, size)
    {
    
    }
    csendes::~csendes()
    {
    
    }
}
}
}