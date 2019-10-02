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
        float result = 0;
        float* params = get_params();
        
        solution::calculate_fitness();
        
        for(int i = 0;i < size() - 1; i++)
        {
           float res1=pow(params[i], 2);
           float res2=pow(params[i+1], 2);
           result += (pow(params[i], 2 * res2 + 2) + pow(res2, res1 + 1));
        }
        return result;
    }
    brown_function::brown_function(generator* generator, unsigned int size)
    :solution(generator, size)
    {
        
    }
    brown_function::~brown_function()
    {
    
    }
    
}//namespace solutions
}//namespace core
}//namespace dnn_opt