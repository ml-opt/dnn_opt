#include <math.h>
#include <core/solutions/chung_reynolds_function.h>

namespace dnn_opt
{
namespace core
{
namespace solution
{
    chung_reynolds* chung_reynolds::make(generator* generator, unsigned int size)
    {
        auto* result= new chung_reynolds(generator, size);
        
        result->init();
        
        return result;
    }
    float chung_reynolds::calculate_fitness()
    {
        float result = 0;
        float* params = get_params();
        
        solution::calculate_fitness();
        
        for(int i = 0;i < size(); i++)
        {
            result += pow(params[i],2);
        }
        return pow(result, 2);
    }
    chung_reynolds::chung_reynolds(generator* generator, unsigned int size):
    solution(generator,size)
    {
    
    }
    chung_reynolds::~chung_reynolds()
    {
    
    }
}//namespace solution
}//namespace core
}//namespace dnn_opt
