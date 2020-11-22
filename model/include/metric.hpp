#ifndef __METRIC__
#define __METRIC__
#include <torch/torch.h>

#ifdef __cplusplus

extern "C"{
#endif // __cplusplus

namespace loss{

    struct arcMarginImpl : torch::nn::Module{

        arcMarginImpl( int64_t N, int64_t M, float s, float m, bool easy_margin);
        torch::Tensor forward(const torch::Tensor& input, const torch::Tensor& label);
        
        void getParamters(){
            for( const auto& pair : this->named_parameters() ){
                std::cout << pair.key() << ": " << pair.value() << std::endl;
            }
    
        }

        torch::nn::Linear fc;
        bool easy_margin;
        torch::Tensor weight;

        int64_t N;
        int64_t M; 
        float s; 
        float m;
        float cos_m;
        float cos_comp;
        float sin_m;
        float mm;
    };

    TORCH_MODULE(arcMargin);

}

#ifdef __cplusplus

}
#endif // __cplusplus

#endif // !__METRIC__