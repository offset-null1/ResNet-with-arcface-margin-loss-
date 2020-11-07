#ifndef __LOAD__
#define __LOAD__

#include <torch/torch.h>
#include <string>
#include <vector>

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus

    class loadData : public torch::data::dataset<loadData> {
        private:
            std::vector<torch::Tensor> images, labels;

        public:

            loadData(vector<std::string> img_path, vector<std::string> label_path){
                images = load(img_path, true);
                labels = load(label_path, false);
            }

            torch::data::Example<> get(size_t index) override {
                
                return {images[index], labels[index]};
            }

            torch::data::optional<size_t> size() const override {
                
                return images.size(0);
            }
    };

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // !__LOAD__