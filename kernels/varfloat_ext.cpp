#include <torch/extension.h>


// CUDA declarations
torch::Tensor fromFloat(torch::Tensor in, int exp_bits, int fraction_bits, bool round);
torch::Tensor satBounds(int exp_bits, int fraction_bits);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fromFloat", &fromFloat, "Convert float to varfloat");
    m.def("satBounds", &satBounds, "Get saturation bounds");
}
