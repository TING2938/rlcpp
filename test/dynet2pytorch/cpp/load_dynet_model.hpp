#ifndef __RLCPP_LOAD_DYNET_MODEL_HPP__
#define __RLCPP_LOAD_DYNET_MODEL_HPP__

#include <torch/torch.h>

namespace rlcpp
{
inline void load_from_dynet(torch::nn::Module& model, std::string fnm, std::string key = "")
{
    auto parameters = model.parameters();
    std::string line, head;
    std::stringstream ss;
    int cur = 0;
    float value;
    std::ifstream fid(fnm);
    if (fid) {
        while (std::getline(fid, line)) {
            ss.str(line);
            ss >> head >> head;
            ss.clear();
            std::getline(fid, line);
            if (key == head.substr(0, head.rfind('/'))) {
                auto param_cur = parameters[cur].data();
                if (cur % 2 == 0) {
                    int row = param_cur.size(0);
                    ss.str(line);
                    int i = 0;
                    while (ss >> value) {
                        param_cur[i % row][i / row] = value;
                        i++;
                    }
                    ss.clear();
                } else {
                    ss.str(line);
                    int i = 0;
                    while (ss >> value) {
                        param_cur[i] = value;
                        i++;
                    }
                    ss.clear();
                }
                cur++;
            }
        }
    }
    fid.close();
}
}  // namespace rlcpp

#endif  // !__RLCPP_LOAD_DYNET_MODEL_HPP__