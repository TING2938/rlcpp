#pragma once

#include <string.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>
#include <fnmatch.h>

#include <libgen.h>

#include <cpptools/ct.hpp>
#include <cpptools/linux/file_name_match.hpp>

namespace rlcpp
{
inline std::string build_reward_model_name(const std::string& basename, float reward)
{
    std::stringstream ss;
    ss << basename << "_Reward_" << std::fixed << std::setprecision(2) << reward << ".params";
    return ss.str();
}

inline std::pair<std::string, float> load_best_reward_model_name(const std::string& basename)
{
    auto total_file_name = ct::file_match(basename + "*.params");
    float best_reward    = -1e6f;
    std::string best_fnm = "";

    for (auto&& fnm : total_file_name) {
        auto pos = fnm.rfind("_Reward_");
        if (pos != std::string::npos) {
            auto sub    = fnm.substr(pos + 8, fnm.size() - pos - 15);
            auto reward = std::stof(sub);
            if (reward > best_reward) {
                best_fnm    = fnm;
                best_reward = reward;
            }
        }
    }
    return {best_fnm, best_reward};
}

}  // namespace rlcpp
