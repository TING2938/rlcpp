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

namespace rlcpp
{

/**
 * @brief 获取本地时间日期字符串
 * @param fmt 时间日期格式，参见 https://en.cppreference.com/w/cpp/io/manip/put_time
 * @return 时间日期字符串
 */
inline std::string localTime(const char* fmt = "%Y-%m-%d %H:%M:%S %A")
{
    time_t t;
    char buf[500];
    t = time(NULL);
    strftime(buf, 500, fmt, localtime(&t));
    return buf;
}

std::vector<std::string> file_match(std::string pattern)
{
#ifdef basename
#undef basename
#endif
    char file_name[1024] = {0};
    std::copy(pattern.begin(), pattern.end(), file_name);
    file_name[pattern.size()] = '\0';
    auto base_name            = basename(static_cast<const char*>(file_name));
    auto dir_name             = dirname(file_name);
    auto dir                  = opendir(dir_name);
    dirent* entry;
    int ret;

    std::vector<std::string> matched;

    if (dir != nullptr) {
        while ((entry = readdir(dir)) != nullptr) {
            ret = fnmatch(base_name, entry->d_name, FNM_PATHNAME | FNM_PERIOD);
            if (ret == 0) {
                matched.push_back(std::string(dir_name) + "/" + entry->d_name);
            }
        }
    }
    closedir(dir);
    return matched;
}

std::string build_reward_model_name(const std::string& basename, float reward)
{
    std::stringstream ss;
    ss << basename << "_Reward_" << std::fixed << std::setprecision(2) << reward << ".params";
    return ss.str();
}

std::pair<std::string, float> load_best_reward_model_name(const std::string& basename)
{
    auto total_file_name = file_match(basename + "*.params");
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
