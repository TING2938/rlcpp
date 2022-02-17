#define RLCPP_STATE_TYPE 1
#define RLCPP_ACTION_TYPE 0

#include <fstream>
#include <iostream>
#include <regex>
#include "common/rl_config.h"
#include "tools/memory_reply.h"
#include "tools/random_tools.h"
#include "tools/vector_tools.h"

int main()
{
    rlcpp::RandomReply memory;
    memory.init(200);
    std::ifstream fid("/home/yeting/work/project/rlcpp/build/dqn_memory.dat");
    if (!fid) {
        return -1;
    }

    while (fid >> memory) {
    }
    fid.close();
}
