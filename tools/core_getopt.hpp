#ifndef __CORE_GETOPT_H__
#define __CORE_GETOPT_H__

#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstdio>

namespace itp
{
    /**
     * 获取命令行选项
     * 用法示例：
     *      itp::Getopt getopt(argc, argv);
     *      int a = 1;
     *      double b = 1.3;
     *      getopt(a, "-a", TRUE, "get value of a");
     *      getopt(b, "-b", TRUE, "get value of b");
     *      getopt.finish();
    */
    class Getopt
    {
    public:

        Getopt() = default;

        /**
         * @brief 构造函数
         * @param gc argc
         * @param gv argv
         * @param msg 程序说明
         * @return
        */
        Getopt(int gc, char** gv, std::string msg) : _mainMessage(msg)
        {
            for (int i = 0; i != gc; i++) {
                argv.push_back(gv[i]);
            }
            auto p = findArgs("-h");
            _printHelpMain = (p == argv.begin() + 1);
            _printHelpSub = (argv.size() == 3 && p == argv.begin() + 2);

        }

        Getopt(std::string str, std::string msg) : _mainMessage(msg)
        {
            if (str[0] == '-') {
                argv.push_back(" ");
            }
            std::stringstream ss(str);
            std::string tmp;
            while (ss >> tmp) {
                argv.push_back(tmp);
            }
            auto p = findArgs("-h");
            _printHelpMain = (p == argv.begin() + 1);
            _printHelpSub = (argv.size() == 3 && p == argv.begin() + 2);
        }

        // set command from string
        void setCommand(std::string str, std::string msg)
        {
            if (str[0] == '-') {
                argv.push_back(" ");
            }
            _mainMessage = msg;
            std::stringstream ss(str);
            std::string tmp;
            while (ss >> tmp) {
                argv.push_back(tmp);
            }
            auto p = findArgs("-h");
            _printHelpMain = (p == argv.begin() + 1);
            _printHelpSub = (argv.size() == 3 && p == argv.begin() + 2);
        }

        void setCommand(int gc, char** gv, std::string msg)
        {
            _mainMessage = msg;
            for (int i = 0; i != gc; i++) {
                argv.push_back(gv[i]);
            }
            auto p = findArgs("-h");
            _printHelpMain = (p == argv.begin() + 1);
            _printHelpSub = (argv.size() == 3 && p == argv.begin() + 2);
        }

        /**
         * @brief 获取标量类型命令行参数
         * @tparam T 标量类型
         * @param x 输入数据
         * @param str 命令行参数指示字符串
         * @param required 是否是必需参数
         * @param msg 参数说明
        */
        template<typename T>
        void operator()(T& x, std::string str, bool required, std::string msg)
        {
            if (addHelpInfo(required, str, x, msg))
                return;

            auto p = findArgs(str);
            if (p != argv.end()) {
                std::stringstream ss{ *(p + 1) };
                ss >> x;
            } else checkRequired(required, str);
        }

        /**
         * @brief 获取布尔类型命令行参数
         * @param x 输入数据
         * @param str 命令行参数指示字符串
         * @param required 是否是必需参数
         * @param msg 参数说明
        */
        void operator()(bool& x, std::string str, bool required, std::string msg)
        {
            if (addHelpInfo(required, str, x, msg))
                return;

            auto p = findArgs(str);
            if (p != argv.end()) {
                std::stringstream ss{ *(p + 1) };
                int tmp;
                ss >> tmp;
                if (tmp == 0 || tmp == 1) {
                    x = tmp;
                } else {
                    std::printf("Error of input, 1 means true and 0 means false\n");
                    std::exit(-1);
                }
            }
            else checkRequired(required, str);
        }

        /**
         * @brief 获取数组类型命令行参数
         * @tparam T 标量类型
         * @param x 输入数据
         * @param str 命令行参数指示字符串
         * @param length 数组的长度，0表示数组长度没有限制
         * @param required 是否是必需参数
         * @param msg 参数说明
        */
        template <typename T>
        void getArray(std::vector<T>& x, std::string str, int length, bool required, std::string msg)
        {
            std::stringstream ss;
            ss << "[";
            if (x.size() > 0) {
                ss << x[0];
            }
            for (int i = 1; i < x.size(); i++) {
                ss << ", " << x[i];
            }
            ss << "]";
            if (addHelpInfo(required, str+(length ? "(len:"+std::to_string(length)+")" : "(len:auto)"), 
                ss.str(), msg))
                return;

            auto b = findArgs(str);
            if (b != argv.end()) {
                x.clear();
                auto beg = std::find_if(b + 1, argv.end(),
                    [](const std::string& chr) { return chr.front() == '['; });
                if (beg == argv.end())
                {
                    std::fprintf(stderr, "Error of command %s!\n", str.c_str());
                    std::exit(-1);
                }
                auto end = std::find_if(b + 1, argv.end(),
                    [](const std::string& chr) { return chr.back() == ']'; });
                if (end == argv.end() || end - beg < 0)
                {
                    std::fprintf(stderr, "Error of command %s!\n", str.c_str());
                    std::exit(-1);
                }

                std::string tmpStr;
                for (auto i = beg; i != end + 1; i++)
                {
                    tmpStr += " ";
                    tmpStr += *i;
                }
                for (auto&& s : tmpStr)
                {
                    if (s == '[' || s == ']' || s == ',')
                        s = ' ';
                }
                std::string tmpString;
                T tmpValue;
                std::stringstream ss { tmpStr };
                while (ss >> tmpString)
                {
                    std::stringstream ss2{ tmpString };
                    ss2 >> tmpValue;
                    x.push_back(tmpValue);
                }
                if (length) {
                    if (x.size() != length) {
                        std::fprintf(stderr, "Option \"%s\": require length of %d, "
                            "but give %zd value, please check your commandline!\n", 
                            str.c_str(), length, x.size());
                        std::exit(-1);
                    }
                }
            } else checkRequired(required, str);
        }

        /**
         * @brief 获取固定位置命令行参数，从1开始表示第一个命令行
         * @tparam T 标量类型
         * @param x 输入数据
         * @param pos 参数位置
         * @param required 是否是必需参数
         * @param msg 参数说明
        */
        template <typename T>
        void getFixPos(T& x, int pos, bool required, std::string msg)
        {
            if (addHelpInfo(required, "pos: " + std::to_string(pos), x, msg))
                return;

            std::stringstream ss{ argv[pos] };
            ss >> x;
        }

        /**
         * @brief 添加子程序
         * @tparam Func 子程序执行的函数类型
         * @param str 命令行参数指示字符串
         * @param msg 子程序说明
         * @param func 子程序执行的函数
        */
        template <typename Func>
        void addSubProgram(std::string str, std::string msg, Func&& func)
        {
            if (_printHelpMain) {
                std::stringstream ss;
                ss << std::setiosflags(std::ios::right) << std::setw(15) << str 
                   << "     " << std::setiosflags(std::ios::left) << msg << "\n";
                _subprogramInfo += ss.str();
                return;
            }

            if (str == argv[1]) {
                func();
            }
        }

        /**
         * @brief 获取完所有参数后调用
        */
        void finish()
        {
            if (_printHelpMain) {
                if (!_mainMessage.empty()) {
                    std::printf("%s\n", _mainMessage.c_str());
                }
                std::printf("Command line option:\n");
                std::printf("%s\n\n", toString().c_str());

                if (!(_requiredInfo.empty() && _optionalInfo.empty())) {
                    std::printf("\033[31m%11s%14s%36s\033[0m\n", "Option", "Value", "Description");
                    std::printf("%s\n", std::string(61, '-').c_str());

                    if (!_requiredInfo.empty()) {
                        std::printf("\033[33m(Required)\033[0m\n");
                        std::printf("%s", _requiredInfo.c_str());
                    }
                    if (!_optionalInfo.empty()) {
                        std::printf("\033[33m(Optional)\033[0m\n");
                        std::printf("%s", _optionalInfo.c_str());
                    }
                    std::printf("%s\n\n", std::string(61, '-').c_str());
                }

                if (!_subprogramInfo.empty()) {
                    std::printf("Sub-program Message:\n");
                    std::printf("\033[31m%15s     %s\033[0m\n", "Function", "Description");
                    std::printf("%s\n", std::string(56, '-').c_str());
                    std::printf("%s", _subprogramInfo.c_str());
                    std::printf("%s\n\n", std::string(56, '-').c_str());
                    std::printf("For help on a sub-program, use '%s <Function> -h'\n", argv[0].c_str());
                    std::printf("To  access  a sub-program, use '%s <Function> [commandline]'\n\n", argv[0].c_str());
                }
            }
            if (_printHelpSub) {
                if (!_mainMessage.empty()) {
                    std::printf("%s\n", _mainMessage.c_str());
                }
                std::printf("Command line option:\n");
                std::printf("%s\n\n", toString().c_str());

                if (!(_requiredInfo.empty() && _optionalInfo.empty())) {
                    std::printf("\033[31m%11s%14s%36s\033[0m\n", "Option", "Value", "Description");
                    std::printf("%s\n", std::string(56, '-').c_str());

                    if (!_requiredInfo.empty()) {
                        std::printf("\033[33m(Required)\033[0m\n");
                        std::printf("%s", _requiredInfo.c_str());
                    }
                    if (!_optionalInfo.empty()) {
                        std::printf("\033[33m(Optional)\033[0m\n");
                        std::printf("%s", _optionalInfo.c_str());
                    }
                    std::printf("%s\n\n", std::string(56, '-').c_str());
                }
            }
            if (_printHelpMain || _printHelpSub) {
                std::exit(0);
            }
        }

        /**
         * @brief 将命令行参数转化为字符串
         * @return 命令行参数字符串
        */
        std::string toString()
        {
            std::string str(argv[0]);
            for (int i = 1; i != argv.size(); ++i) {
                str += " ";
                str += argv[i];
            }
            return str;
        }

    private:
        void checkRequired(bool required, std::string str)
        {
            if (required) {
                std::printf("Command Line Error: ");
                std::printf("\"%s\"", str.c_str());
                std::printf("option NOT Found, type \"-h\" for help.\nexit!\n");
                std::exit(-2);
            }
        }

        template <typename T>
        bool addHelpInfo(bool required, std::string str, const T& x, std::string msg)
        {
            if (_printHelpMain || _printHelpSub) {
                std::stringstream ss;
                ss << std::setiosflags(std::ios::left) << std::string(5, ' ') 
                    << std::setw(15) << str 
                    << std::setw(30) << x << msg << "\n";
                if (required) {
                    _requiredInfo += ss.str();
                } else {
                    _optionalInfo += ss.str();
                }
                return true;
            }
            return false;
        }

        /*! \brief
         *  return argument position, or (argv+argc) if not found.
         */
        std::vector<std::string>::iterator findArgs(std::string str)
        {
            return std::find(argv.begin() + 1, argv.end(), str);
        }

    private:
        std::vector<std::string> argv;
        bool _printHelpMain;
        bool _printHelpSub;
        std::string _mainMessage;
        std::string _requiredInfo;
        std::string _optionalInfo;
        std::string _subprogramInfo;  // for sub program;

    }; // class Getopt;
}

#endif // !__CORE_GETOPT_H__
