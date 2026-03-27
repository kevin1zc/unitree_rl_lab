// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <stdint.h>
#include <chrono>
#include <iostream>
#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>
#include <iomanip>
#include <algorithm>
#include <vector>

/* ---------- logger ---------- */
namespace spdlog
{
inline void create_logger(std::string log_path)
{
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(log_path, 5 * 1024 * 1024, 5);

    std::vector<spdlog::sink_ptr> sinks {console_sink, rotating_sink};
    auto logger = std::make_shared<spdlog::logger>("unitree", sinks.begin(), sinks.end());

    logger->set_pattern("[%Y-%m-%d %H:%M:%S] [%^%l%$] %v");
    logger->flush_on(spdlog::level::info);

    spdlog::set_default_logger(logger);
}

} // namespace spdlog


namespace param
{
namespace po = boost::program_options;

inline std::string VERSION = "1.0.0.1";
inline std::filesystem::path bin_path;
inline std::filesystem::path proj_dir;
inline std::filesystem::path config_dir;
inline YAML::Node config;
inline po::variables_map cli_vm;

inline std::filesystem::path get_bin_path() {
    std::vector<char> path(1024);
    ssize_t len = readlink("/proc/self/exe", &path[0], path.size());
    if (len != -1) {
        path[len] = '\0';  // Null-terminate the result
        return std::filesystem::path(&path[0]);
    } else {
        spdlog::error("Failed to get executable path.");
        exit(1);
    }
}

/* ---------- config.yaml ---------- */
inline void load_config_file()
{
    assert(std::filesystem::exists(bin_path)); // run param::helper before this function
    if(bin_path.parent_path().filename() == "bin" || bin_path.parent_path().filename() == "build")
    {
        proj_dir = bin_path.parent_path().parent_path();
        config_dir = proj_dir / "config";
    }
    else
    {
        proj_dir = bin_path.parent_path();
        config_dir = proj_dir;
    }

    try {
        std::string config_file = (config_dir / "config.yaml").string();
        if(std::filesystem::exists(config_file))
        {
            config = YAML::LoadFile(config_file);
        }
    } catch (const YAML::BadFile& e) {
        spdlog::error("Failed to load config.yaml: {}", e.what());
        exit(1);
    }
}

inline std::filesystem::path parser_policy_dir(std::filesystem::path policy_dir)
{
    auto has_deploy_artifacts = [](const std::filesystem::path& dir)->bool {
        return std::filesystem::exists(dir / "params" / "deploy.yaml") &&
               std::filesystem::exists(dir / "exported" / "policy.onnx");
    };
    auto has_checkpoint_artifacts = [](const std::filesystem::path& dir)->bool {
        return std::filesystem::exists(dir / "params" / "deploy.yaml") &&
               std::filesystem::exists(dir / "checkpoints");
    };

    if (policy_dir.is_relative()) {
        policy_dir = param::proj_dir / policy_dir;
    }
    policy_dir = policy_dir.lexically_normal();

    if (std::filesystem::is_regular_file(policy_dir)) {
        policy_dir = policy_dir.parent_path();
    }

    if (has_deploy_artifacts(policy_dir)) {
        spdlog::info("Policy directory: {}", policy_dir.string());
        return policy_dir;
    }

    if (!std::filesystem::exists(policy_dir)) {
        spdlog::critical("Policy directory does not exist: {}", policy_dir.string());
        exit(1);
    }

    std::vector<std::filesystem::path> dir_list;
    if (std::filesystem::is_directory(policy_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(policy_dir)) {
            if (entry.is_directory()) {
                dir_list.push_back(entry.path());
            }
        }
    }

    std::sort(dir_list.begin(), dir_list.end());
    std::filesystem::path latest_checkpoint_dir;
    for (auto it = dir_list.rbegin(); it != dir_list.rend(); ++it) {
        if (has_deploy_artifacts(*it)) {
            policy_dir = *it;
            spdlog::info("Policy directory: {}", policy_dir.string());
            return policy_dir;
        }
        if (latest_checkpoint_dir.empty() && has_checkpoint_artifacts(*it)) {
            latest_checkpoint_dir = *it;
        }
    }

    if (has_checkpoint_artifacts(policy_dir)) {
        spdlog::critical(
            "Policy directory '{}' has checkpoints but no exported/policy.onnx. "
            "Export the policy first or choose another run with --policy-run/--policy-dir.",
            policy_dir.string()
        );
        exit(1);
    }
    if (!latest_checkpoint_dir.empty()) {
        spdlog::critical(
            "No deployable policy found under '{}'. Latest checkpoint run '{}' is missing exported/policy.onnx. "
            "Export it first or choose another run with --policy-run/--policy-dir.",
            policy_dir.string(),
            latest_checkpoint_dir.filename().string()
        );
        exit(1);
    }

    spdlog::critical(
        "No deployable policy found under '{}'. Expected params/deploy.yaml and exported/policy.onnx.",
        policy_dir.string()
    );
    exit(1);
}

inline std::filesystem::path resolve_policy_dir(const YAML::Node& cfg)
{
    std::filesystem::path policy_dir = cfg["policy_dir"].as<std::string>();
    if (cli_vm.count("policy-dir") && !cli_vm["policy-dir"].as<std::string>().empty()) {
        policy_dir = cli_vm["policy-dir"].as<std::string>();
    }
    if (policy_dir.is_relative()) {
        policy_dir = param::proj_dir / policy_dir;
    }

    if (cli_vm.count("policy-run") && !cli_vm["policy-run"].as<std::string>().empty()) {
        policy_dir /= cli_vm["policy-run"].as<std::string>();
    }

    return parser_policy_dir(policy_dir);
}

//※ This function must be called at the beginning of main() function
inline po::variables_map helper(int argc, char** argv) 
{
    bin_path = get_bin_path();
    load_config_file();

    po::options_description desc("Unitree Controller");
    desc.add_options()
        ("help,h", "produce help message")
        ("version,v", "show version")
        ("log", "record log file")
        ("domain-id,d", po::value<int>()->default_value(0), "dds domain id")
        ("network,n", po::value<std::string>()->default_value(""), "dds network interface")
        ("policy-dir,p", po::value<std::string>()->default_value(""), "override policy directory or experiment root")
        ("policy-run", po::value<std::string>()->default_value(""), "specific policy run directory inside the policy directory")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    cli_vm = vm;

    if (vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(0);
    }
    if (vm.count("version"))
    {
        std::cout << "Version: " << VERSION << std::endl;
        exit(0);
    }

#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#else
    spdlog::set_level(spdlog::level::info);
#endif
    if(vm.count("log"))
    {
        std::filesystem::create_directories(proj_dir / "log");
        spdlog::create_logger(proj_dir.string() + "/log/log.txt");
    }

    return vm;
}

}
