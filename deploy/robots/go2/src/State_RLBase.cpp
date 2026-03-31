#include "FSM/State_RLBase.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

namespace {

namespace fs = std::filesystem;

struct HardwareExperimentLogger
{
    explicit HardwareExperimentLogger(const fs::path& policy_dir)
    : sportstate(std::make_shared<unitree::robot::go2::subscription::SportModeState>())
    {
        run_dir = make_run_dir();
        fs::create_directories(run_dir);

        csv.open(run_dir / "tracking_log.csv", std::ios::out | std::ios::trunc);
        csv << "time_sec,"
            << "command_vx_mps,"
            << "command_vy_mps,"
            << "command_yaw_radps,"
            << "tracked_vx_mps,"
            << "tracked_vy_mps,"
            << "tracked_yaw_radps,"
            << "tracked_linear_velocity_available\n";
        csv << std::fixed << std::setprecision(6);

        std::ofstream metadata(run_dir / "metadata.txt", std::ios::out | std::ios::trunc);
        metadata << "policy_dir: " << policy_dir << "\n";
        metadata << "deploy_project_dir: " << param::proj_dir << "\n";
        metadata << "log_created_at: " << timestamp_string() << "\n";

        sportstate->set_timeout_ms(200);
        t0 = std::chrono::steady_clock::now();

        spdlog::info("Hardware experiment logging to {}", run_dir.string());
    }

    void log_sample(isaaclab::ManagerBasedRLEnv* env)
    {
        if (!csv.is_open() || !env || !env->robot || !env->robot->data.joystick) {
            return;
        }

        const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
        auto* joystick = env->robot->data.joystick;

        const float command_vx = std::clamp(
            joystick->ly(),
            ranges["lin_vel_x"][0].as<float>(),
            ranges["lin_vel_x"][1].as<float>()
        );
        const float command_vy = std::clamp(
            -joystick->lx(),
            ranges["lin_vel_y"][0].as<float>(),
            ranges["lin_vel_y"][1].as<float>()
        );
        const float command_yaw = std::clamp(
            -joystick->rx(),
            ranges["ang_vel_z"][0].as<float>(),
            ranges["ang_vel_z"][1].as<float>()
        );

        float tracked_vx = std::numeric_limits<float>::quiet_NaN();
        float tracked_vy = std::numeric_limits<float>::quiet_NaN();
        const float tracked_yaw = env->robot->data.root_ang_vel_b[2];
        bool tracked_linear_velocity_available = false;

        if (sportstate && !sportstate->isTimeout()) {
            std::lock_guard<std::mutex> lock(sportstate->mutex_);
            const Eigen::Vector3f velocity_w = sportstate->velocity();
            const Eigen::Vector3f velocity_b = env->robot->data.root_quat_w.conjugate() * velocity_w;
            tracked_vx = velocity_b[0];
            tracked_vy = velocity_b[1];
            tracked_linear_velocity_available = true;
        } else if (!sportstate_timeout_warned) {
            spdlog::warn(
                "rt/sportmodestate is not available. Forward/lateral tracked velocity will be logged as NaN until it becomes available."
            );
            sportstate_timeout_warned = true;
        }

        const double time_sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        csv << time_sec << ","
            << command_vx << ","
            << command_vy << ","
            << command_yaw << ","
            << tracked_vx << ","
            << tracked_vy << ","
            << tracked_yaw << ","
            << (tracked_linear_velocity_available ? 1 : 0) << "\n";
        csv.flush();
    }

private:
    static fs::path experiment_root()
    {
        return param::proj_dir.parent_path().parent_path().parent_path() / "cdc_paper" / "experiment";
    }

    static std::string timestamp_string()
    {
        const auto now = std::chrono::system_clock::now();
        const auto time = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&time);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
        return oss.str();
    }

    static fs::path make_run_dir()
    {
        const fs::path root = experiment_root();
        fs::create_directories(root);

        const std::string base = timestamp_string();
        fs::path run_dir = root / base;
        int suffix = 1;
        while (fs::exists(run_dir)) {
            run_dir = root / (base + "_" + std::to_string(suffix));
            ++suffix;
        }
        return run_dir;
    }

    fs::path run_dir;
    std::ofstream csv;
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> sportstate;
    std::chrono::steady_clock::time_point t0;
    bool sportstate_timeout_warned = false;
};

std::unordered_map<const State_RLBase*, std::unique_ptr<HardwareExperimentLogger>> g_experiment_loggers;
std::mutex g_experiment_loggers_mutex;

HardwareExperimentLogger* get_experiment_logger(const State_RLBase* state)
{
    std::lock_guard<std::mutex> lock(g_experiment_loggers_mutex);
    auto it = g_experiment_loggers.find(state);
    return it == g_experiment_loggers.end() ? nullptr : it->second.get();
}

} // namespace

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::resolve_policy_dir(cfg);

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );

    {
        std::lock_guard<std::mutex> lock(g_experiment_loggers_mutex);
        g_experiment_loggers[this] = std::make_unique<HardwareExperimentLogger>(policy_dir);
    }
}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }

    if (auto* logger = get_experiment_logger(this)) {
        logger->log_sample(env.get());
    }
}
