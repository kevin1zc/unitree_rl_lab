#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;
std::shared_ptr<Gamepad> FSMState::gamepad = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    if (param::config["keyboard_joystick"] && param::config["keyboard_joystick"]["enabled"] &&
        param::config["keyboard_joystick"]["enabled"].as<bool>()) {
        FSMState::keyboard = std::make_shared<Keyboard>();
    }
    if (param::config["gamepad_joystick"] && param::config["gamepad_joystick"]["enabled"] &&
        param::config["gamepad_joystick"]["enabled"].as<bool>()) {
        std::string device = "/dev/input/js0";
        if (param::config["gamepad_joystick"]["device"]) {
            device = param::config["gamepad_joystick"]["device"].as<std::string>();
        }
        FSMState::gamepad = std::make_shared<Gamepad>(device);
        if (param::config["gamepad_joystick"]["deadzone"]) {
            FSMState::gamepad->deadzone = param::config["gamepad_joystick"]["deadzone"].as<float>();
        }
    }

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    const int domain_id = vm["domain-id"].as<int>();
    const std::string network = vm["network"].as<std::string>();
    unitree::robot::ChannelFactory::Instance()->Init(domain_id, network);
    std::cout << "DDS settings: domain " << domain_id
              << ", interface '" << (network.empty() ? "<default>" : network) << "'\n";
    if (!vm["policy-dir"].as<std::string>().empty()) {
        std::cout << "Policy override: " << vm["policy-dir"].as<std::string>() << "\n";
    }
    if (!vm["policy-run"].as<std::string>().empty()) {
        std::cout << "Policy run override: " << vm["policy-run"].as<std::string>() << "\n";
    }

    init_fsm_state();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Keyboard controls:\n";
    std::cout << "  f : enter FixStand  (maps to [L2 + A])\n";
    std::cout << "  v : start policy    (maps to [Start])\n";
    std::cout << "  b : return Passive  (maps to [L2 + B])\n";
    std::cout << "  hold w/s : ramp forward/backward velocity\n";
    std::cout << "  hold a/d : ramp left/right velocity\n";
    std::cout << "  hold left/right arrow : ramp yaw velocity\n";
    std::cout << "  space : clear all velocity commands\n";
    std::cout << "Connection note: keyboard control only works after rt/lowstate is connected.\n";
    if (FSMState::gamepad) {
        std::cout << "Gamepad controls:\n";
        std::cout << "  left stick : x-y velocity\n";
        std::cout << "  right stick x : yaw velocity\n";
        std::cout << "  LT + A : enter FixStand\n";
        std::cout << "  LT + B : return Passive\n";
        std::cout << "  X or Menu : start policy\n";
        std::cout << "  device : " << FSMState::gamepad->device_path()
                  << (FSMState::gamepad->connected() ? " (connected)\n" : " (not connected)\n");
    }
    std::cout << "Policy note: default uses the newest exported run under logs/randpol/unitree_go2_forwardyaw_velocity.\n";
    std::cout << "  Override base dir with --policy-dir <path>\n";
    std::cout << "  Pin a run with --policy-run <YYYY-MM-DD_HH-MM-SS>\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}
