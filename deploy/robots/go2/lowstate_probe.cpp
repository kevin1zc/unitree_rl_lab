#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include <boost/program_options.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/dds_wrapper/robots/go2/go2_sub.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
    po::options_description desc("Go2 lowstate probe");
    desc.add_options()
        ("help,h", "show help")
        ("domain-id,d", po::value<int>()->default_value(0), "dds domain id")
        ("network,n", po::value<std::string>()->default_value("eno1"), "dds network interface")
        ("samples,s", po::value<int>()->default_value(5), "number of samples to print");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    const int domain_id = vm["domain-id"].as<int>();
    const std::string network = vm["network"].as<std::string>();
    const int samples = vm["samples"].as<int>();

    unitree::robot::ChannelFactory::Instance()->Init(domain_id, network);
    std::cout << "Probing rt/lowstate on domain " << domain_id
              << ", interface '" << network << "'" << std::endl;

    auto lowstate = std::make_shared<unitree::robot::go2::subscription::LowState>();
    lowstate->wait_for_connection();
    std::cout << "Connected to rt/lowstate" << std::endl;

    for (int i = 0; i < samples; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        lowstate->update();
        std::lock_guard<std::mutex> lock(lowstate->mutex_);

        const auto& motors = lowstate->msg_.motor_state();
        const auto& imu = lowstate->msg_.imu_state();

        std::cout << "sample " << (i + 1)
                  << " tick=" << lowstate->msg_.tick()
                  << " q0=" << motors[0].q()
                  << " dq0=" << motors[0].dq()
                  << " quat_w=" << imu.quaternion()[0]
                  << std::endl;
    }

    return 0;
}
