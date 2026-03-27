#pragma once

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/gamepad/gamepad.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) {
            keyboard->update();
            apply_keyboard_joystick_override();
        }
        if(gamepad) {
            gamepad->update();
            apply_gamepad_joystick_override();
        }
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
    static std::shared_ptr<Gamepad> gamepad;

protected:
    void configure_joystick_axes(unitree::common::UnitreeJoystick& joystick, float axis_smooth)
    {
        joystick.lx.smooth = axis_smooth;
        joystick.ly.smooth = axis_smooth;
        joystick.rx.smooth = axis_smooth;
        joystick.ry.smooth = axis_smooth;
        joystick.LT.smooth = 1.0f;
        joystick.RT.smooth = 1.0f;
    }

    void reset_joystick_buttons(unitree::common::UnitreeJoystick& joystick)
    {
        joystick.back(0);
        joystick.start(0);
        joystick.LS(0);
        joystick.RS(0);
        joystick.LB(0);
        joystick.RB(0);
        joystick.A(0);
        joystick.B(0);
        joystick.X(0);
        joystick.Y(0);
        joystick.up(0);
        joystick.down(0);
        joystick.left(0);
        joystick.right(0);
        joystick.F1(0);
        joystick.F2(0);
    }

    void log_velocity_command(
        const char* source, float lx, float ly, float rx,
        float& last_report_lx, float& last_report_ly, float& last_report_rx,
        std::chrono::steady_clock::time_point& last_report_time) const
    {
        const auto now = std::chrono::steady_clock::now();
        const bool report_due = std::chrono::duration<float>(now - last_report_time).count() >= 0.12f;
        const bool command_changed =
            std::fabs(lx - last_report_lx) >= 0.05f ||
            std::fabs(ly - last_report_ly) >= 0.05f ||
            std::fabs(rx - last_report_rx) >= 0.05f;
        const bool command_active =
            std::fabs(lx) > 1e-3f ||
            std::fabs(ly) > 1e-3f ||
            std::fabs(rx) > 1e-3f ||
            std::fabs(last_report_lx) > 1e-3f ||
            std::fabs(last_report_ly) > 1e-3f ||
            std::fabs(last_report_rx) > 1e-3f;

        if (command_active && report_due && command_changed) {
            spdlog::info(
                "{} velocity command: forward={:.2f}, lateral={:.2f}, yaw={:.2f}",
                source, ly, -lx, -rx
            );
            last_report_lx = lx;
            last_report_ly = ly;
            last_report_rx = rx;
            last_report_time = now;
        }
    }

    bool keyboard_joystick_enabled() const
    {
        auto cfg = param::config["keyboard_joystick"];
        if (!cfg || !cfg["enabled"]) {
            return false;
        }
        return cfg["enabled"].as<bool>();
    }

    bool gamepad_joystick_enabled() const
    {
        auto cfg = param::config["gamepad_joystick"];
        if (!cfg || !cfg["enabled"]) {
            return false;
        }
        return cfg["enabled"].as<bool>();
    }

    void apply_keyboard_joystick_override()
    {
        if (!keyboard_joystick_enabled()) {
            return;
        }

        struct KeyboardJoystickState
        {
            float cmd_lx = 0.0f;
            float cmd_ly = 0.0f;
            float cmd_rx = 0.0f;
            bool initialized = false;
            std::chrono::steady_clock::time_point last_update;
            std::chrono::steady_clock::time_point left_time;
            std::chrono::steady_clock::time_point right_time;
            std::chrono::steady_clock::time_point forward_time;
            std::chrono::steady_clock::time_point backward_time;
            std::chrono::steady_clock::time_point yaw_left_time;
            std::chrono::steady_clock::time_point yaw_right_time;
        };

        static KeyboardJoystickState state;

        auto normalize_key = [](std::string key_value)->std::string {
            for (char& c : key_value) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            return key_value;
        };

        std::vector<std::string> keys = keyboard->keys();
        for (auto& key_value : keys) {
            key_value = normalize_key(key_value);
        }

        const auto cfg = param::config["keyboard_joystick"];
        const float hold_timeout = cfg["hold_timeout"] ? cfg["hold_timeout"].as<float>() : 0.6f;
        const float linear_ramp_rate = cfg["linear_ramp_rate"] ? cfg["linear_ramp_rate"].as<float>() : 2.5f;
        const float yaw_ramp_rate = cfg["yaw_ramp_rate"] ? cfg["yaw_ramp_rate"].as<float>() : 3.0f;

        const auto now = std::chrono::steady_clock::now();
        if (!state.initialized) {
            state.initialized = true;
            state.last_update = now;
            state.left_time = now - std::chrono::seconds(1);
            state.right_time = state.left_time;
            state.forward_time = state.left_time;
            state.backward_time = state.left_time;
            state.yaw_left_time = state.left_time;
            state.yaw_right_time = state.left_time;
        }

        float dt = std::chrono::duration<float>(now - state.last_update).count();
        state.last_update = now;
        dt = std::clamp(dt, 0.0f, 0.05f);

        if (keyboard->on_pressed) {
            for (const auto& pressed_key : keys) {
                if (pressed_key == "w") {
                    state.forward_time = now;
                } else if (pressed_key == "s") {
                    state.backward_time = now;
                } else if (pressed_key == "a") {
                    state.left_time = now;
                } else if (pressed_key == "d") {
                    state.right_time = now;
                } else if (pressed_key == "left") {
                    state.yaw_left_time = now;
                } else if (pressed_key == "right") {
                    state.yaw_right_time = now;
                } else if (pressed_key == " ") {
                    const auto expired = now - std::chrono::seconds(1);
                    state.cmd_lx = 0.0f;
                    state.cmd_ly = 0.0f;
                    state.cmd_rx = 0.0f;
                    state.left_time = expired;
                    state.right_time = expired;
                    state.forward_time = expired;
                    state.backward_time = expired;
                    state.yaw_left_time = expired;
                    state.yaw_right_time = expired;
                }
            }
        }

        const auto is_active = [&](const std::chrono::steady_clock::time_point& last_time)->bool {
            return std::chrono::duration<float>(now - last_time).count() <= hold_timeout;
        };
        const auto choose_direction = [&](bool negative_active, bool positive_active,
                                          const std::chrono::steady_clock::time_point& negative_time,
                                          const std::chrono::steady_clock::time_point& positive_time)->int {
            if (negative_active && positive_active) {
                return positive_time > negative_time ? 1 : -1;
            }
            if (negative_active) {
                return -1;
            }
            if (positive_active) {
                return 1;
            }
            return 0;
        };
        const auto ramp_towards = [&](float current, float target, float rate)->float {
            const float max_step = rate * dt;
            if (target > current) {
                return std::min(current + max_step, target);
            }
            return std::max(current - max_step, target);
        };

        const int lateral_dir = choose_direction(
            is_active(state.left_time), is_active(state.right_time), state.left_time, state.right_time
        );
        const int forward_dir = choose_direction(
            is_active(state.backward_time), is_active(state.forward_time), state.backward_time, state.forward_time
        );
        const int yaw_dir = choose_direction(
            is_active(state.yaw_left_time), is_active(state.yaw_right_time), state.yaw_left_time, state.yaw_right_time
        );

        state.cmd_lx = (lateral_dir == 0) ? 0.0f : ramp_towards(state.cmd_lx, static_cast<float>(lateral_dir), linear_ramp_rate);
        state.cmd_ly = (forward_dir == 0) ? 0.0f : ramp_towards(state.cmd_ly, static_cast<float>(forward_dir), linear_ramp_rate);
        state.cmd_rx = (yaw_dir == 0) ? 0.0f : ramp_towards(state.cmd_rx, static_cast<float>(yaw_dir), yaw_ramp_rate);

        auto contains_key = [&](const std::string& target)->bool {
            return std::find(keys.begin(), keys.end(), target) != keys.end();
        };

        const bool fixstand_pressed = keyboard->on_pressed && contains_key("f");
        const bool start_pressed = keyboard->on_pressed && contains_key("v");
        const bool passive_pressed = keyboard->on_pressed && contains_key("b");

        static float last_report_lx = 0.0f;
        static float last_report_ly = 0.0f;
        static float last_report_rx = 0.0f;
        static auto last_report_time = std::chrono::steady_clock::now();

        auto& joystick = lowstate->joystick;

        configure_joystick_axes(joystick, 1.0f);
        reset_joystick_buttons(joystick);

        joystick.start(start_pressed ? 1 : 0);
        joystick.A(fixstand_pressed ? 1 : 0);
        joystick.B(passive_pressed ? 1 : 0);
        joystick.LT((fixstand_pressed || passive_pressed) ? 1.0f : 0.0f);
        joystick.RT(0.0f);
        joystick.lx(state.cmd_lx);
        joystick.ly(state.cmd_ly);
        joystick.rx(state.cmd_rx);
        joystick.ry(0.0f);

        log_velocity_command(
            "Keyboard", state.cmd_lx, state.cmd_ly, state.cmd_rx,
            last_report_lx, last_report_ly, last_report_rx, last_report_time
        );
    }

    void apply_gamepad_joystick_override()
    {
        if (!gamepad_joystick_enabled() || !gamepad || !gamepad->connected()) {
            return;
        }

        const auto cfg = param::config["gamepad_joystick"];
        const float axis_smooth = cfg["axis_smooth"] ? cfg["axis_smooth"].as<float>() : 0.08f;
        const float trigger_threshold = cfg["trigger_threshold"] ? cfg["trigger_threshold"].as<float>() : 0.5f;

        const bool fixstand_pressed = gamepad->left_trigger() > trigger_threshold && gamepad->button_a();
        const bool passive_pressed = gamepad->left_trigger() > trigger_threshold && gamepad->button_b();
        const bool start_pressed = gamepad->button_x() || gamepad->button_start();

        static float cmd_lx = 0.0f;
        static float cmd_ly = 0.0f;
        static float cmd_rx = 0.0f;
        cmd_lx = cmd_lx * (1.0f - axis_smooth) + gamepad->left_x() * axis_smooth;
        cmd_ly = cmd_ly * (1.0f - axis_smooth) + (-gamepad->left_y()) * axis_smooth;
        cmd_rx = cmd_rx * (1.0f - axis_smooth) + gamepad->right_x() * axis_smooth;

        auto& joystick = lowstate->joystick;

        configure_joystick_axes(joystick, 1.0f);
        reset_joystick_buttons(joystick);
        joystick.back(gamepad->button_back() ? 1 : 0);
        joystick.start(start_pressed ? 1 : 0);
        joystick.LB(gamepad->button_lb() ? 1 : 0);
        joystick.RB(gamepad->button_rb() ? 1 : 0);
        joystick.A(fixstand_pressed ? 1 : 0);
        joystick.B(passive_pressed ? 1 : 0);
        joystick.LT(gamepad->left_trigger());
        joystick.RT(gamepad->right_trigger());
        joystick.lx(cmd_lx);
        joystick.ly(cmd_ly);
        joystick.rx(cmd_rx);
        joystick.ry(0.0f);

        static float last_report_lx = 0.0f;
        static float last_report_ly = 0.0f;
        static float last_report_rx = 0.0f;
        static auto last_report_time = std::chrono::steady_clock::now();

        log_velocity_command(
            "Gamepad", joystick.lx(), joystick.ly(), joystick.rx(),
            last_report_lx, last_report_ly, last_report_rx, last_report_time
        );
    }
};
