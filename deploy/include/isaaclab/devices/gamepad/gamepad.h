#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fcntl.h>
#include <linux/input-event-codes.h>
#include <linux/joystick.h>
#include <string>
#include <sys/ioctl.h>
#include <unistd.h>

class Gamepad
{
public:
  explicit Gamepad(const std::string& device_path = "/dev/input/js0")
  : device_path_(device_path)
  {
    open_device();
  }

  ~Gamepad()
  {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  void update()
  {
    if (fd_ < 0) {
      open_device();
      return;
    }

    js_event event{};
    while (read(fd_, &event, sizeof(event)) == sizeof(event)) {
      const uint8_t event_type = event.type & ~JS_EVENT_INIT;
      if (event_type == JS_EVENT_BUTTON && event.number < button_states_.size()) {
        button_states_[event.number] = event.value;
      } else if (event_type == JS_EVENT_AXIS && event.number < axis_states_.size()) {
        axis_states_[event.number] = event.value;
      }
    }

    if (!calibrated_) {
      calibrate_center();
    }
  }

  bool connected() const { return fd_ >= 0; }
  const std::string& device_path() const { return device_path_; }

  float left_x() const { return normalized_axis(axis_lx_); }
  float left_y() const { return normalized_axis(axis_ly_); }
  float right_x() const { return normalized_axis(axis_rx_); }
  float right_y() const { return normalized_axis(axis_ry_); }
  float left_trigger() const { return normalized_trigger(axis_lt_); }
  float right_trigger() const { return normalized_trigger(axis_rt_); }

  bool button_a() const { return button_state(button_a_); }
  bool button_b() const { return button_state(button_b_); }
  bool button_x() const { return button_state(button_x_); }
  bool button_start() const { return button_state(button_start_); }
  bool button_back() const { return button_state(button_back_); }
  bool button_lb() const { return button_state(button_lb_); }
  bool button_rb() const { return button_state(button_rb_); }

  float deadzone = 0.08f;

private:
  void open_device()
  {
    if (fd_ >= 0) {
      return;
    }

    fd_ = open(device_path_.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd_ < 0) {
      return;
    }

    query_mappings();
  }

  void query_mappings()
  {
    constexpr int kAxisMapSize = ABS_CNT;
    constexpr int kButtonMapSize = KEY_MAX - BTN_MISC + 1;

    uint8_t axis_count = 0;
    uint8_t button_count = 0;
    ioctl(fd_, JSIOCGAXES, &axis_count);
    ioctl(fd_, JSIOCGBUTTONS, &button_count);
    axis_count_ = axis_count;
    button_count_ = button_count;

    std::array<uint8_t, kAxisMapSize> axis_map{};
    std::array<uint16_t, kButtonMapSize> button_map{};
    ioctl(fd_, JSIOCGAXMAP, axis_map.data());
    ioctl(fd_, JSIOCGBTNMAP, button_map.data());

    axis_lx_ = find_axis(axis_map, {0});           // ABS_X
    axis_ly_ = find_axis(axis_map, {1});           // ABS_Y
    axis_rx_ = find_axis(axis_map, {2, 3});        // ABS_Z or ABS_RX
    axis_ry_ = find_axis(axis_map, {5, 4});        // ABS_RZ or ABS_RY
    axis_lt_ = find_axis(axis_map, {10, 5});       // ABS_BRAKE or ABS_RZ
    axis_rt_ = find_axis(axis_map, {9, 2});        // ABS_GAS or ABS_Z

    button_a_ = find_button(button_map, 304);      // BTN_SOUTH
    button_b_ = find_button(button_map, 305);      // BTN_EAST
    button_x_ = find_button(button_map, 307);      // BTN_NORTH/WEST on Xbox layout
    button_lb_ = find_button(button_map, 310);     // BTN_TL
    button_rb_ = find_button(button_map, 311);     // BTN_TR
    button_back_ = find_button(button_map, 314);   // BTN_SELECT
    button_start_ = find_button(button_map, 315);  // BTN_START
  }

  template <size_t N>
  int find_axis(const std::array<uint8_t, N>& axis_map, std::initializer_list<uint8_t> candidates) const
  {
    for (uint8_t candidate : candidates) {
      for (int i = 0; i < axis_count_; ++i) {
        if (axis_map[i] == candidate) {
          return i;
        }
      }
    }
    return -1;
  }

  template <size_t N>
  int find_button(const std::array<uint16_t, N>& button_map, uint16_t button_code) const
  {
    for (int i = 0; i < button_count_; ++i) {
      if (button_map[i] == button_code) {
        return i;
      }
    }
    return -1;
  }

  bool button_state(int index) const
  {
    return index >= 0 && index < static_cast<int>(button_states_.size()) && button_states_[index] != 0;
  }

  float normalized_axis(int index) const
  {
    if (index < 0 || index >= static_cast<int>(axis_states_.size())) {
      return 0.0f;
    }
    float value = static_cast<float>(axis_states_[index]) / 32767.0f;
    value -= axis_offsets_[index];
    if (std::fabs(value) < deadzone) {
      return 0.0f;
    }
    value = std::clamp(value, -1.0f, 1.0f);
    return value;
  }

  void calibrate_center()
  {
    if (any_button_pressed()) {
      return;
    }

    const auto capture_axis = [&](int index) {
      if (index < 0 || index >= static_cast<int>(axis_states_.size())) {
        return;
      }
      const float raw = static_cast<float>(axis_states_[index]) / 32767.0f;
      if (std::fabs(raw) < 0.25f) {
        axis_offsets_[index] = raw;
      }
    };

    capture_axis(axis_lx_);
    capture_axis(axis_ly_);
    capture_axis(axis_rx_);
    capture_axis(axis_ry_);
    calibrated_ = true;
  }

  bool any_button_pressed() const
  {
    for (int i = 0; i < button_count_; ++i) {
      if (button_states_[i] != 0) {
        return true;
      }
    }
    return false;
  }

  float normalized_trigger(int index) const
  {
    if (index < 0 || index >= static_cast<int>(axis_states_.size())) {
      return 0.0f;
    }
    float value = (static_cast<float>(axis_states_[index]) + 32767.0f) / 65534.0f;
    return std::clamp(value, 0.0f, 1.0f);
  }

  std::string device_path_;
  int fd_{-1};
  int axis_count_{0};
  int button_count_{0};

  int axis_lx_{-1};
  int axis_ly_{-1};
  int axis_rx_{-1};
  int axis_ry_{-1};
  int axis_lt_{-1};
  int axis_rt_{-1};

  int button_a_{-1};
  int button_b_{-1};
  int button_x_{-1};
  int button_lb_{-1};
  int button_rb_{-1};
  int button_back_{-1};
  int button_start_{-1};

  std::array<int16_t, ABS_CNT> axis_states_{};
  std::array<int16_t, KEY_MAX - BTN_MISC + 1> button_states_{};
  std::array<float, ABS_CNT> axis_offsets_{};
  bool calibrated_{false};
};
