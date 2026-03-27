#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>


/**
 * @brief Maintain a keyboard reading thread.
 * And get the latest key value.
 */
class Keyboard
{
public:
  Keyboard()
  {
    _tty_available = isatty(fileno(stdin));
    if (!_tty_available) {
      return;
    }

    tcgetattr( fileno( stdin ), &_oldSettings );
    _newSettings = _oldSettings;
    _newSettings.c_lflag &= (~ICANON & ~ECHO);
    _newSettings.c_cc[VMIN] = 0;
    _newSettings.c_cc[VTIME] = 0;

    _startKey();
  }

  ~Keyboard()
  {
    _pauseKey();
  }

  void update()
  {
    _key.clear();
    _keys.clear();
    on_pressed = false;
    on_released = false;

    if (!_running) {
      return;
    }

    std::string latest_key;
    while (_poll(latest_key)) {
      if (!latest_key.empty()) {
        _key = latest_key;
        _keys.push_back(latest_key);
        on_pressed = true;
      }
    }
  }

  /**
   * @brief Get the current key value
   * 
   * @return std::string 
   */
  std::string key() const { return _key; };
  const std::vector<std::string>& keys() const { return _keys; };

  /**
   * @brief Get the String object from keyboard 
   * 
   * @param slogan Used to prompt the user for input
   * @return std::string 
   */
  std::string getString(std::string slogan)
  {
    // Stop reading keyboard value
    _running = false;
    _pauseKey();

    std::string stringtemp;
    std::cout << slogan << std::endl;// prompt
    std::getline(std::cin, stringtemp);

    // Restart reading keyboard value
    _startKey();
    _running = true;

    return stringtemp;
  }

  /**
   * flags; available after update()
   */
  bool on_pressed = false;
  bool on_released = false;

  private:
  bool _tty_available = false;
  bool _running = false;

  bool _poll(std::string& key)
  {
    key.clear();
    if (!_running || !_tty_available) {
      return false;
    }

    FD_ZERO(&_fd_set);
    FD_SET(fileno(stdin), &_fd_set);

    _tv.tv_sec = 0;
    _tv.tv_usec = 0;

    if (select(fileno(stdin) + 1, &_fd_set, NULL, NULL, &_tv) <= 0) {
      return false;
    }

    if (read(fileno(stdin), &_c, 1) <= 0) {
      return false;
    }

    if (_c != '\033') {
      key.assign(1, _c);
      return true;
    }

    if (read(fileno(stdin), &_c, 1) <= 0) {
      return false;
    }

    if (_c == '[') {
      if (read(fileno(stdin), &_c, 1) <= 0) {
        return false;
      }
      switch (_c)
      {
      case 'A': key = "up";    break;
      case 'B': key = "down";  break;
      case 'C': key = "right"; break;
      case 'D': key = "left";  break;
      default:  key.clear();   break;
      }
    }

    return true;
  }

  /**
   * @brief Restore keyboard default settings.
   */
  void _pauseKey()
  {
    if (_tty_available) {
      tcsetattr(fileno(stdin), TCSANOW, &_oldSettings);
    }
    _running = false;
  }

  /**
   * @brief Disable canonical mode and echoing of input characters.
   */
  void _startKey()
  {
    if (!_tty_available) {
      return;
    }
    tcsetattr(fileno(stdin), TCSANOW, &_newSettings);
    _running = true;
  }

  fd_set _fd_set;
  char _c = '\0';
  std::string _key;
  std::vector<std::string> _keys;
  
  termios _oldSettings, _newSettings;
  timeval _tv;
};
