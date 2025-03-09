#include "core/temporary_directory.h"
#include <unistd.h>

namespace FunGPU {
TemporaryDirectory::TemporaryDirectory()
    : path_([&] {
        const auto temp_dir = std::filesystem::temp_directory_path();
        const auto pid = getpid();
        const auto current_time = std::chrono::high_resolution_clock::now();
        const auto dir_suffix = [&] {
          std::stringstream ss;
          ss << "fgpu_" << pid << "_"
             << current_time.time_since_epoch().count();
          return ss.str();
        }();
        const auto result_directory = temp_dir / dir_suffix;
        std::filesystem::create_directories(result_directory);
        return result_directory;
      }()) {}

TemporaryDirectory::~TemporaryDirectory() {
  std::filesystem::remove_all(path_);
}
} // namespace FunGPU