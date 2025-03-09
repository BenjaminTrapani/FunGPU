#pragma once

#include <filesystem>

namespace FunGPU {
    class TemporaryDirectory {
    public:
        TemporaryDirectory();
        ~TemporaryDirectory();
        const std::filesystem::path& path() const noexcept { return path_; }

    private:
        std::filesystem::path path_;  
    };
}
