#pragma once

#include <string>

#include "doodle.h"

namespace doodle {
std::vector<tensor> read_mnist_label(std::string filename);
std::vector<tensor> read_mnist_image(std::string filename);
std::pair<std::vector<tensor>, std::vector<tensor>> read_cifar10(
    std::string filename);
}  // namespace doodle
