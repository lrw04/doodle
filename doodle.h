#pragma once

using real_t = _Float16;

#include <array>
#include <cstdio>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#define FATAL(...)                    \
    do {                              \
        fprintf(stderr, __VA_ARGS__); \
        fprintf(stderr, "\n");        \
        abort();                      \
    } while (false)

namespace doodle {

using shape = std::array<int, 4>;

struct tensor {
    std::vector<real_t> v;
    shape d;
};

enum node_type {
    placeholder,
    parameter,
    matmul,
    add,
    log,
    reshape,
    relu,
    gelu,
    softmax,
    mul,
};

struct node {
    node_type type;
    tensor v;
    int ints[4];
    real_t floats[4];
};

struct graph {
    std::unordered_map<std::string, int> indexes;
    std::vector<node> nodes;
    void placeholder(shape s);
    void parameter(shape s);
    void matmul(int a, int b);
    void add(int a, int b);
    void log(int a);
    void reshape(int a, shape s);
    void relu(int a);
    void gelu(int a);
    void softmax(int a);
    void mul(int a, real_t k);
};

graph compile(std::istream &st);
void forward(graph &g);
void differentiate(graph &g, std::vector<tensor> &adjoint);

}  // namespace doodle
