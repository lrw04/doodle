#include "doodle.h"

#include <algorithm>
#include <cmath>

using namespace std;
using namespace doodle;

istream& operator>>(istream& st, real_t& f) {
    float fp;
    st >> fp;
    f = fp;
    return st;
}

shape read_shape(istream& st) {
    shape s;
    for (int i = 0; i < 4; i++) {
        st >> s[i];
        if (!s[i]) break;
    }
    return s;
}

int shape_to_size(shape s) {
    int ans = 1, i = 0;
    while (s[i]) ans *= s[i++];
    return ans;
}

int shape_length(shape s) {
    int ans = 0;
    while (s[ans]) ans++;
    return ans;
}

tensor make_tensor(shape s) {
    tensor t;
    t.v.resize(shape_to_size(s));
    t.d = s;
    return t;
}

graph doodle::compile(istream& st) {
    graph g;
    string kw;
    while (st >> kw) {
        if (kw == "def") {
            string name;
            cin >> name;
            g.indexes[name] = g.nodes.size();
            string type;
            cin >> type;
            if (type == "placeholder") {
                g.placeholder(read_shape(st));
            } else if (type == "parameter") {
                g.parameter(read_shape(st));
            } else if (type == "matmul") {
                string name1, name2;
                st >> name1 >> name2;
                g.matmul(g.indexes[name1], g.indexes[name2]);
            } else if (type == "add") {
                string name1, name2;
                st >> name1 >> name2;
                g.add(g.indexes[name1], g.indexes[name2]);
            } else if (type == "log") {
                string name1;
                st >> name1;
                g.log(g.indexes[name1]);
            } else if (type == "reshape") {
                string name1;
                st >> name1;
                g.reshape(g.indexes[name1], read_shape(st));
            } else if (type == "relu") {
                string name1;
                st >> name1;
                g.relu(g.indexes[name1]);
            } else if (type == "gelu") {
                string name1;
                st >> name1;
                g.gelu(g.indexes[name1]);
            } else if (type == "softmax") {
                string name1;
                st >> name1;
                g.softmax(g.indexes[name1]);
            } else if (type == "mul") {
                string name1;
                real_t float1;
                st >> name1 >> float1;
                g.mul(g.indexes[name1], float1);
            } else {
                FATAL("Unknown node type: %s", type.c_str());
            }
        } else {
            FATAL("Unknown keyword: %s", kw.c_str());
        }
    }
    return g;
}

void doodle::forward(graph& g) {
    for (int i = 0; i < (int)g.nodes.size(); i++) {
        node& u = g.nodes[i];
        switch (g.nodes[i].type) {
            case placeholder:
            case parameter:
                break;
            case matmul: {
                node &a = g.nodes[u.ints[0]], &b = g.nodes[u.ints[1]];
                int n = a.v.d[0], m = a.v.d[1], l = b.v.d[1];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < l; j++) u.v.v[i * l + j] = 0;
                for (int i = 0; i < n; i++) {
                    int il = i * l, im = i * m;
                    for (int j = 0; j < m; j++) {
                        int jl = j * l;
                        auto t = a.v.v[im + j];
                        for (int k = 0; k < l; k++)
                            u.v.v[il + k] += t * b.v.v[jl + k];
                    }
                }
            } break;
            case add: {
                node &a = g.nodes[u.ints[0]], &b = g.nodes[u.ints[1]];
                for (int i = 0; i < (int)a.v.v.size(); i++)
                    u.v.v[i] = a.v.v[i] + b.v.v[i];
            } break;
            case log: {
                node& a = g.nodes[u.ints[0]];
                for (int i = 0; i < (int)a.v.v.size(); i++)
                    u.v.v[i] = std::log((float)a.v.v[i]);
            } break;
            case reshape: {
                node& a = g.nodes[u.ints[0]];
                for (int i = 0; i < (int)a.v.v.size(); i++) u.v.v[i] = a.v.v[i];
            } break;
            case relu: {
                node& a = g.nodes[u.ints[0]];
                for (int i = 0; i < (int)a.v.v.size(); i++)
                    u.v.v[i] = max(a.v.v[i], (real_t)0);
            } break;
            case gelu: {
                node& a = g.nodes[u.ints[0]];
                for (int i = 0; i < (int)a.v.v.size(); i++) {
                    auto x = a.v.v[i];
                    u.v.v[i] = 0.5 * x * (1 + std::erf(x / sqrtf(2)));
                }
            } break;
            case softmax: {
                node& a = g.nodes[u.ints[0]];
                int size = a.v.v.size();
                auto maximum = *std::max_element(a.v.v.begin(), a.v.v.end());
                for (int i = 0; i < size; i++)
                    u.v.v[i] = exp((float)(a.v.v[i] - maximum));
                real_t sum = 0;
                for (int i = 0; i < size; i++) sum += u.v.v[i];
                for (int i = 0; i < size; i++) u.v.v[i] /= sum;
            } break;
            case mul: {
                node& a = g.nodes[u.ints[0]];
                int size = a.v.v.size();
                for (int i = 0; i < size; i++)
                    u.v.v[i] = a.v.v[i] * u.floats[0];
            } break;
            default:
                FATAL("Forward: unknown node type");
        }
    }
}

void doodle::differentiate(graph& g, std::vector<tensor>& adjoint) {
    adjoint.resize(g.nodes.size());
    for (int i = 0; i < g.nodes.size(); i++)
        adjoint[i] = make_tensor(g.nodes[i].v.d);
    adjoint.back().v[0] = 1;
    for (int i = g.nodes.size() - 1; i >= 0; i--) {
        node& u = g.nodes[i];
        switch (g.nodes[i].type) {
            case placeholder:
            case parameter:
                break;
            case matmul: {
                node &a = g.nodes[u.ints[0]], &b = g.nodes[u.ints[1]];
                auto &pa = adjoint[u.ints[0]], &pb = adjoint[u.ints[1]];
                int n = a.v.d[0], m = a.v.d[1], l = b.v.d[1];
                for (int ip = 0; ip < n; ip++) {
                    auto im = ip * m, il = ip * l;
                    for (int j = 0; j < m; j++) {
                        auto jl = j * l;
                        auto t = a.v.v[im + j];
                        for (int k = 0; k < l; k++) {
                            pa.v[im + j] +=
                                adjoint[i].v[il + k] * b.v.v[jl + k];
                            pb.v[jl + k] += t * adjoint[i].v[il + k];
                        }
                    }
                }
            } break;
            case add: {
                node &a = g.nodes[u.ints[0]], &b = g.nodes[u.ints[1]];
                auto &pa = adjoint[u.ints[0]], &pb = adjoint[u.ints[1]];
                for (int ip = 0; ip < u.v.v.size(); ip++) {
                    pa.v[ip] += adjoint[i].v[ip];
                    pb.v[ip] += adjoint[i].v[ip];
                }
            } break;
            case log: {
                node& a = g.nodes[u.ints[0]];
                auto& pa = adjoint[u.ints[0]];
                for (int ip = 0; ip < (int)a.v.v.size(); ip++)
                    pa.v[ip] += adjoint[i].v[ip] / a.v.v[i];
            } break;
            case reshape: {
                node& a = g.nodes[u.ints[0]];
                auto& pa = adjoint[u.ints[0]];
                for (int ip = 0; ip < (int)a.v.v.size(); ip++)
                    pa.v[ip] += adjoint[i].v[ip];
            } break;
            case relu: {
                node& a = g.nodes[u.ints[0]];
                auto& pa = adjoint[u.ints[0]];
                for (int ip = 0; ip < (int)a.v.v.size(); ip++)
                    pa.v[ip] += a.v.v[ip] > 0 ? adjoint[i].v[ip] : 0;
            } break;
            case gelu: {
                node& a = g.nodes[u.ints[0]];
                auto& pa = adjoint[u.ints[0]];
                for (int ip = 0; ip < (int)a.v.v.size(); ip++) {
                    auto x = a.v.v[ip];
                    pa.v[ip] += adjoint[i].v[ip] *
                                ((1 + std::erf(x / sqrtf(2))) / 2 +
                                 x * expf(-powf(x, 2) / 2) / sqrtf(2 * M_PI));
                }
            } break;
            case softmax: {
                node& a = g.nodes[u.ints[0]];
                auto& pa = adjoint[u.ints[0]];
                int size = a.v.v.size();
                for (int ip = 0; ip < size; ip++) {
                    for (int j = 0; j < size; j++) {
                        pa.v[i] += adjoint[i].v[j] *
                                   (i == j ? u.v.v[i] - u.v.v[i] * u.v.v[i]
                                           : -u.v.v[i] * u.v.v[j]);
                    }
                }
            } break;
            case mul: {
                node& a = g.nodes[u.ints[0]];
                auto& pa = adjoint[u.ints[0]];
                for (int ip = 0; ip < (int)a.v.v.size(); ip++)
                    pa.v[ip] += u.floats[0] * adjoint[i].v[ip];
            } break;
            default:
                FATAL("Forward: unknown node type");
        }
    }
}

void doodle::graph::placeholder(shape s) {
    node u;
    u.type = node_type::placeholder;
    u.v = make_tensor(s);
    nodes.push_back(u);
}

void doodle::graph::parameter(shape s) {
    node u;
    u.type = node_type::parameter;
    u.v = make_tensor(s);
    nodes.push_back(u);
}

void doodle::graph::matmul(int a, int b) {
    node u;
    u.type = node_type::matmul;
    if (shape_length(nodes[a].v.d) != 2) FATAL("Matmul: not a matrix");
    if (shape_length(nodes[b].v.d) != 2) FATAL("Matmul: not a matrix");
    if (nodes[a].v.d[1] != nodes[b].v.d[0]) FATAL("Matmul: shape mismatch");
    u.ints[0] = a;
    u.ints[1] = b;
    u.v = make_tensor({nodes[a].v.d[0], nodes[b].v.d[1]});
    nodes.push_back(u);
}

void doodle::graph::add(int a, int b) {
    node u;
    u.type = node_type::add;
    if (nodes[a].v.d != nodes[b].v.d) FATAL("Add: shape mismatch");
    u.ints[0] = a;
    u.ints[1] = b;
    u.v = make_tensor(nodes[a].v.d);
    nodes.push_back(u);
}

void doodle::graph::log(int a) {
    node u;
    u.type = node_type::log;
    u.ints[0] = a;
    u.v = make_tensor(nodes[a].v.d);
    nodes.push_back(u);
}

void doodle::graph::reshape(int a, shape s) {
    node u;
    u.type = node_type::reshape;
    u.ints[0] = a;
    if (shape_to_size(nodes[a].v.d) != shape_to_size(s))
        FATAL("Reshape: shape mismatch");
    u.v = make_tensor(s);
    nodes.push_back(u);
}

void doodle::graph::relu(int a) {
    node u;
    u.type = node_type::relu;
    u.ints[0] = a;
    u.v = make_tensor(nodes[a].v.d);
    nodes.push_back(u);
}

void doodle::graph::gelu(int a) {
    node u;
    u.type = node_type::gelu;
    u.ints[0] = a;
    u.v = make_tensor(nodes[a].v.d);
    nodes.push_back(u);
}

void doodle::graph::softmax(int a) {
    node u;
    u.type = node_type::softmax;
    u.ints[0] = a;
    u.v = make_tensor(nodes[a].v.d);
    nodes.push_back(u);
}

void doodle::graph::mul(int a, real_t k) {
    node u;
    u.type = node_type::mul;
    u.ints[0] = a;
    u.floats[0] = k;
    u.v = make_tensor(nodes[a].v.d);
    nodes.push_back(u);
}
