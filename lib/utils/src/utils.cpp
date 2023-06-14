#include <vector>
#include <utils.h>
#include <parameter.h>

namespace nano_mind {
    std::pair<float, int> argmax(std::vector<Parameter*> x) {
        float max_value = x[0]->data_;
        int max_idx = 0;
        for (int i = 0; i < x.size(); i++) {
            if (x[i]->data_ > max_value) {
                max_idx = i;
                max_value = x[i]->data_;
            }
        }
        return {max_value, max_idx};
    }
}
