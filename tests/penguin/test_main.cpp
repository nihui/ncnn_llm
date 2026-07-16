// SPDX-License-Identifier: Apache-2.0
#include <cstdio>

#include "test_util.h"

int main() {
    int passed = 0;
    for (auto& tc : pvltest::registry()) {
        int before = pvltest::failures();
        std::printf("[ RUN  ] %s\n", tc.name.c_str());
        tc.fn();
        if (pvltest::failures() == before) {
            std::printf("[  OK  ] %s\n", tc.name.c_str());
            ++passed;
        } else {
            std::printf("[ FAIL ] %s\n", tc.name.c_str());
        }
    }
    std::printf("\n%d/%zu tests passed, %d checks failed\n", passed,
                pvltest::registry().size(), pvltest::failures());
    return pvltest::failures() == 0 ? 0 : 1;
}
