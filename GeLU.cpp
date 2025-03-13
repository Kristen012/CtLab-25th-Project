#include <iostream>
#include <cmath>

using namespace std;

double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2.0 / 3.1415926535) * (x + 0.044715 * pow(x, 3))));
}

int main() {
    cout << gelu(0.3) << endl;

    return 0;
}
