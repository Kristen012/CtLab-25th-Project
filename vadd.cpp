#include <iostream>
#include <fstream>

using namespace std;

const int SIZE = 127 * 768;

void addArrays(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float a[SIZE];
    float b[SIZE];
    float c[SIZE];

    ifstream fileA("a.txt");
    if (!fileA) {
        cerr << "Unable to open file: a.txt" << endl;
        return 1;
    }
    for (int i = 0; i < SIZE; i++) {
        if (!(fileA >> a[i])) {
            cerr << "Error reading from file: a.txt" << endl;
            return 1;
        }
    }
    fileA.close();

    ifstream fileB("b.txt");
    if (!fileB) {
        cerr << "Unable to open file: b.txt" << endl;
        return 1;
    }
    for (int i = 0; i < SIZE; i++) {
        if (!(fileB >> b[i])) {
            cerr << "Error reading from file: b.txt" << endl;
            return 1;
        }
    }
    fileB.close();

    addArrays(a, b, c, SIZE);

    return 0;
}
