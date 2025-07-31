#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <stdexcept>

#include <cutlass/util/host_tensor.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/layout/matrix.h>

using ElementE = uint16_t;
using LayoutInputE = cutlass::layout::RowMajor;
using ReorderedLayoutInputE = cutlass::layout::RowMajor;

const int M = 128;
const int MetaCols = 128 / 2 / 8;

// Load space-separated values from .txt file
std::vector<uint16_t> load_txt(const std::string& filename, int rows, int cols) {
    std::ifstream file(filename);
    std::vector<uint16_t> values;
    std::string token;
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + filename);
    }
    while (file >> token) {
        values.push_back(std::stoi(token));
    }
    if (values.size() != size_t(rows * cols)) {
        throw std::runtime_error("Invalid size in " + filename + ": expected " +
                                 std::to_string(rows * cols) + ", got " + std::to_string(values.size()));
    }
    return values;
}

// Save flattened tensor data to txt (space-separated)
void save_txt(const std::string& filename, const uint16_t* data, int rows, int cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open " + filename + " for writing");
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << data[i * cols + j];
            if (j < cols - 1) file << " ";
        }
        file << "\n";
    }
}

int main() {
    // Load original and user-provided reordered meta
    std::vector<uint16_t> plain_meta = load_txt("meta.txt", M, MetaCols);
    std::vector<uint16_t> your_reordered = load_txt("reorder_meta.txt", M, MetaCols);

    // CUTLASS host tensors
    cutlass::HostTensor<ElementE, LayoutInputE> tensor_e({M, MetaCols});
    cutlass::HostTensor<ElementE, ReorderedLayoutInputE> tensor_e_reordered({M, MetaCols});

    std::memcpy(tensor_e.host_data(), plain_meta.data(), M * MetaCols * sizeof(uint16_t));

    // Perform CUTLASS reorder
    cutlass::reorder_meta(
        tensor_e_reordered.host_ref(),
        tensor_e.host_ref(),
        {M, 0, MetaCols}
    );

    // Save official CUTLASS result to file
    save_txt("official_reorder_meta.txt", tensor_e_reordered.host_data(), M, MetaCols);

    // Compare against user's reordered version
    bool passed = true;
    for (int i = 0; i < M * MetaCols; ++i) {
        uint16_t expected = tensor_e_reordered.host_data()[i];
        uint16_t actual = your_reordered[i];
        if (expected != actual) {
            std::cout << "Mismatch at index " << i
                      << ": expected " << expected
                      << ", got " << actual << std::endl;
            passed = false;
        }
    }

    std::cout << (passed ? "Meta reorder verification PASSED!" : "Meta reorder verification FAILED!") << std::endl;
    return passed ? 0 : 1;
}