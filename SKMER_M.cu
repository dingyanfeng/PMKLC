#include <iostream>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include "cnpy.h"
#include <vector>
#include <unordered_map>


std::string getBaseName(const std::string& path) {
    return std::filesystem::path(path).filename().string();
}

void generateCombinationsHelper(const std::vector<char>& alphabet, int k, std::string currentCombination, int remainingLength, std::vector<std::string>& combinations) {
    if (remainingLength == 0) {
        combinations.push_back(currentCombination);
        return;
    }
    for (char letter : alphabet) {
        generateCombinationsHelper(alphabet, k, currentCombination + letter, remainingLength - 1, combinations);
    }
}

void generateCombinations(const std::vector<char>& alphabet, int k, std::vector<std::string>& combinations) {
    generateCombinationsHelper(alphabet, k, "", k, combinations);
}

void generateAndSaveDictionaries(const std::vector<char>& alphabet, int k, const std::string& paramFile, const std::string& write_char) {
    std::vector<std::string> combinations;
    generateCombinations(alphabet, k, combinations);

    // Create the char2id_dict and id2char_dict
    std::unordered_map<std::string, int> char2idDict;
    std::unordered_map<int, std::string> id2charDict;
    for (int i = 0; i < combinations.size(); ++i) {
        char2idDict[combinations[i]] = i;
        id2charDict[i] = combinations[i];
    }

    // Open file for writing the dictionaries
    std::ofstream paramFileStream(paramFile);
    if (!paramFileStream) {
        std::cerr << "Failed to open file for writing: " << paramFile << std::endl;
        return;
    }

    // Start writing JSON structure manually
    paramFileStream << "{\n";

    // Write char2id_dict to file
    paramFileStream << "  \"char2id_dict\": {\n";
    for (auto it = char2idDict.begin(); it != char2idDict.end(); ++it) {
        paramFileStream << "    \"" << it->first << "\": " << it->second;
        if (std::next(it) != char2idDict.end()) {
            paramFileStream << ",";
        }
        paramFileStream << "\n";
    }
    paramFileStream << "  },\n";

    // Write id2char_dict to file
    paramFileStream << "  \"id2char_dict\": {\n";
    for (auto it = id2charDict.begin(); it != id2charDict.end(); ++it) {
        paramFileStream << "    \"" << it->first << "\": \"" << it->second << "\"";
        if (std::next(it) != id2charDict.end()) {
            paramFileStream << ",";
        }
        paramFileStream << "\n";
    }
    paramFileStream << "  },\n";

    // Write Write-Chars to file
    paramFileStream << "  \"Write-Chars\": \"" << write_char << "\"\n";

    // Close the JSON structure
    paramFileStream << "}\n";

    paramFileStream.close();

    std::cout << "Dictionaries have been saved to " << paramFile << std::endl;
}

bool isValidEncoding(int encoding) {
    return (encoding == 1 || encoding == 2 || encoding == 3 || encoding == 4);
}

__device__ int charToIndex(char c) {
    switch (c) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: return -1; // Invalid character
    }
}

// Kernel to map non-overlapping triplets to integers
__global__ void mapTripletsToIntegers(const char* input, int* output, int length, int k, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int start = idx * w;

    if (start + k - 1 < length) {
        int idxValue = 0;

        for (int i = 0; i < k; ++i) {
            int charIndex = charToIndex(input[start + i]);
            idxValue += charIndex * pow(4, k - 1 - i);
        }

        output[idx] = idxValue;
    }
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <input_file_path> <encoding>\n";
        return 1;
    }

    const char* input_file = argv[1];
    std::string base_name = getBaseName(input_file);
    std::cout << "Base name of the file: " << base_name << std::endl;
    int dictionary_encoding_k = std::stoi(argv[2]);
    if (!isValidEncoding(dictionary_encoding_k)) {
        std::cerr << "dictionary_encoding_k Error!\n";
        return 1;
    }
    int dictionary_encoding_w = std::stoi(argv[3]);
    std::string param_file = "params_" + base_name + "_" + std::to_string(dictionary_encoding_k) + "_" + std::to_string(dictionary_encoding_w);
    std::string output_file = base_name + "_" + std::to_string(dictionary_encoding_k) + "_" + std::to_string(dictionary_encoding_w) + ".npy";
    int gpu_id = std::stoi(argv[4]);
    
    cudaError_t err = cudaSetDevice(gpu_id);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    int split_num = std::stoi(argv[5]);

    std::cout << "Using GPU ID: " << gpu_id << std::endl;
    std::cout << "input_file :" << input_file << std::endl;
    std::cout << "(w, k)-Mer :" << dictionary_encoding_w <<", " << dictionary_encoding_k << std::endl;
    std::cout << "param_file :" << param_file << std::endl;
    std::cout << "output_file :" << output_file << std::endl;


    std::ifstream infile(input_file);

    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << input_file << std::endl;
        return 1;
    }
    std::string data;
    infile >> data;
    infile.close();


    size_t start_pos = (data.length() - dictionary_encoding_k + dictionary_encoding_w) / dictionary_encoding_w * dictionary_encoding_w + dictionary_encoding_k - dictionary_encoding_w;
    std::string write_char = data.substr(start_pos);
    std::vector<char> alphabet = {'A', 'C', 'G', 'T'};
    generateAndSaveDictionaries(alphabet, dictionary_encoding_k, param_file, write_char);


    unsigned long long int sequence_length = data.length();
    std::cout << "Seq Length " << sequence_length << std::endl;
    unsigned long long int skmer_length = (sequence_length - dictionary_encoding_k + dictionary_encoding_w) / dictionary_encoding_w;
    char* data_cpu = new char[sequence_length];
    memcpy(data_cpu, data.c_str(), sequence_length);
    int* skmer_data_cpu = new int[skmer_length];

    char* data_gpu;
    int* skmer_data_gpu;
    cudaMalloc((void**)&data_gpu, sequence_length * sizeof(char));
    cudaMalloc((void**)&skmer_data_gpu, skmer_length * sizeof(int));

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    std::cout << "GPU memory uses: " << (total_mem - free_mem) / 1024 << " KB" << std::endl;

    cudaMemcpy(data_gpu, data_cpu, sequence_length * sizeof(char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (skmer_length + blockSize - 1) / blockSize;

    mapTripletsToIntegers<<<gridSize, blockSize>>>(data_gpu, skmer_data_gpu, sequence_length, dictionary_encoding_k, dictionary_encoding_w);
    cudaDeviceSynchronize();
    
    cudaMemcpy(skmer_data_cpu, skmer_data_gpu, skmer_length * sizeof(int), cudaMemcpyDeviceToHost);

    cnpy::npy_save(output_file, skmer_data_cpu, {skmer_length}, "w");
    std::cout << "Results saved to " << output_file << std::endl;

    size_t subarray_size = skmer_length / split_num;
    size_t remainder = skmer_length % split_num;
    size_t start = 0;
    for(int i=0; i<split_num; i++) {
        size_t current_subarray_size = subarray_size + (i < remainder ? 1 : 0);
        std::string sub_output_file = std::to_string(i) + "_" + output_file;
        cnpy::npy_save(sub_output_file, skmer_data_cpu + start, {current_subarray_size}, "w");
        std::cout << "Subarray saved to " << sub_output_file << std::endl;
        start += current_subarray_size;
    }

    delete[] data_cpu;
    delete[] skmer_data_cpu;
    cudaFree(data_gpu);
    cudaFree(skmer_data_gpu);
    return 0;
}