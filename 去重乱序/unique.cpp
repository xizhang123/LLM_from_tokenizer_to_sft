#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_set>
#include <string>

// 简单的哈希函数，用于示例
size_t simpleHash(const std::string& str) {
    std::hash<std::string> hash_fn;
    return hash_fn(str);
}

int main() {
    std::string inputFilename = "high_ff.txt";  // 输入文件名
    std::string outputFilename = "high_ff_unique.txt"; // 输出文件名

    std::ifstream inputFile(inputFilename);
    std::ofstream outputFile(outputFilename);
    std::unordered_set<size_t> hashSet;

    if (!inputFile.is_open()) {
        std::cerr << "无法打开输入文件: " << inputFilename << std::endl;
        return 1;
    }

    if (!outputFile.is_open()) {
        std::cerr << "无法打开输出文件: " << outputFilename << std::endl;
        return 1;
    }

    std::string line;
    while (getline(inputFile, line)) {
        size_t lineHash = simpleHash(line);
        // 如果哈希集合中不存在当前行的哈希值，则追加到新文件中
        if (hashSet.find(lineHash) == hashSet.end()) {
            outputFile << line << std::endl;
            hashSet.insert(lineHash);
        }
    }

    inputFile.close();
    outputFile.close();

    std::cout << "去重完成，结果已保存到 " << outputFilename << std::endl;

    return 0;
}
