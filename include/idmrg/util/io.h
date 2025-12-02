//
// I/O utilities for iDMRG
//

#ifndef IDMRG_UTIL_IO_H
#define IDMRG_UTIL_IO_H

#include "itensor/all.h"
#include "idmrg/idmrg.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>

namespace idmrg {

using namespace itensor;

//
// Save iDMRG result to file
//
inline void
saveResult(std::string const& filename, 
          iDMRGResult const& result,
          MPS const& psi) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write result
    write(file, result);
    
    // Write MPS
    write(file, psi);
    
    file.close();
}

//
// Load iDMRG result from file
//
inline void
loadResult(std::string const& filename,
          iDMRGResult& result,
          MPS& psi) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Read result
    read(file, result);
    
    // Read MPS
    read(file, psi);
    
    file.close();
}

//
// Write energy data to CSV
//
inline void
writeEnergyCSV(std::string const& filename,
              std::vector<double> const& energies,
              int unit_cell_size) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    file << "step,N_sites,energy_total,energy_per_site\n";
    
    for (size_t i = 0; i < energies.size(); ++i) {
        int N = (i + 1) * unit_cell_size * 2;  // Effective system size
        file << i + 1 << "," << N << "," 
             << std::setprecision(14) << energies[i] << ","
             << energies[i] / (2 * unit_cell_size) << "\n";
    }
    
    file.close();
}

//
// Write correlation data to CSV
//
inline void
writeCorrelationCSV(std::string const& filename,
                   std::vector<double> const& correlation,
                   std::string const& op1 = "Sz",
                   std::string const& op2 = "Sz") {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    file << "distance,correlation\n";
    file << "# " << op1 << "-" << op2 << " correlation function\n";
    
    for (size_t i = 0; i < correlation.size(); ++i) {
        file << i + 1 << "," << std::setprecision(14) << correlation[i] << "\n";
    }
    
    file.close();
}

//
// Create output directory if it doesn't exist
//
inline void
ensureDirectory(std::string const& path) {
    std::filesystem::create_directories(path);
}

//
// Generate timestamped filename
//
inline std::string
timestampedFilename(std::string const& base, std::string const& ext) {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    
    std::ostringstream oss;
    oss << base << "_" 
        << std::put_time(&tm, "%Y%m%d_%H%M%S")
        << "." << ext;
    
    return oss.str();
}

//
// Progress bar for console output
//
class ProgressBar {
public:
    ProgressBar(int total, int width = 50) 
        : total_(total), width_(width), current_(0) {}
    
    void update(int current) {
        current_ = current;
        print();
    }
    
    void increment() {
        ++current_;
        print();
    }
    
    void complete() {
        current_ = total_;
        print();
        std::cout << std::endl;
    }

private:
    int total_;
    int width_;
    int current_;
    
    void print() const {
        float progress = static_cast<float>(current_) / total_;
        int filled = static_cast<int>(progress * width_);
        
        std::cout << "\r[";
        for (int i = 0; i < width_; ++i) {
            if (i < filled) std::cout << "=";
            else if (i == filled) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << static_cast<int>(progress * 100) << "%" << std::flush;
    }
};

} // namespace idmrg

#endif // IDMRG_UTIL_IO_H
