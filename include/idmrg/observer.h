//
// iDMRG Observer for Measurements
//

#ifndef IDMRG_OBSERVER_H
#define IDMRG_OBSERVER_H

#include "itensor/all.h"
#include <vector>
#include <functional>
#include <map>
#include <string>
#include <iostream>
#include <iomanip>

namespace idmrg {

using namespace itensor;

//
// Custom observer for iDMRG calculations
// Tracks convergence and enables custom measurements
//
class iDMRGObserver : public DMRGObserver {
public:
    iDMRGObserver(MPS const& psi, Args const& args = Args::global())
        : DMRGObserver(psi, args),
          psi_(psi),
          energy_threshold_(args.getReal("EnergyThreshold", 1E-8)),
          entropy_threshold_(args.getReal("EntropyThreshold", 1E-6)),
          max_sweeps_(args.getInt("MaxSweeps", 100)),
          check_interval_(args.getInt("CheckInterval", 2))
    {
        energies_.reserve(100);
        entropies_.reserve(100);
    }
    
    // Called at each DMRG step
    void measure(Args const& args) override {
        DMRGObserver::measure(args);
        
        auto sweep = args.getInt("Sweep", 0);
        auto energy = args.getReal("Energy", 0.0);
        auto at_center = args.getBool("AtCenter", false);
        
        if (at_center) {
            // Record energy
            if (sweep >= static_cast<int>(energies_.size())) {
                energies_.push_back(energy);
            } else {
                energies_[sweep] = energy;
            }
            
            // Calculate and record entanglement entropy at center
            auto bond = args.getInt("AtBond", psi_.length() / 2);
            if (bond > 0 && bond < psi_.length()) {
                auto s = entanglement_entropy(bond);
                if (sweep >= static_cast<int>(entropies_.size())) {
                    entropies_.push_back(s);
                } else {
                    entropies_[sweep] = s;
                }
            }
        }
    }
    
    // Check if DMRG has converged
    bool checkDone(Args const& args) override {
        auto sweep = args.getInt("Sweep", 0);
        
        if (sweep < 2) return false;
        if (sweep >= max_sweeps_) {
            std::cout << "Reached maximum number of sweeps." << std::endl;
            return true;
        }
        
        // Only check every check_interval sweeps
        if (sweep % check_interval_ != 0) return false;
        
        // Check energy convergence
        if (energies_.size() >= 2) {
            double de = std::abs(energies_.back() - energies_[energies_.size()-2]);
            if (de < energy_threshold_) {
                std::cout << "Energy converged: ΔE = " << std::scientific 
                         << std::setprecision(2) << de << std::endl;
                return true;
            }
        }
        
        return false;
    }
    
    // Get recorded energies
    std::vector<double> const& energies() const { return energies_; }
    
    // Get recorded entropies
    std::vector<double> const& entropies() const { return entropies_; }
    
    // Calculate entanglement entropy at a bond
    double entanglement_entropy(int bond) const {
        auto psi_orth = psi_;
        psi_orth.position(bond);
        
        auto [U, S, V] = svd(psi_orth(bond) * psi_orth(bond + 1), 
                            leftLinkIndex(psi_orth, bond + 1));
        
        double entropy = 0.0;
        for (int n = 1; n <= dim(S.inds()[0]); ++n) {
            double p = sqr(elt(S, n, n));
            if (p > 1E-15) {
                entropy -= p * std::log(p);
            }
        }
        
        return entropy;
    }
    
    // Calculate correlation length from transfer matrix
    double correlation_length() const {
        // TODO: Implement transfer matrix analysis
        return -1.0;
    }
    
    // Add custom measurement function
    using MeasureFn = std::function<double(MPS const&, int)>;
    void addMeasurement(std::string const& name, MeasureFn fn) {
        custom_measurements_[name] = fn;
    }
    
    // Perform all custom measurements
    std::map<std::string, std::vector<double>> 
    performMeasurements(int start_site = 1, int end_site = -1) const {
        std::map<std::string, std::vector<double>> results;
        
        if (end_site < 0) end_site = psi_.length();
        
        for (const auto& [name, fn] : custom_measurements_) {
            std::vector<double> values;
            for (int i = start_site; i <= end_site; ++i) {
                values.push_back(fn(psi_, i));
            }
            results[name] = values;
        }
        
        return results;
    }

private:
    MPS const& psi_;
    double energy_threshold_;
    double entropy_threshold_;
    int max_sweeps_;
    int check_interval_;
    
    std::vector<double> energies_;
    std::vector<double> entropies_;
    std::map<std::string, MeasureFn> custom_measurements_;
};

//
// Measurement utilities for common observables
//

// Measure local Sz expectation value
inline std::vector<double> 
measureSz(MPS const& psi, SiteSet const& sites) {
    std::vector<double> sz(psi.length());
    for (int i = 1; i <= psi.length(); ++i) {
        auto ket = psi(i);
        auto bra = dag(prime(ket, "Site"));
        auto Sz = op(sites, "Sz", i);
        sz[i-1] = elt(bra * Sz * ket);
    }
    return sz;
}

// Measure two-point correlation function <O_1 O_j>
template<typename SiteSetType>
std::vector<double>
measureCorrelation(MPS const& psi, 
                   SiteSetType const& sites,
                   std::string const& op1,
                   std::string const& op2,
                   int site1 = 1) {
    auto psi_work = psi;
    psi_work.position(site1);
    
    std::vector<double> corr(psi.length() - site1);
    
    auto wf1 = psi_work(site1);
    auto oi = uniqueIndex(wf1, psi_work(site1 + 1), "Link");
    
    auto lcorr = prime(wf1, oi) * op(sites, op1, site1) * dag(prime(wf1));
    
    for (int j = site1 + 1; j <= psi.length(); ++j) {
        int n = j;  // For infinite systems: (j - 1) % N + 1
        auto ui = uniqueIndex(psi_work(n), lcorr, "Link");
        
        corr[j - site1 - 1] = elt(dag(prime(psi_work(n))) * lcorr * 
                                  prime(psi_work(n), ui) * 
                                  op(sites, op2, n));
        
        lcorr *= psi_work(n);
        lcorr *= dag(prime(psi_work(n), "Link"));
    }
    
    return corr;
}

// Measure structure factor S(k) = (1/N) Σ_j e^{ikj} <S_0 S_j>
inline std::vector<std::complex<double>>
structureFactor(std::vector<double> const& corr, int num_k = 100) {
    std::vector<std::complex<double>> sk(num_k);
    const double pi = 3.14159265358979323846;
    
    for (int ik = 0; ik < num_k; ++ik) {
        double k = 2.0 * pi * ik / num_k;
        std::complex<double> sum(0.0, 0.0);
        
        for (size_t j = 0; j < corr.size(); ++j) {
            sum += corr[j] * std::exp(std::complex<double>(0.0, k * (j + 1)));
        }
        
        sk[ik] = sum / static_cast<double>(corr.size());
    }
    
    return sk;
}

} // namespace idmrg

#endif // IDMRG_OBSERVER_H
