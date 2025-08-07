#ifndef DMRG_SOLVER_H
#define DMRG_SOLVER_H

#include "itensor/all.h"
#include "hamiltonian.h"
#include "lattice.h"

class DMRGSolver {
public:
    DMRGSolver(const Hamiltonian& hamiltonian, const Lattice& lattice)
        : hamiltonian_(hamiltonian), lattice_(lattice) {}

    std::pair<double, itensor::MPS> run() {
        auto H = hamiltonian_.get_mpo();
        auto sites = lattice_.get_sites();
        auto psi0 = itensor::randomMPS(sites);

        auto sweeps = itensor::Sweeps(5);
        sweeps.maxdim() = 10, 20, 100, 100, 200;
        sweeps.cutoff() = 1E-10;

        auto [energy, psi] = itensor::dmrg(H, psi0, sweeps, {"Quiet", true});

        return {energy, psi};
    }

private:
    const Hamiltonian& hamiltonian_;
    const Lattice& lattice_;
};

#endif // DMRG_SOLVER_H
