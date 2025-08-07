#include "include/lattice.h"
#include "include/hamiltonian.h"
#include "include/dmrg_solver.h"
#include "itensor/all.h"

int main() {
    // Define the lattice
    int L = 100;
    Chain lattice(L);

    // Define the Hamiltonian
    Heisenberg hamiltonian(lattice);

    // Create and run the DMRG solver
    DMRGSolver solver(hamiltonian, lattice);
    auto [energy, psi] = solver.run();

    // Print the final ground state energy
    itensor::println("Ground state energy = ", energy);

    return 0;
}
