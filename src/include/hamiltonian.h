#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "itensor/all.h"
#include "lattice.h"

class Hamiltonian {
public:
    Hamiltonian(const Lattice& lattice) : lattice_(lattice) {}

    virtual ~Hamiltonian() = default;

    virtual itensor::MPO get_mpo() const = 0;

protected:
    const Lattice& lattice_;
};

class Heisenberg : public Hamiltonian {
public:
    Heisenberg(const Lattice& lattice) : Hamiltonian(lattice) {}

    itensor::MPO get_mpo() const override {
        auto ampo = itensor::AutoMPO(lattice_.get_sites());
        for (int j = 1; j < lattice_.L(); ++j) {
            ampo += 0.5, "S+", j, "S-", j + 1;
            ampo += 0.5, "S-", j, "S+", j + 1;
            ampo += "Sz", j, "Sz", j + 1;
        }
        return itensor::toMPO(ampo);
    }
};

#endif // HAMILTONIAN_H
