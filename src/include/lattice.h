#ifndef LATTICE_H
#define LATTICE_H

#include "itensor/all.h"

class Lattice {
public:
    Lattice(int L) : L_(L) {}

    virtual ~Lattice() = default;

    int L() const { return L_; }

    virtual itensor::SiteSet get_sites() const = 0;

protected:
    int L_;
};

class Chain : public Lattice {
public:
    Chain(int L) : Lattice(L) {}

    itensor::SiteSet get_sites() const override {
        return itensor::SpinHalf(L_, {"ConserveQNs", false});
    }
};

#endif // LATTICE_H
