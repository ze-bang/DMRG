//
// iDMRG - Infinite Density Matrix Renormalization Group
// High-performance implementation using ITensor v3
//
// This header includes all components of the iDMRG library.
//

#ifndef IDMRG_ALL_H
#define IDMRG_ALL_H

// Configuration
#include "idmrg/config.h"

// Core algorithm
#include "idmrg/idmrg.h"
#include "idmrg/observer.h"

// Lattice geometries
#include "idmrg/lattice/lattice.h"
#include "idmrg/lattice/chain.h"
#include "idmrg/lattice/square.h"
#include "idmrg/lattice/triangular.h"
#include "idmrg/lattice/honeycomb.h"

// Physical models
#include "idmrg/models/heisenberg.h"
#include "idmrg/models/hubbard.h"
#include "idmrg/models/tj.h"

// Utilities
#include "idmrg/util/timer.h"
#include "idmrg/util/io.h"

#endif // IDMRG_ALL_H
