/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_PARAM_RIESZ_HPP
#define MRHYDE_PARAM_RIESZ_HPP

#include "trilinos.hpp"
#include "preferences.hpp"
#include "Amesos2.hpp"

// P maps primal to dual and P^-1 back. ROL's default precond is v.dual(),
// so P^-1 also preconditions the trust-region CG.
class MrHyDE_ParamRiesz {

  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SolverNode>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> LA_MultiVector;
  typedef Teuchos::RCP<LA_MultiVector>                  vector_RCP;
  typedef Teuchos::RCP<LA_CrsMatrix>                    matrix_RCP;

public:

  MrHyDE_ParamRiesz(const matrix_RCP & P_)
  : P(P_) {}

  // y <- P x
  void apply(const vector_RCP & x, const vector_RCP & y) const {
    P->apply(*x, *y);
  }

  // y <- P^-1 x. One factorization serves every slab and every CG iteration.
  void applyInverse(const vector_RCP & x, const vector_RCP & y) const {
    if (solver.is_null()) {
      solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>("KLU2", P, y, x);
      // The parameter map is built from a gid list, so KLU2 must be told whether it is contiguous.
      Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::rcp(new Teuchos::ParameterList());
      plist->sublist("KLU2").set("IsContiguous", P->getRowMap()->isContiguous());
      solver->setParameters(plist);
      solver->symbolicFactorization();
      solver->numericFactorization();
    }
    solver->setX(y);
    solver->setB(x);
    solver->solve();
  }

private:

  matrix_RCP P;
  mutable Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > solver;

};

#endif
