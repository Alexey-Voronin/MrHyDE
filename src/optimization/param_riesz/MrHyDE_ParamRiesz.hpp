/***********************************************************************
 MrHyDE - The parameter Riesz operator P: applies P (primal to dual) and
 P^-1 (dual back to primal), by sparse factorization or RefMaxwell-CG.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_PARAM_RIESZ_HPP
#define MRHYDE_PARAM_RIESZ_HPP

#include "trilinos.hpp"
#include "preferences.hpp"

// P maps primal to dual and P^-1 back. ROL's default precond is v.dual(),
// so P^-1 also preconditions the trust-region CG.
class MrHyDE_ParamRiesz {

protected:

  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SolverNode>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> LA_MultiVector;
  typedef Tpetra::Operator<ScalarT,LO,GO,SolverNode>    LA_Operator;
  typedef Teuchos::RCP<LA_MultiVector>                  vector_RCP;
  typedef Teuchos::RCP<LA_CrsMatrix>                    matrix_RCP;

public:

  explicit MrHyDE_ParamRiesz(const matrix_RCP & P_) : P(P_) {}

  virtual ~MrHyDE_ParamRiesz() {}

  // ys[s] <- P xs[s]
  void apply(const std::vector<vector_RCP> & xs, const std::vector<vector_RCP> & ys) const {
    for (size_t s = 0; s < xs.size(); ++s) P->apply(*xs[s], *ys[s]);
  }

  // ys[s] <- P^-1 xs[s]
  virtual void applyInverse(const std::vector<vector_RCP> & xs, const std::vector<vector_RCP> & ys) const = 0;

protected:

  matrix_RCP P;
  Teuchos::RCP<Teuchos::Time> solvetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParamRiesz::applyInverse()");

};

#endif
