/***********************************************************************
 MrHyDE - The parameter Riesz operator P: applies P (primal to dual) and
 P^-1 (dual back to primal), by sparse factorization or RefMaxwell-CG.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_PARAM_RIESZ_HPP
#define MRHYDE_PARAM_RIESZ_HPP

#include "trilinos.hpp"
#include "preferences.hpp"
#include "ParamRieszConfig.hpp"
#include "Amesos2.hpp"
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <iostream>

// P maps primal to dual and P^-1 back. ROL's default precond is v.dual(),
// so P^-1 also preconditions the trust-region CG.
class MrHyDE_ParamRiesz {

  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SolverNode>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> LA_MultiVector;
  typedef Tpetra::Operator<ScalarT,LO,GO,SolverNode>    LA_Operator;
  typedef Teuchos::RCP<LA_MultiVector>                  vector_RCP;
  typedef Teuchos::RCP<LA_CrsMatrix>                    matrix_RCP;

public:

  // P_factor is P with a sparser stored pattern, or null to factor P itself.
  MrHyDE_ParamRiesz(const matrix_RCP & P_, const std::string & solver_name_ = "KLU2",
                    const matrix_RCP & P_factor_ = Teuchos::null)
  : P(P_), P_factor(P_factor_.is_null() ? P_ : P_factor_), solver_name(solver_name_) {}

  // Krylov on the control rows P_live; P is diagonal on the other rows, given by diag.
  MrHyDE_ParamRiesz(const matrix_RCP & P_, const matrix_RCP & P_live_, const vector_RCP & diag,
                    const Teuchos::RCP<LA_Operator> & prec_, const MrHyDE::ParamRieszConfig & cfg,
                    const int verbosity_ = 0)
  : P(P_), P_factor(P_), solver_name(cfg.krylov == "GMRES" ? "Pseudo Block GMRES" : "Pseudo Block CG"),
    P_live(P_live_), prec(prec_), belos_params(cfg.belos), verbosity(verbosity_), strict(cfg.strict),
    slab_cap(cfg.krylov == "GMRES" ? 1 : (cfg.slabs_per_solve > 0 ? size_t(cfg.slabs_per_solve) : max_auto_slabs)) {
    inv_diag = Teuchos::rcp(new LA_MultiVector(diag->getMap(), 1));
    inv_diag->reciprocal(*diag);
    to_live = Teuchos::rcp(new Tpetra::Import<LO,GO,SolverNode>(P->getRowMap(), P_live->getRowMap()));
    rank = P_live->getRowMap()->getComm()->getRank();
  }

  // y <- P x
  void apply(const vector_RCP & x, const vector_RCP & y) const {
    P->apply(*x, *y);
  }

  // y <- P^-1 x. One factorization serves every slab and every CG iteration.
  void applyInverse(const vector_RCP & x, const vector_RCP & y) const {
    Teuchos::TimeMonitor localtimer(*solvetimer);
    if (!P_live.is_null()) {
      this->solveControlRows({x}, {y}, true);
      return;
    }
    if (solver.is_null()) {
      Teuchos::TimeMonitor localfactortimer(*factortimer);
      solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>(solver_name, P_factor, y, x);
      solver->symbolicFactorization();
      solver->numericFactorization();
    }
    solver->setX(y);
    solver->setB(x);
    solver->solve();
  }

  // ys[s] <- P^-1 xs[s]; the iterative CG apply solves the slabs in groups of at most slab_cap.
  void applyInverse(const std::vector<vector_RCP> & xs, const std::vector<vector_RCP> & ys) const {
    std::vector<vector_RCP> bx, by;
    for (size_t s = 0; s < xs.size(); ++s) {
      // An all-zero slab maps to zero; in a batch it would break CG under a Deflation Quorum above 1.
      if (slab_cap > 1 && allZero(*xs[s])) { ys[s]->putScalar(0.0); }
      else { bx.push_back(xs[s]); by.push_back(ys[s]); }
    }
    const size_t groups = (bx.size() + slab_cap - 1) / slab_cap;
    for (size_t g = 0, s0 = 0; g < groups; ++g) {
      const size_t nb = bx.size() / groups + (g < bx.size() % groups ? 1 : 0);
      bool done = false;
      if (nb > 1) {
        Teuchos::TimeMonitor localtimer(*solvetimer);
        done = this->solveControlRows(std::vector<vector_RCP>(bx.begin() + s0, bx.begin() + s0 + nb),
                                      std::vector<vector_RCP>(by.begin() + s0, by.begin() + s0 + nb), false);
      }
      // A batch that does not converge is redone slab by slab, so one bad slab cannot spoil the others.
      for (size_t s = s0; s < s0 + nb && !done; ++s) { this->applyInverse(bx[s], by[s]); }
      s0 += nb;
    }
  }

private:

  static bool allZero(const LA_MultiVector & x) {
    Teuchos::Array<typename LA_MultiVector::mag_type> nrm(x.getNumVectors());
    x.normInf(nrm());
    for (const auto v : nrm) { if (v != 0) return false; }
    return true;
  }

  // One multi-column Krylov solve of the control rows of xs; true if it converged, else report throws or warns.
  bool solveControlRows(const std::vector<vector_RCP> & xs, const std::vector<vector_RCP> & ys,
                        const bool report) const {
    size_t nv = 0;
    for (const vector_RCP & x : xs) nv += x->getNumVectors();
    if (b_work.is_null() || b_work->getNumVectors() != nv) {
      b_work = Teuchos::rcp(new LA_MultiVector(P_live->getRowMap(), nv));
      z_work = Teuchos::rcp(new LA_MultiVector(P_live->getRowMap(), nv));
    }
    const vector_RCP b = b_work, z = z_work;
    z->putScalar(0.0);   // cold start, so the apply does not depend on the previous solve
    size_t c = 0;
    for (size_t s = 0; s < xs.size(); ++s) {
      const size_t k = xs[s]->getNumVectors();
      ys[s]->elementWiseMultiply(1.0, *inv_diag->getVector(0), *xs[s], 0.0);
      b->subViewNonConst(Teuchos::Range1D(c, c + k - 1))->doImport(*xs[s], *to_live, Tpetra::INSERT);
      c += k;
    }
    if (krylov.is_null()) {
      problem = Teuchos::rcp(new Belos::LinearProblem<ScalarT,LA_MultiVector,LA_Operator>(P_live, z, b));
      if (!prec.is_null()) { problem->setLeftPrec(prec); }
      problem->setProblem();
      Belos::SolverFactory<ScalarT,LA_MultiVector,LA_Operator> factory;
      krylov = factory.create(solver_name, belos_params);
      krylov->setProblem(problem);
    }
    else {
      problem->setProblem(z, b);
    }
    const Belos::ReturnType ret = krylov->solve();
    if (ret != Belos::Converged && report) {
      TEUCHOS_TEST_FOR_EXCEPTION(strict, std::runtime_error,
        "Riesz map: iterative solve stopped at " << krylov->achievedTol() << " after "
        << krylov->getNumIters() << " iterations (strict is set).");
      if (!warned && verbosity >= 1 && rank == 0) {
        std::cout << "WARNING: Riesz map: iterative solve stopped at " << krylov->achievedTol()
                  << " after " << krylov->getNumIters() << " iterations; continuing. "
                  << "Set Iterative Settings -> strict: true to make this an error." << std::endl;
      }
      warned = true;
    }
    c = 0;
    for (size_t s = 0; s < ys.size(); ++s) {
      const size_t k = ys[s]->getNumVectors();
      ys[s]->doExport(*z->subView(Teuchos::Range1D(c, c + k - 1)), *to_live, Tpetra::INSERT);
      c += k;
    }
    return ret == Belos::Converged;
  }

  matrix_RCP P, P_factor;
  std::string solver_name;
  matrix_RCP P_live;
  Teuchos::RCP<LA_Operator> prec;
  Teuchos::RCP<Teuchos::ParameterList> belos_params;
  int verbosity = 0;
  bool strict = false;
  size_t slab_cap = 1;                           // most slabs per Krylov solve
  static constexpr size_t max_auto_slabs = 8;    // most slabs per solve when the deck sets none
  int rank = 0;
  mutable bool warned = false;
  mutable vector_RCP b_work, z_work;   // control-row RHS and solution, reused across applies
  vector_RCP inv_diag;
  Teuchos::RCP<Tpetra::Import<LO,GO,SolverNode> > to_live;
  mutable Teuchos::RCP<Belos::LinearProblem<ScalarT,LA_MultiVector,LA_Operator> > problem;
  mutable Teuchos::RCP<Belos::SolverManager<ScalarT,LA_MultiVector,LA_Operator> > krylov;
  mutable Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > solver;
  Teuchos::RCP<Teuchos::Time> factortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParamRiesz::factor()");
  Teuchos::RCP<Teuchos::Time> solvetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParamRiesz::applyInverse()");

};

#endif
