/***********************************************************************
 MrHyDE - P^-1 for the parameter Riesz map by one sparse factorization
 of P (Amesos2).

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_PARAM_RIESZ_EXACT_HPP
#define MRHYDE_PARAM_RIESZ_EXACT_HPP

#include "MrHyDE_ParamRiesz.hpp"
#include "Amesos2.hpp"

// One factorization serves every slab and every CG iteration.
class MrHyDE_ParamRieszExact : public MrHyDE_ParamRiesz {

public:

  // P_factor is P with a sparser stored pattern, or null to factor P itself.
  MrHyDE_ParamRieszExact(const matrix_RCP & P_, const std::string & amesos_name_ = "KLU2",
                         const matrix_RCP & P_factor_ = Teuchos::null)
  : MrHyDE_ParamRiesz(P_), P_factor(denseIds(P_factor_.is_null() ? P_ : P_factor_)), amesos_name(amesos_name_) {}

  // The slabs are the columns of one solve.
  void applyInverse(const std::vector<vector_RCP> & xs, const std::vector<vector_RCP> & ys) const override {
    if (xs.empty()) return;
    Teuchos::TimeMonitor localtimer(*solvetimer);
    size_t nv = 0;
    for (const vector_RCP & x : xs) nv += x->getNumVectors();
    if (b_work.is_null() || b_work->getNumVectors() < nv) {
      b_work = Teuchos::rcp(new LA_MultiVector(P_factor->getRangeMap(), nv, false));
      x_work = Teuchos::rcp(new LA_MultiVector(P_factor->getDomainMap(), nv, false));
    }
    const vector_RCP b = b_work->subViewNonConst(Teuchos::Range1D(0, nv - 1));
    const vector_RCP x = x_work->subViewNonConst(Teuchos::Range1D(0, nv - 1));
    size_t c = 0;
    for (size_t s = 0; s < xs.size(); ++s) {
      const size_t k = xs[s]->getNumVectors();
      b->subViewNonConst(Teuchos::Range1D(c, c + k - 1))->assign(*xs[s]);
      c += k;
    }
    if (solver.is_null()) {
      Teuchos::TimeMonitor localfactortimer(*factortimer);
      solver = Amesos2::create<LA_CrsMatrix,LA_MultiVector>(amesos_name, P_factor, x, b);
      solver->symbolicFactorization();
      solver->numericFactorization();
    }
    solver->setX(x);
    solver->setB(b);
    solver->solve();
    c = 0;
    for (size_t s = 0; s < ys.size(); ++s) {
      const size_t k = ys[s]->getNumVectors();
      ys[s]->assign(*x->subView(Teuchos::Range1D(c, c + k - 1)));
      c += k;
    }
  }

private:

  // Amesos2 indexes the gathered matrix by global id, so gapped ids become [0,n); the local data are shared.
  static matrix_RCP denseIds(const matrix_RCP & A) {
    typedef Tpetra::Map<LO,GO,SolverNode> Map;
    const Teuchos::RCP<const Map> rows = A->getRowMap();
    if (rows->getMaxAllGlobalIndex() + 1 == static_cast<GO>(rows->getGlobalNumElements())) return A;
    const Teuchos::RCP<const Map> dense = Teuchos::rcp(new Map(rows->getGlobalNumElements(),
      rows->getLocalNumElements(), 0, rows->getComm()));
    LA_MultiVector ids(rows, 1), col_ids(A->getColMap(), 1);
    {
      auto v = ids.getLocalViewHost(Tpetra::Access::OverwriteAll);
      for (size_t i = 0; i < v.extent(0); ++i) v(i,0) = static_cast<ScalarT>(dense->getGlobalElement(i));
    }
    col_ids.doImport(ids, Tpetra::Import<LO,GO,SolverNode>(rows, A->getColMap()), Tpetra::INSERT);
    std::vector<GO> cols(col_ids.getLocalLength());
    {
      auto v = col_ids.getLocalViewHost(Tpetra::Access::ReadOnly);
      for (size_t c = 0; c < cols.size(); ++c) cols[c] = static_cast<GO>(v(c,0));
    }
    const Teuchos::RCP<const Map> dense_cols = Teuchos::rcp(new Map(
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), Teuchos::ArrayView<const GO>(cols), 0, rows->getComm()));
    return Teuchos::rcp(new LA_CrsMatrix(A->getLocalMatrixDevice(), dense, dense_cols, dense, dense));
  }

  matrix_RCP P_factor;
  std::string amesos_name;
  mutable vector_RCP b_work, x_work;   // slabs side by side; grows to the widest apply seen
  mutable Teuchos::RCP<Amesos2::Solver<LA_CrsMatrix,LA_MultiVector> > solver;
  Teuchos::RCP<Teuchos::Time> factortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParamRiesz::factor()");

};

#endif
