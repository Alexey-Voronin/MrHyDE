/***********************************************************************
 MrHyDE - One row scale per split, so the Krylov stopping test weighs every
 split of a block system against its own right-hand side.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_BLOCK_PREC_SPLIT_SCALING_HPP
#define MRHYDE_BLOCK_PREC_SPLIT_SCALING_HPP

#include "block_prec/BlockTypes.hpp"
#include "block_prec/TekoAdapter.hpp"
#include "linearSolverContext.hpp"

#include <BelosLinearProblem.hpp>
#include <Teuchos_CommHelpers.hpp>

#include <cmath>
#include <vector>

namespace MrHyDE {
namespace block_prec {

// Y = alpha * post .* op(pre .* X) + beta * Y, with a null scale meaning the identity.
template<class Node>
class ScaledOperator : public Tpetra::Operator<ScalarT,LO,GO,Node> {
public:
  using LA_Map = typename BlockTypes<Node>::Map;
  using LA_MultiVector = typename BlockTypes<Node>::MultiVector;
  using LA_Vector = typename BlockTypes<Node>::Vector;
  using LA_Operator = Tpetra::Operator<ScalarT,LO,GO,Node>;

  ScaledOperator(const Teuchos::RCP<const LA_Operator> & op,
                 const Teuchos::RCP<const LA_Vector> & pre,
                 const Teuchos::RCP<const LA_Vector> & post)
    : op_(op), pre_(pre), post_(post) {}

  Teuchos::RCP<const LA_Map> getDomainMap() const override { return op_->getDomainMap(); }
  Teuchos::RCP<const LA_Map> getRangeMap() const override { return op_->getRangeMap(); }
  bool hasTransposeApply() const override { return false; }

  void apply(const LA_MultiVector & X, LA_MultiVector & Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta = Teuchos::ScalarTraits<ScalarT>::zero()) const override {
    TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS, std::runtime_error,
      "ScaledOperator: only NO_TRANS is supported.");
    const size_t nvec = X.getNumVectors();
    const LA_MultiVector * in = &X;
    if (!pre_.is_null()) {
      ensureWorkspace(preX_, X.getMap(), nvec);
      preX_->elementWiseMultiply(1.0, *pre_, X, 0.0);
      in = preX_.getRawPtr();
    }
    if (post_.is_null()) {
      op_->apply(*in, Y, mode, alpha, beta);
      return;
    }
    ensureWorkspace(opX_, op_->getRangeMap(), nvec);
    op_->apply(*in, *opX_);
    Y.elementWiseMultiply(alpha, *post_, *opX_, beta);
  }

private:
  static void ensureWorkspace(Teuchos::RCP<LA_MultiVector> & v, const Teuchos::RCP<const LA_Map> & map,
                              const size_t nvec) {
    if (v.is_null() || v->getNumVectors() != nvec) v = Teuchos::rcp(new LA_MultiVector(map, nvec, false));
  }

  Teuchos::RCP<const LA_Operator> op_;
  Teuchos::RCP<const LA_Vector> pre_, post_;
  mutable Teuchos::RCP<LA_MultiVector> preX_, opX_;
};

// d_k = 1 / ||r_k|| on the rows of split k (rowSplit[lid] = k), or 1 / ||r|| where r_k = 0;
// the norms are collective.
template<class Node>
Teuchos::RCP<typename BlockTypes<Node>::Vector>
splitRowScales(const typename BlockTypes<Node>::MultiVector & r, const std::vector<int> & rowSplit,
               const size_t numSplits) {
  using LA_Vector = typename BlockTypes<Node>::Vector;
  auto slot = [&](const size_t i) { return rowSplit[i] < 0 ? numSplits : static_cast<size_t>(rowSplit[i]); };
  std::vector<ScalarT> localSq(numSplits + 1, 0.0), globalSq(numSplits + 1, 0.0);
  {
    auto rv = r.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t i = 0; i < rowSplit.size(); ++i) {
      for (size_t j = 0; j < r.getNumVectors(); ++j) {
        localSq[slot(i)] += rv(i, j) * rv(i, j);
      }
    }
  }
  Teuchos::reduceAll<int, ScalarT>(*r.getMap()->getComm(), Teuchos::REDUCE_SUM,
                                   static_cast<int>(numSplits + 1), localSq.data(), globalSq.data());
  ScalarT total = 0.0;
  for (const ScalarT s : globalSq) total += s;
  std::vector<ScalarT> scale(numSplits + 1, total > 0.0 ? 1.0 / std::sqrt(total) : 1.0);
  for (size_t k = 0; k < numSplits; ++k) {
    if (globalSq[k] > 0.0) scale[k] = 1.0 / std::sqrt(globalSq[k]);
  }
  Teuchos::RCP<LA_Vector> d = Teuchos::rcp(new LA_Vector(r.getMap(), false));
  {
    auto dv = d->getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (size_t i = 0; i < rowSplit.size(); ++i) dv(i, 0) = scale[slot(i)];
  }
  return d;
}

// Belos solves (D J)(M D^{-1}) y = D r, so it minimizes and tests D(r - J x).
template<class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> >
maybeScaleSplits(const Teuchos::RCP<LinearSolverContext<Node> > & cntxt,
                 Belos::LinearProblem<ScalarT, typename BlockTypes<Node>::MultiVector,
                                      Tpetra::Operator<ScalarT,LO,GO,Node> > & problem,
                 const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > & prec) {
  using LA_MultiVector = typename BlockTypes<Node>::MultiVector;
  using LA_Vector = typename BlockTypes<Node>::Vector;
  if (!cntxt->schur.split_residual_scaling[cntxt->sweep_kind]) return prec;
  Teuchos::RCP<const TekoTpetraAdapter<Node> > adapter =
    Teuchos::rcp_dynamic_cast<const TekoTpetraAdapter<Node> >(prec);
  TEUCHOS_TEST_FOR_EXCEPTION(adapter.is_null(), std::runtime_error,
    "'split residual scaling' needs the block-triangular preconditioner.");
  const LA_MultiVector & r = *problem.getRHS();
  Teuchos::RCP<LA_Vector> d = splitRowScales<Node>(r, adapter->rowSplit(), adapter->numSplits());
  Teuchos::RCP<LA_Vector> dinv = Teuchos::rcp(new LA_Vector(r.getMap(), false));
  dinv->reciprocal(*d);
  Teuchos::RCP<LA_MultiVector> rhs = Teuchos::rcp(new LA_MultiVector(r.getMap(), r.getNumVectors(), false));
  rhs->elementWiseMultiply(1.0, *d, r, 0.0);
  problem.setOperator(Teuchos::rcp(new ScaledOperator<Node>(problem.getOperator(), Teuchos::null, d)));
  problem.setRHS(rhs);
  return Teuchos::rcp(new ScaledOperator<Node>(prec, dinv, Teuchos::null));
}

} // namespace block_prec
} // namespace MrHyDE

#endif
