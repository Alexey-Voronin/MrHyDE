#ifndef MRHYDE_INNERKRYLOVOPERATOR_H
#define MRHYDE_INNERKRYLOVOPERATOR_H

#include "trilinos.hpp"
#include "preferences.hpp"

#include "Tpetra_Operator.hpp"
#include "Tpetra_MultiVector.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosPseudoBlockCGSolMgr.hpp"

namespace MrHyDE {

// InnerKrylovOperator.hpp owns the Tpetra::Operator that runs Belos pseudo-block CG with a
// left preconditioner, and the factory that allocates it.
// It does not own InnerKrylovConfig (see linearSolverContext.hpp), block assembly, or Schur builders.

// InnerKrylovConfig is defined in linearSolverContext.hpp.
struct InnerKrylovConfig;

/** Tpetra::Operator that applies an inexact solve: matrix * Y = X via CG with a left preconditioner. */
template<class ScalarT, class LO, class GO, class Node>
class InnerKrylovOperator : public Tpetra::Operator<ScalarT, LO, GO, Node> {
public:
  using Operator = Tpetra::Operator<ScalarT, LO, GO, Node>;
  using MultiVector = Tpetra::MultiVector<ScalarT, LO, GO, Node>;
  using Map = Tpetra::Map<LO, GO, Node>;
  using LinearProblem = Belos::LinearProblem<ScalarT, MultiVector, Operator>;
  using SolverManager = Belos::PseudoBlockCGSolMgr<ScalarT, MultiVector, Operator>;

private:
  Teuchos::RCP<const Operator> matrix_;      // Block matrix (J00 or Schur approximation).
  Teuchos::RCP<const Operator> prec_;        // Base preconditioner (AMG, RefMaxwell, etc.).
  InnerKrylovConfig config_;

  mutable Teuchos::RCP<SolverManager> solver_;
  mutable Teuchos::RCP<Teuchos::ParameterList> solverParams_;
  mutable Teuchos::RCP<LinearProblem> problem_;
  mutable bool initialized_;
  mutable int total_iters_;
  mutable int num_applies_;

public:
  InnerKrylovOperator(
    const Teuchos::RCP<const Operator>& matrix,
    const Teuchos::RCP<const Operator>& prec,
    const InnerKrylovConfig& config)
    : matrix_(matrix)
    , prec_(prec)
    , config_(config)
    , initialized_(false)
    , total_iters_(0)
    , num_applies_(0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(matrix_.is_null(), std::runtime_error,
      "InnerKrylovOperator: matrix cannot be null");
    TEUCHOS_TEST_FOR_EXCEPTION(prec_.is_null(), std::runtime_error,
      "InnerKrylovOperator: preconditioner cannot be null");
  }

  virtual ~InnerKrylovOperator() {
    if (num_applies_ > 0 && config_.verbose) {
      std::cout << "InnerKrylovOperator: Applied " << num_applies_
                << " times, average inner iterations = "
                << (double)total_iters_ / num_applies_ << std::endl;
    }
  }

  Teuchos::RCP<const Map> getDomainMap() const override {
    return matrix_->getDomainMap();
  }

  Teuchos::RCP<const Map> getRangeMap() const override {
    return matrix_->getRangeMap();
  }

  void apply(const MultiVector& X, MultiVector& Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             ScalarT alpha = Teuchos::ScalarTraits<ScalarT>::one(),
             ScalarT beta = Teuchos::ScalarTraits<ScalarT>::zero()) const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(mode != Teuchos::NO_TRANS, std::logic_error,
      "InnerKrylovOperator::apply only supports NO_TRANS mode");
    TEUCHOS_TEST_FOR_EXCEPTION(alpha != Teuchos::ScalarTraits<ScalarT>::one() ||
                               beta != Teuchos::ScalarTraits<ScalarT>::zero(),
                               std::logic_error,
      "InnerKrylovOperator::apply only supports alpha=1, beta=0");

    // Initialize solver on first call
    if (!initialized_) {
      initializeSolver();
    }

    // Create copies for the linear problem (Belos requires non-const RCPs)
    Teuchos::RCP<MultiVector> X_rcp = Teuchos::rcp(new MultiVector(X));
    Teuchos::RCP<MultiVector> Y_rcp = Teuchos::rcp(&Y, false); // Non-owning RCP

    // Set zero initial guess
    Y.putScalar(Teuchos::ScalarTraits<ScalarT>::zero());

    // Set problem
    problem_->setProblem(Y_rcp, X_rcp);

    // Solve
    Belos::ReturnType ret = solver_->solve();

    // Track statistics
    num_applies_++;
    int iters = solver_->getNumIters();
    total_iters_ += iters;

    // Handle convergence status
    if (ret != Belos::Converged) {
      if (config_.warn_on_failure) {
        std::cout << "WARNING: InnerKrylovOperator: CG did not converge in "
                  << iters << " iterations (tol=" << config_.tolerance
                  << "). Using best iterate." << std::endl;
      }
      // Best iterate is already in Y, so we just continue
    }
  }

  bool hasTransposeApply() const override {
    return false;
  }

  int getTotalInnerIterations() const {
    return total_iters_;
  }

  int getNumApplies() const {
    return num_applies_;
  }

  double getAverageInnerIterations() const {
    return num_applies_ > 0 ? (double)total_iters_ / num_applies_ : 0.0;
  }

private:
  void initializeSolver() const {
    // Create parameter list for Belos CG
    solverParams_ = Teuchos::rcp(new Teuchos::ParameterList("Inner CG"));
    solverParams_->set("Convergence Tolerance", config_.tolerance);
    solverParams_->set("Maximum Iterations", config_.max_iterations);
    solverParams_->set("Verbosity", config_.verbose ?
                       Belos::Errors + Belos::Warnings + Belos::IterationDetails :
                       Belos::Errors + Belos::Warnings);
    solverParams_->set("Output Frequency", 1);

    // Create linear problem
    problem_ = Teuchos::rcp(new LinearProblem());
    problem_->setOperator(matrix_);
    problem_->setLeftPrec(prec_);

    // Create solver
    solver_ = Teuchos::rcp(new SolverManager(problem_, solverParams_));

    initialized_ = true;
  }
};

/**
 * Wraps a preconditioner with an inner Krylov solver (CG).
 *
 * Takes a block matrix and preconditioner, and wraps them in an InnerKrylovOperator
 * that applies the preconditioner iteratively using CG instead of a single application.
 *
 * matrix is the block matrix (J00 for pivot or SchurApprox for Schur).
 * prec is the base preconditioner (AMG, RefMaxwell, etc.).
 * config holds inner Krylov solver options.
 * Returns an operator that solves matrix*y=x using CG with prec.
 */
template<class ScalarT, class LO, class GO, class Node>
Teuchos::RCP<Tpetra::Operator<ScalarT, LO, GO, Node>>
createInnerKrylovOperator(
  const Teuchos::RCP<const Tpetra::Operator<ScalarT, LO, GO, Node>>& matrix,
  const Teuchos::RCP<const Tpetra::Operator<ScalarT, LO, GO, Node>>& prec,
  const InnerKrylovConfig& config)
{
  return Teuchos::rcp(new InnerKrylovOperator<ScalarT, LO, GO, Node>(matrix, prec, config));
}

} // namespace MrHyDE

#endif // MRHYDE_INNERKRYLOVOPERATOR_H
