#ifndef MRHYDE_LINEARALGEBRAINTERFACE_MASSMATRIX_HPP
#define MRHYDE_LINEARALGEBRAINTERFACE_MASSMATRIX_HPP

#include "trilinos.hpp"
#include "preferences.hpp"
#include "block_prec/BlockOperators.hpp"
#include <Amesos2.hpp>

namespace MrHyDE {

template<class Node> class LinearAlgebraInterface;

// M and approximate M^{-1} as Tpetra::Operator wrappers for ROL (diagonal, lumped, sparse).
template<class Node>
struct MassMatrixOperators {
  using OperatorRCP = Teuchos::RCP<Tpetra::Operator<ScalarT, LO, GO, Node>>;

  OperatorRCP forwardOp;
  OperatorRCP inverseOp;
  std::string type;
};

// mass_type: none, default, diagonal, lumped, sparse_direct, sparse_iterative.
template<class Node>
MassMatrixOperators<Node>
buildMassMatrixOperators(LinearAlgebraInterface<Node> & linalg,
                         const Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node>> & mass,
                         const Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node>> & diagMass,
                         const std::string & mass_type,
                         const Teuchos::ParameterList & solver_params = Teuchos::ParameterList()) {

  using Types = block_prec::BlockTypes<Node>;
  using LA_Vector = typename Types::Vector;
  using LA_MultiVector = typename Types::MultiVector;
  using LA_CrsMatrix = typename Types::CrsMatrix;
  using OperatorRCP = typename MassMatrixOperators<Node>::OperatorRCP;

  MassMatrixOperators<Node> result;
  result.type = mass_type;

  if (mass_type == "none" || mass_type == "default") {
    result.forwardOp = Teuchos::null;
    result.inverseOp = Teuchos::null;
    return result;
  }

  if (mass_type == "diagonal" || mass_type == "lumped") {
    TEUCHOS_TEST_FOR_EXCEPTION(diagMass.is_null(), std::runtime_error,
                               "buildMassMatrixOperators: diagMass required for diagonal/lumped type.");

    auto diag_vec = diagMass->getVectorNonConst(0);

    auto inv_diag_vec = Teuchos::rcp(new LA_Vector(diagMass->getMap()));
    inv_diag_vec->reciprocal(*diag_vec);

    result.forwardOp = Teuchos::rcp(new block_prec::DiagonalMultiplyOperator<Node>(diag_vec));
    result.inverseOp = Teuchos::rcp(new block_prec::DiagonalInverseOperator<Node>(inv_diag_vec));

    return result;
  }

  if (mass_type == "sparse_direct") {
    TEUCHOS_TEST_FOR_EXCEPTION(mass.is_null(), std::runtime_error,
                               "buildMassMatrixOperators: mass matrix required for sparse_direct type.");

    result.forwardOp = mass;

    auto solver = Amesos2::create<LA_CrsMatrix, LA_MultiVector>("KLU2", mass);
    solver->symbolicFactorization();
    solver->numericFactorization();

    result.inverseOp = Teuchos::rcp(new block_prec::DirectSolveOperator<Node>(
      solver, mass->getRowMap()));

    return result;
  }

  if (mass_type == "sparse_iterative") {
    TEUCHOS_TEST_FOR_EXCEPTION(mass.is_null(), std::runtime_error,
                               "buildMassMatrixOperators: mass matrix required for sparse_iterative type.");

    result.forwardOp = mass;

    Teuchos::ParameterList belos_params;

    int max_iters = 200;
    double tol = 1e-10;
    std::string solver_type = "GMRES";

    if (solver_params.isParameter("mass matrix max iterations")) {
      max_iters = solver_params.get<int>("mass matrix max iterations");
    }
    if (solver_params.isParameter("mass matrix tolerance")) {
      tol = solver_params.get<double>("mass matrix tolerance");
    }
    if (solver_params.isParameter("mass matrix solver")) {
      solver_type = solver_params.get<std::string>("mass matrix solver");
    }

    belos_params.set("Maximum Iterations", max_iters);
    belos_params.set("Convergence Tolerance", tol);
    belos_params.set("Verbosity", Belos::Errors + Belos::Warnings);
    belos_params.set("Output Style", Belos::Brief);

    result.inverseOp = Teuchos::rcp(new block_prec::IterativeSolveOperator<Node>(
      mass, solver_type, belos_params));

    return result;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "buildMassMatrixOperators: Unknown mass_type '" + mass_type + "'. "
                             "Valid options: none, diagonal, lumped, sparse_direct, sparse_iterative.");

  return result; // unreachable
}

} // namespace MrHyDE

#endif // MRHYDE_LINEARALGEBRAINTERFACE_MASSMATRIX_HPP
