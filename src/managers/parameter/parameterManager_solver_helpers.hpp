#ifndef MRHYDE_PARAMETERMANAGER_SOLVER_HELPERS_H
#define MRHYDE_PARAMETERMANAGER_SOLVER_HELPERS_H

#include "trilinos.hpp"
#include "preferences.hpp"
#include "Amesos2_Factory.hpp"
#include "BelosSolverFactory.hpp"
#include "../linear_algebra/block_prec/BlockOperators.hpp"

namespace MrHyDE {

namespace ParameterManager_detail {

// ========================================================================
// DATA STRUCTURES FOR SOLVER CONFIGURATIONS
// ========================================================================

/**
 * @brief Configuration for direct solvers (sparse_direct)
 */
struct DirectSolverConfig {
  double regularization = 0.0;
  std::string solver_type = "KLU2";
};

/**
 * @brief Configuration for iterative solvers (sparse_iterative)
 */
struct IterativeSolverConfig {
  std::string solver_type = "CG";
  std::string prec_type = "diagonal";
  int max_iters = 100;
  double tolerance = 1e-8;
  bool strict_convergence = false;
  Teuchos::ParameterList prec_params;
};

// ========================================================================
// PARAMETER EXTRACTION HELPERS
// ========================================================================

/**
 * @brief Extract a single parameter from Analysis sublist with debug output
 * @tparam T Parameter type (string, double, int, bool)
 * @param settings Main settings parameter list
 * @param param_name Parameter name to extract
 * @param default_value Default value if parameter not found
 * @param comm MPI communicator for rank checking
 * @param verbosity Verbosity level
 * @param debug_prefix Optional prefix for debug messages
 * @return Extracted or default value
 */
template<typename T>
inline T extractAnalysisParam(
    const Teuchos::RCP<Teuchos::ParameterList> & settings,
    const std::string & param_name,
    const T & default_value,
    const Teuchos::RCP<const Teuchos::Comm<int>> & comm,
    int verbosity,
    const std::string & debug_prefix = "")
{
  T value = default_value;

  if (settings->isSublist("Analysis")) {
    auto& analysis_list = settings->sublist("Analysis");
    if (analysis_list.isParameter(param_name)) {
      value = analysis_list.get<T>(param_name);

      if (comm->getRank() == 0 && verbosity >= 8) {
        std::string prefix = debug_prefix.empty() ? "" : debug_prefix + ": ";
        std::cout << "DEBUG: " << prefix << "Using " << param_name
                  << " = " << value << std::endl;
      }
    }
  }

  return value;
}

/**
 * @brief Extract direct solver configuration (regularization + solver type)
 */
inline DirectSolverConfig extractDirectSolverConfig(
    const Teuchos::RCP<Teuchos::ParameterList> & settings,
    const Teuchos::RCP<const Teuchos::Comm<int>> & comm,
    int verbosity,
    const std::string & param_prefix = "mass matrix")
{
  DirectSolverConfig config;

  config.regularization = extractAnalysisParam<double>(
      settings, param_prefix + " regularization", 0.0, comm, verbosity);

  config.solver_type = extractAnalysisParam<std::string>(
      settings, param_prefix + " solver type", "KLU2", comm, verbosity);

  return config;
}

/**
 * @brief Extract iterative solver configuration
 */
inline IterativeSolverConfig extractIterativeSolverConfig(
    const Teuchos::RCP<Teuchos::ParameterList> & settings,
    const Teuchos::RCP<const Teuchos::Comm<int>> & comm,
    int verbosity,
    const std::string & param_prefix = "mass matrix")
{
  IterativeSolverConfig config;

  config.solver_type = extractAnalysisParam<std::string>(
      settings, param_prefix + " solver", "CG", comm, verbosity);

  config.prec_type = extractAnalysisParam<std::string>(
      settings, param_prefix + " preconditioner", "diagonal", comm, verbosity);

  config.max_iters = extractAnalysisParam<int>(
      settings, param_prefix + " max iterations", 100, comm, verbosity);

  config.tolerance = extractAnalysisParam<double>(
      settings, param_prefix + " tolerance", 1e-8, comm, verbosity);

  config.strict_convergence = extractAnalysisParam<bool>(
      settings, param_prefix + " strict convergence", false, comm, verbosity);

  // Extract preconditioner-specific settings
  if (settings->isSublist("Analysis")) {
    auto& analysis_list = settings->sublist("Analysis");

    if (analysis_list.isSublist("AMG settings")) {
      config.prec_params.set("AMG settings", analysis_list.sublist("AMG settings"));
    }

    if (analysis_list.isSublist("ILU settings")) {
      config.prec_params.set("ILU settings", analysis_list.sublist("ILU settings"));
    }
  }

  return config;
}

// ========================================================================
// MATRIX MANIPULATION HELPERS
// ========================================================================

/**
 * @brief Create a regularized copy of a matrix by adding epsilon*I to diagonal
 * @param source_matrix Original matrix
 * @param regularization Regularization parameter (epsilon)
 * @param comm MPI communicator for debug output
 * @param verbosity Verbosity level
 * @param matrix_name Name for debug messages (e.g., "mass matrix", "H(curl) matrix")
 * @return Regularized matrix (or original if regularization == 0)
 */
template<class Node>
inline Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, Node>>
createRegularizedMatrix(
    const Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, Node>> & source_matrix,
    double regularization,
    const Teuchos::RCP<const Teuchos::Comm<int>> & comm,
    int verbosity,
    const std::string & matrix_name = "matrix")
{
  if (regularization <= 0.0) {
    return source_matrix;
  }

  // Create a copy of the matrix for regularization
  auto regularized = Teuchos::rcp(
      new Tpetra::CrsMatrix<ScalarT, LO, GO, Node>(source_matrix->getCrsGraph()));
  regularized->resumeFill();

  // Copy the original matrix values
  typename Tpetra::CrsMatrix<ScalarT, LO, GO, Node>::local_inds_host_view_type indices;
  typename Tpetra::CrsMatrix<ScalarT, LO, GO, Node>::values_host_view_type values;

  for (LO localRow = 0; localRow < source_matrix->getLocalNumRows(); ++localRow) {
    source_matrix->getLocalRowView(localRow, indices, values);
    regularized->replaceLocalValues(localRow, indices, values);
  }

  // Add regularization to diagonal
  for (LO localRow = 0; localRow < regularized->getLocalNumRows(); ++localRow) {
    GO globalRow = regularized->getRowMap()->getGlobalElement(localRow);
    GO globalCol = globalRow;  // Diagonal entry
    Teuchos::Array<GO> cols(1, globalCol);
    Teuchos::Array<ScalarT> vals(1, regularization);
    regularized->sumIntoGlobalValues(globalRow, cols(), vals());
  }

  regularized->fillComplete();

  if (comm->getRank() == 0 && verbosity >= 8) {
    std::cout << "DEBUG: Applied regularization " << regularization
              << " to " << matrix_name << " diagonal" << std::endl;
  }

  return regularized;
}

/**
 * @brief Create and configure a Belos parameter list
 * @param max_iters Maximum iterations
 * @param tolerance Convergence tolerance
 * @param verbose Enable verbose output (default: warnings+errors only)
 * @return Configured parameter list
 */
inline Teuchos::ParameterList createBelosParamList(
    int max_iters,
    double tolerance,
    bool verbose = false)
{
  Teuchos::ParameterList belos_params;
  belos_params.set("Maximum Iterations", max_iters);
  belos_params.set("Convergence Tolerance", tolerance);

  if (verbose) {
    belos_params.set("Verbosity", Belos::Errors + Belos::Warnings + Belos::TimingDetails);
  } else {
    belos_params.set("Verbosity", Belos::Errors + Belos::Warnings);
  }

  belos_params.set("Output Style", Belos::Brief);

  return belos_params;
}

// ========================================================================
// SOLVER CREATION HELPERS
// ========================================================================

/**
 * @brief Create, configure, and factorize an Amesos2 direct solver
 * @param solver_type Solver type (KLU2, SuperLU, etc.)
 * @param matrix Matrix to factor
 * @param settings Main settings (for SuperLU parameters)
 * @param comm MPI communicator
 * @param verbosity Verbosity level
 * @return Factorized solver object
 */
template<class Node>
inline Teuchos::RCP<Amesos2::Solver<Tpetra::CrsMatrix<ScalarT, LO, GO, Node>,
                                     Tpetra::MultiVector<ScalarT, LO, GO, Node>>>
createDirectSolver(
    const std::string & solver_type,
    const Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, Node>> & matrix,
    const Teuchos::RCP<Teuchos::ParameterList> & settings,
    const Teuchos::RCP<const Teuchos::Comm<int>> & comm,
    int verbosity)
{
  using Matrix = Tpetra::CrsMatrix<ScalarT, LO, GO, Node>;
  using Vector = Tpetra::MultiVector<ScalarT, LO, GO, Node>;

  auto solver = Amesos2::create<Matrix, Vector>(solver_type, matrix);

  // Apply solver-specific parameters if SuperLU is selected
  if (solver_type == "SuperLU" && settings->isSublist("Analysis")) {
    auto& analysis_list = settings->sublist("Analysis");
    if (analysis_list.isSublist("SuperLU settings")) {
      Teuchos::ParameterList superlu_params = analysis_list.sublist("SuperLU settings");
      solver->setParameters(Teuchos::rcpFromRef(superlu_params));

      if (comm->getRank() == 0 && verbosity >= 8) {
        std::cout << "DEBUG: Applied SuperLU settings from parameter list" << std::endl;
      }
    }
  }

  solver->symbolicFactorization();
  solver->numericFactorization();

  return solver;
}

// ========================================================================
// HIGH-LEVEL OPERATOR BUILDERS
// ========================================================================

/**
 * @brief Build a sparse_direct inverse operator
 * @param matrix Matrix to invert
 * @param config Direct solver configuration
 * @param settings Main settings
 * @param comm MPI communicator
 * @param verbosity Verbosity level
 * @param matrix_name Name for debug messages
 * @return DirectSolveOperator wrapped in Tpetra::Operator interface
 */
template<class Node>
inline Teuchos::RCP<Tpetra::Operator<ScalarT, LO, GO, Node>>
buildSparseDirectOperator(
    const Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, Node>> & matrix,
    const DirectSolverConfig & config,
    const Teuchos::RCP<Teuchos::ParameterList> & settings,
    const Teuchos::RCP<const Teuchos::Comm<int>> & comm,
    int verbosity,
    const std::string & matrix_name = "matrix")
{
  // Apply regularization if requested
  auto matrix_to_factor = createRegularizedMatrix(
      matrix, config.regularization, comm, verbosity, matrix_name);

  // Create and factorize the direct solver
  auto solver = createDirectSolver(
      config.solver_type, matrix_to_factor, settings, comm, verbosity);

  // Wrap in operator interface
  return Teuchos::rcp(new block_prec::DirectSolveOperator<Node>(
      solver, matrix->getRowMap()));
}

/**
 * @brief Build a sparse_iterative inverse operator
 * @param matrix Matrix to invert
 * @param config Iterative solver configuration
 * @param comm MPI communicator
 * @param verbosity Verbosity level
 * @param matrix_name Name for debug messages
 * @return IterativeSolveOperator wrapped in Tpetra::Operator interface
 */
template<class Node>
inline Teuchos::RCP<Tpetra::Operator<ScalarT, LO, GO, Node>>
buildSparseIterativeOperator(
    const Teuchos::RCP<Tpetra::CrsMatrix<ScalarT, LO, GO, Node>> & matrix,
    const IterativeSolverConfig & config,
    const Teuchos::RCP<const Teuchos::Comm<int>> & comm,
    int verbosity,
    const std::string & matrix_name = "matrix")
{
  auto belos_params = createBelosParamList(
      config.max_iters, config.tolerance, verbosity >= 6);

  if (comm->getRank() == 0 && verbosity >= 8) {
    std::cout << "DEBUG: Creating IterativeSolveOperator for " << matrix_name
              << " with solver=" << config.solver_type
              << ", prec=" << config.prec_type
              << ", maxiters=" << config.max_iters
              << ", tol=" << config.tolerance << std::endl;
  }

  auto op = Teuchos::rcp(new block_prec::IterativeSolveOperator<Node>(
      matrix, config.solver_type, belos_params,
      config.prec_type, config.prec_params, config.strict_convergence));

  if (comm->getRank() == 0 && verbosity >= 8) {
    std::cout << "DEBUG: " << matrix_name
              << " IterativeSolveOperator created successfully" << std::endl;
  }

  return op;
}

} // namespace ParameterManager_detail

} // namespace MrHyDE

#endif
