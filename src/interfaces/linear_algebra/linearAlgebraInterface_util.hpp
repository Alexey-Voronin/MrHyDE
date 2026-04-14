/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#include "linearAlgebraInterface.hpp"
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBiCGStabSolMgr.hpp>
#include <BelosGCRODRSolMgr.hpp>
#include <BelosPCPGSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosPseudoBlockStochasticCGSolMgr.hpp>
#include <BelosPseudoBlockTFQMRSolMgr.hpp>
#include <BelosRCGSolMgr.hpp>
#include <BelosTFQMRSolMgr.hpp>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace MrHyDE;

// ========================================================================================

template<class Node>
bool LinearAlgebraInterface<Node>::getJacobianReuse(const size_t & set) {
  bool reuse = false;
  if (context[set]->reuse_matrix && context[set]->have_matrix) {
    reuse = true;
  }
  return reuse;
}

// ========================================================================================

template<class Node>
bool LinearAlgebraInterface<Node>::getPrevJacobianReuse(const size_t & step) {
  bool reuse = true;
  if (context_prev.size() == 0) {
    reuse = false;
  }
  else {
    for (size_t set=0; set<context_prev[step].size(); ++set) {
      if (!context_prev[step][set]->reuse_matrix || !context_prev[step][set]->have_matrix) {
        reuse = false;
      }
    }
  }
  return reuse;
}

// ========================================================================================

template<class Node>
bool LinearAlgebraInterface<Node>::getParamJacobianReuse() {
  bool reuse = false;
  if (context_param->reuse_matrix && context_param->have_matrix) {
    reuse = true;
  }
  return reuse;
}

// ========================================================================================

template<class Node>
bool LinearAlgebraInterface<Node>::getParamStateJacobianReuse(const size_t & set) {
  bool reuse = false;
  if (context_param_state[set]->reuse_matrix && context_param_state[set]->have_matrix) {
    reuse = true;
  }
  return reuse;
}

// ========================================================================================
// All iterative solvers use the same Belos list.  This would be easy to specialize.
// ========================================================================================

template<class Node>
Teuchos::RCP<Teuchos::ParameterList> LinearAlgebraInterface<Node>::getBelosParameterList(Teuchos::RCP<LinearSolverContext<Node> > & cntxt) {
  Teuchos::RCP<Teuchos::ParameterList> belosList = Teuchos::rcp(new Teuchos::ParameterList());
  belosList->set("Maximum Iterations",    maxLinearIters); // Maximum number of iterations allowed
  belosList->set("Num Blocks", maxLinearIters);
  belosList->set("Convergence Tolerance", linearTOL);    // Relative convergence tolerance requested
  if (cntxt->belos_type != "MINRES") {
    belosList->set("Estimate Condition Number", doCondEst); // Only implemented in Belos for Pseudo Block CG, based on AztecOO
  }
  if (verbosity >= 9) {
    belosList->set("Verbosity", Belos::Errors + Belos::Warnings + Belos::StatusTestDetails);
  }
  else {
    belosList->set("Verbosity", Belos::Errors);
  }
  if (verbosity > 8) {
    belosList->set("Output Frequency",10);
  }
  else {
    belosList->set("Output Frequency",0);
  }
  int numEqns = 1;
  if (disc->block_names.size() == 1) {
    numEqns = disc->physics->num_vars[0][0];
  }
  if (cntxt->belos_type != "MINRES") {
    belosList->set("number of equations", numEqns);
  }
  
  belosList->set("Output Style", Belos::Brief);
  belosList->set("Implicit Residual Scaling", belos_residual_scaling);
  
  if (cntxt->belos_sublist.name() != "empty") {
    //Teuchos::ParameterList inputParams = settings->sublist("Solver").sublist(belosSublist);
    belosList->setParameters(cntxt->belos_sublist);
  }
  
  return belosList;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetAllJacobian() {
  this->resetJacobian();
  this->resetL2Jacobian();
  this->resetBndryL2Jacobian();
  this->resetParamJacobian();
  this->resetParamStateJacobian();
  this->resetPrevJacobian();
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetJacobian() {
  for (size_t set=0; set<context.size(); ++set) {
    context[set]->reset();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetL2Jacobian() {
  for (size_t set=0; set<context_L2.size(); ++set) {
    context_L2[set]->reset();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetBndryL2Jacobian() {
  for (size_t set=0; set<context_BndryL2.size(); ++set) {
    context_BndryL2[set]->reset();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetParamJacobian() {
  context_param->reset();
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetParamStateJacobian() {
  for (size_t set=0; set<context_param_state.size(); ++set) {
    context_param_state[set]->reset();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::resetPrevJacobian() {
  for (size_t step=0; step<context_prev.size(); ++step) {
    for (size_t set=0; set<context_prev[step].size(); ++set) {
      context_prev[step][set]->reset();
    }
  }
}


// ========================================================================================
// ========================================================================================

template<class Node>
size_t LinearAlgebraInterface<Node>::getLocalNumElements(const size_t & set) {
  size_t numElem = 0;
  if (have_overlapped) {
    numElem = overlapped_map[set]->getLocalNumElements();
  }
  else {
    numElem = owned_map[set]->getLocalNumElements();
  }
  return numElem;
}


// ========================================================================================
// ========================================================================================

template<class Node>
size_t LinearAlgebraInterface<Node>::getLocalNumParamElements() {
  size_t numElem = 0;
  if (have_overlapped) {
    numElem = param_overlapped_map->getLocalNumElements();
  }
  else {
    numElem = param_owned_map->getLocalNumElements();
  }
  return numElem;
}


// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsGraph<LO,GO,Node> > LinearAlgebraInterface<Node>::getNewOverlappedGraph(const size_t & set, vector<size_t> & maxEntriesPerRow) {
  Teuchos::RCP<LA_CrsGraph> newgraph;
  if (have_overlapped) {
    newgraph = Teuchos::rcp(new LA_CrsGraph(overlapped_map[set], maxEntriesPerRow));
  }
  else {
    newgraph = Teuchos::rcp(new LA_CrsGraph(owned_map[set], maxEntriesPerRow));
  }
  return newgraph;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::CrsGraph<LO,GO,Node> > LinearAlgebraInterface<Node>::getNewParamOverlappedGraph(vector<size_t> & maxEntriesPerRow) {
  Teuchos::RCP<LA_CrsGraph> newgraph;
  if (have_overlapped) {
    newgraph = Teuchos::rcp(new LA_CrsGraph(param_overlapped_map, maxEntriesPerRow));
  }
  else {
    newgraph = Teuchos::rcp(new LA_CrsGraph(param_owned_map, maxEntriesPerRow));
  }
  return newgraph;
}

// ========================================================================================
// ========================================================================================

template<class Node>
GO LinearAlgebraInterface<Node>::getGlobalElement(const size_t & set, const LO & lid) {
  GO gid = 0;
  if (have_overlapped) {
    gid = overlapped_map[set]->getGlobalElement(lid);
  }
  else {
    gid = owned_map[set]->getGlobalElement(lid);
  }
  return gid;
}

// ========================================================================================
// ========================================================================================

template<class Node>
GO LinearAlgebraInterface<Node>::getGlobalParamElement(const LO & lid) {
  GO gid = 0;
  if (have_overlapped) {
    gid = param_overlapped_map->getGlobalElement(lid);
  }
  else {
    gid = param_owned_map->getGlobalElement(lid);
  }
  return gid;
}

// ========================================================================================
// ========================================================================================

template<class Node>
bool LinearAlgebraInterface<Node>::getHaveOverlapped() {
  return have_overlapped;
}

// ========================================================================================
// ========================================================================================

template<class Node>
LO LinearAlgebraInterface<Node>::getOverlappedLID(const size_t & set, const GO & gid) {
  LO lid = 0;
  if (have_overlapped) {
    lid = overlapped_map[set]->getLocalElement(gid);
  }
  else {
    lid = owned_map[set]->getLocalElement(gid);
  }
  return lid;
}

// ========================================================================================
// ========================================================================================

template<class Node>
LO LinearAlgebraInterface<Node>::getOwnedLID(const size_t & set, const GO & gid) {
  return owned_map[set]->getLocalElement(gid);
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::writeToFile(matrix_RCP &J, vector_RCP &r, vector_RCP &soln,
                                               const std::string &jac_filename,
                                               const std::string &res_filename,
                                               const std::string &sol_filename) {
  Teuchos::TimeMonitor localtimer(*writefiletimer);
  
  if (do_dump_jacobian) {
    Tpetra::MatrixMarket::Writer<LA_CrsMatrix>::writeSparseFile(jac_filename,*J);
  }
  if (do_dump_residual) {
    Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(res_filename,*r);
  }
  if (do_dump_solution) {
    Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(sol_filename,*soln);
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::writeStateToFile(vector<vector_RCP> & soln,
                                                    const std::string & filebase, const int & stepnum) {
  Teuchos::TimeMonitor localtimer(*writefiletimer);
  
  for (size_t set=0; set<soln.size(); ++set) {
    std::stringstream ss;
    ss << filebase << "." << set << "." << stepnum << ".mm";
    
    Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(ss.str(),*(soln[set]));
  }
}
// ========================================================================================
// ========================================================================================

template<class Node>
void LinearAlgebraInterface<Node>::writeVectorToFile(ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > & vec, string & filename) {
  Teuchos::TimeMonitor localtimer(*writefiletimer);
  
  Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(filename,vec->getVector());
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::readParameterVectorFromFile(const std::string & filename) {
  Teuchos::TimeMonitor localtimer(*readfiletimer);
  
  vector_RCP vec = Tpetra::MatrixMarket::Reader<LA_MultiVector>::readDenseFile(filename, comm, param_owned_map);
  return vec;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> > LinearAlgebraInterface<Node>::readStateVectorFromFile(const std::string & filename, const size_t & set) {
  Teuchos::TimeMonitor localtimer(*readfiletimer);

  vector_RCP vec = Tpetra::MatrixMarket::Reader<LA_MultiVector>::readDenseFile(filename, comm, owned_map[set]);
  return vec;
}

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<Tpetra::MultiVector<ScalarT,LO,GO,Node> >
LinearAlgebraInterface<Node>::extractLumpedDiagonal(const Teuchos::RCP<LA_CrsMatrix> & matrix,
                                                     const bool use_lumped) {

  auto diag = Teuchos::rcp(new LA_MultiVector(matrix->getRowMap(), 1));
  diag->putScalar(0.0);
  auto diag_data = diag->getLocalViewHost(Tpetra::Access::ReadWrite);
  const auto numLocalRows = matrix->getLocalNumRows();

  if (use_lumped) {
    // Lumped diagonal: sum all entries in each row
    for (size_t i = 0; i < numLocalRows; ++i) {
      typename LA_CrsMatrix::local_inds_host_view_type indices;
      typename LA_CrsMatrix::values_host_view_type values;
      matrix->getLocalRowView(i, indices, values);

      ScalarT rowSum = 0.0;
      for (size_t j = 0; j < indices.extent(0); ++j) {
        rowSum += values[j];
      }
      diag_data(i, 0) = rowSum;
    }
  }
  else {
    // diag entries only
    for (size_t i = 0; i < numLocalRows; ++i) {
      typename LA_CrsMatrix::local_inds_host_view_type indices;
      typename LA_CrsMatrix::values_host_view_type values;
      matrix->getLocalRowView(i, indices, values);

      ScalarT diagVal = 0.0;
      for (size_t j = 0; j < indices.extent(0); ++j) {
        if (indices[j] == static_cast<LO>(i)) {
          diagVal = values[j];
          break;
        }
      }
      diag_data(i, 0) = diagVal;
    }
  }

  return diag;
}

template<class Node>
typename LinearAlgebraInterface<Node>::MatrixScaleStats
LinearAlgebraInterface<Node>::printMatrixScaleDiagnostics(const Teuchos::RCP<LA_CrsMatrix> & matrix,
                                                          const std::string & label) const {
  MatrixScaleStats stats;
  if (matrix.is_null()) {
    return stats;
  }

  using LA_Vector = Tpetra::Vector<ScalarT, LO, GO, Node>;
  using Teuchos::as;
  const size_t nlocal = matrix->getLocalNumRows();
  const size_t nglobal = matrix->getGlobalNumRows();

  auto ones = Teuchos::rcp(new LA_Vector(matrix->getDomainMap()));
  auto rowSums = Teuchos::rcp(new LA_Vector(matrix->getRowMap()));
  ones->putScalar(static_cast<ScalarT>(1.0));
  matrix->apply(*ones, *rowSums);
  auto row_view = rowSums->getLocalViewHost(Tpetra::Access::ReadOnly);

  auto diag = Teuchos::rcp(new LA_Vector(matrix->getRowMap()));
  matrix->getLocalDiagCopy(*diag);
  auto diag_view = diag->getLocalViewHost(Tpetra::Access::ReadOnly);

  double local_row_sum_min = std::numeric_limits<double>::infinity();
  double local_row_sum_max = -std::numeric_limits<double>::infinity();
  double local_row_sum_sum = 0.0;
  double local_abs_row_sum_min = std::numeric_limits<double>::infinity();
  double local_abs_row_sum_max = -std::numeric_limits<double>::infinity();
  double local_abs_row_sum_sum = 0.0;
  double local_diag_min = std::numeric_limits<double>::infinity();
  double local_diag_max = -std::numeric_limits<double>::infinity();
  double local_diag_sum = 0.0;
  long long local_nonpos_diag = 0;
  double local_max_abs_entry = 0.0;

  for (size_t i = 0; i < nlocal; ++i) {
    const double rs = as<double>(row_view(i, 0));
    local_row_sum_min = std::min(local_row_sum_min, rs);
    local_row_sum_max = std::max(local_row_sum_max, rs);
    local_row_sum_sum += rs;

    typename LA_CrsMatrix::local_inds_host_view_type idx;
    typename LA_CrsMatrix::values_host_view_type vals;
    matrix->getLocalRowView(i, idx, vals);
    double abs_rs = 0.0;
    for (size_t j = 0; j < idx.extent(0); ++j) {
      const double av = std::abs(as<double>(vals[j]));
      abs_rs += av;
      local_max_abs_entry = std::max(local_max_abs_entry, av);
    }
    local_abs_row_sum_min = std::min(local_abs_row_sum_min, abs_rs);
    local_abs_row_sum_max = std::max(local_abs_row_sum_max, abs_rs);
    local_abs_row_sum_sum += abs_rs;

    const double dv = as<double>(diag_view(i, 0));
    local_diag_min = std::min(local_diag_min, dv);
    local_diag_max = std::max(local_diag_max, dv);
    local_diag_sum += dv;
    if (dv <= 0.0) {
      local_nonpos_diag += 1;
    }
  }

  if (nlocal == 0) {
    local_row_sum_min = 0.0;
    local_row_sum_max = 0.0;
    local_abs_row_sum_min = 0.0;
    local_abs_row_sum_max = 0.0;
    local_diag_min = 0.0;
    local_diag_max = 0.0;
  }

  double global_row_sum_min = 0.0, global_row_sum_max = 0.0, global_row_sum_sum = 0.0;
  double global_abs_row_sum_min = 0.0, global_abs_row_sum_max = 0.0, global_abs_row_sum_sum = 0.0;
  double global_diag_min = 0.0, global_diag_max = 0.0, global_diag_sum = 0.0;
  double global_max_abs_entry = 0.0;
  long long global_nonpos_diag = 0;
  long long global_rows_ll = static_cast<long long>(nglobal);

  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_row_sum_min, &global_row_sum_min);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_row_sum_max, &global_row_sum_max);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_row_sum_sum, &global_row_sum_sum);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_abs_row_sum_min, &global_abs_row_sum_min);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_abs_row_sum_max, &global_abs_row_sum_max);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_abs_row_sum_sum, &global_abs_row_sum_sum);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &local_diag_min, &global_diag_min);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_diag_max, &global_diag_max);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_diag_sum, &global_diag_sum);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_SUM, 1, &local_nonpos_diag, &global_nonpos_diag);
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &local_max_abs_entry, &global_max_abs_entry);

  stats.valid = true;
  stats.global_rows = static_cast<size_t>(global_rows_ll);
  if (nglobal > 0) {
    const double inv_rows = 1.0 / static_cast<double>(nglobal);
    stats.row_sum_min = global_row_sum_min;
    stats.row_sum_max = global_row_sum_max;
    stats.row_sum_mean = global_row_sum_sum * inv_rows;
    stats.abs_row_sum_min = global_abs_row_sum_min;
    stats.abs_row_sum_max = global_abs_row_sum_max;
    stats.abs_row_sum_mean = global_abs_row_sum_sum * inv_rows;
    stats.diag_min = global_diag_min;
    stats.diag_max = global_diag_max;
    stats.diag_mean = global_diag_sum * inv_rows;
  }
  stats.nonpos_diag_count = static_cast<size_t>(global_nonpos_diag);
  stats.max_abs_entry = global_max_abs_entry;

  if (comm->getRank() == 0) {
    const double row_ratio = (std::abs(stats.row_sum_min) > 0.0) ? stats.row_sum_max / std::abs(stats.row_sum_min) : 0.0;
    const double abs_row_ratio = (stats.abs_row_sum_min > 0.0) ? stats.abs_row_sum_max / stats.abs_row_sum_min : 0.0;
    std::cout << "[MetricOpStats] label=" << label
              << " rows=" << stats.global_rows
              << " row_sum(min,max,mean)=(" << stats.row_sum_min << ","
              << stats.row_sum_max << "," << stats.row_sum_mean << ")"
              << " row_sum_ratio_max_over_absmin=" << row_ratio
              << " abs_row_sum(min,max,mean)=(" << stats.abs_row_sum_min << ","
              << stats.abs_row_sum_max << "," << stats.abs_row_sum_mean << ")"
              << " abs_row_sum_ratio_max_over_min=" << abs_row_ratio
              << " diag(min,max,mean)=(" << stats.diag_min << ","
              << stats.diag_max << "," << stats.diag_mean << ")"
              << " nonpos_diag=" << stats.nonpos_diag_count
              << " max_abs_entry=" << stats.max_abs_entry
              << std::endl;
  }

  return stats;
}
