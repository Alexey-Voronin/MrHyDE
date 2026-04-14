/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#include <algorithm>
#include <cmath>
#include <limits>

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setInitialParams() {
  
  debugger->print("**** Starting ParameterManager::setInitialParams ...");
  
  //auto Psol = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
  //auto Psol_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
  //Psol->putScalar(0.0);
  //Psol_over->putScalar(0.0); // TMW: why is this hard-coded???
  
  int numsols = 1;
  if (have_dynamic_discretized) {
    numsols = numTimeSteps;
  }
  
  for (int i=0; i<numsols; ++i) {
    vector_RCP dyninit = Teuchos::rcp(new LA_MultiVector(param_owned_map,1));
    vector_RCP dyninit_over = Teuchos::rcp(new LA_MultiVector(param_overlapped_map,1));
    dyninit->putScalar(0.0); // TMW: why is this hard-coded???
    dyninit_over->putScalar(0.0); // TMW: why is this hard-coded???
    discretized_params.push_back(dyninit);
    discretized_params_over.push_back(dyninit_over);
  }
  
  /*
  if (scalarInitialData) {
    // This will be done on the host for now
    auto initial_kv = initial->getLocalView<HostDevice>();
    for (size_t block=0; block<assembler->groups.size(); block++) {
      Kokkos::View<int**,AssemblyDevice> offsets = assembler->wkset[block]->offsets;
      auto host_offsets = Kokkos::create_mirror_view(offsets);
      Kokkos::deep_copy(host_offsets,offsets);
      for (size_t group=0; group<assembler->groups[block].size(); group++) {
        Kokkos::View<LO**,HostDevice> LIDs = assembler->groups[block][group]->LIDs_host;
        Kokkos::View<LO*,HostDevice> numDOF = assembler->groups[block][group]->group_data->numDOF_host;
        //parallel_for("solver initial scalar",RangePolicy<HostExec>(0,LIDs.extent(0)), MRHYDE_LAMBDA (const int e ) {
        for (int e=0; e<LIDs.extent(0); e++) {
          for (size_t n=0; n<numDOF.extent(0); n++) {
            for (size_t i=0; i<numDOF(n); i++ ) {
              initial_kv(LIDs(e,host_offsets(n,i)),0) = scalarInitialValues[block][n];
            }
          }
        }
      }
    }
  }
  else {
  
    initial->putScalar(0.0);
    
    vector_RCP glinitial = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1));
    
    if (initial_type == "L2-projection") {
      
      // Compute the L2 projection of the initial data into the discrete space
      vector_RCP rhs = Teuchos::rcp(new LA_MultiVector(LA_overlapped_map,1)); // reset residual
      matrix_RCP mass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_overlapped_graph));//Tpetra::createCrsMatrix<ScalarT>(LA_overlapped_map); // reset Jacobian
      vector_RCP glrhs = Teuchos::rcp(new LA_MultiVector(LA_owned_map,1)); // reset residual
      matrix_RCP glmass = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,HostNode>(LA_owned_map, maxEntries));//Tpetra::createCrsMatrix<ScalarT>(LA_owned_map); // reset Jacobian
      assembler->setInitial(rhs, mass, useadjoint);
      
      glmass->setAllToScalar(0.0);
      glmass->doExport(*mass, *exporter, Tpetra::ADD);
      
      glrhs->putScalar(0.0);
      glrhs->doExport(*rhs, *exporter, Tpetra::ADD);
      
      glmass->fillComplete();
      
      this->linearSolver(glmass, glrhs, glinitial);
      have_preconditioner = false; // resetting this because mass matrix may not have same connectivity as Jacobians
      initial->doImport(*glinitial, *importer, Tpetra::ADD);
      
    }
    else if (initial_type == "interpolation") {
      
      assembler->setInitial(initial, useadjoint);
      
    }
  }
  */
  
  debugger->print("**** Finished ParameterManager::setInitialParams ...");
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setParam(const vector<ScalarT> & newparams, const std::string & name) {
  size_t pprog = 0;
  // perhaps add a check that the size of newparams equals the number of parameters of the
  // requested type
  
  int index = 0;
  if (have_dynamic_scalar) {
    index = dynamic_timeindex;
  }
  
  if (paramvals.size() > index) {
    for (size_t i=0; i<paramvals[index].size(); i++) {
      if (paramnames[i] == name) {
        for (size_t j=0; j<paramvals[index][i].size(); j++) {
          if (Comm->getRank() == 0 && verbosity > 0) {
            cout << "Updated Params: " << paramvals[index][i][j] << " (old value)   " << newparams[pprog] << " (new value)" << endl;
          }
          paramvals[index][i][j] = newparams[pprog];
          pprog++;
        }
      }
    }
  }
  
}

// ========================================================================================
// ========================================================================================

template<class Node>
bool ParameterManager<Node>::isParameter(const string & name) {
  bool isparam = false;
  if (paramvals.size() > 0) {
    for (size_t i=0; i<paramvals[0].size(); i++) { // just first is fine
      if (paramnames[i] == name) {
        isparam = true;
      }
    }
  }
  return isparam;
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::stashParams(){
  if (batchID == 0 && Comm->getRank() == 0){
    string outname = "param_stash.dat";
    std::ofstream respOUT(outname);
    respOUT.precision(16);
    for (size_t i=0; i<paramvals.size(); i++) {
      for (size_t k=0; k<paramvals[i].size(); k++) {
        if (paramtypes[k] == 1) {
          for (size_t j=0; j<paramvals[i][k].size(); j++) {
            respOUT << paramvals[i][k][j] << endl;
          }
        }
      }
    }
    respOUT.close();
  }
}

// ========================================================================================
// ========================================================================================

template<class Node>
void ParameterManager<Node>::setParamMass(Teuchos::RCP<LA_MultiVector> diag,
                                          matrix_RCP mass) {

  buildMassOperators(diag, mass);

}

template<class Node>
void ParameterManager<Node>::buildMassOperators(Teuchos::RCP<LA_MultiVector> diagParamMass,
                                                 matrix_RCP paramMass) {

  using LA_Vector = Tpetra::Vector<ScalarT, LO, GO, SolverNode>;

  std::string mass_type = "none";
  double diag_floor = 1.0e-14;
  std::string safety_mode = "clamp";

  if (settings->isSublist("Analysis")) {
    auto& analysis_list = settings->sublist("Analysis");

    // Primary name (new): "parameter gradient preconditioner type"
    // Fallback (backward compatibility): "parameter mass matrix type"
    if (analysis_list.isParameter("parameter gradient preconditioner type")) {
      mass_type = analysis_list.get<std::string>("parameter gradient preconditioner type");
    } else if (analysis_list.isParameter("parameter mass matrix type")) {
      mass_type = analysis_list.get<std::string>("parameter mass matrix type");
      std::cout << "WARNING: 'parameter mass matrix type' is deprecated. "
                << "Use 'parameter gradient preconditioner type' instead.\n"
                << "The mass matrix is used for GRADIENT PRECONDITIONING ONLY, "
                << "not for inner products in dot()." << std::endl;
    }

    // Parse diagonal safety configuration
    diag_floor = analysis_list.get<double>("diagonal floor", 1.0e-14);
    safety_mode = analysis_list.get<std::string>("diagonal safety mode", "clamp");

    if (safety_mode != "clamp" && safety_mode != "error") {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "diagonal safety mode must be 'clamp' or 'error', got: " + safety_mode);
    }
  }

  massForwardOp = Teuchos::null;
  massInvOperator = Teuchos::null;

  if (mass_type == "none" || mass_type == "default") {
    return;
  }

  if (mass_type == "lumped") {
    if (diagParamMass.is_null()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "ParameterManager::buildMassOperators: diagParamMass is null for mass_type='"
        + mass_type + "'. Call setParamMass() before buildMassOperators().");
    }

    auto diag_vec = diagParamMass->getVectorNonConst(0);

    // Apply diagonal safety before reciprocal
    {
      auto diag_view = diag_vec->getLocalViewHost(Tpetra::Access::ReadWrite);
      bool has_small = false;

      for (size_t i = 0; i < diag_view.extent(0); ++i) {
        double val = diag_view(i, 0);
        if (std::abs(val) < diag_floor) {
          has_small = true;
          if (safety_mode == "error") {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
              "Diagonal entry " + std::to_string(i) + " = " +
              std::to_string(val) + " < floor " + std::to_string(diag_floor));
          } else { // clamp mode
            diag_view(i, 0) = (val >= 0) ? diag_floor : -diag_floor;
          }
        }
      }

      if (has_small && Comm->getRank() == 0 && verbosity >= 4) {
        std::cout << "WARNING: Clamped small diagonal entries to +/- "
                  << diag_floor << std::endl;
      }
    }

    // Now safe to compute reciprocal
    auto inv_diag_vec = Teuchos::rcp(new LA_Vector(diagParamMass->getMap()));
    inv_diag_vec->reciprocal(*diag_vec);

    massForwardOp = Teuchos::rcp(new block_prec::DiagonalMultiplyOperator<SolverNode>(diag_vec));
    massInvOperator = Teuchos::rcp(new block_prec::DiagonalInverseOperator<SolverNode>(inv_diag_vec));

    return;
  }

  if (mass_type == "sparse_direct") {
    if (paramMass.is_null()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "ParameterManager::buildMassOperators: paramMass is null for mass_type='sparse_direct'. "
        "Call setParamMass() before buildMassOperators().");
    }

    massForwardOp = paramMass;

    // Extract configuration and build operator using helpers
    auto config = ParameterManager_detail::extractDirectSolverConfig(
        settings, Comm, verbosity, "mass matrix");

    massInvOperator = ParameterManager_detail::buildSparseDirectOperator<SolverNode>(
        paramMass, config, settings, Comm, verbosity, "mass matrix");

    return;
  }

  if (mass_type == "sparse_iterative") {
    if (paramMass.is_null()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "ParameterManager::buildMassOperators: paramMass is null for mass_type='sparse_iterative'. "
        "Call setParamMass() before buildMassOperators().");
    }

    massForwardOp = paramMass;

    // Extract configuration and build operator using helpers
    auto config = ParameterManager_detail::extractIterativeSolverConfig(
        settings, Comm, verbosity, "mass matrix");

    massInvOperator = ParameterManager_detail::buildSparseIterativeOperator<SolverNode>(
        paramMass, config, Comm, verbosity, "mass matrix");

    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
    "ParameterManager::buildMassOperators: Unknown mass_type='" + mass_type + "'. "
    "Valid options: none, lumped, sparse_direct, sparse_iterative.");
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Check if mass inverse operator is available for Sobolev gradient preconditioning
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
bool ParameterManager<Node>::hasMassInverseOperator() const {
  return !massInvOperator.is_null();
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Apply mass inverse operator for Sobolev gradient preconditioning
// This converts dual-space gradients (functionals) to primal-space Sobolev gradients
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void ParameterManager<Node>::applyMassInverse(const vector_RCP & in, vector_RCP & out) const {
  TEUCHOS_TEST_FOR_EXCEPTION(massInvOperator.is_null(), std::runtime_error,
    "ParameterManager::applyMassInverse: massInvOperator is null. "
    "Enable with 'parameter gradient preconditioner type' in Analysis settings.");

  massInvOperator->apply(*in, *out);
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Build H(curl) operators: (M + K) and (M + K)^{-1}
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void ParameterManager<Node>::buildHcurlOperators(matrix_RCP paramMass, matrix_RCP paramStiffness) {

  using LA_CrsMatrix = Tpetra::CrsMatrix<ScalarT, LO, GO, SolverNode>;
  using LA_Vector = Tpetra::Vector<ScalarT, LO, GO, SolverNode>;

  // Parse diagonal safety configuration from settings
  double diag_floor = 1.0e-14;
  std::string safety_mode = "clamp";

  if (settings->isSublist("Analysis")) {
    auto& analysis_list = settings->sublist("Analysis");
    diag_floor = analysis_list.get<double>("diagonal floor", 1.0e-14);
    safety_mode = analysis_list.get<std::string>("diagonal safety mode", "clamp");
  }

  // Check parameters
  if (paramMass.is_null() || paramStiffness.is_null()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "ParameterManager::buildHcurlOperators: Both mass and stiffness matrices are required");
  }

  // Build weighted (alpha1*M + alpha2*K) matrix for H(curl) Riesz map.
  // Weights default to 1.0 but can be overridden from Analysis.
  ScalarT alpha1 = 1.0;
  ScalarT alpha2 = 1.0;

  // Try to read weights from regularization sublists
  if (settings->isSublist("Postprocess")) {
    auto& postproc_list = settings->sublist("Postprocess");
    if (postproc_list.isSublist("Objective functions")) {
      auto& obj_list = postproc_list.sublist("Objective functions");
      if (obj_list.isSublist("RegObj")) {
        auto& regobj_list = obj_list.sublist("RegObj");
        if (regobj_list.isSublist("Regularization functions")) {
          auto& reg_list = regobj_list.sublist("Regularization functions");
          if (reg_list.isSublist("l2reg")) {
            alpha1 = reg_list.sublist("l2reg").get<ScalarT>("weight", 1.0);
          }
          if (reg_list.isSublist("curlreg")) {
            alpha2 = reg_list.sublist("curlreg").get<ScalarT>("weight", 1.0);
          }
        }
      }
    }
  }

  // Allow direct override from Analysis section
  if (settings->isSublist("Analysis")) {
    auto& analysis_list = settings->sublist("Analysis");
    alpha1 = analysis_list.get<ScalarT>("hcurl alpha1", alpha1);
    alpha2 = analysis_list.get<ScalarT>("hcurl alpha2", alpha2);
  }

  // Trick A: when alpha1 == alpha2, factor (M + K) with unit weights instead of
  // (alpha*M + alpha*K). KLU roundoff then scales with ||M+K|| ~ O(1) instead of
  // O(alpha), eliminating the Riesz metric noise floor that plateaus L-BFGS when
  // alpha is large (e.g. 1e20). The assumption alpha1 == alpha2 means the factor
  // alpha pulls out as a pure scalar: (alpha*(M+K))^{-1} = (1/alpha)*(M+K)^{-1}.
  // We cache alpha in hcurlMetricScale_ and compensate at apply time.
  ScalarT build_a1 = alpha1;
  ScalarT build_a2 = alpha2;
  hcurlMetricScale_ = static_cast<ScalarT>(1);
  if (alpha1 == alpha2 && alpha1 != static_cast<ScalarT>(1)) {
    build_a1 = static_cast<ScalarT>(1);
    build_a2 = static_cast<ScalarT>(1);
    hcurlMetricScale_ = alpha1;
  }

  if (Comm->getRank() == 0 && verbosity >= 5) {
    std::cout << "Building H(curl) operator: (" << alpha1 << "*M + " << alpha2 << "*K)";
    if (hcurlMetricScale_ != static_cast<ScalarT>(1)) {
      std::cout << " [factored as (M+K); scale " << hcurlMetricScale_
                << " applied at apply() time]";
    }
    std::cout << std::endl;
  }

  // Create a new empty matrix with the same graph structure
  auto hcurlMatrix = Teuchos::rcp(new LA_CrsMatrix(paramMass->getRowMap(),
                                                     paramMass->getGlobalMaxNumRowEntries()));

  // Use Tpetra's matrix addition: C = build_a1*M + build_a2*K
  // Note: both paramMass and paramStiffness must be fillComplete'd
  Tpetra::MatrixMatrix::Add(*paramMass, false, build_a1, *paramStiffness, false, build_a2, hcurlMatrix);

  std::string mass_type = "none";
  if (settings->isSublist("Analysis")) {
    auto& analysis_list = settings->sublist("Analysis");
    if (analysis_list.isParameter("parameter gradient preconditioner type")) {
      mass_type = analysis_list.get<std::string>("parameter gradient preconditioner type");
    }
  }

  // MatrixMatrix::Add does not call fillComplete, so we need to do it
  hcurlMatrix->fillComplete(paramMass->getDomainMap(), paramMass->getRangeMap());
  hcurlForwardOp = hcurlMatrix;

  {
    auto ones = Teuchos::rcp(new LA_Vector(hcurlMatrix->getDomainMap()));
    auto rowSums = Teuchos::rcp(new LA_Vector(hcurlMatrix->getRowMap()));
    ones->putScalar(static_cast<ScalarT>(1.0));
    hcurlMatrix->apply(*ones, *rowSums);
    auto row_view = rowSums->getLocalViewHost(Tpetra::Access::ReadOnly);

    auto diag = Teuchos::rcp(new LA_Vector(hcurlMatrix->getRowMap()));
    hcurlMatrix->getLocalDiagCopy(*diag);
    auto diag_view = diag->getLocalViewHost(Tpetra::Access::ReadOnly);

    const size_t nlocal = hcurlMatrix->getLocalNumRows();
    const size_t nglobal = hcurlMatrix->getGlobalNumRows();
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
      const double rs = static_cast<double>(row_view(i, 0));
      local_row_sum_min = std::min(local_row_sum_min, rs);
      local_row_sum_max = std::max(local_row_sum_max, rs);
      local_row_sum_sum += rs;

      typename LA_CrsMatrix::local_inds_host_view_type idx;
      typename LA_CrsMatrix::values_host_view_type vals;
      hcurlMatrix->getLocalRowView(i, idx, vals);
      double abs_rs = 0.0;
      for (size_t j = 0; j < idx.extent(0); ++j) {
        const double av = std::abs(static_cast<double>(vals[j]));
        abs_rs += av;
        local_max_abs_entry = std::max(local_max_abs_entry, av);
      }
      local_abs_row_sum_min = std::min(local_abs_row_sum_min, abs_rs);
      local_abs_row_sum_max = std::max(local_abs_row_sum_max, abs_rs);
      local_abs_row_sum_sum += abs_rs;

      const double dv = static_cast<double>(diag_view(i, 0));
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
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MIN, 1, &local_row_sum_min, &global_row_sum_min);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &local_row_sum_max, &global_row_sum_max);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &local_row_sum_sum, &global_row_sum_sum);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MIN, 1, &local_abs_row_sum_min, &global_abs_row_sum_min);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &local_abs_row_sum_max, &global_abs_row_sum_max);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &local_abs_row_sum_sum, &global_abs_row_sum_sum);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MIN, 1, &local_diag_min, &global_diag_min);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &local_diag_max, &global_diag_max);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &local_diag_sum, &global_diag_sum);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_SUM, 1, &local_nonpos_diag, &global_nonpos_diag);
    Teuchos::reduceAll(*Comm, Teuchos::REDUCE_MAX, 1, &local_max_abs_entry, &global_max_abs_entry);

    if (Comm->getRank() == 0) {
      const double inv_rows = (nglobal > 0) ? (1.0 / static_cast<double>(nglobal)) : 0.0;
      const double row_mean = global_row_sum_sum * inv_rows;
      const double abs_row_mean = global_abs_row_sum_sum * inv_rows;
      const double diag_mean = global_diag_sum * inv_rows;
      const double row_ratio = (std::abs(global_row_sum_min) > 0.0) ? global_row_sum_max / std::abs(global_row_sum_min) : 0.0;
      const double abs_row_ratio = (global_abs_row_sum_min > 0.0) ? global_abs_row_sum_max / global_abs_row_sum_min : 0.0;
      std::cout << "[MetricOpStats] label=Hcurl_H rows=" << nglobal
                << " row_sum(min,max,mean)=(" << global_row_sum_min << ","
                << global_row_sum_max << "," << row_mean << ")"
                << " row_sum_ratio_max_over_absmin=" << row_ratio
                << " abs_row_sum(min,max,mean)=(" << global_abs_row_sum_min << ","
                << global_abs_row_sum_max << "," << abs_row_mean << ")"
                << " abs_row_sum_ratio_max_over_min=" << abs_row_ratio
                << " diag(min,max,mean)=(" << global_diag_min << ","
                << global_diag_max << "," << diag_mean << ")"
                << " nonpos_diag=" << global_nonpos_diag
                << " max_abs_entry=" << global_max_abs_entry
                << " alpha1=" << alpha1
                << " alpha2=" << alpha2
                << " hcurlMetricScale=" << hcurlMetricScale_
                << " solver_type=" << mass_type
                << std::endl;
    }
  }

  if (mass_type == "sparse_iterative") {
    auto config = ParameterManager_detail::extractIterativeSolverConfig(
        settings, Comm, verbosity, "mass matrix");

    hcurlInvOperator = ParameterManager_detail::buildSparseIterativeOperator<SolverNode>(
        hcurlMatrix, config, Comm, verbosity, "H(curl) matrix");
  }
  else if (mass_type == "sparse_direct") {
    auto config = ParameterManager_detail::extractDirectSolverConfig(
        settings, Comm, verbosity, "mass matrix");

    hcurlInvOperator = ParameterManager_detail::buildSparseDirectOperator<SolverNode>(
        hcurlMatrix, config, settings, Comm, verbosity, "H(curl) matrix");
  }
  else if (mass_type == "lumped") {
    auto lumped_diag_vec = Teuchos::rcp(new LA_Vector(hcurlMatrix->getRowMap()));
    auto ones = Teuchos::rcp(new LA_Vector(hcurlMatrix->getDomainMap()));
    ones->putScalar(1.0);

    // row sums
    hcurlMatrix->apply(*ones, *lumped_diag_vec);

    // check for zeros before inverting..
    {
      auto diag_view = lumped_diag_vec->getLocalViewHost(Tpetra::Access::ReadWrite);
      bool has_small = false;

      for (size_t i = 0; i < diag_view.extent(0); ++i) {
        double val = diag_view(i, 0);
        if (std::abs(val) < diag_floor) {
          has_small = true;
          if (safety_mode == "error") {
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
              "H(curl) lumped diagonal entry " + std::to_string(i) + " = " +
              std::to_string(val) + " < floor " + std::to_string(diag_floor));
          } else { // clamp mode
            diag_view(i, 0) = (val >= 0) ? diag_floor : -diag_floor;
          }
        }
      }

      if (has_small && Comm->getRank() == 0 && verbosity >= 4) {
        std::cout << "WARNING: Clamped small H(curl) lumped diagonal entries to +/- "
                  << diag_floor << std::endl;
      }
    }

    hcurlInvOperator = Teuchos::rcp(new block_prec::DiagonalInverseOperator<SolverNode>(lumped_diag_vec));
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
      "ParameterManager::buildHcurlOperators: Unsupported solver type for H(curl): " + mass_type);
  }

  // Trick A wire-up: if we factored unit-weighted (M+K) instead of alpha*(M+K),
  // wrap both operators so that callers transparently see the correct behavior:
  //   hcurlForwardOp    callers expect alpha*(M+K)*x
  //   hcurlInvOperator  callers expect (1/alpha)*(M+K)^{-1}*x
  // This must happen after ALL branches above have assigned hcurlInvOperator.
  // It covers sparse_direct, sparse_iterative, and lumped uniformly. The wrap
  // is essential because MrHyDE_OptVector::dot()/dual() call the raw operator
  // apply() directly (via analysisManager_solve.hpp setMassOperators), bypassing
  // ParameterManager::applyHcurlInverse().
  if (hcurlMetricScale_ != static_cast<ScalarT>(1)) {
    using ScaledOp = block_prec::ScaledOperator<SolverNode>;
    // Forward op alpha*(M+K)*x maps primal (O(1)) -> dual (O(alpha)).
    // The inner matvec accumulates O(1) sums with no cancellation, then the
    // scalar multiply by alpha is exact per element => PostScale.
    hcurlForwardOp = Teuchos::rcp(new ScaledOp(
        hcurlMatrix, hcurlMetricScale_, ScaledOp::Mode::PostScale));
    // Inverse op (1/alpha)*(M+K)^{-1}*g maps dual (O(alpha)) -> primal (O(1)).
    // The backsolve against O(1)-entry (M+K) with an O(alpha) RHS suffers
    // catastrophic cancellation in each row update; PreScale collapses the
    // RHS to O(1) first so the backsolve stays in the clean regime.
    hcurlInvOperator = Teuchos::rcp(new ScaledOp(
        hcurlInvOperator, static_cast<ScalarT>(1) / hcurlMetricScale_,
        ScaledOp::Mode::PreScale));
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Check if H(curl) inverse operator is available
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
bool ParameterManager<Node>::hasHcurlInverseOperator() const {
  return !hcurlInvOperator.is_null();
}

/////////////////////////////////////////////////////////////////////////////////////////////
// Apply H(curl) inverse operator: (M + K)^{-1}
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void ParameterManager<Node>::applyHcurlInverse(const vector_RCP & in, vector_RCP & out) const {
  TEUCHOS_TEST_FOR_EXCEPTION(hcurlInvOperator.is_null(), std::runtime_error,
    "ParameterManager::applyHcurlInverse: hcurlInvOperator is null. "
    "Call buildHcurlOperators() first.");

  // hcurlInvOperator already folds the Trick A 1/alpha scale (if any) internally
  // via ScaledOperator; callers see (alpha*(M+K))^{-1}*in regardless.
  hcurlInvOperator->apply(*in, *out);
}

/////////////////////////////////////////////////////////////////////////////////////////////
// After the setup phase, we can get rid of a few things
/////////////////////////////////////////////////////////////////////////////////////////////

template<class Node>
void ParameterManager<Node>::purgeMemory() {
  // nothing here
}
