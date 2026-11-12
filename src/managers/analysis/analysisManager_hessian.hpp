/***********************************************************************
Extracts the reduced Hessian, either column by column through hessVec or by handing
MrHyDE_HessianOperator to Trilinos, or runs PCG on it (the Krylov sublist).

 Questions? Contact Alexey Voronin
 ************************************************************************/

// ========================================================================================
// ========================================================================================

void AnalysisManager::hessianSolve() {

  TEUCHOS_TEST_FOR_EXCEPTION(settings_->get<bool>("enable memory purge",false),
                             std::runtime_error,
                             "Error: analysis type 'hessian' requires 'enable memory purge: false' "
                             "(the control column filter needs the per-block element lists).");

  Teuchos::ParameterList & hset = settings_->sublist("Analysis").sublist("Hessian");

  const string  filebase  = hset.get<string>("output file","hess");
  const int     first_col = hset.get<int>("first column",0);
  const int     max_col   = hset.get<int>("max columns",8000);
  const ScalarT fd_step   = block_prec::scalarParam(hset,"FD step",1.0e-7);
  const bool    ctrl_only = hset.get<bool>("control columns only",false);
  const string  ctrl_list = hset.get<string>("control blocks","");
  const bool    probe_set = hset.isParameter("probe mode");
  const string  probe     = hset.get<string>("probe mode","columns");
  const int     probing   = hset.get<int>("probing size",10);
  const int     neig      = hset.get<int>("eigenvalues",0);
  const string  metric    = hset.get<string>("metric","euclidean");
  const bool    krylov    = hset.isSublist("Krylov");

  TEUCHOS_TEST_FOR_EXCEPTION(krylov && (neig > 0 || probe_set || first_col != 0
                                        || hset.get<int>("number of columns",0) > 0),
                             std::runtime_error,
                             "Error: Analysis->Hessian->Krylov runs alone; remove eigenvalues, 'probe mode', "
                             "'first column' and 'number of columns'.");
  TEUCHOS_TEST_FOR_EXCEPTION(probe != "columns" && probe != "operator",std::runtime_error,
                             "Error: Analysis->Hessian->'probe mode' must be columns or operator.");
  TEUCHOS_TEST_FOR_EXCEPTION(metric != "euclidean" && metric != "riesz",std::runtime_error,
                             "Error: Analysis->Hessian->'metric' must be euclidean or riesz.");
  TEUCHOS_TEST_FOR_EXCEPTION(neig > 0 && ((probe_set && probe == "columns") || first_col != 0
                                          || hset.get<int>("number of columns",0) > 0),
                             std::runtime_error,
                             "Error: Analysis->Hessian->eigenvalues > 0 skips the column dump; remove "
                             "'probe mode: columns', 'first column' and 'number of columns', or set eigenvalues to 0.");
  TEUCHOS_TEST_FOR_EXCEPTION(metric == "riesz" && probe == "columns" && neig <= 0,std::runtime_error,
                             "Error: Analysis->Hessian->metric = riesz applies only to probe mode operator "
                             "or eigenvalues > 0; the column dump writes H itself.");
  TEUCHOS_TEST_FOR_EXCEPTION(metric == "riesz" && params_->paramRiesz.is_null(),
                             std::runtime_error,
                             "Error: Analysis->Hessian->metric = riesz needs Solver->'Parameter "
                             "Riesz map'->apply = exact so P^-1 is available.");
  for (size_t r=0; r<postproc_->objectives.size(); ++r) {
    TEUCHOS_TEST_FOR_EXCEPTION(postproc_->objectives[r].type == "discrete control",std::runtime_error,
                               "Error: analysis type 'hessian' cannot evaluate the 'discrete control' objective '"
                               + postproc_->objectives[r].name
                               + "'; its data come from the ROL/ROL2 'Generate data' step, which hessian does not run.");
  }

  const int rank = comm_->getRank();
  const bool exact = params_->isParameter("src_gate");
  const string prefix = filebase + (exact ? "_exact" : "_fd");
  const bool use_operator = (probe == "operator" || neig > 0);

  postproc_->write_solution = false;

  Teuchos::RCP<ROL::Objective_MILO<ScalarT> > obj
    = Teuchos::rcp(new ROL::Objective_MILO<ScalarT>(solver_,postproc_,params_));

  MrHyDE_OptVector xtmp = params_->getCurrentVector();
  Teuchos::RCP<ROL::Vector<ScalarT> > x = xtmp.clone();
  x->set(xtmp);

  std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > > xfield
    = Teuchos::dyn_cast<MrHyDE_OptVector>(*x).getField();

  TEUCHOS_TEST_FOR_EXCEPTION(xfield.size() == 0, std::runtime_error,
                             "Error: analysis type 'hessian' needs a discretized parameter.");

  const int nslabs = static_cast<int>(xfield.size());
  const int nspatial = static_cast<int>(params_->param_owned_map->getGlobalNumElements());
  const int ctrl_dim = nslabs*nspatial;

  TEUCHOS_TEST_FOR_EXCEPTION(params_->param_owned_map->getMinAllGlobalIndex() != 0 ||
                             params_->param_owned_map->getMaxAllGlobalIndex() != static_cast<GO>(nspatial)-1,
                             std::runtime_error,
                             "Error: analysis type 'hessian' needs parameter global ids dense in [0,n); "
                             "basis(j) addresses a parameter dof by its global id.");

  if (krylov) {
    TEUCHOS_TEST_FOR_EXCEPTION(!exact,std::runtime_error,
                               "Error: Analysis->Hessian->Krylov needs the exact hessVec; declare the "
                               "inactive scalar parameter src_gate.");
    ScalarT tol = fd_step;
    const ScalarT J0 = obj->value(*x,tol);
    this->hessianKrylov(hset.sublist("Krylov"),obj,x,J0,tol);
    postproc_->resetErrors();
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(probe == "operator" && ctrl_dim > max_col,std::runtime_error,
                             "Error: probe mode operator probes all " + std::to_string(ctrl_dim)
                             + " Hessian columns, which exceeds Analysis->Hessian->'max columns' ("
                             + std::to_string(max_col)
                             + "). Raise the cap, or use probe mode columns with 'number of columns'.");

  std::vector<int> ctrl_mask;
  if (ctrl_only && !use_operator) {
    ctrl_mask = this->controlDofMask(ctrl_list, nspatial);
  }

  std::vector<int> cols;
  int num_col = 0;

  if (!use_operator) {

    num_col = hset.get<int>("number of columns",0);
    if (num_col <= 0) { num_col = ctrl_dim - first_col; }
    num_col = std::min(num_col, ctrl_dim - first_col);

    // Columns whose DoF touches no control block are identically zero, so skip them.
    for (int c=0; c<num_col; ++c) {
      const int j = first_col + c;
      if (ctrl_only && ctrl_mask[j % nspatial] == 0) { continue; }
      cols.push_back(j);
    }

    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(cols.size()) > max_col, std::runtime_error,
                               "Error: Hessian column count " + std::to_string(cols.size())
                               + " (dimension " + std::to_string(ctrl_dim)
                               + ") exceeds Analysis->Hessian->'max columns' ("
                               + std::to_string(max_col)
                               + "). Set 'number of columns' to extract a sub-block, or raise the cap.");
  }

  const int ncols = use_operator ? ctrl_dim : static_cast<int>(cols.size());
  const int sweeps_per_col = exact ? 2 : 4;

  if (rank == 0) {
    printSectionBanner(rank,"Hessian column dump");
    std::cout << "[HESS-DUMP] path = " << (exact ? "exact (src_gate declared)" : "fd (no src_gate)")
              << std::endl;
    std::cout << "[HESS-DUMP] dim = " << ctrl_dim << " (slabs = " << nslabs
              << ", spatial = " << nspatial << ")" << std::endl;
    if (probe == "operator" || !use_operator) {
      std::cout << "[HESS-DUMP] " << ncols << " columns, " << sweeps_per_col
                << " sweeps/column -> " << ncols*sweeps_per_col << " transient sweeps";
      if (probe == "operator") { std::cout << ", probed in blocks of " << probing; }
      std::cout << std::endl;
    }
    if (neig > 0) {
      std::cout << "[HESS-DUMP] " << neig << " eigenpairs of "
                << (metric == "riesz" ? "inv(P)*H" : "H") << " via Anasazi" << std::endl;
    }
    if (ctrl_only && !use_operator) {
      std::cout << "[HESS-DUMP] control columns only: " << ncols << " of " << num_col
                << " columns carry support" << std::endl;
    }
    if (!exact) {
      std::cout << "[HESS-DUMP] FD step = " << fd_step << std::endl;
    }
  }

  // Pin the stored forward trajectory at x so the column loop does not re-solve it.
  {
    ScalarT tol = fd_step;
    obj->value(*x,tol);
  }

  typedef Tpetra::MatrixMarket::Writer<LA_MultiVector> MMWriter;

  if (use_operator) {

    Teuchos::RCP<MrHyDE_ParamRiesz> rz;
    if (metric == "riesz") { rz = params_->paramRiesz; }

    Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,SolverNode> > Hop
      = Teuchos::rcp(new MrHyDE_HessianOperator(obj,x,params_->param_owned_map,
                                                fd_step,rz));

    if (probe == "operator") {
      Teuchos::ParameterList wp;
      wp.set("probing size",probing);
      wp.set("precision",16);
      MMWriter::writeOperator(prefix+"_op.mtx",*Hop,wp);
      if (rank == 0) {
        std::cout << "[HESS-DUMP] wrote " << prefix
                  << "_op.mtx (coordinate format, true global ids)" << std::endl;
      }
    }

    if (neig > 0) {
      this->hessianEigenvalues(hset,Hop,prefix,neig,ctrl_dim,rz.is_null() && exact);
    }

    postproc_->resetErrors();
    return;
  }

  // One row block per slab; the writer does the gather and the formatting.
  std::vector<Teuchos::RCP<LA_MultiVector> > Hblocks;
  for (int t=0; t<nslabs; ++t) {
    Hblocks.push_back(solver_->linalg->getNewParamVector(ncols));
  }

  Teuchos::RCP<ROL::Vector<ScalarT> > hv = x->clone();
  std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > > hvf
    = Teuchos::dyn_cast<MrHyDE_OptVector>(*hv).getField();

  for (int c=0; c<ncols; ++c) {

    const int j = cols[c];
    ROL::Ptr<ROL::Vector<ScalarT> > ej = x->basis(j);

    ScalarT tol = fd_step;
    obj->hessVec(*hv,*ej,*x,tol);

    for (int t=0; t<nslabs; ++t) {
      Hblocks[t]->getVectorNonConst(c)->update(1.0,*(hvf[t]->getVector()),0.0);
    }

    if (rank == 0 && ncols > 20 && (c+1)%(ncols/20) == 0) {
      std::cout << "[HESS-DUMP] column " << c+1 << " of " << ncols << std::endl;
    }
  }

  // Row order follows the gather map, not global id, so the map file is what recovers it.
  for (int t=0; t<nslabs; ++t) {
    std::stringstream fn, note;
    fn << prefix << "_slab" << t << ".mtx";
    note << (exact ? "exact" : "fd") << " hessVec, slab " << t << " row block, "
         << "row i = line i of " << prefix << ".map, "
         << "column j = line j of " << prefix << ".cols";
    MMWriter::writeDenseFile(fn.str(),*(Hblocks[t]),"reduced Hessian",note.str());
  }
  MMWriter::writeMapFile(prefix+".map",*(params_->param_owned_map));

  int cols_open = 1;
  if (rank == 0) {

    std::cout << "[HESS-DUMP] wrote " << nslabs << " row blocks " << prefix
              << "_slab*.mtx (" << nspatial << " x " << ncols << " each), "
              << prefix << ".map and " << prefix << ".cols" << std::endl;

    std::ofstream colout(prefix+".cols");
    cols_open = colout.is_open() ? 1 : 0;
    for (int c=0; c<ncols; ++c) { colout << cols[c] << std::endl; }
    colout.close();
  }
  Teuchos::broadcast(*comm_,0,1,&cols_open);
  TEUCHOS_TEST_FOR_EXCEPTION(cols_open == 0,std::runtime_error,
                             "Error: could not open the data file: " + prefix + ".cols");

  postproc_->resetErrors();
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::hessianEigenvalues(Teuchos::ParameterList & hset,
                                         const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,SolverNode> > & Hop,
                                         const string & prefix, const int & neig,
                                         const int & ctrl_dim, const bool & hermitian) {

  typedef Tpetra::Operator<ScalarT,LO,GO,SolverNode> EigOp;
  typedef Anasazi::BasicEigenproblem<ScalarT,LA_MultiVector,EigOp> EigProblem;
  typedef Anasazi::BlockKrylovSchurSolMgr<ScalarT,LA_MultiVector,EigOp> EigSolver;
  typedef Tpetra::MatrixMarket::Writer<LA_MultiVector> MMWriter;

  // Which and Block Size keep Anasazi's defaults.
  Teuchos::ParameterList def;
  def.set("Num Blocks",std::min(ctrl_dim-1,std::max(2*neig,10)));
  def.set("Maximum Restarts",100);
  def.set("Convergence Tolerance",1.0e-8);
  def.set("Verbosity",Anasazi::Errors+Anasazi::Warnings+Anasazi::FinalSummary);

  Teuchos::ParameterList ap = hset.sublist("Anasazi");
  ap.setParametersNotAlreadySet(def);

  Teuchos::RCP<LA_MultiVector> ivec
    = Teuchos::rcp(new LA_MultiVector(Hop->getDomainMap(),ap.get("Block Size",1)));
  ivec->randomize();

  Teuchos::RCP<EigProblem> prob = Teuchos::rcp(new EigProblem(Hop,ivec));
  prob->setHermitian(hermitian);
  prob->setNEV(neig);
  TEUCHOS_TEST_FOR_EXCEPTION(!prob->setProblem(),std::runtime_error,
                             "Error: Anasazi rejected the Hessian eigenproblem.");

  EigSolver mgr(prob,ap);
  const Anasazi::ReturnType eret = mgr.solve();
  Anasazi::Eigensolution<ScalarT,LA_MultiVector> sol = prob->getSolution();

  if (comm_->getRank() == 0 && eret != Anasazi::Converged) {
    std::cout << "[HESS-DUMP] WARNING: Anasazi did not converge; "
              << sol.numVecs << " of " << neig << " eigenpairs returned" << std::endl;
  }

  if (sol.numVecs > 0 && !sol.Evecs.is_null()) {
    MMWriter::writeDenseFile(prefix+"_evec.mtx",*(sol.Evecs),"reduced Hessian eigenvectors",
                             "row i = line i of " + prefix + "_control.map");
    MMWriter::writeMapFile(prefix+"_control.map",*(Hop->getDomainMap()));
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::hessianKrylov(Teuchos::ParameterList & kset,
                                    const Teuchos::RCP<ROL::Objective<ScalarT> > & obj,
                                    const Teuchos::RCP<ROL::Vector<ScalarT> > & x,
                                    const ScalarT & J0, const ScalarT & tol_in) {

  typedef Teuchos::RCP<ROL::Vector<ScalarT> > VecRCP;

  const int     maxit     = kset.get<int>("iteration limit",1000);
  const ScalarT eta       = block_prec::scalarParam(kset,"relative tolerance",1.0e-10);
  const bool    reorth    = kset.get<bool>("reorthogonalize",false);
  const string  outfile   = kset.get<string>("output file","krylov.txt");
  const string  basisfile = kset.get<string>("basis file","");
  const ScalarT cloak     = block_prec::scalarParam(kset,"cloaking tolerance",0.0);

  TEUCHOS_TEST_FOR_EXCEPTION(maxit < 1 || eta < 0.0,std::runtime_error,
                             "Error: Analysis->Hessian->Krylov needs 'iteration limit' >= 1 and "
                             "'relative tolerance' >= 0.");
  TEUCHOS_TEST_FOR_EXCEPTION(!(cloak >= 0.0 && cloak < 1.0),std::runtime_error,
                             "Error: Analysis->Hessian->Krylov->'cloaking tolerance' must lie in [0,1).");

  // Ratio q = w'/w of the misfit weight turns H s = -g into (H + sigma P) s = -g, sigma = c (1/q - 1), R = c P;
  // R = c P needs each regularization term to be 0.5 weight times a square, as the regularizer map assumes.
  std::vector<ScalarT> q;
  if (kset.isParameter("weight ratios") && !kset.getEntry("weight ratios").isType<string>()) {
    q.push_back(block_prec::scalarParam(kset,"weight ratios",0.0));
  }
  else {
    string list = kset.get<string>("weight ratios","");
    std::replace(list.begin(),list.end(),',',' ');
    std::istringstream in(list);
    ScalarT v;
    while (in >> v) { q.push_back(v); }
    TEUCHOS_TEST_FOR_EXCEPTION(!in.eof(),std::runtime_error,
                               "Error: Analysis->Hessian->Krylov->'weight ratios' must be a list of numbers.");
  }
  ScalarT c = 0.0;
  if (!q.empty() || cloak > 0.0) {
    TEUCHOS_TEST_FOR_EXCEPTION(params_->paramRiesz.is_null() || solver_->param_riesz.type != "regularizer",
                               std::runtime_error,
                               "Error: Analysis->Hessian->Krylov->'weight ratios' and 'cloaking tolerance' need "
                               "Solver->'Parameter Riesz map' with type regularizer, which makes the regularizer "
                               "a multiple of the metric.");
    // R = c P with one regularizer term of weight deltat per time step, all in one slab for a static control.
    const ScalarT cdef = (solver_->isTransient && !params_->have_dynamic_discretized)
                         ? params_->numTimeSteps*solver_->deltat : solver_->deltat;
    c = block_prec::scalarParam(kset,"metric scale",cdef);
    TEUCHOS_TEST_FOR_EXCEPTION(!(c > 0.0),std::runtime_error,
                               "Error: Analysis->Hessian->Krylov->'metric scale' must be positive.");
    // q > 1 gives -c < sigma < 0, still SPD since H >= R = c P; those systems converge after the base.
    for (size_t j=0; j<q.size(); ++j) {
      TEUCHOS_TEST_FOR_EXCEPTION(!(q[j] > 0.0 && q[j] != 1.0),std::runtime_error,
                                 "Error: Analysis->Hessian->Krylov->'weight ratios' must be positive and differ "
                                 "from 1.");
    }
  }

  const int rank = comm_->getRank();
  ScalarT tol = tol_in;

  // g(w') = (w'/w) g(w) only at zero control, where the regularizer term R x of g vanishes.
  TEUCHOS_TEST_FOR_EXCEPTION((!q.empty() || cloak > 0.0) && x->norm() > 0.0,std::runtime_error,
                             "Error: Analysis->Hessian->Krylov->'weight ratios' and 'cloaking tolerance' need zero "
                             "initial parameters.");
  // At zero control J0 is the misfit alone, w_EM E0 / 2, which the cloaking bracket divides by.
  TEUCHOS_TEST_FOR_EXCEPTION(cloak > 0.0 && !(J0 > 0.0),std::runtime_error,
                             "Error: Analysis->Hessian->Krylov->'cloaking tolerance' needs a positive objective at "
                             "zero control.");

  // Opened before the loop so that a bad path fails before the products are spent.
  std::ofstream bout;
  if (!basisfile.empty()) {
    std::stringstream fn;
    fn << basisfile << "." << rank << ".bin";
    bout.open(fn.str().c_str(),std::ios::binary);
    int ok = bout.is_open() ? 1 : 0, all_ok = 0;
    Teuchos::reduceAll(*comm_,Teuchos::REDUCE_MIN,1,&ok,&all_ok);
    TEUCHOS_TEST_FOR_EXCEPTION(all_ok == 0,std::runtime_error,
                               "Error: could not open the basis file: " + fn.str());
  }

  // r is a functional and y = P^-1 r its Riesz representative; every pairing below is apply().
  VecRCP g = x->dual().clone();
  obj->gradient(*g,*x,tol);
  VecRCP r  = g->clone(); r->set(*g); r->scale(-1.0);
  VecRCP y  = x->clone(); y->set(r->dual());
  VecRCP p  = x->clone(); p->set(*y);
  VecRCP s  = x->clone(); s->zero();
  VecRCP Hp = g->clone();

  ScalarT rr = y->apply(*r);
  const ScalarT gnorm = std::sqrt(rr);
  TEUCHOS_TEST_FOR_EXCEPTION(!(gnorm > 0.0),std::runtime_error,
                             "Error: Analysis->Hessian->Krylov: the gradient at the current parameters is zero.");

  // Z holds the P-orthonormal Lanczos vectors y_k/||r_k|| as raw slab vectors; (z_j, y)_P = z_j^T r, and a
  // reorthogonalization pass applies P once, to sum_j c_j z_j.
  typedef std::vector<vector_RCP> Slabs;
  auto slabs = [](const VecRCP & v) {
    Slabs out;
    std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > f
      = Teuchos::dyn_cast<MrHyDE_OptVector>(*v).getField();
    for (size_t t=0; t<f.size(); ++t) { out.push_back(f[t]->getVector()); }
    return out;
  };
  const Slabs yf = slabs(y), rf = slabs(r);
  size_t fdim = 0;
  for (size_t t=0; t<yf.size(); ++t) { fdim += yf[t]->getGlobalLength(); }
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<size_t>(x->dimension()) != fdim,std::runtime_error,
                             "Error: Analysis->Hessian->Krylov needs a control of discretized parameters only.");
  VecRCP wv = x->clone();
  const Slabs wf = slabs(wv);
  const bool store = reorth || !basisfile.empty();
  std::vector<Slabs> Z;
  auto push = [&](const ScalarT nrm) {
    Slabs z;
    for (size_t t=0; t<yf.size(); ++t) {
      z.push_back(Teuchos::rcp(new LA_MultiVector(yf[t]->getMap(),1)));
      z[t]->update(1.0/nrm,*yf[t],0.0);
    }
    Z.push_back(z);
  };
  auto zdot = [&](const Slabs & z, const Slabs & v) {
    ScalarT d = 0.0;
    Teuchos::Array<ScalarT> dt(1);
    for (size_t t=0; t<z.size(); ++t) { z[t]->dot(*v[t],dt()); d += dt[0]; }
    return d;
  };
  if (store) { push(gnorm); }

  // Multi-shift CG: the residual of shift j is zeta_j times the base residual.
  const size_t ns = q.size();
  std::vector<ScalarT> sigma(ns), zeta(ns,1.0), zeta_old(ns,1.0), zres(ns,1.0);
  std::vector<int> sexit(ns,0);
  std::vector<VecRCP> xs(ns), ps(ns);
  for (size_t j=0; j<ns; ++j) {
    sigma[j] = c*(1.0/q[j] - 1.0);
    xs[j] = x->clone(); xs[j]->zero();
    ps[j] = x->clone(); ps[j]->set(*y);
  }
  ScalarT alpha_old = 1.0, beta_old = 0.0;

  std::ofstream out;
  int out_open = 1;
  if (rank == 0) {
    out.open(outfile.c_str());
    out_open = out.is_open() ? 1 : 0;
  }
  Teuchos::broadcast(*comm_,0,1,&out_open);
  TEUCHOS_TEST_FOR_EXCEPTION(out_open == 0,std::runtime_error,
                             "Error: could not open the data file: " + outfile);
  if (rank == 0) {
    out << std::scientific << std::setprecision(17);
    out << "# PCG on H s = -g from s = 0 in the metric P of the parameters; rnorm = ||r||_{P^-1}" << std::endl;
    out << "J0 " << J0 << std::endl;
    out << "gnorm " << gnorm << std::endl;
    out << "reorthogonalize " << (reorth ? 1 : 0) << std::endl;
    out << "relative_tolerance " << eta << std::endl;
    out << "metric_scale " << c << std::endl;
    for (size_t j=0; j<ns; ++j) {
      out << "shift " << j << " ratio " << q[j] << " sigma " << sigma[j] << std::endl;
    }
    out << "# it k alpha beta rnorm pred snorm2 orth, then rnorm of each shift" << std::endl;
    if (cloak > 0.0) {
      out << "cloaking_tolerance " << cloak << std::endl;
      out << "# cloak k delta, then the lower bound on E1/E0 at the base weight and at each shift;"
          << " E1/E0 of the iterate is the lower bound plus delta" << std::endl;
    }
  }

  // Lanczos matrix T_k of P^-1 H from the CG coefficients, and solves with s T_k + t I (SPD for every use
  // below); a non-positive pivot returns NaN, which turns the cloaking test off for that step.
  std::vector<ScalarT> td, te;
  auto trisolve = [&](const ScalarT sc, const ScalarT sh, std::vector<ScalarT> b) {
    const size_t n = td.size();
    std::vector<ScalarT> cp(n,0.0);
    for (size_t i=0; i<n; ++i) {
      const ScalarT m = sc*td[i] + sh - (i > 0 ? sc*te[i-1]*cp[i-1] : 0.0);
      if (!(m > 0.0)) { return std::vector<ScalarT>(n,std::numeric_limits<ScalarT>::quiet_NaN()); }
      cp[i] = (i+1 < n ? sc*te[i]/m : 0.0);
      b[i] = (b[i] - (i > 0 ? sc*te[i-1]*b[i-1] : 0.0))/m;
    }
    for (size_t i=n-1; i-- > 0;) { b[i] -= cp[i]*b[i+1]; }
    return b;
  };

  // Run past convergence ('relative tolerance' 0), r, y and p would underflow. They are kept at 2^ex times
  // their true size, rescaled by powers of 2, which is exact, so alpha and beta do not change.
  // count is the number of products; base is the iteration at which the base system met the tolerance, after
  // which s stays fixed and the recurrence runs on for the shifts with larger weights.
  int count = 0, flag = 1, ex = 0, base = 0;
  ScalarT pred = 0.0;
  for (int k=1; k<=maxit; ++k) {

    obj->hessVec(*Hp,*p,*x,tol);
    count = k;
    const ScalarT kappa = p->apply(*Hp);
    if (!(kappa > 0.0)) { flag = 2; break; }
    const ScalarT alpha = rr/kappa;
    if (base == 0) {
      s->axpy(std::ldexp(alpha,-ex),*p);
      pred += 0.5*alpha*std::ldexp(rr,-2*ex);
    }
    r->axpy(-alpha,*Hp);
    y->set(r->dual());

    // orth: largest |(z_j, y)_P|/||y||_P before reorthogonalization, the loss of orthogonality.
    ScalarT orth = 0.0;
    if (store) {
      const ScalarT rnrm = std::sqrt(y->apply(*r));
      std::vector<ScalarT> cf(Z.size());
      for (int pass=0; pass<(reorth ? 2 : 1); ++pass) {
        for (size_t j=0; j<Z.size(); ++j) { cf[j] = zdot(Z[j],rf); }
        if (pass == 0) {
          for (size_t j=0; j<Z.size(); ++j) { orth = std::max(orth,std::abs(cf[j])/rnrm); }
        }
        if (reorth) {
          wv->zero();
          for (size_t j=0; j<Z.size(); ++j) {
            for (size_t t=0; t<wf.size(); ++t) { wf[t]->update(cf[j],*Z[j][t],1.0); }
          }
          y->axpy(-1.0,*wv);
          r->axpy(-1.0,wv->dual());
        }
      }
    }

    const ScalarT rr_new = y->apply(*r);
    // r = 0 exactly: the Krylov space is invariant and s is the solution.
    if (!(rr_new > 0.0)) { flag = 0; break; }
    const ScalarT beta = rr_new/rr;
    if (cloak > 0.0) {
      td.push_back(1.0/alpha + (k > 1 ? beta_old/alpha_old : 0.0));
      if (k > 1) { te.push_back(std::sqrt(beta_old)/alpha_old); }
    }

    for (size_t j=0; j<ns; ++j) {
      if (sexit[j] > 0) { continue; }
      const ScalarT z1 = zeta[j]*zeta_old[j]*alpha_old
                         /(alpha*beta_old*(zeta_old[j] - zeta[j]) + zeta_old[j]*alpha_old*(1.0 + sigma[j]*alpha));
      // With 'relative tolerance' 0 zeta underflows; the shift keeps its last iterate.
      if (!(std::isfinite(z1) && z1 != 0.0)) { sexit[j] = k; continue; }
      xs[j]->axpy(alpha*z1/zeta[j],*ps[j]);
      ps[j]->scale(beta*(z1/zeta[j])*(z1/zeta[j]));
      ps[j]->axpy(std::ldexp(z1,-ex),*y);
      zeta_old[j] = zeta[j];
      zeta[j] = z1;
      zres[j] = std::abs(z1)*std::ldexp(std::sqrt(rr_new),-ex);
      if (zres[j] < eta*gnorm) { sexit[j] = k; }
    }
    alpha_old = alpha;
    beta_old = beta;

    p->scale(beta);
    p->axpy(1.0,*y);
    rr = rr_new;
    const ScalarT rnorm = std::ldexp(std::sqrt(rr),-ex);
    const ScalarT snorm2 = s->dot(*s);

    if (rank == 0) {
      out << "it " << k << " " << alpha << " " << beta << " " << rnorm << " " << pred << " "
          << snorm2 << " " << orth;
      for (size_t j=0; j<ns; ++j) { out << " " << zres[j]; }
      out << std::endl;
    }

    // Gauss rule of T_k with L = T_k/c - I (eigenvalues lambda_i) and M = I + q L: E1/E0 of the k-step iterate at
    // weight ratio q is K e1^T L^-1 M^-2 e1 + delta, and the exact E1/E0 lies between that and delta below it.
    bool certified = false;
    if (cloak > 0.0) {
      std::vector<ScalarT> e1(td.size(),0.0);
      e1[0] = 1.0;
      const ScalarT K = gnorm*gnorm/(2.0*c*J0);
      const ScalarT delta = 1.0 - K*trisolve(1.0/c,-1.0,e1)[0];
      std::vector<ScalarT> lower(ns+1);
      // delta is a difference of O(1) numbers: at or below zero it is rounding or R != c P, not a certificate.
      certified = std::isfinite(delta) && delta > 0.0;
      for (size_t j=0; j<=ns; ++j) {
        const ScalarT qj = (j == 0 ? 1.0 : q[j-1]);
        const std::vector<ScalarT> v = trisolve(qj/c,1.0-qj,trisolve(qj/c,1.0-qj,e1));
        lower[j] = K*trisolve(1.0/c,-1.0,v)[0];
        const bool active = (j == 0 ? base == 0 : sexit[j-1] == 0);
        if (!std::isfinite(lower[j]) || (active && delta > cloak*(lower[j] + delta))) { certified = false; }
      }
      if (rank == 0) {
        out << "cloak " << k << " " << delta;
        for (size_t j=0; j<=ns; ++j) { out << " " << lower[j]; }
        out << std::endl;
      }
    }

    if (base == 0 && rnorm < eta*gnorm) { base = k; flag = 0; }
    if (base > 0 && std::find(sexit.begin(),sexit.end(),0) == sexit.end()) { break; }
    // Every system still iterating has E1 certified to the cloaking tolerance: stop all of them here.
    if (certified) {
      if (base == 0) { base = k; flag = 3; }
      for (size_t j=0; j<ns; ++j) { if (sexit[j] == 0) { sexit[j] = k; } }
      break;
    }
    if (store && k < maxit) { push(std::sqrt(rr)); }
    if (rr < std::ldexp(gnorm*gnorm,-200)) {
      r->scale(std::ldexp(1.0,100));
      y->scale(std::ldexp(1.0,100));
      p->scale(std::ldexp(1.0,100));
      rr = std::ldexp(rr,200);
      ex += 100;
    }
  }

  // The recurrence residual drifts from -g - H s; one more product, not in the count, measures it.
  obj->hessVec(*Hp,*s,*x,tol);
  VecRCP rt = g->clone(); rt->set(*g); rt->axpy(1.0,*Hp); rt->scale(-1.0);
  VecRCP yt = x->clone(); yt->set(rt->dual());
  const ScalarT true_rnorm = std::sqrt(yt->apply(*rt));

  ScalarT defect = 0.0;
  if (!Z.empty()) {
    VecRCP pz = g->clone();
    const Slabs pzf = slabs(pz);
    for (size_t j=0; j<Z.size(); ++j) {
      for (size_t t=0; t<wf.size(); ++t) { wf[t]->update(1.0,*Z[j][t],0.0); }
      pz->set(wv->dual());
      for (size_t i=0; i<=j; ++i) {
        defect = std::max(defect,std::abs(zdot(Z[i],pzf) - (i == j ? 1.0 : 0.0)));
      }
    }
  }

  // J after the step, J(x + s); the shifts start from x = 0.
  VecRCP xp = x->clone(); xp->set(*x); xp->axpy(1.0,*s);
  const ScalarT Js = obj->value(*xp,tol);
  const ScalarT ss = s->dot(*s);
  std::vector<ScalarT> Jj(ns), sj(ns);
  for (size_t j=0; j<ns; ++j) {
    Jj[j] = obj->value(*xs[j],tol);
    sj[j] = xs[j]->dot(*xs[j]);
  }

  if (rank == 0) {
    out << "exit " << (base > 0 ? base : count) << " flag " << flag << std::endl;
    out << "products " << count << std::endl;
    out << "true_rnorm " << true_rnorm << std::endl;
    out << "J " << Js << " snorm2 " << ss << std::endl;
    out << "basis " << Z.size() << " orthonormality " << defect << std::endl;
    for (size_t j=0; j<ns; ++j) {
      out << "shift " << j << " exit " << sexit[j] << " rnorm " << zres[j] << " J " << Jj[j]
          << " snorm2 " << sj[j] << std::endl;
    }
    out.close();
    out_open = out.fail() ? 0 : 1;
  }
  Teuchos::broadcast(*comm_,0,1,&out_open);
  TEUCHOS_TEST_FOR_EXCEPTION(out_open == 0,std::runtime_error,
                             "Error: writing the data file failed: " + outfile);

  if (!basisfile.empty()) {
    // One file per rank: nvec, nslabs, nloc as int64, the nloc parameter global ids, then each vector slab by slab.
    Teuchos::RCP<const Tpetra::Map<LO,GO,SolverNode> > pmap = params_->param_owned_map;
    const std::int64_t nloc = static_cast<std::int64_t>(pmap->getLocalNumElements());
    const std::int64_t nvec = static_cast<std::int64_t>(Z.size());
    const std::int64_t nsl  = static_cast<std::int64_t>(yf.size());
    bout.write(reinterpret_cast<const char*>(&nvec),sizeof(nvec));
    bout.write(reinterpret_cast<const char*>(&nsl),sizeof(nsl));
    bout.write(reinterpret_cast<const char*>(&nloc),sizeof(nloc));
    for (std::int64_t i=0; i<nloc; ++i) {
      const std::int64_t gid = static_cast<std::int64_t>(pmap->getGlobalElement(static_cast<LO>(i)));
      bout.write(reinterpret_cast<const char*>(&gid),sizeof(gid));
    }
    // A slab off the owned map is reported through ok, so no rank throws alone.
    int ok = 1, all_ok = 0;
    for (size_t v=0; v<Z.size() && ok; ++v) {
      for (size_t t=0; t<Z[v].size() && ok; ++t) {
        if (static_cast<std::int64_t>(Z[v][t]->getLocalLength()) != nloc) { ok = 0; break; }
        Teuchos::ArrayRCP<const ScalarT> data = Z[v][t]->getData(0);
        bout.write(reinterpret_cast<const char*>(data.getRawPtr()),nloc*sizeof(ScalarT));
      }
    }
    bout.close();
    if (bout.fail()) { ok = 0; }
    Teuchos::reduceAll(*comm_,Teuchos::REDUCE_MIN,1,&ok,&all_ok);
    TEUCHOS_TEST_FOR_EXCEPTION(all_ok == 0,std::runtime_error,
                               "Error: writing the basis file " + basisfile + ".<rank>.bin failed.");
  }
}

// ========================================================================================
// ========================================================================================

std::vector<int> AnalysisManager::controlDofMask(const string & control_blocks, const size_t nspatial) {

  Teuchos::RCP<DiscretizationInterface> disc = postproc_->disc;
  const size_t nblocks = disc->block_names.size();

  // A regularization can miss blocks where the control acts, so the deck names them.
  const std::vector<bool> is_ctrl = matchNamedBlocks(control_blocks,disc->block_names,
                                                     "Error: Analysis->Hessian->'control blocks'");
  TEUCHOS_TEST_FOR_EXCEPTION(std::find(is_ctrl.begin(),is_ctrl.end(),true) == is_ctrl.end(),
                             std::runtime_error,
                             "Error: Analysis->Hessian->'control columns only' needs 'control blocks' "
                             "to name the element blocks the control acts on.");

  std::vector<int> local(nspatial,0), mask(nspatial,0);
  for (size_t b=0; b<nblocks; ++b) {
    if (!is_ctrl[b]) { continue; }
    for (size_t e=0; e<disc->my_elements[b].extent(0); ++e) {
      std::vector<GO> gids;
      params_->paramDOF->getElementGIDs(disc->my_elements[b](e),gids,disc->block_names[b]);
      for (size_t i=0; i<gids.size(); ++i) { local[gids[i]] = 1; }
    }
  }
  Teuchos::reduceAll(*comm_,Teuchos::REDUCE_MAX,static_cast<int>(nspatial),local.data(),mask.data());
  return mask;
}
