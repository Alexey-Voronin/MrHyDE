/***********************************************************************
Extracts the reduced Hessian, either column by column through hessVec or by handing
MrHyDE_HessianOperator to Trilinos, and writes the DoF geometry needed to read it.

 Questions? Contact Alexey Voronin
 ************************************************************************/

// ========================================================================================
// ========================================================================================

void AnalysisManager::hessianSolve() {

  TEUCHOS_TEST_FOR_EXCEPTION(settings_->get<bool>("enable memory purge",false),
                             std::runtime_error,
                             "Error: analysis type 'hessian' requires 'enable memory purge: false' "
                             "(the DoF dump needs the mesh and the DOF local-ID views).");

  Teuchos::ParameterList & hset = settings_->sublist("Analysis").sublist("Hessian");

  const string  filebase  = hset.get<string>("output file","hess");
  const int     first_col = hset.get<int>("first column",0);
  const int     max_col   = hset.get<int>("max columns",8000);
  const ScalarT fd_step   = hset.get<double>("FD step",1.0e-7);
  const bool    dump_dofs = hset.get<bool>("write dof data",true);
  const bool    ctrl_only = hset.get<bool>("control columns only",false);
  const string  ctrl_list = hset.get<string>("control blocks","");
  const int     trace_col = hset.get<int>("tangent trace column",-1);
  const string  probe     = hset.get<string>("probe mode","columns");
  const int     probing   = hset.get<int>("probing size",10);
  const int     neig      = hset.get<int>("eigenvalues",0);
  const string  metric    = hset.get<string>("metric","euclidean");

  TEUCHOS_TEST_FOR_EXCEPTION(probe != "columns" && probe != "operator",std::runtime_error,
                             "Error: Analysis->Hessian->'probe mode' must be columns or operator.");
  TEUCHOS_TEST_FOR_EXCEPTION(metric != "euclidean" && metric != "riesz",std::runtime_error,
                             "Error: Analysis->Hessian->'metric' must be euclidean or riesz.");
  TEUCHOS_TEST_FOR_EXCEPTION(metric == "riesz" && params_->paramRiesz.is_null(),
                             std::runtime_error,
                             "Error: Analysis->Hessian->metric = riesz needs Solver->'Parameter "
                             "Riesz map'->apply = exact so P^-1 is available.");

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

  std::vector<int> ctrl_mask;
  if (dump_dofs || (ctrl_only && !use_operator)) {
    this->dumpDofData(ctrl_list, dump_dofs, ctrl_mask);
  }

  std::vector<int> cols;
  int num_col = 0;

  if (!use_operator) {

    num_col = hset.get<int>("number of columns",0);
    if (num_col <= 0) { num_col = ctrl_dim - first_col; }
    num_col = std::min(num_col, ctrl_dim - first_col);

    // dumpDofData fills ctrl_mask on rank 0; every rank needs it to agree on the column list.
    int nmask = static_cast<int>(ctrl_mask.size());
    Teuchos::broadcast(*comm_,0,1,&nmask);
    ctrl_mask.resize(nmask,0);
    if (nmask > 0) { Teuchos::broadcast(*comm_,0,nmask,&ctrl_mask[0]); }

    // Columns whose DoF touches no control block are identically zero, so skip them.
    for (int c=0; c<num_col; ++c) {
      const int j = first_col + c;
      if (ctrl_only && nmask > 0 && ctrl_mask[j % nspatial] == 0) { continue; }
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
      this->hessianEigenvalues(hset,Hop,prefix,neig,ctrl_dim,rz.is_null());
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

    // Only the exact path runs a tangent sweep, so only it has a trace to write.
    if (j == trace_col && exact) { this->writeTangentTrace(j, j/nspatial); }

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

  // Same row order as the slab blocks, so the metric does not have to be rebuilt offline.
  if (!params_->diagParamMass.is_null()) {
    string dname = prefix + "_riesz_diag.mtx";
    solver_->linalg->writeVectorToFile(params_->diagParamMass,dname);
  }

  if (rank == 0) {

    std::cout << "[HESS-DUMP] wrote " << nslabs << " row blocks " << prefix
              << "_slab*.mtx (" << nspatial << " x " << ncols << " each), "
              << prefix << ".map and " << prefix << ".cols" << std::endl;

    std::ofstream colout = DofDump::openDataFile(prefix+".cols",16);
    for (int c=0; c<ncols; ++c) { colout << cols[c] << std::endl; }
    colout.close();

    const double t0 = solver_->initial_time;
    const double tf = solver_->final_time;
    const double dt = (tf - t0)/static_cast<double>(nslabs);

    std::ofstream jout = DofDump::openDataFile(prefix+".json",16);
    jout << "{" << std::endl;
    jout << "  \"path\": \"" << (exact ? "exact" : "fd") << "\"," << std::endl;
    jout << "  \"dim\": " << ctrl_dim << "," << std::endl;
    jout << "  \"n_slabs\": " << nslabs << "," << std::endl;
    jout << "  \"n_spatial\": " << nspatial << "," << std::endl;
    jout << "  \"first_column\": " << first_col << "," << std::endl;
    jout << "  \"num_columns\": " << ncols << "," << std::endl;
    jout << "  \"control_columns_only\": " << (ctrl_only ? "true" : "false") << "," << std::endl;
    jout << "  \"fd_step\": " << fd_step << "," << std::endl;
    jout << "  \"initial_time\": " << t0 << "," << std::endl;
    jout << "  \"final_time\": " << tf << "," << std::endl;
    jout << "  \"dt\": " << dt << std::endl;
    jout << "}" << std::endl;
    jout.close();

    std::ofstream sout = DofDump::openDataFile("slabs.csv",16);
    sout << "slab,t_apply" << std::endl;
    for (int t=0; t<nslabs; ++t) { sout << t << "," << t0 + (t+1)*dt << std::endl; }
    sout.close();
  }

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

  if (comm_->getRank() == 0) {
    if (eret != Anasazi::Converged) {
      std::cout << "[HESS-DUMP] WARNING: Anasazi did not converge; "
                << sol.numVecs << " of " << neig << " eigenpairs returned" << std::endl;
    }
    std::ofstream eout = DofDump::openDataFile(prefix+"_eig.csv",16);
    eout << std::scientific;
    eout << "index,real,imag" << std::endl;
    for (int i=0; i<sol.numVecs; ++i) {
      eout << i << "," << sol.Evals[i].realpart << "," << sol.Evals[i].imagpart << std::endl;
    }
    eout.close();
    std::cout << "[HESS-DUMP] wrote " << prefix << "_eig.csv ("
              << sol.numVecs << " eigenvalues)" << std::endl;
  }

  if (sol.numVecs > 0 && !sol.Evecs.is_null()) {
    MMWriter::writeDenseFile(prefix+"_evec.mtx",*(sol.Evecs),"reduced Hessian eigenvectors",
                             "row i = line i of " + prefix + "_control.map");
    MMWriter::writeMapFile(prefix+"_control.map",*(Hop->getDomainMap()));
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::writeTangentTrace(const int & col, const int & slab) {

  const int rank = comm_->getRank();

  // norm2 is collective, so every rank walks the trajectory while only rank 0 writes.
  std::ofstream tout;
  if (rank == 0) {
    tout = DofDump::openDataFile("tangent_trace.csv",16);
    tout << std::scientific;
    tout << "column,slab,set,step,time,norm" << std::endl;
  }

  for (size_t set=0; set<postproc_->incr_soln.size(); ++set) {
    std::vector<std::vector<Teuchos::RCP<LA_MultiVector> > > wd
      = postproc_->incr_soln[set]->extractAllData();
    std::vector<std::vector<ScalarT> > wt = postproc_->incr_soln[set]->extractAllTimes();
    for (size_t idx=0; idx<wd.size(); ++idx) {
      for (size_t k=0; k<wd[idx].size(); ++k) {
        Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> nrm(1);
        wd[idx][k]->norm2(nrm());
        if (rank == 0) {
          tout << col << "," << slab << "," << set << "," << k << ","
               << wt[idx][k] << "," << nrm[0] << std::endl;
        }
      }
    }
  }

  if (rank == 0) {
    tout.close();
    std::cout << "[HESS-DUMP] wrote tangent_trace.csv for column " << col
              << " (slab " << slab << ")" << std::endl;
  }
}

// ========================================================================================
// ========================================================================================

void AnalysisManager::dumpDofData(const string & control_blocks, bool write_files,
                                  std::vector<int> & ctrl_mask) {

  Teuchos::RCP<DiscretizationInterface> disc = postproc_->disc;
  const size_t nblocks = disc->block_names.size();
  const int dim = disc->dimension;
  const int rank = comm_->getRank();

  // A zero-weight term contributes nothing to the Hessian, so it does not mark a region.
  std::vector<int> is_obj(nblocks,0), is_reg(nblocks,0), is_ctrl(nblocks,0);
  for (size_t r=0; r<postproc_->objectives.size(); ++r) {
    const size_t b = postproc_->objectives[r].block;
    if (postproc_->objectives[r].weight != 0.0) { is_obj[b] = 1; }
    for (size_t k=0; k<postproc_->objectives[r].regularizations.size(); ++k) {
      if (postproc_->objectives[r].regularizations[k].weight != 0.0) {
        is_reg[postproc_->objectives[r].regularizations[k].block] = 1;
      }
    }
  }

  if (control_blocks.empty()) {
    is_ctrl = is_reg;
  }
  else {
    for (size_t b=0; b<nblocks; ++b) {
      if (control_blocks.find(disc->block_names[b]) != string::npos) { is_ctrl[b] = 1; }
    }
  }

  // Control space
  for (size_t p=0; p<params_->discretized_param_names.size(); ++p) {

    const string pname = params_->discretized_param_names[p];
    const int fieldnum = params_->paramDOF->getFieldNum(pname);
    basis_RCP pbasis = params_->discretized_param_basis[params_->discretized_param_usebasis[p]];

    Teuchos::RCP<ParameterManager<SolverNode> > pm = params_;
    DofDump::GidLookup lookup =
      [pm,disc,fieldnum](size_t block, size_t e, std::vector<std::pair<int,GO> > & out) {
        std::vector<GO> gids;
        pm->paramDOF->getElementGIDs(disc->my_elements[block](e),gids,disc->block_names[block]);
        const std::vector<int> & offs
          = pm->paramDOF->getGIDFieldOffsets(disc->block_names[block],fieldnum);
        for (size_t d=0; d<offs.size(); ++d) {
          out.push_back(std::make_pair(static_cast<int>(d),gids[offs[d]]));
        }
      };

    ScalarT scale = 0.0;
    std::map<GO,DofDump::Record> local = DofDump::collect(comm_,disc,pbasis,lookup,nblocks,dim,scale);
    std::map<GO,DofDump::Record> all = DofDump::gatherToRoot(comm_,local,nblocks);

    if (rank == 0 && p == 0) {
      // Sorted global-id order, matching the Hessian column index.
      ctrl_mask.assign(all.size(),0);
      size_t k = 0;
      for (std::map<GO,DofDump::Record>::const_iterator it=all.begin(); it!=all.end(); ++it, ++k) {
        for (size_t b=0; b<nblocks; ++b) {
          if (it->second.blocks[b] == 1 && is_ctrl[b] == 1) { ctrl_mask[k] = 1; }
        }
      }
    }

    if (rank == 0 && write_files) {
      const string fname = "dofs_" + pname + ".csv";
      DofDump::writeTable(all,is_ctrl,is_obj,scale,fname);
      std::cout << "[DOF-DUMP] " << pname << ": " << all.size() << " global DoFs, "
                << params_->discretized_param_basis_types[p] << " order "
                << params_->discretized_param_basis_orders[p]
                << " -> " << fname << std::endl;
    }
  }

  if (!write_files) { return; }

  // State spaces
  for (size_t set=0; set<disc->physics->set_names.size(); ++set) {
    for (size_t v=0; v<disc->physics->var_list[set][0].size(); ++v) {

      const string vname = disc->physics->var_list[set][0][v];

      // The variable list can differ by block, so locate it per block by name.
      std::vector<int> vb(nblocks,-1);
      for (size_t b=0; b<nblocks; ++b) {
        for (size_t k=0; k<disc->physics->var_list[set][b].size(); ++k) {
          if (disc->physics->var_list[set][b][k] == vname) { vb[b] = static_cast<int>(k); }
        }
      }
      if (vb[0] < 0) { continue; }

      basis_RCP vbasis
        = disc->basis_pointers[0][disc->physics->getUniqueIndex(static_cast<int>(set),0,vname)];

      DofDump::GidLookup lookup =
        [disc,set,vb](size_t block, size_t e, std::vector<std::pair<int,GO> > & out) {
          if (vb[block] < 0) { return; }
          const std::vector<int> & offs = disc->offsets[set][block][vb[block]];
          const LO eid = disc->my_elements[block](e);
          for (size_t d=0; d<offs.size(); ++d) {
            out.push_back(std::make_pair(static_cast<int>(d),
                          disc->dof_owned_and_shared[set](disc->dof_lids[set](eid,offs[d]))));
          }
        };

      ScalarT scale = 0.0;
      std::map<GO,DofDump::Record> local = DofDump::collect(comm_,disc,vbasis,lookup,nblocks,dim,scale);
      std::map<GO,DofDump::Record> all = DofDump::gatherToRoot(comm_,local,nblocks);

      if (rank == 0) {
        std::stringstream ss;
        ss << "dofs_set" << set << "_" << vname << ".csv";
        DofDump::writeTable(all,is_ctrl,is_obj,scale,ss.str());
        std::cout << "[DOF-DUMP] " << vname << " (set " << set << "): " << all.size()
                  << " global DoFs, " << disc->physics->types[set][0][vb[0]] << " order "
                  << disc->physics->orders[set][0][vb[0]]
                  << " -> " << ss.str() << std::endl;
      }
    }
  }

  std::vector<ScalarT> zmin, zmax;
  DofDump::blockExtentsZ(comm_,disc,nblocks,dim,zmin,zmax);

  if (rank == 0) {
    std::ofstream rout = DofDump::openDataFile("regions.csv",12);
    rout << "block_index,block_name,zmin,zmax,is_objective,is_regularization,is_control" << std::endl;
    for (size_t b=0; b<nblocks; ++b) {
      rout << b << "," << disc->block_names[b] << "," << zmin[b] << "," << zmax[b]
           << "," << is_obj[b] << "," << is_reg[b] << "," << is_ctrl[b] << std::endl;
    }
    rout.close();
    std::cout << "[DOF-DUMP] wrote regions.csv" << std::endl;
  }
}
