/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#include <Ifpack2_Factory.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Tpetra_ComputeGatherMap.hpp>
#include <Tpetra_TieBreak.hpp>
#include <Zoltan2_XpetraCrsMatrixAdapter.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_PartitioningHelpers.hpp>

namespace {

// A on the rows of rows_map and the columns where keep (on A's domain map) is nonzero.
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> >
restrictParamCrs(const Tpetra::CrsMatrix<ScalarT,LO,GO,Node> & A,
                 const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & rows_map,
                 const Tpetra::MultiVector<ScalarT,LO,GO,Node> & keep,
                 const Teuchos::RCP<const Tpetra::Map<LO,GO,Node> > & domain_map) {
  using Map = Tpetra::Map<LO,GO,Node>;
  using device = typename Node::device_type;
  const Teuchos::RCP<const Map> src_rows = A.getRowMap();
  const Teuchos::RCP<const Map> src_cols = A.getColMap();

  Tpetra::MultiVector<ScalarT,LO,GO,Node> keep_col(src_cols, 1);
  keep_col.doImport(keep, Tpetra::Import<LO,GO,Node>(A.getDomainMap(), src_cols), Tpetra::INSERT);
  const LO ncols = static_cast<LO>(src_cols->getLocalNumElements());
  Kokkos::View<LO*,device> out_col("param_restrict_col", ncols);
  auto out_col_h = Kokkos::create_mirror_view(out_col);
  std::vector<GO> col_gids;
  {
    auto kc = keep_col.getLocalViewHost(Tpetra::Access::ReadOnly);
    for (LO c = 0; c < ncols; ++c) {
      out_col_h(c) = (kc(c,0) != 0.0) ? static_cast<LO>(col_gids.size()) : LO(-1);
      if (kc(c,0) != 0.0) col_gids.push_back(src_cols->getGlobalElement(c));
    }
  }
  Kokkos::deep_copy(out_col, out_col_h);
  const Teuchos::RCP<const Map> cols_map = Teuchos::rcp(new Map(
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), Teuchos::ArrayView<const GO>(col_gids),
    src_cols->getIndexBase(), src_cols->getComm()));

  const LO nrows = static_cast<LO>(rows_map->getLocalNumElements());
  Kokkos::View<LO*,device> src_row("param_restrict_row", nrows);
  auto src_row_h = Kokkos::create_mirror_view(src_row);
  for (LO i = 0; i < nrows; ++i) {
    src_row_h(i) = src_rows->getLocalElement(rows_map->getGlobalElement(i));
  }
  Kokkos::deep_copy(src_row, src_row_h);

  block_prec::detail::RowsFromTable<Node> rows;
  rows.srcRow = src_row;
  block_prec::detail::ColumnsFromTable<Node> cols;
  cols.outCol = out_col;
  return block_prec::detail::filterCopyCrs<Node>(A, rows_map, cols_map, domain_map, rows_map, rows, cols);
}

template<class Node>
Teuchos::RCP<Xpetra::Matrix<ScalarT,LO,GO,Node> >
wrapParamCrs(const Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > & A) {
  Teuchos::RCP<Xpetra::CrsMatrix<ScalarT,LO,GO,Node> > xc =
    Teuchos::rcp(new Xpetra::TpetraCrsMatrix<ScalarT,LO,GO,Node>(A));
  return Teuchos::rcp(new Xpetra::CrsMatrixWrap<ScalarT,LO,GO,Node>(xc));
}

// Control rows per rank when the deck sets no rank count.
constexpr GO param_riesz_rows_per_rank = 2000;

// Ranks to hold the control rows, from global values only so every rank agrees; 0 keeps the mesh layout.
inline int paramRieszRanks(const int setting, const GO nrows, const int nranks) {
  if (setting < 0 || nranks == 1 || nrows == 0) return 0;
  const GO want = (setting > 0) ? GO(setting)
                                : (nrows + param_riesz_rows_per_rank / 2) / param_riesz_rows_per_rank;
  return static_cast<int>(std::min<GO>(std::min<GO>(nranks, nrows), std::max<GO>(1, want)));
}

// A GID listed by several ranks goes to the lowest of them.
struct LowestRankTieBreak : public Tpetra::Details::TieBreak<LO,GO> {
  std::size_t selectedIndex(GO, const std::vector<std::pair<int,LO> > & pid_lid) const override {
    return static_cast<std::size_t>(std::min_element(pid_lid.begin(), pid_lid.end()) - pid_lid.begin());
  }
};

// A's rows on to_new's target map, left globally indexed so fillComplete finds the column owners in the new domain map.
template<class Node>
Teuchos::RCP<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> >
importParamRows(const Tpetra::CrsMatrix<ScalarT,LO,GO,Node> & A, const Tpetra::Import<LO,GO,Node> & to_new) {
  auto B = Teuchos::rcp(new Tpetra::CrsMatrix<ScalarT,LO,GO,Node>(to_new.getTargetMap(), A.getGlobalMaxNumRowEntries()));
  B->doImport(A, to_new, Tpetra::INSERT);
  return B;
}

// A's rows split into k parts by Zoltan's hypergraph partitioner on A's sparsity; part p goes to rank p.
template<class Node>
Teuchos::RCP<const Tpetra::Map<LO,GO,Node> >
partitionRows(const Teuchos::RCP<const Tpetra::CrsMatrix<ScalarT,LO,GO,Node> > & A, const int k) {
  using Adapter = Zoltan2::XpetraCrsMatrixAdapter<Tpetra::CrsMatrix<ScalarT,LO,GO,Node> >;
  Adapter adapter(A);
  Teuchos::ParameterList zlist;
  zlist.set("algorithm", "phg");
  zlist.set("num_global_parts", k);
  Teuchos::ParameterList & zoltan = zlist.sublist("zoltan_parameters");
  zoltan.set("DEBUG_LEVEL", "0");
  zoltan.set("REMAP", "0");
  Zoltan2::PartitioningProblem<Adapter> problem(&adapter, &zlist, A->getComm());
  problem.solve();
  Teuchos::ArrayRCP<GO> mine;
  Zoltan2::getImportList(problem.getSolution(), &adapter, mine);
  std::vector<GO> gids(mine.getRawPtr(), mine.getRawPtr() + mine.size());
  std::sort(gids.begin(), gids.end());
  return Teuchos::rcp(new Tpetra::Map<LO,GO,Node>(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
    Teuchos::ArrayView<const GO>(gids), A->getRowMap()->getIndexBase(), A->getComm()));
}

} // namespace

// ========================================================================================
// ========================================================================================

template<class Node>
Teuchos::RCP<MrHyDE_ParamRiesz> SolverManager<Node>::buildIterativeParamRiesz(const vector_RCP & assembled_diag) {

  debugger->print("**** Starting SolverManager::buildIterativeParamRiesz()");
  static Teuchos::RCP<Teuchos::Time> setuptimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::ParamRiesz::setup()");
  Teuchos::TimeMonitor localtimer(*setuptimer);

  using Map = Tpetra::Map<LO,GO,Node>;
  using CoordMV = Tpetra::MultiVector<typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node>;
  const Teuchos::RCP<const Map> pmap = paramMass->getRowMap();

  // The padded rows are diagonal, so the metric splits into the control block and a diagonal.
  std::vector<GO> live_gids;
  {
    auto d = assembled_diag->getLocalViewHost(Tpetra::Access::ReadOnly);
    for (size_t i = 0; i < d.extent(0); ++i) {
      if (d(i,0) != 0.0) live_gids.push_back(pmap->getGlobalElement(static_cast<LO>(i)));
    }
  }
  const Teuchos::RCP<const Map> live_map = Teuchos::rcp(new Map(
    Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), Teuchos::ArrayView<const GO>(live_gids),
    pmap->getIndexBase(), pmap->getComm()));
  matrix_RCP P_live = restrictParamCrs<Node>(*paramMass, live_map, *assembled_diag, live_map);

  // The control rows move to k ranks, cut along the couplings of P; k == 0 keeps the mesh layout.
  const int k = paramRieszRanks(param_riesz.ranks, static_cast<GO>(live_map->getGlobalNumElements()),
                                pmap->getComm()->getSize());
  Teuchos::RCP<const Map> edge_map = live_map;
  Teuchos::RCP<const Tpetra::Import<LO,GO,Node> > to_new;
  if (k > 0) {
    edge_map = (k == 1) ? Tpetra::Details::computeGatherMap(live_map, Teuchos::null) : partitionRows<Node>(P_live, k);
    to_new = Teuchos::rcp(new Tpetra::Import<LO,GO,Node>(live_map, edge_map));
    P_live = importParamRows<Node>(*P_live, *to_new);
    P_live->fillComplete(edge_map, edge_map);
  }
  if (verbosity >= 5) {
    const GO here = static_cast<GO>(edge_map->getLocalNumElements());
    const GO holds = (here > 0) ? 1 : 0;
    GO most = 0, holders = 0;
    Teuchos::reduceAll<int, GO>(*(pmap->getComm()), Teuchos::REDUCE_MAX, 1, &here, &most);
    Teuchos::reduceAll<int, GO>(*(pmap->getComm()), Teuchos::REDUCE_SUM, 1, &holds, &holders);
    if (Comm->getRank() == 0) {
      std::cout << "[RieszMap] control rows on " << holders << " of " << Comm->getSize()
                << " ranks, at most " << most << " per rank" << std::endl;
    }
  }

  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,Node> > prec;
  if (param_riesz.prec == "Jacobi") {
    Teuchos::ParameterList ilist;
    ilist.set("relaxation: type", "Jacobi");
    auto jacobi = Ifpack2::Factory::create<Tpetra::RowMatrix<ScalarT,LO,GO,Node> >("RELAXATION", P_live);
    jacobi->setParameters(ilist);
    jacobi->initialize();
    jacobi->compute();
    prec = jacobi;
  }
  else {
    matrix_RCP D0;
    Teuchos::RCP<CoordMV> coords;
    vector_RCP nullspace;
    this->buildParamGradient(D0, coords, nullspace);

    // The nodal space is the nodes a control edge touches.
    LA_MultiVector touched(D0->getColMap(), 1);
    {
      auto t = touched.getLocalViewHost(Tpetra::Access::ReadWrite);
      auto lcl = D0->getLocalMatrixHost();
      for (size_t i = 0; i < live_gids.size(); ++i) {
        const auto row = lcl.row(D0->getRowMap()->getLocalElement(live_gids[i]));
        for (LO k = 0; k < row.length; ++k) t(row.colidx(k), 0) = 1.0;
      }
    }
    const Teuchos::RCP<const Map> nodal_map = D0->getDomainMap();
    LA_MultiVector live_nodes(nodal_map, 1);
    live_nodes.doExport(touched, Tpetra::Export<LO,GO,Node>(D0->getColMap(), nodal_map), Tpetra::ADD);
    std::vector<GO> node_gids;
    {
      auto n = live_nodes.getLocalViewHost(Tpetra::Access::ReadOnly);
      for (size_t i = 0; i < n.extent(0); ++i) {
        if (n(i,0) != 0.0) node_gids.push_back(nodal_map->getGlobalElement(static_cast<LO>(i)));
      }
    }
    const Teuchos::RCP<const Map> live_node_map = Teuchos::rcp(new Map(
      Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), Teuchos::ArrayView<const GO>(node_gids),
      nodal_map->getIndexBase(), nodal_map->getComm()));
    matrix_RCP D0_live = restrictParamCrs<Node>(*D0, live_map, live_nodes, live_node_map);
    Teuchos::RCP<const Map> node_map = live_node_map;
    if (k > 0) {
      // Each control node goes to the lowest rank holding one of its edges.
      D0_live = importParamRows<Node>(*D0_live, *to_new);
      std::vector<GO> held;
      {
        typename LA_CrsMatrix::global_inds_host_view_type inds;
        typename LA_CrsMatrix::values_host_view_type vals;
        for (LO i = 0; i < static_cast<LO>(edge_map->getLocalNumElements()); ++i) {
          D0_live->getGlobalRowView(edge_map->getGlobalElement(i), inds, vals);
          held.insert(held.end(), inds.data(), inds.data() + inds.extent(0));
        }
      }
      std::sort(held.begin(), held.end());
      held.erase(std::unique(held.begin(), held.end()), held.end());
      const Teuchos::RCP<const Map> held_map = Teuchos::rcp(new Map(
        Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), Teuchos::ArrayView<const GO>(held),
        live_node_map->getIndexBase(), live_node_map->getComm()));
      node_map = Tpetra::createOneToOne<LO,GO,Node>(held_map, LowestRankTieBreak());
      D0_live->fillComplete(node_map, edge_map);
    }

    Teuchos::RCP<CoordMV> coords_live = Teuchos::rcp(new CoordMV(node_map, coords->getNumVectors()));
    coords_live->doImport(*coords, Tpetra::Import<LO,GO,Node>(nodal_map, node_map), Tpetra::INSERT);
    vector_RCP ns_live = Teuchos::rcp(new LA_MultiVector(edge_map, nullspace->getNumVectors()));
    ns_live->doImport(*nullspace, Tpetra::Import<LO,GO,Node>(pmap, edge_map), Tpetra::INSERT);

    // P vanishes on gradients except for its mass term, so P serves as RefMaxwell's mass matrices.
    Teuchos::RCP<Teuchos::ParameterList> mlist = Teuchos::getParametersFromXmlFile(param_riesz.xml);
    mlist->set("refmaxwell: disable addon", true);
    mlist->set("refmaxwell: use as preconditioner", true);
    using XMV = Xpetra::TpetraMultiVector<ScalarT,LO,GO,Node>;
    using XCoordMV = Xpetra::TpetraMultiVector<typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node>;
    Teuchos::RCP<Xpetra::MultiVector<ScalarT,LO,GO,Node> > xns = Teuchos::rcp(new XMV(ns_live));
    Teuchos::RCP<Xpetra::MultiVector<typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node> > xco =
      Teuchos::rcp(new XCoordMV(coords_live));
    auto refmaxwell = Teuchos::rcp(new MueLu::RefMaxwell<ScalarT,LO,GO,Node>(
      wrapParamCrs<Node>(P_live), wrapParamCrs<Node>(D0_live), wrapParamCrs<Node>(P_live), Teuchos::null,
      wrapParamCrs<Node>(P_live), xns, xco, *mlist, true));
    prec = Teuchos::rcp(new MueLu::TpetraOperator<ScalarT,LO,GO,Node>(
      Teuchos::rcp_static_cast<Xpetra::Operator<ScalarT,LO,GO,Node> >(refmaxwell)));
  }

  debugger->print("**** Finished SolverManager::buildIterativeParamRiesz()");
  return Teuchos::rcp(new MrHyDE_ParamRiesz(paramMass, P_live, diagParamMass, prec, param_riesz, verbosity));
}

// ========================================================================================
// D0 comes from auxiliary DOF managers because the parameter's own carries no orientations.
// ========================================================================================

template<class Node>
void SolverManager<Node>::buildParamGradient(matrix_RCP & D0,
                                            Teuchos::RCP<Tpetra::MultiVector<typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node> > & coords,
                                            vector_RCP & nullspace) {

  using Map = Tpetra::Map<LO,GO,Node>;
  using CoordMV = Tpetra::MultiVector<typename Teuchos::ScalarTraits<ScalarT>::coordinateType,LO,GO,Node>;
  const Teuchos::RCP<const Map> pmap = paramMass->getRowMap();
  const int porder = params->discretized_param_basis_orders[params->discretized_param_usebasis[0]];
  const std::string hgrad_name = "riesz_hgrad", hcurl_name = "riesz_hcurl";

  Teuchos::RCP<panzer::ConnManager> conn = mesh->getSTKConnManager();
  Teuchos::RCP<panzer::DOFManager> hgrad_dof = Teuchos::rcp(new panzer::DOFManager());
  hgrad_dof->setConnManager(conn, *(Comm->getRawMpiComm()));
  hgrad_dof->setOrientationsRequired(false);
  Teuchos::RCP<panzer::DOFManager> hcurl_dof = Teuchos::rcp(new panzer::DOFManager());
  hcurl_dof->setConnManager(conn, *(Comm->getRawMpiComm()));
  hcurl_dof->setOrientationsRequired(true);
  for (size_t block = 0; block < mesh->block_names.size(); ++block) {
    std::string block_name = mesh->block_names[block];
    topo_RCP cellTopo = mesh->getCellTopology(block_name);
    hgrad_dof->addField(block_name, hgrad_name, Teuchos::rcp(new panzer::Intrepid2FieldPattern(
      disc->getBasis(dimension, cellTopo, "HGRAD", 1))), panzer::FieldType::CG);
    hcurl_dof->addField(block_name, hcurl_name, Teuchos::rcp(new panzer::Intrepid2FieldPattern(
      disc->getBasis(dimension, cellTopo, "HCURL", porder))), panzer::FieldType::CG);
  }
  hgrad_dof->buildGlobalUnknowns();
  hcurl_dof->buildGlobalUnknowns();

  Teuchos::RCP<Thyra::LinearOpBase<ScalarT> > D0_thyra =
    panzer::buildInterpolation(conn, hgrad_dof, hcurl_dof, hgrad_name, hcurl_name,
                               Intrepid2::OPERATOR_GRAD, 1000, false, true, false);
  matrix_RCP D0_aux = Teuchos::rcp_dynamic_cast<LA_CrsMatrix>(
    Thyra::TpetraOperatorVectorExtraction<ScalarT,LO,GO,Node>::getTpetraOperator(D0_thyra), true);
  const Teuchos::RCP<const Map> aux_edge_map = D0_aux->getRangeMap();

  // aux2prm: auxiliary edge GID to parameter GID, matched through the element's field offsets.
  const int hcurl_num = hcurl_dof->getFieldNum(hcurl_name);
  std::unordered_map<GO,GO> aux2prm;
  GO conflictGid = -1;
  for (size_t b = 0; b < mesh->block_names.size(); ++b) {
    const std::string & block_name = mesh->block_names[b];
    const std::vector<int> aux_off = hcurl_dof->getGIDFieldOffsets(block_name, hcurl_num);
    const std::vector<int> prm_off = params->paramDOF->getGIDFieldOffsets(block_name, 0);
    TEUCHOS_TEST_FOR_EXCEPTION(aux_off.size() != prm_off.size(), std::runtime_error,
      "Riesz map: HCURL offset count mismatch on " + block_name +
      " (auxiliary=" + std::to_string(aux_off.size()) + ", parameter=" + std::to_string(prm_off.size()) + ").");
    for (size_t e = 0; e < disc->my_elements[b].extent(0); ++e) {
      const LO elem = disc->my_elements[b](e);
      std::vector<GO> aux_gids, prm_gids;
      hcurl_dof->getElementGIDs(elem, aux_gids, block_name);
      params->paramDOF->getElementGIDs(elem, prm_gids, block_name);
      for (size_t j = 0; j < aux_off.size(); ++j) {
        auto ins = aux2prm.emplace(aux_gids[aux_off[j]], prm_gids[prm_off[j]]);
        if (!ins.second && ins.first->second != prm_gids[prm_off[j]] && conflictGid < 0) {
          conflictGid = aux_gids[aux_off[j]];
        }
      }
    }
  }
  std::unordered_map<GO,LO> prm2aux_lid;
  GO unmappedGid = -1, nonOwnedGid = -1;
  for (LO lid = 0; lid < static_cast<LO>(aux_edge_map->getLocalNumElements()); ++lid) {
    auto it = aux2prm.find(aux_edge_map->getGlobalElement(lid));
    if (it == aux2prm.end()) {
      if (unmappedGid < 0) unmappedGid = aux_edge_map->getGlobalElement(lid);
      continue;
    }
    if (nonOwnedGid < 0 && !pmap->isNodeGlobalElement(it->second)) nonOwnedGid = it->second;
    prm2aux_lid[it->second] = lid;
  }
  GO bad[3] = {conflictGid, unmappedGid, nonOwnedGid}, worst[3] = {-1, -1, -1};
  Teuchos::reduceAll<int, GO>(*(pmap->getComm()), Teuchos::REDUCE_MAX, 3, bad, worst);
  TEUCHOS_TEST_FOR_EXCEPTION(worst[0] >= 0, std::runtime_error,
    "Riesz map: auxiliary edge GID " << worst[0] << " maps to two parameter GIDs.");
  TEUCHOS_TEST_FOR_EXCEPTION(worst[1] >= 0, std::runtime_error,
    "Riesz map: no parameter GID for auxiliary GID " << worst[1] << ".");
  TEUCHOS_TEST_FOR_EXCEPTION(worst[2] >= 0, std::runtime_error,
    "Riesz map: parameter GID " << worst[2] << " is not owned by the rank that "
    "owns its auxiliary row.");

  const LO n_rows = static_cast<LO>(pmap->getLocalNumElements());
  Kokkos::View<LO*, typename Node::device_type> srcRowLid("param_d0_row", n_rows);
  auto srcRowLidHost = Kokkos::create_mirror_view(srcRowLid);
  for (LO lid = 0; lid < n_rows; ++lid) {
    auto it = prm2aux_lid.find(pmap->getGlobalElement(lid));
    srcRowLidHost(lid) = (it == prm2aux_lid.end()) ? LO(-1) : it->second;
  }
  Kokkos::deep_copy(srcRowLid, srcRowLidHost);
  block_prec::detail::RowsFromTable<Node> rows;
  rows.srcRow = srcRowLid;
  D0 = block_prec::detail::filterCopyCrs<Node>(*D0_aux, pmap, D0_aux->getColMap(), D0_aux->getDomainMap(),
                                               pmap, rows, block_prec::detail::KeepAllEntries());

  // Node coordinates: each node's lowest-sum periodic image, the same on every rank.
  const Teuchos::RCP<const Map> nodal_map = D0->getDomainMap();
  std::map<GO, std::vector<double> > gid_to_coords;
  GO global_missing = 0;
  nullspace = this->buildEdgeNullspace(D0, hgrad_dof,
    [&](const size_t b, const LO elem, std::vector<GO> & gids) {
      std::vector<GO> prm_gids;
      params->paramDOF->getElementGIDs(elem, prm_gids, mesh->block_names[b]);
      gids.clear();
      for (const int off : params->paramDOF->getGIDFieldOffsets(mesh->block_names[b], 0)) gids.push_back(prm_gids[off]);
    }, global_missing, &gid_to_coords);
  TEUCHOS_TEST_FOR_EXCEPTION(global_missing > 0, std::runtime_error,
    "Riesz map: " << global_missing << " owned control edges touch no element this rank owns; "
    "the iterative RefMaxwell metric cannot build their nullspace. Use apply = exact, or repartition.");

  coords = Teuchos::rcp(new CoordMV(nodal_map, dimension));
  {
    GO missing_nodes = 0;
    for (LO lid = 0; lid < static_cast<LO>(nodal_map->getLocalNumElements()); ++lid) {
      if (gid_to_coords.find(nodal_map->getGlobalElement(lid)) == gid_to_coords.end()) ++missing_nodes;
    }
    GO global_missing_nodes = 0;
    Teuchos::reduceAll<int, GO>(*(nodal_map->getComm()), Teuchos::REDUCE_SUM, 1, &missing_nodes, &global_missing_nodes);
    TEUCHOS_TEST_FOR_EXCEPTION(global_missing_nodes > 0, std::runtime_error,
      "Riesz map: " << global_missing_nodes << " owned nodes have no coordinates from an "
      "owned element; the iterative RefMaxwell metric needs them. Use apply = exact, or repartition.");
  }
  fillFromLowestImage<LO,GO,Node>(gid_to_coords, *coords);
}
