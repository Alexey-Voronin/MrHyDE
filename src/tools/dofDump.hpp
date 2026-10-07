/***********************************************************************
Per-DoF geometry tables for the Hessian column dump: coordinates, edge tangents or
face normals, and the element blocks each global DoF touches.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_DOF_DUMP_HPP
#define MRHYDE_DOF_DUMP_HPP

#include "trilinos.hpp"
#include "preferences.hpp"
#include "meshInterface.hpp"
#include "discretizationInterface.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace MrHyDE {

namespace DofDump {

  // One global DoF with enough geometry to place it on the mesh.
  struct Record {
    ScalarT pt[3];
    ScalarT dir[3];
    ScalarT measure;
    ScalarT max_offset;
    int n_elem;
    int orient;
    int orient_conflict;
    std::vector<int> blocks;

    Record() : measure(0.0), max_offset(0.0), n_elem(0), orient(0), orient_conflict(0) {
      for (int d=0; d<3; ++d) { pt[d] = 0.0; dir[d] = 0.0; }
    }
  };

  // Inserts r under gid, or folds it into the record of a DoF already seen.
  inline void mergeRecord(std::map<GO,Record> & recs, const GO gid, const Record & r) {
    std::map<GO,Record>::iterator it = recs.find(gid);
    if (it == recs.end()) {
      recs[gid] = r;
      return;
    }
    Record & m = it->second;
    ScalarT off = 0.0;
    for (int d=0; d<3; ++d) {
      const ScalarT del = m.pt[d] - r.pt[d];
      off += del*del;
    }
    m.max_offset = std::max(m.max_offset,std::max(std::sqrt(off),r.max_offset));
    m.n_elem += r.n_elem;
    // Shared edges must agree: a mismatch means the space is non-conforming.
    if ((m.orient != 0 && r.orient != 0 && m.orient != r.orient) || r.orient_conflict == 1) {
      m.orient_conflict = 1;
    }
    for (size_t b=0; b<m.blocks.size(); ++b) {
      if (r.blocks[b] == 1) { m.blocks[b] = 1; }
    }
  }

  // Fills (basis dof ordinal, global id) for one element, identified by block and block-local index.
  typedef std::function<void(size_t, size_t, std::vector<std::pair<int,GO> > &)> GidLookup;

  // Lexicographic order on coordinates, used to orient edge tangents.
  inline bool lexLess(const ScalarT * a, const ScalarT * b, int dim) {
    for (int d=0; d<dim; ++d) {
      if (a[d] < b[d]) { return true; }
      if (a[d] > b[d]) { return false; }
    }
    return false;
  }

  // Places one basis DoF from the subcell its Intrepid2 tag names.
  inline void placeDof(const topo_RCP & cellTopo, const basis_RCP & basis, int dof,
                       const Kokkos::View<ScalarT**,HostDevice> & nodes, int dim,
                       const int * edge_ort,
                       ScalarT pt[3], ScalarT dir[3], ScalarT & measure, int & orient) {

    for (int d=0; d<3; ++d) { pt[d] = 0.0; dir[d] = 0.0; }
    measure = 0.0;
    orient = 0;

    auto tag = basis->getDofTag(dof);
    const int sdim = tag(0);
    const int sord = tag(1);

    if (sdim == 0) {
      for (int d=0; d<dim; ++d) { pt[d] = nodes(sord,d); }
      measure = 1.0;
    }
    else if (sdim == 1) {
      const unsigned na = cellTopo->getNodeMap(1,sord,0);
      const unsigned nb = cellTopo->getNodeMap(1,sord,1);
      ScalarT pa[3] = {0.0,0.0,0.0}, pb[3] = {0.0,0.0,0.0};
      for (int d=0; d<dim; ++d) { pa[d] = nodes(na,d); pb[d] = nodes(nb,d); }
      // Sign of the reference edge direction against the lexicographic one we report.
      const int ref_sign = lexLess(pb,pa,dim) ? -1 : 1;
      if (ref_sign < 0) { std::swap(pa,pb); }
      for (int d=0; d<dim; ++d) {
        pt[d] = 0.5*(pa[d] + pb[d]);
        dir[d] = pb[d] - pa[d];
        measure += dir[d]*dir[d];
      }
      measure = std::sqrt(measure);
      if (measure > 0.0) {
        for (int d=0; d<dim; ++d) { dir[d] /= measure; }
      }
      // Intrepid2 flips the lowest-order HCURL basis when the edge's global nodes are reversed.
      orient = (edge_ort != 0 && edge_ort[sord] != 0) ? -ref_sign : ref_sign;
    }
    else if (sdim == 2) {
      const unsigned nv = cellTopo->getNodeCount(2,sord);
      for (unsigned k=0; k<nv; ++k) {
        const unsigned nk = cellTopo->getNodeMap(2,sord,k);
        for (int d=0; d<dim; ++d) { pt[d] += nodes(nk,d)/static_cast<ScalarT>(nv); }
      }
      if (nv == 4 && dim == 3) {
        // Quad area and normal from the diagonals, exact for a planar face.
        const unsigned n0 = cellTopo->getNodeMap(2,sord,0);
        const unsigned n1 = cellTopo->getNodeMap(2,sord,1);
        const unsigned n2 = cellTopo->getNodeMap(2,sord,2);
        const unsigned n3 = cellTopo->getNodeMap(2,sord,3);
        ScalarT d1[3], d2[3];
        for (int d=0; d<3; ++d) {
          d1[d] = nodes(n2,d) - nodes(n0,d);
          d2[d] = nodes(n3,d) - nodes(n1,d);
        }
        dir[0] = d1[1]*d2[2] - d1[2]*d2[1];
        dir[1] = d1[2]*d2[0] - d1[0]*d2[2];
        dir[2] = d1[0]*d2[1] - d1[1]*d2[0];
        ScalarT len = 0.0;
        for (int d=0; d<3; ++d) { len += dir[d]*dir[d]; }
        len = std::sqrt(len);
        measure = 0.5*len;
        if (len > 0.0) {
          for (int d=0; d<3; ++d) { dir[d] /= len; }
        }
        // Element-independent sign so the two elements sharing a face agree.
        for (int d=0; d<3; ++d) {
          if (dir[d] > 1.0e-12) { break; }
          if (dir[d] < -1.0e-12) {
            for (int dd=0; dd<3; ++dd) { dir[dd] = -dir[dd]; }
            break;
          }
        }
      }
    }
    else {
      ScalarT lo[3], hi[3];
      for (int d=0; d<dim; ++d) { lo[d] = nodes(0,d); hi[d] = nodes(0,d); }
      for (size_type k=0; k<nodes.extent(0); ++k) {
        for (int d=0; d<dim; ++d) {
          lo[d] = std::min(lo[d], nodes(k,d));
          hi[d] = std::max(hi[d], nodes(k,d));
        }
      }
      measure = 1.0;
      for (int d=0; d<dim; ++d) {
        pt[d] = 0.5*(lo[d] + hi[d]);
        measure *= (hi[d] - lo[d]);
      }
    }
  }

  // Host copy of the nodes of every element this rank owns in one block.
  inline auto hostNodes(const Teuchos::RCP<DiscretizationInterface> & disc, size_t block) {
    const size_t nelem = disc->my_elements[block].extent(0);
    Kokkos::View<LO*,AssemblyDevice> eids("block local element ids",nelem);
    auto eids_host = Kokkos::create_mirror_view(eids);
    for (size_t e=0; e<nelem; ++e) { eids_host(e) = static_cast<LO>(e); }
    Kokkos::deep_copy(eids,eids_host);

    DRV allnodes = disc->getMyNodes(block,eids);
    auto allnodes_host = Kokkos::create_mirror_view(allnodes);
    Kokkos::deep_copy(allnodes_host,allnodes);
    return allnodes_host;
  }

  // Walks every owned element of a DoF space and merges shared DoFs by global id.
  inline std::map<GO,Record> collect(const Teuchos::RCP<MpiComm> & comm,
                                     const Teuchos::RCP<DiscretizationInterface> & disc,
                                     const basis_RCP & basis,
                                     const GidLookup & lookup,
                                     size_t numblocks, int dim,
                                     ScalarT & coord_scale) {

    std::map<GO,Record> recs;
    const ScalarT big = std::numeric_limits<ScalarT>::max();
    ScalarT lo[3] = {big,big,big}, hi[3] = {-big,-big,-big};

    std::vector<std::pair<int,GO> > eldofs;

    for (size_t block=0; block<numblocks; ++block) {

      const size_t nelem = disc->my_elements[block].extent(0);
      if (nelem == 0) { continue; }

      topo_RCP cellTopo = disc->mesh->getCellTopology(disc->block_names[block]);

      auto allnodes_host = hostNodes(disc,block);

      const size_type nnodes = allnodes_host.extent(1);
      Kokkos::View<ScalarT**,HostDevice> nodes("element nodes",nnodes,dim);
      const int numedge = static_cast<int>(cellTopo->getSubcellCount(1));
      std::vector<int> edge_ort(numedge,0);

      for (size_t e=0; e<nelem; ++e) {

        const LO stkid = disc->my_elements[block](e);
        if (numedge > 0 && static_cast<int>(disc->panzer_orientations.extent(0)) > stkid) {
          disc->panzer_orientations(stkid).getEdgeOrientation(&edge_ort[0],numedge);
        }

        for (size_type k=0; k<nnodes; ++k) {
          for (int d=0; d<dim; ++d) {
            nodes(k,d) = allnodes_host(e,k,d);
            lo[d] = std::min(lo[d],nodes(k,d));
            hi[d] = std::max(hi[d],nodes(k,d));
          }
        }

        eldofs.clear();
        lookup(block,e,eldofs);

        for (size_t j=0; j<eldofs.size(); ++j) {
          Record r;
          placeDof(cellTopo,basis,eldofs[j].first,nodes,dim,
                   numedge > 0 ? &edge_ort[0] : 0,r.pt,r.dir,r.measure,r.orient);
          r.n_elem = 1;
          r.blocks.assign(numblocks,0);
          r.blocks[block] = 1;
          mergeRecord(recs,eldofs[j].second,r);
        }
      }
    }

    // Global box, so the multi_position tolerance does not depend on the decomposition.
    ScalarT lo_all[3], hi_all[3];
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_MIN,3,lo,lo_all);
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,3,hi,hi_all);

    coord_scale = 0.0;
    for (int d=0; d<dim; ++d) {
      const ScalarT ext = hi_all[d] - lo_all[d];
      coord_scale += ext*ext;
    }
    coord_scale = std::sqrt(coord_scale);

    return recs;
  }

  // Collects every rank's records on rank 0, merging duplicates the same way collect does.
  inline std::map<GO,Record> gatherToRoot(const Teuchos::RCP<MpiComm> & comm,
                                          const std::map<GO,Record> & local,
                                          size_t numblocks) {

    const int nranks = comm->getSize();
    const int myrank = comm->getRank();
    const int nb = static_cast<int>(numblocks);

    std::map<GO,Record> merged;

    std::vector<GO> gids;
    std::vector<ScalarT> vals;
    std::vector<int> ints;
    for (std::map<GO,Record>::const_iterator it=local.begin(); it!=local.end(); ++it) {
      gids.push_back(it->first);
      for (int d=0; d<3; ++d) { vals.push_back(it->second.pt[d]); }
      for (int d=0; d<3; ++d) { vals.push_back(it->second.dir[d]); }
      vals.push_back(it->second.measure);
      vals.push_back(it->second.max_offset);
      ints.push_back(it->second.n_elem);
      ints.push_back(it->second.orient);
      ints.push_back(it->second.orient_conflict);
      for (int b=0; b<nb; ++b) { ints.push_back(it->second.blocks[b]); }
    }

    const int nloc = static_cast<int>(gids.size());

    std::vector<int> counts(nranks,0);
    Teuchos::gather<int,int>(&nloc,1,&counts[0],1,0,*comm);

    std::vector<int> cval(nranks,0), cint(nranks,0);
    std::vector<int> dgid(nranks,0), dval(nranks,0), dint(nranks,0);
    int ntot = 0, vtot = 0, itot = 0;
    for (int r=0; r<nranks; ++r) {
      cval[r] = 8*counts[r];
      cint[r] = (3+nb)*counts[r];
      dgid[r] = ntot;  ntot += counts[r];
      dval[r] = vtot;  vtot += cval[r];
      dint[r] = itot;  itot += cint[r];
    }

    // MPI ignores the receive side off root and the send side when the count is zero,
    // but every buffer still has to be addressable.
    gids.resize(std::max<size_t>(gids.size(),1));
    vals.resize(std::max<size_t>(vals.size(),1));
    ints.resize(std::max<size_t>(ints.size(),1));
    std::vector<GO> rgids(std::max(ntot,1));
    std::vector<ScalarT> rvals(std::max(vtot,1));
    std::vector<int> rints(std::max(itot,1));

    Teuchos::gatherv<int,GO>(&gids[0],nloc,&rgids[0],&counts[0],&dgid[0],0,*comm);
    Teuchos::gatherv<int,ScalarT>(&vals[0],8*nloc,&rvals[0],&cval[0],&dval[0],0,*comm);
    Teuchos::gatherv<int,int>(&ints[0],(3+nb)*nloc,&rints[0],&cint[0],&dint[0],0,*comm);

    if (myrank != 0) { return merged; }

    for (int i=0; i<ntot; ++i) {
      Record r;
      for (int d=0; d<3; ++d) { r.pt[d] = rvals[8*i+d]; r.dir[d] = rvals[8*i+3+d]; }
      r.measure = rvals[8*i+6];
      r.max_offset = rvals[8*i+7];
      r.n_elem = rints[(3+nb)*i];
      r.orient = rints[(3+nb)*i+1];
      r.orient_conflict = rints[(3+nb)*i+2];
      r.blocks.assign(numblocks,0);
      for (int b=0; b<nb; ++b) { r.blocks[b] = rints[(3+nb)*i+3+b]; }
      mergeRecord(merged,rgids[i],r);
    }

    return merged;
  }

  // Rank 0 writes every sidecar, so this throw is deliberately not collective.
  inline std::ofstream openDataFile(const std::string & name, int prec) {
    std::ofstream f(name.c_str());
    TEUCHOS_TEST_FOR_EXCEPTION(!f.is_open(),std::runtime_error,
                               "Error: could not open the data file: " + name);
    f.precision(prec);
    return f;
  }

  // Writes one row per global DoF in sorted global-id order.
  inline void writeTable(const std::map<GO,Record> & recs,
                         const std::vector<int> & is_control,
                         const std::vector<int> & is_observation,
                         ScalarT coord_scale,
                         const std::string & filename) {

    std::ofstream fout = openDataFile(filename,12);
    fout << "idx,gid,x,y,z,dir_x,dir_y,dir_z,measure,n_elem,blocks,"
         << "in_control,in_observation,multi_position,orient,orient_conflict" << std::endl;

    const ScalarT tol = 1.0e-6*coord_scale;
    size_t idx = 0;

    for (std::map<GO,Record>::const_iterator it=recs.begin(); it!=recs.end(); ++it) {

      const Record & r = it->second;
      int ctrl = 0, obs = 0;
      std::string blockstr(r.blocks.size(),'0');
      for (size_t b=0; b<r.blocks.size(); ++b) {
        if (r.blocks[b] == 0) { continue; }
        blockstr[b] = '1';
        if (is_control[b] == 1) { ctrl = 1; }
        if (is_observation[b] == 1) { obs = 1; }
      }

      fout << idx << "," << it->first;
      for (int d=0; d<3; ++d) { fout << "," << r.pt[d]; }
      for (int d=0; d<3; ++d) { fout << "," << r.dir[d]; }
      fout << "," << r.measure << "," << r.n_elem << "," << blockstr
           << "," << ctrl << "," << obs
           << "," << (r.max_offset > tol ? 1 : 0)
           << "," << r.orient << "," << r.orient_conflict << std::endl;
      ++idx;
    }

    fout.close();
  }

  // Per-block z extent, taken from the node coordinates rather than from the mesh deck.
  // Reduced across ranks; one block can span several subdomains.
  inline void blockExtentsZ(const Teuchos::RCP<MpiComm> & comm,
                            const Teuchos::RCP<DiscretizationInterface> & disc,
                            size_t numblocks, int dim,
                            std::vector<ScalarT> & zmin, std::vector<ScalarT> & zmax) {

    if (numblocks == 0) { return; }

    const ScalarT big = std::numeric_limits<ScalarT>::max();
    zmin.assign(numblocks,big);
    zmax.assign(numblocks,-big);

    for (size_t block=0; block<numblocks; ++block) {

      const size_t nelem = disc->my_elements[block].extent(0);
      if (nelem == 0) { continue; }

      auto allnodes_host = hostNodes(disc,block);

      ScalarT lo = allnodes_host(0,0,dim-1), hi = lo;
      for (size_type e=0; e<allnodes_host.extent(0); ++e) {
        for (size_type k=0; k<allnodes_host.extent(1); ++k) {
          lo = std::min(lo,static_cast<ScalarT>(allnodes_host(e,k,dim-1)));
          hi = std::max(hi,static_cast<ScalarT>(allnodes_host(e,k,dim-1)));
        }
      }
      zmin[block] = lo;
      zmax[block] = hi;
    }

    std::vector<ScalarT> rmin(numblocks,0.0), rmax(numblocks,0.0);
    const int nb = static_cast<int>(numblocks);
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_MIN,nb,&zmin[0],&rmin[0]);
    Teuchos::reduceAll(*comm,Teuchos::REDUCE_MAX,nb,&zmax[0],&rmax[0]);
    for (size_t b=0; b<numblocks; ++b) {
      // A block no rank owns keeps the sentinel; report it as empty instead.
      zmin[b] = (rmin[b] ==  big) ? 0.0 : rmin[b];
      zmax[b] = (rmax[b] == -big) ? 0.0 : rmax[b];
    }
  }

} // namespace DofDump

} // namespace MrHyDE

#endif
