/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for 
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov) 
************************************************************************/

#ifndef ROL_MILO_HPP
#define ROL_MILO_HPP

#include "ROL_StdVector.hpp"
#include "ROL_RiskVector.hpp"
#include "ROL_Objective.hpp"
#include "ROL_BoundConstraint.hpp"
#include "Teuchos_ParameterList.hpp"

#include "solverManager.hpp"
#include "postprocessManager.hpp"
#include "parameterManager.hpp"

//#include "ROL_RiskVector.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cctype>

//#include <random> //for normal noise...not sure if this is necessary...

namespace ROL {

  using namespace MrHyDE;
  
  template<class Real>
  class Objective_MILO : public Objective<Real> {
    
  private:
    enum class PrecondMode { Dual, Identity };
    
    Real noise_;                                            //standard deviation of normal additive noise to add to data (0 for now)
    Teuchos::RCP<SolverManager<SolverNode> > solver;                                     // Solver object for MILO (solves FWD, ADJ, computes gradient, etc.)
    Teuchos::RCP<PostprocessManager<SolverNode> > postproc;                              // Postprocessing object for MILO (write solution, computes response, etc.)
    Teuchos::RCP<ParameterManager<SolverNode> > params;
    PrecondMode precondMode_ = PrecondMode::Dual;
    bool didVectorContractCheck_ = false;

    // Riesz energy diagnostics: when enabled, log per-outer-iteration values
    // of g_d^T M g_d, g_d^T K g_d, z^T M z, z^T K z to a CSV file so we can
    // diagnose whether a static (M, K) weighting is optimal late in L-BFGS.
    bool rieszDiag_ = false;
    int rieszDiagLastIter_ = -1;
    Teuchos::RCP<std::ofstream> rieszDiagOut_;
    Teuchos::RCP<MrHyDE_OptVector> lastDualGrad_;
    bool lastDualGradFresh_ = false;

    Teuchos::RCP<Teuchos::Time> valuetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Objective::value()");
    Teuchos::RCP<Teuchos::Time> gradienttimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::Objective::gradient()");

    std::string toLower(std::string val) const {
      std::transform(val.begin(), val.end(), val.begin(),
                     [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
      return val;
    }

    bool isRootRank() const {
      return (solver != Teuchos::null && solver->Comm != Teuchos::null && solver->Comm->getRank() == 0);
    }

    Real maxCheckError(const std::vector<Real> &errs) const {
      Real maxErr = static_cast<Real>(0);
      for (size_t i = 0; i < errs.size(); ++i) {
        maxErr = std::max(maxErr, std::abs(errs[i]));
      }
      return maxErr;
    }

    void printCheckErrors(const std::string &label, const std::vector<Real> &errs) const {
      if (!isRootRank()) {
        return;
      }
      std::cout << "MrHyDE vector contract check (" << label << ") errors:";
      for (size_t i = 0; i < errs.size(); ++i) {
        std::cout << " e[" << i << "]=" << errs[i];
      }
      std::cout << std::endl;
    }

    void runVectorContractCheckOnce(const Vector<Real> &Params, const Vector<Real> &g) {
      if (didVectorContractCheck_) {
        return;
      }

      auto px = Params.clone();
      auto py = Params.clone();
      px->set(Params);
      py->set(Params);
      py->scale(static_cast<Real>(0.5));
      py->axpy(static_cast<Real>(1.0), *px);

      auto gx = g.clone();
      auto gy = g.clone();
      gx->set(g);
      gy->set(g);
      gy->scale(static_cast<Real>(0.5));
      gy->axpy(static_cast<Real>(1.0), *gx);

      const std::vector<Real> pErrs = Params.checkVector(*px, *py, false);
      const std::vector<Real> gErrs = g.checkVector(*gx, *gy, false);
      printCheckErrors("primal", pErrs);
      printCheckErrors("dual", gErrs);

      const Real pMax = maxCheckError(pErrs);
      const Real gMax = maxCheckError(gErrs);
      const Real maxErr = std::max(pMax, gMax);
      const Real tol = static_cast<Real>(1.0e-8);

      if (isRootRank()) {
        std::cout << "MrHyDE vector contract check: max_primal_err=" << pMax
                  << " max_dual_err=" << gMax
                  << " tol=" << tol << std::endl;
      }

      TEUCHOS_TEST_FOR_EXCEPTION(maxErr > tol, std::runtime_error,
                                 "MrHyDE vector contract check failed: max error = "
                                 << maxErr << ", tolerance = " << tol);

      didVectorContractCheck_ = true;
    }
    
  public:
    
    /*!
     \brief A constructor generating data
     */
    Objective_MILO(Teuchos::RCP<SolverManager<SolverNode> > solver_,
                   Teuchos::RCP<PostprocessManager<SolverNode> > postproc_,
                   Teuchos::RCP<ParameterManager<SolverNode> > & params_,
                   const std::string & hessVecPrecondMode = "dual") :
    solver(solver_), postproc(postproc_), params(params_) {
      const std::string precondMode = toLower(hessVecPrecondMode);

      if (precondMode == "identity") {
        precondMode_ = PrecondMode::Identity;
      }
      else if (precondMode == "dual") {
        precondMode_ = PrecondMode::Dual;
      }
      else if (isRootRank()) {
        std::cout << "MrHyDE warning: unrecognized HessVec Precond Mode '" << hessVecPrecondMode
                  << "'. Falling back to 'dual'." << std::endl;
      }
    } //end constructor
    
    ////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////
    
    Real value(const Vector<Real> & Params, Real & tol) override {
      
      Teuchos::TimeMonitor localtimer(*valuetimer);
      
      MrHyDE_OptVector Paramsp =
      Teuchos::dyn_cast<MrHyDE_OptVector >(const_cast<Vector<Real> &>(Params));
      
      params->updateParams(Paramsp);
      
      ScalarT val = 0.0;
      solver->forwardModel(val);
      
      //params->stashParams(); //dumping to file, for long runs...
      return val;
    }
    
    //! Compute gradient of objective function with respect to parameters
    void gradient(Vector<Real> &g, const Vector<Real> &Params, Real &tol) override {

      Teuchos::TimeMonitor localtimer(*gradienttimer);
      bool newparams = this->checkNewParams(Params);

      if (newparams) {
        MrHyDE_OptVector Paramsp =
        Teuchos::dyn_cast<MrHyDE_OptVector >(const_cast<Vector<Real> &>(Params));

        params->updateParams(Paramsp);
        ScalarT val = 0.0;
        solver->forwardModel(val);

      }
      MrHyDE_OptVector sens =
        Teuchos::dyn_cast<MrHyDE_OptVector>(const_cast<Vector<Real> &>(g));
      sens.setDualSpace(true);  // gradient is always in the dual space
      sens.zero();
      solver->adjointModel(sens);
      runVectorContractCheckOnce(Params, g);

      if (rieszDiag_) {
        if (lastDualGrad_.is_null()) {
          ROL::Ptr<Vector<Real>> clone = sens.clone();
          lastDualGrad_ = Teuchos::rcp_dynamic_cast<MrHyDE_OptVector>(clone);
        }
        if (!lastDualGrad_.is_null()) {
          lastDualGrad_->set(sens);
          lastDualGradFresh_ = true;
        }
      }
    }
    
    bool checkNewParams(const Vector<Real> &Params) {
      MrHyDE_OptVector curr_params = params->getCurrentVector();
      auto diff = curr_params.clone();
      diff->zero();
      diff->set(curr_params);
      diff->axpy(-1.0,Params);
      ScalarT dnorm = diff->norm();
      ScalarT refnorm = curr_params.norm();
      dnorm = (refnorm > 0.0) ? dnorm/refnorm : dnorm; // dnorm/ROL_EPSILON<Real>();
      ScalarT reltol = 1.0e-12;
      bool newparams = false;
      if (dnorm > reltol) {
        newparams = true;
      }
      return newparams;
    }

    void hessVec(Vector<Real> &hv, const Vector<Real> &v, const Vector<Real> &Params, Real &tol) override {
      const MrHyDE_OptVector &v_opt =
          Teuchos::dyn_cast<const MrHyDE_OptVector>(v);
      const MrHyDE_OptVector &x_opt =
          Teuchos::dyn_cast<const MrHyDE_OptVector>(Params);

      const Real zero(0), one(1);
      // euclidian is cheaper than metric-aware norm() and converges the same way.
      const Real vnorm = v_opt.euclideanNorm(); // .norm(); 
      if (vnorm == zero) { hv.zero(); return; }

      const Real xnorm = x_opt.euclideanNorm(); // .norm();
      const Real delta = std::max(one, xnorm) * tol;
      if (delta == zero) { hv.zero(); return; }

      ROL::Ptr<Vector<Real>> vhat = v.clone();
      vhat->set(v);
      vhat->scale(one / vnorm);

      ROL::Ptr<Vector<Real>> gbase = hv.clone();
      this->gradient(*gbase, Params, tol);

      ROL::Ptr<Vector<Real>> xplus = Params.clone();
      xplus->set(Params);
      xplus->axpy(delta, *vhat);
      this->update(*xplus, ROL::UpdateType::Temp);
      this->gradient(hv, *xplus, tol);

      hv.axpy(-one, *gbase);
      hv.scale(vnorm / delta);
      this->update(Params, ROL::UpdateType::Revert);
    }


    void precond(Vector<Real> &Pv, const Vector<Real> &v,
                 const Vector<Real> &x, Real &tol) override {
      ROL_UNUSED(x);
      ROL_UNUSED(tol);
      // Identity is the safe default while hessVec is FD.
      //
      // Dual precond calls v.dual(), which in Riesz mode multiplies by M^{-1}.
      // This seems to have some ill-conditioned effect on the HessVec approximation.
      //
      // Analytic H should be less senstive to the choice of precond mode,
      // but it's not implemented yet.
      if (precondMode_ == PrecondMode::Identity) {
        Pv.set(v);
      }
      else {
        Pv.set(v.dual());
      }
    }

    // Enable per-outer-iteration logging of (g_d^T M g_d, g_d^T K g_d,
    // z^T M z, z^T K z, ratios, alpha1, alpha2) to a CSV file. Called by
    // AnalysisManager when the YAML key 'riesz diagnostics' is true.
    void setRieszDiagnostics(bool enable, const std::string & path) {
      rieszDiag_ = enable;
      if (!rieszDiag_) {
        rieszDiagOut_ = Teuchos::null;
        lastDualGrad_ = Teuchos::null;
        return;
      }
      if (isRootRank()) {
        rieszDiagOut_ = Teuchos::rcp(new std::ofstream(path));
        if (rieszDiagOut_->is_open()) {
          rieszDiagOut_->precision(8);
          (*rieszDiagOut_) << std::scientific;
          (*rieszDiagOut_) << "iter,gMg,gKg,zMz,zKz,ratio_g,ratio_z,alpha1,alpha2\n";
          rieszDiagOut_->flush();
        } else {
          std::cout << "MrHyDE: WARNING could not open riesz diagnostics file '"
                    << path << "'" << std::endl;
          rieszDiagOut_ = Teuchos::null;
        }
      }
    }

    // Bring in the inherited bool-flag update overload alongside the
    // UpdateType override below (hessVec uses the 2-arg form).
    using Objective<Real>::update;

    // ROL calls update(z, type, iter) once per outer iteration. We log only
    // on Initial (iter==0) and Accept (committed step) to avoid recording
    // trial line-search states. The default iter=-1 matches the base
    // signature so hessVec's 2-arg call still resolves here.
    void update(const Vector<Real> & Params, UpdateType type, int iter = -1) override {
      Objective<Real>::update(Params, type, iter);
      if (!rieszDiag_) return;
      if (type != UpdateType::Initial && type != UpdateType::Accept) return;
      if (iter == rieszDiagLastIter_) return;
      rieszDiagLastIter_ = iter;
      logRieszEnergies(Params, iter);
    }

    // Compute g_d^T M g_d, g_d^T K g_d, z^T M z, z^T K z by applying M and K
    // to each field block of the iterate and the cached dual gradient.
    void logRieszEnergies(const Vector<Real> & z_vec, int iter) {
      auto M = params->getParamMassMatrix();
      auto K = params->getParamStiffnessMatrix();
      if (M.is_null() || K.is_null()) return;

      const MrHyDE_OptVector & z =
          Teuchos::dyn_cast<const MrHyDE_OptVector>(const_cast<Vector<Real> &>(z_vec));
      const auto & zfields = z.getField();

      // Skip this iter if gradient() has not yet populated the cache. Calling
      // gradient() from within update() drives a fresh adjoint solve through
      // SolverManager state that ROL has not finished setting up at Initial,
      // which segfaults. The very first iter therefore has only zMz / zKz; the
      // first row with all four energies is the Accept after iter 1.
      //
      // The M->apply / dot calls below are collective on the Tpetra row map,
      // so every MPI rank must enter them. Only the file write is root-only.
      if (!lastDualGradFresh_ || lastDualGrad_.is_null()) {
        ScalarT zMz0 = 0.0, zKz0 = 0.0;
        using Multi = Tpetra::MultiVector<ScalarT,LO,GO,SolverNode>;
        Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);
        for (size_t b = 0; b < zfields.size(); ++b) {
          auto zb = zfields[b]->getVector();
          if (zb.is_null()) continue;
          if (zb->getMap()->getGlobalNumElements() != M->getRowMap()->getGlobalNumElements()) continue;
          auto tmp = Teuchos::rcp(new Multi(M->getRangeMap(), zb->getNumVectors()));
          M->apply(*zb, *tmp); zb->dot(*tmp, dotprod()); for (size_t c=0;c<dotprod.size();++c) zMz0 += dotprod[c];
          K->apply(*zb, *tmp); zb->dot(*tmp, dotprod()); for (size_t c=0;c<dotprod.size();++c) zKz0 += dotprod[c];
        }
        if (isRootRank() && !rieszDiagOut_.is_null()) {
          const double rz = (zMz0 > 0.0) ? static_cast<double>(zKz0/zMz0)
                                            : std::numeric_limits<double>::quiet_NaN();
          (*rieszDiagOut_) << iter << ",nan,nan,"
                           << zMz0 << "," << zKz0 << ","
                           << "nan," << rz << ","
                           << params->getHcurlAlpha1() << ","
                           << params->getHcurlAlpha2() << "\n";
          rieszDiagOut_->flush();
        }
        return;
      }
      const auto & gfields = lastDualGrad_->getField();

      using Multi = Tpetra::MultiVector<ScalarT,LO,GO,SolverNode>;
      Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);

      ScalarT gMg = 0.0, gKg = 0.0, zMz = 0.0, zKz = 0.0;
      const size_t nblocks = std::min(zfields.size(), gfields.size());
      for (size_t b = 0; b < nblocks; ++b) {
        auto zb = zfields[b]->getVector();
        auto gb = gfields[b]->getVector();
        if (zb.is_null() || gb.is_null()) continue;
        if (zb->getMap()->getGlobalNumElements() != M->getRowMap()->getGlobalNumElements()) continue;

        auto tmp = Teuchos::rcp(new Multi(M->getRangeMap(), zb->getNumVectors()));

        M->apply(*gb, *tmp);
        gb->dot(*tmp, dotprod());
        for (size_t c = 0; c < dotprod.size(); ++c) gMg += dotprod[c];

        K->apply(*gb, *tmp);
        gb->dot(*tmp, dotprod());
        for (size_t c = 0; c < dotprod.size(); ++c) gKg += dotprod[c];

        M->apply(*zb, *tmp);
        zb->dot(*tmp, dotprod());
        for (size_t c = 0; c < dotprod.size(); ++c) zMz += dotprod[c];

        K->apply(*zb, *tmp);
        zb->dot(*tmp, dotprod());
        for (size_t c = 0; c < dotprod.size(); ++c) zKz += dotprod[c];
      }

      if (isRootRank() && !rieszDiagOut_.is_null()) {
        const double ratio_g = (gMg > 0.0) ? static_cast<double>(gKg / gMg)
                                            : std::numeric_limits<double>::quiet_NaN();
        const double ratio_z = (zMz > 0.0) ? static_cast<double>(zKz / zMz)
                                            : std::numeric_limits<double>::quiet_NaN();
        (*rieszDiagOut_) << iter << ","
                         << gMg << "," << gKg << ","
                         << zMz << "," << zKz << ","
                         << ratio_g << "," << ratio_z << ","
                         << params->getHcurlAlpha1() << ","
                         << params->getHcurlAlpha2() << "\n";
        rieszDiagOut_->flush();
      }
    }

    //print out Hessian (estimated via component-wise FD; to get inverse covariance in linear-Gaussian Bayesian inverse problem)
    void printHess(const string & filename, const Vector<Real> & xin, const int & commrank){
      StdVector<Real> x = Teuchos::dyn_cast<StdVector<Real> >(const_cast<Vector<Real> &>(xin));
      int paramDim = x.getVector()->size();
      
      Teuchos::RCP<StdVector<Real> > g = Teuchos::rcp(new StdVector<Real>(Teuchos::rcp(new vector<Real>(paramDim,0.0))));
      ScalarT gtol = sqrt(ROL_EPSILON<Real>());
      this->gradient(*g,x,gtol);
      Teuchos::RCP<StdVector<Real> > gnew = Teuchos::rcp(new StdVector<Real>(Teuchos::rcp(new vector<Real>(paramDim,0.0))));
      
      vector<vector<ScalarT> > hessStash(paramDim);
      
      //Real h = 1.e-3*x.norm(); //step length
      Real h = std::max(static_cast<Real>(1.0),
                        x.norm())*sqrt(ROL_EPSILON<Real>()); ///step length...more like what ROL has...
      
      //perturb each component
      for(int i=0; i<x.dimension(); i++){
        //compute new step
        Teuchos::RCP<StdVector<Real> > xnew = Teuchos::rcp(new StdVector<Real>(Teuchos::rcp(new vector<Real>(paramDim,0.0))));
        xnew->set(x);
        xnew->axpy(h,*x.basis(i));
        
        //gradient at new step
        gnew->zero();
        this->gradient(*gnew,*xnew,gtol);
        
        //i-th column (or row...) of Hessian
        gnew->axpy(static_cast<Real>(-1.0),*g);
        gnew->scale(static_cast<Real>(1.0)/h);
        Teuchos::RCP<vector<ScalarT> > gnewv = gnew->getVector();
        
        vector<ScalarT> row(paramDim);
        for(int j=0; j<paramDim; j++)
          row[j] = (*gnewv)[j];
        hessStash[i] = row;
      }
      
      if(commrank == 0){
        std::ofstream respOUT(filename);
        respOUT.precision(16);
        for(int i=0; i<paramDim; i++){
          for(int j=0; j<paramDim; j++)
            respOUT << hessStash[i][j] << " ";
          respOUT << endl;
        }
        respOUT.close();
      }
    }
    
    /*!
     \brief Generate data to plot objective function
     */
    void generate_plot(Real difflo, Real diffup, Real diffstep){
      
      Teuchos::RCP<std::vector<ScalarT> > Params_rcp = Teuchos::rcp(new std::vector<ScalarT>(1,0.0) );
      ROL::StdVector<ScalarT> Params(Params_rcp);
      std::ofstream output ("Objective.dat");
      
      Real diff = 0.0;
      Real val = 0.0;
      Real tol = 1.e-16;
      int n = (diffup-difflo)/diffstep + 1;
      for(int i=0;i<n;i++){
        diff = difflo + i*diffstep;
        (*Params_rcp)[0] = diff;
        val = this->value(Params,tol);
        if(output.is_open()){
          output << std::scientific << diff << " " << val << "\n";
        }
      }
      output.close();
    }
    
    /*!
     \brief Generate data to plot objective function
     */
    void generate_plot(Real alo, Real aup, Real astep, Real blo, Real bup, Real bstep){
      
      Teuchos::RCP<std::vector<ScalarT> > Params_rcp = Teuchos::rcp(new std::vector<ScalarT>(2,0.0) );
      ROL::StdVector<ScalarT> Params(Params_rcp);
      std::ofstream output ("Objective.dat");
      
      Real a = 0.0;
      Real b = 0.0;
      Real val = 0.0;
      Real tol = 1.e-16;
      int n = (aup-alo)/astep + 1;
      int m = (bup-blo)/bstep + 1;
      for(int i=0;i<n;i++){
        a = alo + i*astep;
        for(int j=0;j<m;j++){
          b = blo + j*bstep;
          (*Params_rcp)[0] = a;
          (*Params_rcp)[1] = b;
          val = this->value(Params,tol);
          if(output.is_open()){
            output << std::scientific << a << " " << b << " " << val << "\n";
          }
        }
      }
      output.close();
    }
    
  }; //end description of Objective class
  
  /*!
   \brief Inequality constraints on optimization parameters
   
   
   ---
   */
  
}//end namespace ROL

#endif

