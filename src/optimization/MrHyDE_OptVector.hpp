/***********************************************************************
 MrHyDE - a framework for solving Multi-resolution Hybridized
 Differential Equations and enabling beyond forward simulation for
 large-scale multiphysics and multiscale systems.
 
 Questions? Contact Tim Wildey (tmwilde@sandia.gov)
 ************************************************************************/

#ifndef MRHYDE_OPTVEC_HPP
#define MRHYDE_OPTVEC_HPP

#include "trilinos.hpp"
#include "preferences.hpp"

#include "ROL_StdVector.hpp"
#include "ROL_TpetraMultiVector.hpp"
#include "MatrixMarket_Tpetra.hpp"

class MrHyDE_OptVector : public ROL::Vector<ScalarT> {
  
  typedef Tpetra::CrsMatrix<ScalarT,LO,GO,SolverNode>   LA_CrsMatrix;
  typedef Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> LA_MultiVector;
  typedef Teuchos::RCP<LA_MultiVector>            vector_RCP;
  typedef Teuchos::RCP<LA_CrsMatrix>              matrix_RCP;
  typedef typename SolverNode::device_type              LA_device;
  typedef typename SolverNode::execution_space LA_exec;
  
private:
  
  std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > field_vec; // vector for dynamic field
  std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > scalar_vec; // vector for dynamic scalars
  
  const int mpirank;
  bool have_scalar, have_field, have_dynamic_scalar, have_dynamic_field;
  double dyn_dt = 1.0;
  
  mutable std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > dual_field_vec;
  mutable std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > dual_scalar_vec;
  mutable ROL::Ptr<MrHyDE_OptVector> dual_vec;
  mutable bool isDualInitialized;

  mutable bool isDual;
  mutable ScalarT dualityScale;

  // Optional Tpetra M and M^{-1} from setMassOperators(); mass_type comes from Analysis.
  std::string mass_type = "none";
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,SolverNode>> massOperator = Teuchos::null;
  Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,SolverNode>> massInvOperator = Teuchos::null;
  bool have_mass_operator = false;

  // Global flag to control whether to use proper dual space handling with Riesz maps
  // When true: dual() applies (M+K) transformation, gradients should be unpreconditioned
  // When false: dual() is identity, gradients should be preconditioned (legacy behavior)
  // Note: Using mutable non-static to avoid template static member issues in header
  mutable bool use_proper_dual_spaces = false;
  mutable int verbosity = 0;

  Teuchos::RCP<Teuchos::Time> constructortimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::OptVector::constructor()");
  Teuchos::RCP<Teuchos::Time> clonetimer = Teuchos::TimeMonitor::getNewCounter("MrHyDE::OptVector::clone()");
  
public:
  
  ///////////////////////////////////////////////////
  // Constructors for MrHyDE_OptVector
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const std::vector<ROL::Ptr<std::vector<ScalarT> > > & s_vec,
                   const double & dt,
                   const std::string & mass_type_in = "none",
                   const int mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false), isDual(isdual), dualityScale(scale), mass_type(mass_type_in) {

    Teuchos::TimeMonitor localtimer(*constructortimer);

    if (s_vec.size() == 0) {
      have_scalar = false;
      have_dynamic_scalar = false;
    }
    else {
      for (size_t k=0; k<s_vec.size(); ++k) {
        scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec[k]));
      }
      have_scalar = true;
      if (s_vec.size() > 1) {
        have_dynamic_scalar = true;
      }
      else {
        have_dynamic_scalar = false;
      }
    }

    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    for (size_t k=0; k<f_vec.size(); ++k) {
      field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode>>(f_vec[k]));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }

    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }

    if (have_scalar) {
      for (size_t k=0; k<s_vec.size(); ++k) {
        dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
      }
    }

  }
    
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const std::vector<ROL::Ptr<std::vector<ScalarT> > > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    if (s_vec.size() == 0) {
      have_scalar = false;
      have_dynamic_scalar = false;
    }
    else {
      for (size_t k=0; k<s_vec.size(); ++k) {
        scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec[k]));
      }
      have_scalar = true;
      if (s_vec.size() > 1) {
        have_dynamic_scalar = true;
      }
      else {
        have_dynamic_scalar = false;
      }
    }
    
    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    for (size_t k=0; k<f_vec.size(); ++k) {
      field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode>>(f_vec[k]));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }
    
    if (have_scalar) {
      for (size_t k=0; k<s_vec.size(); ++k) {
        dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
      }
    }

  }
  
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const ROL::Ptr<std::vector<ScalarT> > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    have_dynamic_scalar = false;
    if (s_vec == ROL::nullPtr) {
      have_scalar = false;
    }
    else {
      scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec));
      have_scalar = true;
    }
    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }
      
    for (size_t k=0; k<f_vec.size(); ++k) {
      field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode>>(f_vec[k]));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    if (have_scalar) {
      dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));
    }

  }
  
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const std::vector<ROL::Ptr<std::vector<ScalarT> > > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    if (s_vec.size() == 0) {
      //scalar_vec = ROL::nullPtr;
      have_scalar = false;
      have_dynamic_scalar = false;
    }
    else {
      for (size_t k=0; k<s_vec.size(); ++k) {
        scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec[k]));
      }
      have_scalar = true;
      if (s_vec.size() > 1) {
        have_dynamic_scalar = true;
      }
      else {
        have_dynamic_scalar = false;
      }
    }
    
    have_field = true;
    field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode>>(f_vec));
    dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[0]->dual().clone()));
    
    have_dynamic_field = false;
    
    if (have_scalar) {
      for (size_t k=0; k<s_vec.size(); ++k) {
        dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
      }
    }

  }
  
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const ROL::Ptr<std::vector<ScalarT> > & s_vec,
                   const int mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : mpirank(mpirank_), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec));
    field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec));
    
    have_dynamic_scalar = false;
    have_dynamic_field = false;
    
    if (s_vec->size() > 0) {
      have_scalar = true;
    }
    else {//if (s_vec == ROL::nullPtr) {
      have_scalar = false;
    }
    
    have_field = true;
    
    dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[0]->dual().clone()));
    dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));

  }
  
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<Tpetra::MultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const double & dt,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : scalar_vec(ROL::nullPtr), mpirank(0), dyn_dt(dt), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    have_scalar = false;
    have_field = true;
    have_dynamic_scalar = false;
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      field_vec.push_back(ROL::makePtr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec[k]));
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    have_dynamic_field = false;
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }

  }
  
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<std::vector<ScalarT> > & s_vec,
                   const int & mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : field_vec(ROL::nullPtr), mpirank(mpirank_), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    have_scalar = true;
    have_field = false;
    have_dynamic_scalar = false;
    have_dynamic_field = false;
    
    scalar_vec.push_back(ROL::makePtr<ROL::StdVector<ScalarT>>(s_vec));
    dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));

  }
  
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector()
  : mpirank(0), isDualInitialized(false), isDual(false), dualityScale(1.0) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    have_scalar = false;
    have_field = false;
    have_dynamic_scalar = false;
    have_dynamic_field = false;

  }

  ///////////////////////////////////////////////////
  // Constructors for cloning
  ///////////////////////////////////////////////////

  MrHyDE_OptVector(const std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : field_vec(f_vec), scalar_vec(s_vec), mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    have_scalar = true;
    if (s_vec[0]->getVector()->size() == 0) {
      have_scalar = false;
    }
    if (s_vec.size() > 1) {
      have_dynamic_scalar = true;
    }
    else {
      have_dynamic_scalar = false;
    }
    
    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    
    for (size_t k=0; k<s_vec.size(); ++k) {
      dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
    }

  }
  
  ///////////////////////////////////////////////////

  MrHyDE_OptVector(const std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const ROL::Ptr<ROL::StdVector<ScalarT> > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : field_vec(f_vec), mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    scalar_vec.push_back(s_vec);
    have_scalar = true;
    if (s_vec->getVector()->size() == 0) {
      have_scalar = false;
    }
    have_dynamic_scalar = false;
    
    have_field = true;
    if (f_vec.size() == 0) {
      have_field = false;
    }
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }
    dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));

  }

  ///////////////////////////////////////////////////

  MrHyDE_OptVector(const ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > & s_vec,
                   const double & dt,
                   const int mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : scalar_vec(s_vec), mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
  
    Teuchos::TimeMonitor localtimer(*constructortimer);

    have_scalar = true;
    if (s_vec[0]->getVector()->size() == 0) {
      have_scalar = false;
    }
    if (s_vec.size() > 1) {
      have_dynamic_scalar = true;
    }
    else {
      have_dynamic_scalar = false;
    }
  
    have_field = true;
    have_dynamic_field = false;
  
    field_vec.push_back(f_vec);
    
    dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[0]->dual().clone()));
  
    for (size_t k=0; k<s_vec.size(); ++k) {
      dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
    }

  }

  ///////////////////////////////////////////////////

  MrHyDE_OptVector(const ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const ROL::Ptr<ROL::StdVector<ScalarT> > & s_vec,
                   const int mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : mpirank(mpirank_), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    have_scalar = true;
    if (s_vec->getVector()->size() == 0) {
      have_scalar = false;
    }
    have_dynamic_scalar = false;
    
    have_field = true;
    have_dynamic_field = false;
    
    scalar_vec.push_back(s_vec);
    field_vec.push_back(f_vec);
    dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[0]->dual().clone()));
    dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));

  }
  
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > & f_vec,
                   const double & dt,
                   const int & mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : field_vec(f_vec), mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    have_scalar = false;
    have_field = true;
    scalar_vec.push_back(ROL::nullPtr);
    
    if (f_vec.size() > 1) {
      have_dynamic_field = true;
    }
    else {
      have_dynamic_field = false;
    }
    have_dynamic_scalar = false;
    
    for (size_t k=0; k<f_vec.size(); ++k) {
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(f_vec[k]->dual().clone()));
    }

  }
  
  ///////////////////////////////////////////////////

  MrHyDE_OptVector(const ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > & f_vec,
                   const int & mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : mpirank(mpirank_), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    have_scalar = false;
    scalar_vec.push_back(ROL::nullPtr);
    have_dynamic_scalar = false;
    
    have_field = true;
    
    field_vec.push_back(f_vec);
    have_dynamic_field = false;
    
    for (size_t k=0; k<field_vec.size(); ++k) {
      dual_field_vec.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[k]->dual().clone()));
    }

  }
  
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const ROL::Ptr<ROL::StdVector<ScalarT> > & s_vec,
                   const int & mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : mpirank(mpirank_), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    scalar_vec.push_back(s_vec);
    have_scalar = true;
    have_dynamic_scalar = false;
    have_field = false;
    have_dynamic_field = false;
    
    dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[0]->dual().clone()));

  }
  
  ///////////////////////////////////////////////////
  
  MrHyDE_OptVector(const std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > & s_vec,
                   const double & dt,
                   const int & mpirank_ = 0,
                   bool isdual = false,
                   ScalarT scale = 1.0)
  : scalar_vec(s_vec), mpirank(mpirank_), dyn_dt(dt), isDualInitialized(false), isDual(isdual), dualityScale(scale) {
    
    Teuchos::TimeMonitor localtimer(*constructortimer);

    have_scalar = true;
    have_dynamic_scalar = false;
    if (s_vec.size() > 1) {
      have_dynamic_scalar = true;
    }
    have_field = false;
    have_dynamic_field = false;
    
    for (size_t k=0; k<s_vec.size(); ++k) {
      dual_scalar_vec.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[k]->dual().clone()));
    }
    

  }
  
  ///////////////////////////////////////////////////

  ScalarT getInnerProductScaling(const MrHyDE_OptVector &xs) const {
    if (!isDual && !xs.isDual) {
      return dualityScale; // primal-primal
    }
    else if (isDual && xs.isDual) {
      return 1.0 / dualityScale; // dual-dual
    }
    else {
      return 1.0; // cross terms (primal-dual or dual-primal)
    }
  }
  
  ///////////////////////////////////////////////////
  // Virtual functions from ROL::Vector
  ///////////////////////////////////////////////////
  
  void set( const ROL::Vector<ScalarT> &x ) {
    
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    if (field_vec.size() > 0) {
      const auto& xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->set(*(xs_f[i]));
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      const auto& xs_s = xs.getParameter();
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->set(*(xs_s[i]));
        }
      }
    }
  }
  
  ///////////////////////////////////////////////////
  
  void plus( const ROL::Vector<ScalarT> &x ) {
    
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    if (field_vec.size() > 0) {
      const auto& xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->plus(*(xs_f[i]));
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      const auto& xs_s = xs.getParameter();
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->plus(*(xs_s[i]));
        }
      }
    }
    
  }
  
  ///////////////////////////////////////////////////
  
  void scale( const ScalarT alpha ) {
    
    if (field_vec.size() > 0) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->scale(alpha);
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->scale(alpha);
        }
      }
    }
    
  }
  
  ///////////////////////////////////////////////////
  
  void zero() {
    
    if (field_vec.size() > 0) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->zero();
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->zero();
        }
      }
    }
    
  }
  
  ///////////////////////////////////////////////////
  
  void putScalar(const ScalarT & alpha) {
    
    if (field_vec.size() > 0) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->getVector()->putScalar(alpha);
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->getVector()->assign(scalar_vec[i]->dimension(),alpha);
        }
      }
    }
    
  }
  
  ///////////////////////////////////////////////////
  
  void axpy( const ScalarT alpha, const ROL::Vector<ScalarT> &x ) {
    
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    if (field_vec.size() > 0) {
      const auto& xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          field_vec[i]->axpy(alpha,*(xs_f[i]));
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      const auto& xs_s = xs.getParameter();
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          scalar_vec[i]->axpy(alpha,*(xs_s[i]));
        }
      }
    }
    
  }
  
  /**
   * Compute mass-weighted inner product between two vectors
   *
   * Implements the inner product <x, y>_M where M is a mass matrix.
   * The specific form depends on whether vectors are primal or dual:
   *
   * - Primal-Primal: <x, y>_M = x^T M y
   * - Dual-Dual:     <x, y>_M = x^T M^{-1} y
   * - Primal-Dual:   <x, y>   = x^T y (no scaling)
   *
   * input: x The other vector in the inner product
   * output: Scalar value of <this, x>_M
   *
   */
  ScalarT dot( const ROL::Vector<ScalarT> &x ) const {

    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    ScalarT val(0);
    if (field_vec.size() > 0) {
      const auto& xs_f = xs.getField();

      bool inverse_scaling = false;
      bool no_scaling = false;
      
      if (!isDual && !xs.isDual) {
        // no changes needed (primal-primal)
      }
      else if (isDual && xs.isDual) {
        inverse_scaling = true; // (dual-dual)
      }
      else {
        no_scaling = true; // (primal-dual or dual-primal)
      }
      
      if (no_scaling) {
        for (size_t i=0; i<field_vec.size(); ++i) {
          if ( field_vec[i] != ROL::nullPtr ) {
            val += field_vec[i]->dot(*(xs_f[i]));
          }
        }
      }
      else {
        if (have_mass_operator) {
          if (inverse_scaling) {
            TEUCHOS_TEST_FOR_EXCEPTION(massInvOperator.is_null(), std::runtime_error,
                                       "MrHyDE_OptVector::dot: massInvOperator required for dual-dual inner product.");
            applyMassOperatorAndDot(massInvOperator, xs, val);
          }
          else {
            TEUCHOS_TEST_FOR_EXCEPTION(massOperator.is_null(), std::runtime_error,
                                       "MrHyDE_OptVector::dot: massOperator required for primal-primal inner product.");
            applyMassOperatorAndDot(massOperator, xs, val);
          }
        }
        else {
          TEUCHOS_TEST_FOR_EXCEPTION(!inverse_scaling && !have_field,
              std::logic_error,
              "MrHyDE_OptVector::dot: Fallback scaled-Euclidean path requires mass operators for "
              "mixed discretizations. Call setMassOperators() to provide M and M^{-1}.");
          for (size_t i=0; i<field_vec.size(); ++i) {
            if ( field_vec[i] != ROL::nullPtr ) {
              val += field_vec[i]->dot(*(xs_f[i]));
            }
          }
          if (inverse_scaling) {
            val *= 1.0/dualityScale;
          }
          else {
            val *= dualityScale;
          }
        }
      }
    }
    
    if (scalar_vec.size() > 0) {
      const auto& xs_s = xs.getParameter();
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          val += scalar_vec[i]->dot(*(xs_s[i]));
        }
      }
    }
    
    return val;
  }

  ScalarT norm() const {
    ScalarT val = this->dot(*this);
    return std::sqrt(val);
  }

  // Euclidean dot product ignoring mass operators.
  ScalarT euclideanDot(const ROL::Vector<ScalarT> &x) const {
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    ScalarT val(0);
    if (field_vec.size() > 0) {
      const auto& xs_f = xs.getField();
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          val += field_vec[i]->dot(*(xs_f[i]));
        }
      }
    }
    if (scalar_vec.size() > 0) {
      const auto& xs_s = xs.getParameter();
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          val += scalar_vec[i]->dot(*(xs_s[i]));
        }
      }
    }
    return val;
  }

  // Euclidean norm ignoring mass operators. Used for FD step size computation
  // where metric-weighted norms would distort the finite-difference scaling.
  ScalarT euclideanNorm() const {
    ScalarT val(0);
    if (field_vec.size() > 0) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          val += field_vec[i]->dot(*(field_vec[i]));
        }
      }
    }
    if (scalar_vec.size() > 0) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if ( scalar_vec[i] != ROL::nullPtr ) {
          val += scalar_vec[i]->dot(*(scalar_vec[i]));
        }
      }
    }
    return std::sqrt(val);
  }

  ///////////////////////////////////////////////////

  ROL::Ptr<ROL::Vector<ScalarT> > clone(void) const {
    
    Teuchos::TimeMonitor localtimer(*clonetimer);

    ROL::Ptr<ROL::Vector<ScalarT> > clonevec;
    if ( !have_scalar ) {
      std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > fvecs;
      
      for (size_t i=0; i<field_vec.size(); ++i) {
        fvecs.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[i]->clone()));
      }
      clonevec = ROL::makePtr<MrHyDE_OptVector>(fvecs, dyn_dt, mpirank);

    }
    else if ( !have_field) {
      std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > svecs;
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        svecs.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[i]->clone()));
      }
      clonevec = ROL::makePtr<MrHyDE_OptVector>(svecs, dyn_dt, mpirank);
    }
    else {
      std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> > > fvecs;
      for (size_t i=0; i<field_vec.size(); ++i) {
        fvecs.push_back(ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode> >(field_vec[i]->clone()));
      }
      std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > > svecs;
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        svecs.push_back(ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(scalar_vec[i]->clone()));
      }

      clonevec = ROL::makePtr<MrHyDE_OptVector>(fvecs, svecs, dyn_dt, mpirank);
    }

    auto cloned = ROL::dynamicPtrCast<MrHyDE_OptVector>(clonevec);
    propagateMassOperators(cloned);
    cloned->isDual = this->isDual;
    cloned->use_proper_dual_spaces = this->use_proper_dual_spaces;

    return clonevec;
  }
  
  ///////////////////////////////////////////////////
  
  const ROL::Vector<ScalarT> & dual(void) const {

    if ( !isDualInitialized ) {
      if ( !have_field) {
        dual_vec = ROL::makePtr<MrHyDE_OptVector>(dual_scalar_vec, dyn_dt);
      }
      else if ( !have_scalar ) {
        dual_vec = ROL::makePtr<MrHyDE_OptVector>(dual_field_vec, dyn_dt);
      }
      else {
        dual_vec = ROL::makePtr<MrHyDE_OptVector>(dual_field_vec, dual_scalar_vec, dyn_dt);
      }

      auto dual_optvec = ROL::dynamicPtrCast<MrHyDE_OptVector>(dual_vec);
      propagateMassOperators(dual_optvec);

      // CRITICAL: Set the dual flag to indicate this is in the dual space
      dual_optvec->isDual = !this->isDual;

      isDualInitialized = true;
    }

    // Riesz map: primal->dual applies (M+K), dual->primal applies (M+K)^{-1}.
    // Falls back to identity when mass operators are unavailable.
    if (use_proper_dual_spaces && have_mass_operator && !massOperator.is_null() && !massInvOperator.is_null()) {
      auto & op = isDual ? massInvOperator : massOperator;
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          op->apply(*(field_vec[i]->getVector()), *(dual_field_vec[i]->getVector()));
        }
      }
    } else {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if ( field_vec[i] != ROL::nullPtr ) {
          dual_field_vec[i]->set(field_vec[i]->dual());
        }
      }
    }

    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        dual_scalar_vec[i]->set(scalar_vec[i]->dual());
      }
    }

    return *dual_vec;
  }
  
  ///////////////////////////////////////////////////
  
  ScalarT apply(const ROL::Vector<ScalarT> &x) const {
    // Duality pairing <this, x> = dot(x.dual())
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);

    if (isDual != xs.isDual) {
      // Different spaces (primal-dual): standard dot product suffices
      return this->dot(xs);
    } else {
      // Same space: apply dual transformation for metric-aware pairing
      return this->dot(xs.dual());
    }
  }
  
  ///////////////////////////////////////////////////
  
  ROL::Ptr<ROL::Vector<ScalarT> > basis( const int i )  const {
    ROL::Ptr<ROL::Vector<ScalarT> > e;

    /*
     if ( field_vec != ROL::nullPtr && scalar_vec != ROL::nullPtr ) {
     int n1 = field_vec->dimension();
     ROL::Ptr<ROL::Vector<ScalarT> > e1, e2;
     if ( i < n1 ) {
     e1 = field_vec->basis(i);
     e2 = scalar_vec->clone(); e2->zero();
     }
     else {
     e1 = field_vec->clone(); e1->zero();
     e2 = scalar_vec->basis(i-n1);
     }
     e = ROL::makePtr<MrHyDE_OptVector>(
     ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT> >(e1),
     ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(e2));
     }
     if ( field_vec != ROL::nullPtr && scalar_vec == ROL::nullPtr ) {
     int n1 = field_vec->dimension();
     ROL::Ptr<ROL::Vector<ScalarT> > e1;
     if ( i < n1 ) {
     e1 = field_vec->basis(i);
     }
     else {
     e1->zero();
     }
     e = ROL::makePtr<MrHyDE_OptVector>(
     ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT> >(e1));
     }
     if ( field_vec == ROL::nullPtr && scalar_vec != ROL::nullPtr ) {
     int n2 = scalar_vec->dimension();
     ROL::Ptr<ROL::Vector<ScalarT> > e2;
     if ( i < n2 ) {
     e2 = scalar_vec->basis(i);
     }
     else {
     e2->zero();
     }
     e = ROL::makePtr<MrHyDE_OptVector>(
     ROL::dynamicPtrCast<ROL::StdVector<ScalarT> >(e2));
     }*/
    return e;
  }
  
  ///////////////////////////////////////////////////
  
  void applyUnary( const ROL::Elementwise::UnaryFunction<ScalarT> &f ) {
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        field_vec[i]->applyUnary(f);
      }
    }
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        scalar_vec[i]->applyUnary(f);
      }
    }
  }
  
  ///////////////////////////////////////////////////
  
  void applyBinary( const ROL::Elementwise::BinaryFunction<ScalarT> &f, const ROL::Vector<ScalarT> &x ) {
    const MrHyDE_OptVector &xs = dynamic_cast<const MrHyDE_OptVector&>(x);
    const auto& xs_f = xs.getField();
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        field_vec[i]->applyBinary(f,*(xs_f[i]));
      }
    }
    
    const auto& xs_s = xs.getParameter();
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        scalar_vec[i]->applyBinary(f,*xs_s[i]);
      }
    }
  }

  ///////////////////////////////////////////////////
  
  ScalarT reduce( const ROL::Elementwise::ReductionOp<ScalarT> &r ) const {
    ScalarT result = r.initialValue();
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        r.reduce(field_vec[i]->reduce(r),result);
      }
    }
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        r.reduce(scalar_vec[i]->reduce(r),result);
      }
    }
    return result;
  }
  
  ///////////////////////////////////////////////////
  
  int dimension() const {
    int dim(0);
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        dim += field_vec[i]->dimension();
      }
    }
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        dim += scalar_vec[i]->dimension();
      }
    }
    return dim;
  }
  
  ///////////////////////////////////////////////////
  
  void randomize(const ScalarT l = 0.0, const ScalarT u = 1.0) {
    for (size_t i=0; i<field_vec.size(); ++i) {
      if (field_vec[i] != ROL::nullPtr) {
        field_vec[i]->randomize(l,u);
      }
    }
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if (scalar_vec[i] != ROL::nullPtr) {
        scalar_vec[i]->randomize(l,u);
      }
    }
  }
  
  ///////////////////////////////////////////////////
  
  void print(std::ostream &outStream) const {
    if (have_field) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if (field_vec[i] != ROL::nullPtr) {
          field_vec[i]->print(outStream);
        }
      }
    }
    if (have_scalar) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if (mpirank == 0 && scalar_vec[i] != ROL::nullPtr) {
          scalar_vec[i]->print(outStream);
        }
      }
    }
  }
  
  ///////////////////////////////////////////////////
  
  void print(string & filebase) const {
    if (have_field) {
      for (size_t i=0; i<field_vec.size(); ++i) {
        if (field_vec[i] != ROL::nullPtr) {
          std::stringstream ss;
          ss << filebase << ".field." << i << ".mm";
          Tpetra::MatrixMarket::Writer<LA_MultiVector>::writeDenseFile(ss.str(),*(field_vec[i]->getVector()));
        }
      }
    }
    if (have_scalar) {
      for (size_t i=0; i<scalar_vec.size(); ++i) {
        if (mpirank == 0 && scalar_vec[i] != ROL::nullPtr) {
          std::stringstream ss;
          ss << filebase << ".scalar." << i << ".dat";
          std::ofstream scalarOUT(ss.str());
          scalarOUT.precision(12);
          auto vec = scalar_vec[i]->getVector();
          for (size_t j=0; j<vec->size(); ++j) {
            scalarOUT << (*vec)[j] << endl;
          }
        }
      }
    }
  }
  
  ///////////////////////////////////////////////////
  // Extra functions
  ///////////////////////////////////////////////////
  
  const std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > >& getField(void) const {
    return field_vec;
  }

  ///////////////////////////////////////////////////

  const std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > >& getParameter(void) const {
    return scalar_vec;
  }

  ///////////////////////////////////////////////////

  std::vector<ROL::Ptr<ROL::TpetraMultiVector<ScalarT> > >& getField(void) {
    return field_vec;
  }

  ///////////////////////////////////////////////////

  std::vector<ROL::Ptr<ROL::StdVector<ScalarT> > >& getParameter(void) {
    return scalar_vec;
  }
  
  ///////////////////////////////////////////////////
  
  void setField(const ROL::Vector<ScalarT>& vec) {
    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        field_vec[i]->set(vec);
      }
    }
  }
  
  ///////////////////////////////////////////////////
  
  void setParameter(const ROL::Vector<ScalarT>& vec) {
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        scalar_vec[i]->set(vec);
      }
    }
  }
  
  ///////////////////////////////////////////////////

  void setParameter(const std::vector<ROL::Vector<ScalarT> >& vec) {
    for (size_t i=0; i<scalar_vec.size(); ++i) {
      if ( scalar_vec[i] != ROL::nullPtr ) {
        scalar_vec[i]->set(vec[i]);
      }
    }
  }

  ///////////////////////////////////////////////////
  
  bool haveDynamicField() {
    return have_dynamic_field;
  }
  
  ///////////////////////////////////////////////////

  bool haveDynamicScalar() {
    return have_dynamic_scalar;
  }

  ///////////////////////////////////////////////////
  
  bool haveScalar() const {
    return have_scalar;
  }
  
  ///////////////////////////////////////////////////
  
  bool haveField() const {
    return have_field;
  }

  // Sets Tpetra M and M^{-1}; clears cached dual so clones stay consistent.
  void setMassOperators(const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,SolverNode>> & forward_op,
                        const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,SolverNode>> & inverse_op) {
    massOperator = forward_op;
    massInvOperator = inverse_op;
    have_mass_operator = !massOperator.is_null() || !massInvOperator.is_null();

    isDualInitialized = false;
  }

  ///////////////////////////////////////////////////

  std::string getMassType() const {
    return mass_type;
  }

  void setVerbosity(int v) const {
    verbosity = v;
  }

private:
  void propagateMassOperators(ROL::Ptr<MrHyDE_OptVector> target) const {
    target->setVerbosity(verbosity);
    if (have_mass_operator) {
      target->setMassOperators(massOperator, massInvOperator);
    }
    target->use_proper_dual_spaces = this->use_proper_dual_spaces;
  }

  // Apply op (M or M^{-1}) per field block and add <op x_this, xs> into val.
  void applyMassOperatorAndDot(
      const Teuchos::RCP<Tpetra::Operator<ScalarT,LO,GO,SolverNode>>& op,
      const MrHyDE_OptVector& xs,
      ScalarT& val) const {

    const auto& xs_f = xs.getField();

    Teuchos::RCP<LA_MultiVector> result_tpetra;

    for (size_t i=0; i<field_vec.size(); ++i) {
      if ( field_vec[i] != ROL::nullPtr ) {
        auto xs_rol = ROL::dynamicPtrCast<ROL::TpetraMultiVector<ScalarT,LO,GO,SolverNode>>(xs_f[i]);
        auto xs_tpetra = xs_rol->getVector();

        if (result_tpetra.is_null() ||
            !result_tpetra->getMap()->isSameAs(*(xs_tpetra->getMap())) ||
            result_tpetra->getNumVectors() != xs_tpetra->getNumVectors()) {
          result_tpetra = Teuchos::rcp(new LA_MultiVector(xs_tpetra->getMap(),
                                                           xs_tpetra->getNumVectors()));
        }

        op->apply(*xs_tpetra, *result_tpetra);

        auto field_tpetra = field_vec[i]->getVector();
        const size_t numVecs = field_tpetra->getNumVectors();
        std::vector<ScalarT> dots(numVecs);
        field_tpetra->dot(*result_tpetra, Teuchos::arrayViewFromVector(dots));
        for (size_t j = 0; j < numVecs; ++j) {
          val += dots[j];
        }
      }
    }
  }

public:

  // ========================================================================================
  // Specialized PCG
  // ========================================================================================

  void PCG(matrix_RCP & J, vector_RCP & b, vector_RCP & x,
           vector_RCP & M, const ScalarT & tol, const int & maxiter) const {
    
    //Teuchos::TimeMonitor localtimer(*PCGtimer);
    
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> dotprod(1);
    
    ScalarT rho = 1.0, rho1 = 0.0, alpha = 0.0, beta = 1.0, pq = 0.0;
    ScalarT one = 1.0, zero = 0.0;
    
    vector_RCP p, q, r, z;
    
    p = Teuchos::rcp(new LA_MultiVector(x->getMap(),1));
    q = Teuchos::rcp(new LA_MultiVector(x->getMap(),1));
    r = Teuchos::rcp(new LA_MultiVector(x->getMap(),1));
    z = Teuchos::rcp(new LA_MultiVector(x->getMap(),1));
    
    
    p->putScalar(zero);
    q->putScalar(zero);
    r->putScalar(zero);
    z->putScalar(zero);
    
    int iter=0;
    Teuchos::Array<typename Teuchos::ScalarTraits<ScalarT>::magnitudeType> rnorm(1);
    {
      //Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
      J->apply(*x,*q);
    }
    
    r->assign(*b);
    r->update(-one,*q,one);
    
    r->norm2(rnorm);
    ScalarT r0 = rnorm[0];
    
    auto M_view = M->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto r_view = r->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    auto z_view = z->template getLocalView<LA_device>(Tpetra::Access::ReadWrite);
    
    while (iter<maxiter && rnorm[0]/r0>tol) {
      
      {
        //Teuchos::TimeMonitor localtimer(*PCGApplyPrectimer);
        parallel_for("PCG apply prec",
                     RangePolicy<LA_exec>(0,z_view.extent(0)),
                     KOKKOS_LAMBDA (const int k ) {
          z_view(k,0) = r_view(k,0)/M_view(k,0);
        });
      }
      
      rho1 = rho;
      r->dot(*z, dotprod);
      rho = dotprod[0];
      if (iter == 0) {
        p->assign(*z);
      }
      else {
        beta = rho/rho1;
        p->update(one,*z,beta);
      }
      
      {
        //Teuchos::TimeMonitor localtimer(*PCGApplyOptimer);
        J->apply(*p,*q);
      }
      
      p->dot(*q,dotprod);
      pq = dotprod[0];
      alpha = rho/pq;
      
      x->update(alpha,*p,one);
      r->update(-one*alpha,*q,one);
      r->norm2(rnorm);
      
      iter++;
    }
    if (mpirank == 0 && verbosity >= 6) {
      cout << " ******* PCG Convergence Information: " << endl;
      cout << " *******     Iter: " << iter << "   " << "rnorm = " << rnorm[0]/r0 << endl;
    }
  }

  // Method to control dual space behavior for this instance
  void setProperDualSpaces(bool use_proper) const {
    use_proper_dual_spaces = use_proper;
    if (mpirank == 0 && verbosity >= 6) {
      if (use_proper) {
        std::cout << "MrHyDE_OptVector: Enabled proper dual space handling with Riesz maps" << std::endl;
      } else {
        std::cout << "MrHyDE_OptVector: Using legacy dual space behavior (identity transformation)" << std::endl;
      }
    }
  }

  // Method to set whether this vector is in the dual space
  void setDualSpace(bool is_dual) const {
    isDual = is_dual;
  }

}; // class MrHyDE_OptVector

#endif

