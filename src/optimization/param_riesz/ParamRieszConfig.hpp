/***********************************************************************
 MrHyDE - Parsed configuration of the parameter Riesz metric, the ROL
 control inner product.

 Questions? Contact Alexey Voronin
 ************************************************************************/

#ifndef MRHYDE_PARAM_RIESZ_CONFIG_HPP
#define MRHYDE_PARAM_RIESZ_CONFIG_HPP

#include "trilinos.hpp"
#include "preferences.hpp"
#include "block_prec/ParamUtils.hpp"

namespace MrHyDE {

  // Parsed configuration of the parameter Riesz metric (the ROL control inner product).
  struct ParamRieszConfig {
    std::string type = "mass";        // mass, sobolev, or regularizer
    std::string apply = "diagonal";   // diagonal, exact or iterative
    std::string solver = "KLU2";      // Amesos2 solver for apply exact
    bool drop_zeros = false;          // factor only the stored nonzeros
    bool strict = false;              // iterative apply: abort on non-convergence instead of warning
    std::string krylov = "CG";        // CG or GMRES for apply iterative
    std::string prec = "RefMaxwell";  // RefMaxwell or Jacobi for apply iterative
    std::string xml;                  // MueLu list for RefMaxwell
    int slabs_per_solve = 0;          // iterative apply: most time slabs per Krylov solve; 0 is 'auto', see slabCap
    int ranks = 0;                    // iterative apply: ranks holding the control rows; 0 is 'all', -1 is 'mesh' (the mesh layout)
    int hgrad_order = 0;              // iterative apply: order of RefMaxwell's H(grad) space; 0 is 'control', the control's order
    Teuchos::RCP<Teuchos::ParameterList> belos;  // tolerance, iterations, Belos Settings
    std::string blocks;               // blank means every block
    std::string file;                 // blank means the operator is not written
    ScalarT l2_weight = 1.0, semi_weight = 0.0;

    // present: a "Parameter Riesz map" sublist exists, which flips the type/apply defaults.
    void parse(Teuchos::ParameterList & rset, const bool present) {
      type = rset.get<std::string>("type", present ? "regularizer" : "mass");
      apply = rset.get<std::string>("apply", present ? "exact" : "diagonal");
      solver = rset.get<std::string>("direct solver", "KLU2");
      drop_zeros = rset.get<bool>("drop explicit zeros", false);
      strict = false;
      slabs_per_solve = 0;
      ranks = 0;
      hgrad_order = 0;
      if (apply == "iterative") {
        Teuchos::ParameterList & iset = rset.sublist("Iterative Settings");
        krylov = iset.get<std::string>("solver", "CG");
        prec = iset.get<std::string>("preconditioner", "RefMaxwell");
        strict = iset.get<bool>("strict", false);
        slabs_per_solve = countOrName(iset, "slabs per solve", {{"auto", 0}});
        TEUCHOS_TEST_FOR_EXCEPTION(krylov == "GMRES" && slabs_per_solve > 1, std::runtime_error,
                                   "Riesz map: 'slabs per solve' above 1 needs solver CG; GMRES solves one slab at a time.");
        ranks = countOrName(iset, "ranks", {{"all", 0}, {"mesh", -1}});
        hgrad_order = countOrName(iset, "hgrad order", {{"control", 0}});
        xml = iset.get<std::string>("xml param file", "");
        belos = Teuchos::rcp(new Teuchos::ParameterList("Belos"));
        belos->set("Convergence Tolerance",
                   static_cast<double>(block_prec::scalarParam(iset, "tolerance", 1.0e-14)));
        belos->set("Maximum Iterations", iset.get<int>("max iterations", 500));
        if (krylov == "GMRES") { belos->set("Num Blocks", 50); }
        belos->setParameters(iset.sublist("Belos Settings"));
      }
      l2_weight = block_prec::scalarParam(rset, "l2 weight", 1.0);
      semi_weight = block_prec::scalarParam(rset, "seminorm weight", 0.0);
      blocks = rset.get<std::string>("blocks", "");
      file = rset.get<std::string>("output file", "");
    }

    // Most slabs per Krylov solve: one for GMRES, up to 8 for 'auto'.
    size_t slabCap() const {
      return krylov == "GMRES" ? 1 : (slabs_per_solve > 0 ? static_cast<size_t>(slabs_per_solve) : 8);
    }

    // A positive count or one of the names; absent means the first name.
    static int countOrName(const Teuchos::ParameterList & list, const std::string & key,
                           const std::vector<std::pair<std::string, int> > & names) {
      if (!list.isParameter(key)) return names[0].second;
      std::string allowed = "a positive count";
      for (const auto & n : names) allowed += " or '" + n.first + "'";
      if (list.isType<int>(key)) {
        const int k = list.get<int>(key);
        TEUCHOS_TEST_FOR_EXCEPTION(k < 1, std::runtime_error, "Riesz map: '" << key << "' must be " << allowed);
        return k;
      }
      const std::string v = list.isType<std::string>(key) ? list.get<std::string>(key) : std::string();
      for (const auto & n : names) {
        if (v == n.first) return n.second;
      }
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Riesz map: '" << key << "' must be " << allowed);
      return names[0].second;
    }
  };

} // namespace MrHyDE

#endif
