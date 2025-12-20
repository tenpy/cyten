
#include "check.h"

#include <vector>
#include <string>


namespace cyten {

  int add(int i, int j) {
    return i +j;
  }

  // typedef unsigned int charge;
  // template <typename Sector>
  // class Symmetry {
  //   public:
  //     typedef std::vector<Sector> SectorArray;
  //     virtual SectorArray fusion_outcomes(Sector a, Sector b) = 0;
  // };

  // class U1Symmetry : public Symmetry<charge> {
  //   public:
  //     SectorArray fusion_outcomes(charge a, charge b) override{
  //       return SectorArray{a + b};
  //     }
  // } u1_symmetry;

  // template<int N>
  // class SUNSymmetry : public Symmetry<std::array<charge, N>> {
  //   public:
  //     using Sector = std::array<charge, N>;
  //     using SectorArray = std::vector<Sector>;
  //     SectorArray fusion_outcomes(Sector a, Sector b) override{
  //       SectorArray result;
  //       for (size_t i = 0; i < 10; i++)
  //       {
  //         Sector c;
  //         for (size_t j = 0; j < 10; j++)
  //           c[j] = a[j] + b[j] + j;  // TODO: that's wrong, just to check...
  //         result.push_back(c);
  //       }
  //     }
  // };
  // SUNSymmetry<2> SU2;
  // SUNSymmetry<3> SU3;

  // class ProductSymmetry: public Symmetry<std::vector<charge>> {
  //     using Sector = std::vector<charge>;
  //     using SectorArray = std::vector<Sector>;
  //     SectorArray fusion_outcomes(Sector a, Sector b) override{
  //       SectorArray result;
  //       for (size_t i = 0; i < 10; i++)
  //       {
  //         Sector c;
  //         for (size_t j = 0; j < 10; j++)
  //           c[j] = a[j] + b[j] + j;
  //         result.push_back(c);
  //       }
  //     }
  // };


}
