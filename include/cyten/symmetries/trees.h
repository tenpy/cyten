#pragma once

#include "../block_backend/block_backend.h"
#include "../block_backend/dtypes.h"
#include "exceptions.h"
#include "sector.h"
#include "styles.h"
#include "symmetry.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cyten {

/// Linear combination of fusion trees: ``sum_i coeff_i * tree_i``.
using FusionTreeLinearCombination = std::map<class FusionTree, complex128>;

/// Linear combination of tree pairs ``(Y_i, X_i)`` (as in :meth:`FusionTree.bend_leg`).
using FusionTreePairLinearCombination =
  std::map<std::pair<class FusionTree, class FusionTree>, complex128>;

/// A fusion tree, which represents the map from uncoupled to coupled sectors.
///
/// Consider the following example tree::
///
///     FusionTree(
///         symmetry=symmetry,
///         coupled=coupled,
///         uncoupled=[a, b, c, d],
///         are_dual=[False, True, True, False],
///         inner_sectors=[x, y],
///         multiplicities=[i, j, k],
///     )
///
/// Graphically::
///
///     |    a     b     c     d     <- isomorphic to pre_Z_uncoupled
///     |    v     ^     ^     v        e.g. dual(b) iso to pre_Z_uncoupled[1]
///     |    │     Z     Z     │
///     |    v     v     v     v
///     |    a     b     c     d     <- uncoupled
///     |    ╰──i──╯     │     │
///     |      x│        │     │
///     |       ╰───j────╯     │
///     |          y│          │
///     |           ╰────k─────╯
///     |                │
///     |                coupled
class FusionTree
{
  public:
    Symmetry::Ptr symmetry;
    SectorArray uncoupled;
    std::size_t num_uncoupled = 0;
    std::size_t num_vertices = 0;
    std::size_t num_inner_edges = 0;
    Sector coupled;
    /// Length ``num_uncoupled``; ``1`` means a Z isomorphism above that uncoupled sector.
    std::vector<std::uint8_t> are_dual;
    SectorArray inner_sectors;
    std::vector<int64> multiplicities;
    FusionStyle fusion_style{};
    bool is_abelian = false;
    BraidingStyle braiding_style{};

    FusionTree(Symmetry::Ptr symmetry,
               SectorArray uncoupled,
               Sector coupled,
               std::vector<std::uint8_t> are_dual,
               SectorArray inner_sectors,
               std::optional<std::vector<int64>> multiplicities = std::nullopt);

    /// Perform sanity checks.
    void test_sanity() const;

    /// Assume an abelian symmetry and build the unique tree with the given `uncoupled`.
    static FusionTree from_abelian_symmetry(Symmetry::Ptr symmetry,
                                            SectorArray const& uncoupled,
                                            std::vector<std::uint8_t> const& are_dual);

    /// The empty tree with no uncoupled sectors.
    static FusionTree from_empty(Symmetry::Ptr symmetry);

    /// A tree with a single uncoupled sector and no nodes.
    static FusionTree from_sector(Symmetry::Ptr symmetry, Sector sector, bool is_dual);

    /// The uncoupled sectors *above* any Z isomorphisms.
    [[nodiscard]] SectorArray pre_Z_uncoupled() const;

    [[nodiscard]] std::size_t hash() const;
    [[nodiscard]] bool operator==(FusionTree const& other) const;
    [[nodiscard]] bool operator<(FusionTree const& other) const;

    /// Visual representation of the tree as ASCII art.
    [[nodiscard]] std::string ascii_diagram(bool dagger = false) const;

    /// Helper function for string representation (also used by ``fusion_trees``).
    static std::string str_uncoupled_coupled(Symmetry const& symmetry,
                                             SectorArray const& uncoupled,
                                             Sector coupled,
                                             std::vector<std::uint8_t> const& are_dual);

    /// Bend a leg on a tree-pair, return the resulting linear combination of tree-pairs.
    static FusionTreePairLinearCombination bend_leg(FusionTree const& X,
                                                    FusionTree const& Y,
                                                    bool bend_downward,
                                                    bool do_conj = false);

    /// Braid a leg on a fusion tree, return the resulting linear combination of trees.
    [[nodiscard]] FusionTreeLinearCombination braid(int64 j,
                                                    bool overbraid,
                                                    float64 cutoff = 1e-16,
                                                    bool do_conj = false) const;

    /// For the ``n``-th fusion vertex, get the respective sectors ``(a, b, mu, c)``.
    [[nodiscard]] std::tuple<Sector, Sector, int64, Sector> vertex_labels(int64 n) const;

    /// Update the multiplicity and the three sectors around the ``n``-th vertex.
    FusionTree modify_vertex_labels(int64 n,
                                    Sector a,
                                    Sector b,
                                    int64 mu,
                                    Sector c,
                                    bool copy = true);

    [[nodiscard]] std::string str() const;
    [[nodiscard]] std::string repr() const;

    /// Get the matrix elements of the map as a backend Block.
    ///
    /// If ``backend`` is null, uses :class:`NumpyBlockBackend` on CPU.
    /// Optional Python ``TensorBackend`` may be passed via the pybind binding instead.
    [[nodiscard]] BlockBackend::BlockPtr to_dense_block(BlockBackend* backend = nullptr,
                                                        std::optional<Dtype> dtype = std::nullopt,
                                                        bool understood_braiding = false) const;

    /// Return a shallow (or deep) copy.
    [[nodiscard]] FusionTree copy(bool deep = true) const;

    /// A new tree, from adding a new fusion node at the bottom, below the coupled sector.
    [[nodiscard]] FusionTree extended(Sector new_uncoupled,
                                      int64 mu,
                                      Sector new_coupled,
                                      bool is_dual) const;

    /// Insert a tree `t2` above the first uncoupled sector.
    [[nodiscard]] FusionTree insert(FusionTree const& t2) const;

    /// Insert a tree `t2` above the `n`-th uncoupled sector.
    [[nodiscard]] FusionTreeLinearCombination insert_at(int64 n,
                                                        FusionTree const& t2,
                                                        float64 eps = 1.0e-14) const;

    /// Outer product with another tree.
    [[nodiscard]] FusionTreeLinearCombination outer(FusionTree const& right_tree,
                                                    float64 eps = 1.0e-14) const;

    /// Split into two separate fusion trees.
    [[nodiscard]] std::pair<FusionTree, FusionTree> split(int64 n) const;

    /// Split off the bottom vertex. Returns ``(rest_tree, c, mu, z)``.
    [[nodiscard]] std::tuple<FusionTree, Sector, int64, Sector> split_bottom_vertex() const;

    /// Twist some legs above a tree, return the resulting linear combination of trees.
    [[nodiscard]] FusionTreeLinearCombination twist(std::vector<int64> const& idcs,
                                                    bool overtwist) const;

  private:
    /// The :meth:`ascii_diagram` as a 2D array of single Unicode characters (cols × rows).
    /// Each cell is a UTF-8 string of one codepoint (box-drawing / sector labels).
    [[nodiscard]] std::vector<std::vector<std::string>> ascii_diagram_chars(
      bool dagger,
      int uncoupled_padding = 2,
      int inner_sector_padding = 0) const;
};

/// Iterable over all :class:`FusionTree`\ s with given uncoupled and coupled sectors.
///
/// Efficient ``len`` and :meth:`index` avoid generating all intermediate trees.
///
/// TODO elaborate on canonical order of trees -> reference in module level docstring.
class fusion_trees
{
  public:
    Symmetry::Ptr symmetry;
    SectorArray uncoupled;
    std::size_t num_uncoupled = 0;
    Sector coupled;
    std::vector<std::uint8_t> are_dual;

    fusion_trees(Symmetry::Ptr symmetry,
                 SectorArray uncoupled,
                 Sector coupled,
                 std::optional<std::vector<std::uint8_t>> are_dual = std::nullopt);

    /// Materialize all trees (used by Python ``__iter__``).
    [[nodiscard]] std::vector<FusionTree> all_trees() const;

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::string str() const;
    [[nodiscard]] std::string repr() const;

    /// The index of a given tree in the iterator.
    [[nodiscard]] std::size_t index(FusionTree const& tree) const;

  private:
    [[nodiscard]] std::size_t compute_index(FusionTree const& tree) const;
};

} // namespace cyten

template<>
struct std::hash<cyten::FusionTree>
{
    std::size_t operator()(cyten::FusionTree const& t) const noexcept { return t.hash(); }
};

/// Hash for tree-pair keys used by :class:`cyten::SparseMappingFusionTreePair`.
template<>
struct std::hash<std::pair<cyten::FusionTree, cyten::FusionTree>>
{
    std::size_t operator()(std::pair<cyten::FusionTree, cyten::FusionTree> const& p) const noexcept
    {
        // Boost-style hash_combine of the two tree hashes.
        std::size_t seed = std::hash<cyten::FusionTree>{}(p.first);
        std::size_t h2 = std::hash<cyten::FusionTree>{}(p.second);
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
