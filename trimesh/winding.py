"""
winding.py
----------

Generalized winding number queries for triangle meshes.

The generalized winding number of a point is the total signed solid
angle of all oriented triangles as seen from that point, divided by
4*pi. Unlike a boolean inside/outside ('contains') query it does not
assume the mesh is a closed watertight solid: open surfaces, meshes
with holes, and overlapping shells all produce meaningful continuous
values.

No attempt is made to repair, close, or re-orient the mesh: the
returned values reflect the triangle orientations exactly as stored,
so reversing every face flips the sign of the result.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from . import proximity
from .constants import tol
from .triangles import cross as _tri_cross
from .typed import ArrayLike, NDArray, float64, int64
from .util import diagonal_dot


class BoundaryType(IntEnum):
    """
    Classification of a query point relative to the mesh surface,
    used for the 'boundary' field of 'WindingNumberResult'.
    """

    NONE = 0  # not near the surface
    FACE = 1  # near the interior of a triangle
    EDGE = 2  # near a triangle edge
    VERTEX = 3  # near a mesh vertex


@dataclass
class WindingNumberResult:
    """
    Result of a generalized winding number query.

    Attributes
    ----------
    winding : (n,) float
      Continuous generalized winding number for each query point,
      i.e. the total signed solid angle divided by 4*pi. Values are
      NOT thresholded into inside/outside booleans.
    boundary : (n,) BoundaryType
      Flag marking points close to a face, edge, or vertex where
      the winding number is ill-defined or jumps.
    """

    winding: NDArray[float64]
    boundary: NDArray[int64]

    @property
    def on_boundary(self) -> NDArray[np.bool_]:
        """
        Boolean mask for query points flagged as near the surface.
        """
        return self.boundary != BoundaryType.NONE

    def __iter__(self):
        # allow 'winding, boundary = mesh.winding_number(...)'
        return iter((self.winding, self.boundary))


def solid_angle(
    triangles: NDArray[float64], points: NDArray[float64]
) -> NDArray[float64]:
    """
    Compute the exact signed solid angle of each triangle as seen
    from each query point using the formula of
    Van Oosterom and Strackee (1983).

    Parameters
    ----------
    triangles : (m, 3, 3) float
      Oriented triangle vertices.
    points : (n, 3) float
      Query points.

    Returns
    -------
    angles : (n, m) float
      Signed solid angle of triangle j seen from point i in
      steradians, in the range [-2*pi, 2*pi]. The sign follows the
      stored orientation of each triangle.
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    points = np.asanyarray(points, dtype=np.float64)

    # vertices relative to each query point: (n, m, 3)
    a = triangles[None, :, 0] - points[:, None]
    b = triangles[None, :, 1] - points[:, None]
    c = triangles[None, :, 2] - points[:, None]

    la = np.sqrt((a * a).sum(axis=2))
    lb = np.sqrt((b * b).sum(axis=2))
    lc = np.sqrt((c * c).sum(axis=2))

    numerator = (a * np.cross(b, c)).sum(axis=2)
    denominator = (
        la * lb * lc
        + (a * b).sum(axis=2) * lc
        + (b * c).sum(axis=2) * la
        + (c * a).sum(axis=2) * lb
    )
    return 2.0 * np.arctan2(numerator, denominator)


def winding_number_exact(mesh, points: ArrayLike, chunk: int = 2**20) -> NDArray[float64]:
    """
    The exact generalized winding number computed by directly
    summing the signed solid angle of every triangle.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Source mesh, used exactly as stored.
    points : (n, 3) float
      Query points.
    chunk : int
      Maximum number of point-triangle pairs evaluated at once,
      bounding memory use for large queries.

    Returns
    -------
    winding : (n,) float
      Continuous winding number for each query point.
    """
    points = np.asanyarray(points, dtype=np.float64)
    triangles = mesh.triangles.view(np.ndarray)

    winding = np.zeros(len(points), dtype=np.float64)
    if len(points) == 0 or len(triangles) == 0:
        return winding

    # evaluate in blocks so the (n, m) product stays bounded
    step = max(1, int(chunk) // len(triangles))
    for start in range(0, len(points), step):
        block = points[start : start + step]
        winding[start : start + step] = solid_angle(triangles, block).sum(axis=1)
    return winding / (4.0 * np.pi)


class WindingTree:
    """
    A binary bounding-volume hierarchy over triangles used for
    approximate generalized winding number queries.

    Each node stores the dipole of its triangles (total area vector
    and area-weighted centroid). Clusters that are far enough from a
    query point are approximated by their dipole instead of summing
    every triangle, giving O(log n) behavior per point.
    """

    def __init__(self, triangles: NDArray[float64], leaf_size: int = 8):
        """
        Build a hierarchy from oriented triangles.

        Parameters
        ----------
        triangles : (m, 3, 3) float
          Triangle vertices, orientation is significant.
        leaf_size : int
          Maximum number of triangles evaluated exactly per leaf.
        """
        triangles = np.asanyarray(triangles, dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("triangles must be (m, 3, 3)")
        self._leaf_size = max(1, int(leaf_size))

        # per-triangle dipole data
        center = triangles.mean(axis=1)
        area_vector = 0.5 * _tri_cross(triangles)
        area = np.sqrt(diagonal_dot(area_vector, area_vector))
        # radius of a sphere around each triangle center
        # containing the whole triangle
        tri_radius = np.sqrt(((triangles - center[:, None]) ** 2).sum(axis=2)).max(axis=1)

        # node storage, appended during the recursive build
        node_centroid = []
        node_area_vector = []
        node_radius = []
        node_left = []
        node_right = []
        node_start = []
        node_count = []
        # triangle indices of every leaf, concatenated at the end so
        # each leaf references a contiguous span of reordered triangles
        leaf_order = []
        leaf_offset = [0]

        def build(indices: NDArray[int64]) -> int:
            node = len(node_centroid)
            weights = area[indices]
            total = weights.sum()
            if total > tol.zero:
                centroid = (center[indices] * weights[:, None]).sum(axis=0) / total
            else:
                # all triangles are degenerate: fall back to a mean
                centroid = center[indices].mean(axis=0)
            node_centroid.append(centroid)
            node_area_vector.append(area_vector[indices].sum(axis=0))
            node_radius.append(
                (
                    np.sqrt(((center[indices] - centroid) ** 2).sum(axis=1))
                    + tri_radius[indices]
                ).max()
            )
            # append placeholders so child nodes can be appended
            # during the recursion without misaligning the arrays
            node_left.append(-1)
            node_right.append(-1)
            node_start.append(-1)
            node_count.append(0)
            if len(indices) <= self._leaf_size:
                node_start[node] = leaf_offset[0]
                node_count[node] = len(indices)
                leaf_order.append(indices)
                leaf_offset[0] += len(indices)
                return node
            # split on the axis with the largest spread of centers
            axis = np.ptp(center[indices], axis=0).argmax()
            median = np.median(center[indices, axis])
            left_mask = center[indices, axis] <= median
            # degenerate split (all centers equal): split in half
            if left_mask.all() or not left_mask.any():
                left_mask = np.zeros(len(indices), dtype=bool)
                left_mask[: len(indices) // 2] = True
            left = build(indices[left_mask])
            right = build(indices[~left_mask])
            node_left[node] = left
            node_right[node] = right
            return node

        if len(triangles) > 0:
            build(np.arange(len(triangles)))
        # reorder triangles so every leaf is a contiguous span
        if len(leaf_order) > 0:
            self._triangles = triangles[np.concatenate(leaf_order)]
        else:
            self._triangles = triangles[:0]
        self._centroid = np.array(node_centroid, dtype=np.float64).reshape((-1, 3))
        self._area_vector = np.array(node_area_vector, dtype=np.float64).reshape((-1, 3))
        self._radius = np.array(node_radius, dtype=np.float64)
        self._left = np.array(node_left, dtype=np.int64)
        self._right = np.array(node_right, dtype=np.int64)
        self._start = np.array(node_start, dtype=np.int64)
        self._count = np.array(node_count, dtype=np.int64)

    def winding_number(self, points: ArrayLike, beta: float = 2.0) -> NDArray[float64]:
        """
        Approximate generalized winding number for a batch of points.

        A node is approximated by its dipole when the query point is
        farther than 'beta * node_radius' from the node centroid, so
        larger 'beta' values are more accurate and slower; the error
        of the dipole approximation decays roughly like beta**-2.

        Parameters
        ----------
        points : (n, 3) float
          Query points.
        beta : float
          Error control parameter, must be positive.

        Returns
        -------
        winding : (n,) float
          Approximate continuous winding number per point.
        """
        points = np.asanyarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be (n, 3)")
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta must be a positive number")

        winding = np.zeros(len(points), dtype=np.float64)
        if len(points) == 0 or len(self._centroid) == 0:
            return winding

        # depth-first traversal processing points in vectorized
        # batches: each stack item is a node and the indices of the
        # query points that still need to visit it
        stack = [(0, np.arange(len(points)))]
        while len(stack) > 0:
            node, index = stack.pop()
            diff = self._centroid[node] - points[index]
            distance = np.sqrt(diagonal_dot(diff, diff))
            # far enough to use the dipole approximation
            far = distance > beta * self._radius[node]
            if far.any():
                idx = index[far]
                winding[idx] += (diff[far] @ self._area_vector[node]) / (
                    distance[far] ** 3
                )
            index = index[~far]
            if len(index) == 0:
                continue
            if self._left[node] < 0:
                # leaf: sum the stored triangles exactly
                start = self._start[node]
                leaf = self._triangles[start : start + self._count[node]]
                winding[index] += solid_angle(leaf, points[index]).sum(axis=1)
            else:
                stack.append((self._left[node], index))
                stack.append((self._right[node], index))
        return winding / (4.0 * np.pi)


def boundary_type(mesh, points: ArrayLike, boundary_tol: float) -> NDArray[int64]:
    """
    Classify query points that are close to the mesh surface, where
    the winding number is ill-defined or discontinuous.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Source mesh.
    points : (n, 3) float
      Query points.
    boundary_tol : float
      Distance below which a point is flagged as being on a face,
      edge, or vertex.

    Returns
    -------
    boundary : (n,) BoundaryType
      Classification for each query point.
    """
    points = np.asanyarray(points, dtype=np.float64)
    boundary = np.zeros(len(points), dtype=np.int64)
    if len(points) == 0 or len(mesh.faces) == 0:
        return boundary

    closest, distance, triangle_id = proximity.closest_point(mesh, points)
    near = distance <= boundary_tol
    if not near.any():
        return boundary

    tri = mesh.triangles.view(np.ndarray)[triangle_id[near]]
    point = closest[near]

    # distance from the closest point to each triangle vertex
    vertex_dist = np.sqrt(((tri - point[:, None]) ** 2).sum(axis=2)).min(axis=1)

    # distance from the closest point to each triangle edge
    edge_dist = np.full(len(tri), np.inf)
    for k in range(3):
        origin = tri[:, k]
        segment = tri[:, (k + 1) % 3] - origin
        length_sq = diagonal_dot(segment, segment)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                length_sq > tol.zero,
                ((point - origin) * segment).sum(axis=1) / length_sq,
                0.0,
            )
        on_segment = origin + np.clip(t, 0.0, 1.0)[:, None] * segment
        edge_dist = np.minimum(
            edge_dist, np.sqrt(((point - on_segment) ** 2).sum(axis=1))
        )

    flags = np.full(len(tri), BoundaryType.FACE, dtype=np.int64)
    flags[edge_dist <= boundary_tol] = BoundaryType.EDGE
    flags[vertex_dist <= boundary_tol] = BoundaryType.VERTEX
    boundary[near] = flags
    return boundary


def winding_number(
    mesh,
    points: ArrayLike,
    method: str = "tree",
    beta: float = 2.0,
    boundary_tol: float | None = None,
    tree: WindingTree | None = None,
) -> WindingNumberResult:
    """
    Compute the generalized winding number of a mesh at query points.

    The result is the total signed solid angle of all oriented
    triangles divided by 4*pi, kept as a continuous value rather than
    a thresholded inside/outside boolean. The mesh is used exactly as
    stored: no holes are filled and no faces are re-oriented, so open
    surfaces produce fractional values and reversing every face flips
    the sign of the result.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Source mesh, used exactly as stored.
    points : (n, 3) float
      Query points.
    method : 'tree' or 'exact'
      'exact' sums the solid angle of every triangle directly,
      'tree' uses a cached bounding volume hierarchy with a dipole
      approximation which is much faster for batched queries.
    beta : float
      Error control for the 'tree' method: clusters farther than
      'beta * cluster_radius' are approximated by their dipole.
      Larger values are more accurate and slower.
    boundary_tol : float or None
      Distance below which points are flagged as being on a face,
      edge, or vertex. If None a small multiple of the mesh scale
      is used; pass 0 to disable boundary classification.
    tree : WindingTree or None
      Precomputed hierarchy for the 'tree' method, built from the
      current mesh triangles if not provided.

    Returns
    -------
    result : WindingNumberResult
      Continuous 'winding' values and 'boundary' flags per point.
    """
    points = np.asanyarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (n, 3)")

    if method == "exact":
        winding = winding_number_exact(mesh, points)
    elif method == "tree":
        if tree is None:
            tree = WindingTree(mesh.triangles.view(np.ndarray))
        winding = tree.winding_number(points, beta=beta)
    else:
        raise ValueError(f"method must be 'tree' or 'exact', got {method!r}")

    if boundary_tol is None:
        boundary_tol = tol.merge * mesh.scale
    if boundary_tol > 0.0:
        boundary = boundary_type(mesh, points, boundary_tol)
    else:
        boundary = np.zeros(len(points), dtype=np.int64)

    return WindingNumberResult(winding=winding, boundary=boundary)
