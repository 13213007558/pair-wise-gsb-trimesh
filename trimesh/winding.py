"""
winding.py
----------

Generalized winding number queries for triangle meshes.

The generalized winding number of a point is the sum of the signed
solid angles of every triangle as seen from that point, divided by
4*pi. Unlike 'Trimesh.contains' this does not assume the mesh is a
closed, consistently oriented volume: the result is a continuous
value which is meaningful for meshes with holes, self-intersections,
or flipped faces.

No attempt is made to fill holes or re-orient faces: the winding
number is computed from the mesh exactly as it is stored.
"""

import enum
from dataclasses import dataclass

import numpy as np

from . import proximity, triangles, util
from .constants import tol
from .typed import ArrayLike, NDArray
from .util import diagonal_dot


class BoundaryType(enum.IntFlag):
    """
    Flag describing what a query point is close to, used to mark
    points where the winding number is discontinuous or ill-defined.
    """

    NONE = 0
    FACE = 1
    EDGE = 2
    VERTEX = 4


@dataclass
class WindingResult:
    """
    Result of a generalized winding number query.

    Attributes
    ----------
    winding : (n, ) float
      Continuous generalized winding number for each query point.
      A closed outward-oriented surface gives values near 1.0 inside
      and 0.0 outside; open or self-intersecting meshes give
      intermediate continuous values.
    boundary : (n, ) int
      A 'BoundaryType' flag for each query point marking points
      close to a face, edge, or vertex where the winding number
      jumps or is ill-defined.
    """

    winding: NDArray[np.float64]
    boundary: NDArray[np.int64]

    def __iter__(self):
        # allow 'winding, boundary = mesh.winding_numbers(...)'
        return iter((self.winding, self.boundary))


def solid_angles(triangles: ArrayLike, point: ArrayLike) -> NDArray[np.float64]:
    """
    Compute the signed solid angle of every triangle as seen from
    a single query point using the formula of
    Van Oosterom and Strackee (1983).

    The sign of each angle depends on the orientation of the
    triangle: no re-orientation is performed.

    Parameters
    ----------
    triangles : (m, 3, 3) float
      Triangle vertices in space.
    point : (3, ) float
      Query point in space.

    Returns
    -------
    angles : (m, ) float
      Signed solid angle of each triangle in steradians.
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    point = np.asanyarray(point, dtype=np.float64)

    # vectors from the query point to each triangle vertex
    a = triangles[:, 0] - point
    b = triangles[:, 1] - point
    c = triangles[:, 2] - point

    la = np.sqrt(diagonal_dot(a, a))
    lb = np.sqrt(diagonal_dot(b, b))
    lc = np.sqrt(diagonal_dot(c, c))

    numerator = diagonal_dot(a, np.cross(b, c))
    denominator = (
        la * lb * lc
        + diagonal_dot(a, b) * lc
        + diagonal_dot(b, c) * la
        + diagonal_dot(c, a) * lb
    )

    with np.errstate(all="ignore"):
        angles = 2.0 * np.arctan2(numerator, denominator)

    # degenerate configurations (query point on a vertex, zero-area
    # triangles) produce undefined angles: contribute nothing here
    # and rely on the boundary flags to mark the query point
    return np.nan_to_num(angles)


def boundary_flags(
    mesh, points: ArrayLike, boundary_tol: float | None = None
) -> NDArray[np.int64]:
    """
    Mark query points which are close to the mesh surface, where
    the winding number is discontinuous or ill-defined.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to query.
    points : (n, 3) float
      Points in space.
    boundary_tol : float or None
      Distance below which a point is considered to be on the
      boundary. Defaults to 'tol.merge * mesh.scale'.

    Returns
    -------
    flags : (n, ) int
      A 'BoundaryType' value for each point: FACE, EDGE, or VERTEX
      for points near the surface, NONE otherwise.
    """
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError("points must be (n, 3)")

    if boundary_tol is None:
        boundary_tol = tol.merge * mesh.scale

    flags = np.zeros(len(points), dtype=np.int64)
    if len(points) == 0 or len(mesh.faces) == 0:
        return flags

    # closest point on the mesh for every query point
    closest, distance, triangle_id = proximity.closest_point(mesh, points)

    # barycentric coordinates of the closest point tell us whether
    # it lies in a face interior, on an edge, or on a vertex
    barycentric = triangles.points_to_barycentric(
        mesh.triangles.view(np.ndarray)[triangle_id], closest
    )
    zeros = (np.abs(barycentric) < tol.merge).sum(axis=1)

    near = distance <= boundary_tol
    flags[near & (zeros == 0)] = BoundaryType.FACE
    flags[near & (zeros == 1)] = BoundaryType.EDGE
    flags[near & (zeros >= 2)] = BoundaryType.VERTEX
    return flags


def winding_numbers_exact(
    mesh, points: ArrayLike, boundary_tol: float | None = None
) -> WindingResult:
    """
    Compute the generalized winding number of every query point by
    directly summing the signed solid angle of every triangle.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to query, used exactly as stored.
    points : (n, 3) float
      Points in space.
    boundary_tol : float or None
      Distance below which a point is marked as being on the boundary.

    Returns
    -------
    result : WindingResult
      Continuous winding numbers and boundary flags for each point.
    """
    points = np.asanyarray(points, dtype=np.float64)
    if not util.is_shape(points, (-1, 3)):
        raise ValueError("points must be (n, 3)")

    current = mesh.triangles.view(np.ndarray)
    winding = np.array([solid_angles(current, p).sum() for p in points], dtype=np.float64)
    winding /= 4.0 * np.pi

    return WindingResult(
        winding=winding, boundary=boundary_flags(mesh, points, boundary_tol)
    )


class WindingTree:
    """
    A binary hierarchy over triangles for approximate generalized
    winding number queries, using a Barnes-Hut style traversal.

    Nodes whose bounding sphere is far enough from the query point
    (controlled by the opening angle 'theta') are approximated as a
    single dipole: the summed vector area of the node acting at its
    area-weighted centroid. Smaller 'theta' values are more accurate
    and slower; the approximation error decays roughly as theta**2.

    The tree captures the triangles passed at construction time: if
    the mesh geometry changes a new tree must be built. The
    'Trimesh.winding_tree' cached property handles this automatically.
    """

    def __init__(self, triangles: ArrayLike, max_leaf_size: int = 8):
        triangles = np.asanyarray(triangles, dtype=np.float64)
        if triangles.size == 0:
            triangles = triangles.reshape((0, 3, 3))
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            raise ValueError("triangles must be (n, 3, 3)")
        if max_leaf_size < 1:
            raise ValueError("max_leaf_size must be positive")

        self._triangles = triangles

        # per-face vector area: orientation is preserved so flipping
        # faces flips the sign of the resulting winding numbers
        if len(triangles) > 0:
            self._area_vector = 0.5 * np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            self._face_centroid = triangles.mean(axis=1)
        else:
            self._area_vector = np.zeros((0, 3))
            self._face_centroid = np.zeros((0, 3))

        # node storage accumulated during the recursive build
        child = []
        span = []
        area_sum = []
        centroid = []
        radius = []
        order: list = []

        def build(indices):
            node = len(child)
            child.append((-1, -1))
            span.append((-1, -1))

            area_sum.append(self._area_vector[indices].sum(axis=0))

            cents = self._face_centroid[indices]
            area_local = self._area_vector[indices]
            weights = np.sqrt(diagonal_dot(area_local, area_local))
            weight_total = weights.sum()
            if weight_total > tol.zero:
                center = (cents * weights.reshape((-1, 1))).sum(axis=0)
                center /= weight_total
            elif len(cents) > 0:
                center = cents.mean(axis=0)
            else:
                center = np.zeros(3)
            centroid.append(center)

            # bounding sphere of all member vertices around the centroid
            vertices = triangles[indices].reshape((-1, 3))
            if len(vertices) > 0:
                offset = vertices - center
                radius.append(np.sqrt(diagonal_dot(offset, offset)).max())
            else:
                radius.append(0.0)

            if len(indices) <= max_leaf_size:
                start = len(order)
                order.extend(indices.tolist())
                span[node] = (start, len(order))
            else:
                # split the longest axis of the centroid bounds at the median
                axis = np.ptp(cents, axis=0).argmax()
                sort = np.argsort(cents[:, axis], kind="stable")
                sorted_indices = indices[sort]
                middle = len(indices) // 2
                left = build(sorted_indices[:middle])
                right = build(sorted_indices[middle:])
                child[node] = (left, right)
            return node

        build(np.arange(len(triangles)))

        self._child = np.array(child, dtype=np.int64)
        self._span = np.array(span, dtype=np.int64)
        self._node_area = np.array(area_sum, dtype=np.float64)
        self._node_centroid = np.array(centroid, dtype=np.float64)
        self._node_radius = np.array(radius, dtype=np.float64)
        # triangles reordered so leaves own contiguous spans
        self._ordered = triangles[np.array(order, dtype=np.int64)]

    def winding_numbers(
        self, points: ArrayLike, theta: float = 0.5
    ) -> NDArray[np.float64]:
        """
        Approximate the generalized winding number of every query
        point using the hierarchical dipole approximation.

        Parameters
        ----------
        points : (n, 3) float
          Points in space.
        theta : float
          Opening angle controlling the error: a node is approximated
          when its bounding sphere radius is smaller than
          'theta * distance' from the query point. Smaller values
          are more accurate and slower.

        Returns
        -------
        winding : (n, ) float
          Approximate continuous generalized winding numbers.
        """
        points = np.asanyarray(points, dtype=np.float64)
        if not util.is_shape(points, (-1, 3)):
            raise ValueError("points must be (n, 3)")
        if not np.isfinite(theta) or theta <= 0.0:
            raise ValueError("theta must be a positive finite value")

        winding = np.array(
            [self._solid_angle_point(p, theta) for p in points], dtype=np.float64
        )
        return winding / (4.0 * np.pi)

    def _solid_angle_point(self, point: NDArray[np.float64], theta: float) -> float:
        """
        Traverse the hierarchy for a single query point, summing
        exact solid angles for nearby leaves and dipole
        approximations for distant nodes.
        """
        total = 0.0
        stack = [0]
        while len(stack) > 0:
            node = stack.pop()
            start, end = self._span[node]
            if start >= 0:
                # leaf node: sum exact solid angles
                total += solid_angles(self._ordered[start:end], point).sum()
                continue

            offset = self._node_centroid[node] - point
            distance = np.sqrt(offset @ offset)
            if self._node_radius[node] < theta * distance:
                # far field: approximate the node as a single dipole
                total += (self._node_area[node] @ offset) / distance**3
            else:
                stack.extend(self._child[node])
        return total


def winding_numbers_approximate(
    mesh,
    points: ArrayLike,
    theta: float = 0.5,
    boundary_tol: float | None = None,
) -> WindingResult:
    """
    Compute approximate generalized winding numbers using the
    cached hierarchical representation of the mesh.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to query, used exactly as stored.
    points : (n, 3) float
      Points in space.
    theta : float
      Opening angle controlling the approximation error.
    boundary_tol : float or None
      Distance below which a point is marked as being on the boundary.

    Returns
    -------
    result : WindingResult
      Approximate continuous winding numbers and boundary flags.
    """
    points = np.asanyarray(points, dtype=np.float64)
    winding = mesh.winding_tree.winding_numbers(points, theta=theta)
    return WindingResult(
        winding=winding, boundary=boundary_flags(mesh, points, boundary_tol)
    )
