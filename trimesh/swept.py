"""
swept.py
---------

Continuous (swept) collision queries against triangular meshes.

Currently implements a sphere swept along a straight line segment,
reporting the earliest time of contact in the unit interval rather
than sampling fixed timesteps which can skip over thin features.
"""

import numpy as np

from . import triangles as _triangles
from .constants import tol
from .typed import ArrayLike, NDArray

# tolerance on the segment parameter for accepting
# roots and for grouping simultaneous contacts as ties
_TOL_TIME = 1e-9

# relative tolerance on quadratic discriminants
# so exact tangencies are accepted as contacts
_TOL_DISC = 1e-9

# tolerance on barycentric coordinates for a contact
# point to be considered inside a triangle
_TOL_BARY = 1e-8


class SweptSphereResult:
    """
    The result of a swept sphere query against a mesh.

    Attributes
    ----------
    hit : bool
      True if the sphere makes contact for some parameter in [0, 1].
    t : float
      Parameter of the earliest contact where the sphere center is
      (start + t * (end - start)); nan if there was no contact.
    point : (3,) float
      Contact position on the mesh surface.
    normal : (3,) float
      Contact normal pointing from the surface toward the sphere center.
    face : int
      Index of the primary contact face: the lowest face
      index among the faces tied at the earliest contact time.
    faces : (k,) int
      Indices of every face in contact at the earliest time, sorted
      ascending, so simultaneous contacts are preserved and reproducible.
    center : (3,) float
      Sphere center at the moment of first contact.
    """

    def __init__(
        self,
        hit: bool,
        t: float,
        point: NDArray,
        normal: NDArray,
        face: int,
        faces: NDArray,
        center: NDArray,
    ):
        self.hit = hit
        self.t = t
        self.point = point
        self.normal = normal
        self.face = face
        self.faces = faces
        self.center = center

    def __repr__(self):
        if not self.hit:
            return "<SweptSphereResult no contact>"
        return f"<SweptSphereResult t={self.t:.6f} face={self.face}>"


def _unitize_fallback(vectors: NDArray, fallback: NDArray) -> NDArray:
    """
    Unitize rows of a vector array, substituting a fallback
    row wherever the vector is too short to normalize.
    """
    length = np.sqrt((vectors * vectors).sum(axis=1))
    result = np.array(fallback, dtype=np.float64, copy=True)
    ok = length > tol.zero
    if ok.any():
        result[ok] = vectors[ok] / length[ok].reshape((-1, 1))
    return result


def swept_sphere(
    mesh, start: ArrayLike, end: ArrayLike, radius: float
) -> SweptSphereResult:
    """
    Sweep a sphere along a straight line segment and find the
    earliest contact with the triangles of a mesh.

    Contact against triangle interiors, edges, and vertices is
    computed analytically so no fixed timestep sampling is used
    and thin features cannot be skipped. A sphere already
    intersecting the mesh at the start reports t = 0.0, and an
    exactly tangent configuration counts as a contact.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to query against.
    start : (3,) float
      Center of the sphere at parameter t = 0.
    end : (3,) float
      Center of the sphere at parameter t = 1.
    radius : float
      Radius of the swept sphere.

    Returns
    -------
    result : SweptSphereResult
      Earliest contact parameter in [0, 1], contact position,
      normal, primary face index, and all tied face indices.
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("start and end must both be (3,) points")
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be a nonnegative finite value")

    direction = end - start
    # squared length of the swept segment
    segment2 = float(np.dot(direction, direction))

    # broadphase: the AABB of the segment inflated by the radius
    # contains every position occupied by the sphere during the
    # motion so no triangle the sweep can touch is excluded
    lower = np.minimum(start, end) - radius
    upper = np.maximum(start, end) + radius
    try:
        candidates = np.array(
            list(mesh.triangles_tree.intersection(np.concatenate((lower, upper)))),
            dtype=np.int64,
        )
    except BaseException:
        # if no bounds tree is available fall back to every triangle
        candidates = np.arange(len(mesh.faces), dtype=np.int64)

    nan3 = np.full(3, np.nan)
    if len(candidates) == 0:
        return SweptSphereResult(
            hit=False,
            t=np.nan,
            point=nan3.copy(),
            normal=nan3.copy(),
            face=-1,
            faces=np.array([], dtype=np.int64),
            center=nan3.copy(),
        )

    # sort candidates so results do not depend on tree return order
    candidates = np.unique(candidates)
    tri = np.array(mesh.triangles[candidates], dtype=np.float64)
    count = len(tri)

    face_normal, face_ok = _triangles.normals(triangles=tri)

    # per-triangle best contact found so far
    best_t = np.full(count, np.inf)
    best_point = np.full((count, 3), np.nan)
    best_normal = np.zeros((count, 3))

    def consider(valid, t_hit, point_hit, normal_hit):
        # merge a batch of contact candidates into the per-triangle best
        ok = valid & (t_hit > -_TOL_TIME) & (t_hit < 1.0 + _TOL_TIME)
        if not ok.any():
            return
        t_clip = np.clip(np.where(ok, t_hit, np.inf), 0.0, 1.0)
        better = t_clip < best_t
        if not better.any():
            return
        best_t[better] = t_clip[better]
        best_point[better] = point_hit[better]
        best_normal[better] = normal_hit[better]

    # a sphere already intersecting the triangle contacts at t = 0
    closest = _triangles.closest_point(tri, np.tile(start, (count, 1)))
    offset = start - closest
    overlapped = (offset * offset).sum(axis=1) <= (radius + tol.merge) ** 2
    if overlapped.any():
        consider(
            overlapped,
            np.zeros(count),
            closest,
            _unitize_fallback(offset, face_normal),
        )

    if segment2 > tol.zero:
        # sphere versus each vertex: solve |start + t * d - v|^2 = r^2
        for k in range(3):
            vertex = tri[:, k, :]
            relative = start - vertex
            qb = 2.0 * (relative * direction).sum(axis=1)
            qc = (relative * relative).sum(axis=1) - radius**2
            disc = qb * qb - 4.0 * segment2 * qc
            valid = disc >= -_TOL_DISC * np.maximum(qb * qb, 1.0)
            if not valid.any():
                continue
            t_hit = np.full(count, np.inf)
            t_hit[valid] = (-qb[valid] - np.sqrt(np.clip(disc[valid], 0.0, None))) / (
                2.0 * segment2
            )
            center = np.full((count, 3), np.nan)
            center[valid] = start + t_hit[valid].reshape((-1, 1)) * direction
            consider(
                valid, t_hit, vertex, _unitize_fallback(center - vertex, face_normal)
            )

        # sphere versus each edge: solve for the center entering the
        # cylinder of radius radius around the edge segment, with
        # endpoints handled by the vertex tests above
        for index_a, index_b in ((0, 1), (1, 2), (2, 0)):
            edge_start = tri[:, index_a, :]
            axis = tri[:, index_b, :] - edge_start
            edge_length = np.sqrt((axis * axis).sum(axis=1))
            live = edge_length > tol.zero
            if not live.any():
                continue
            unit = np.zeros_like(axis)
            unit[live] = axis[live] / edge_length[live].reshape((-1, 1))
            # split offset and motion into axial and radial components
            relative = start - edge_start
            axial_offset = (relative * unit).sum(axis=1)
            axial_motion = (direction * unit).sum(axis=1)
            radial_offset = relative - axial_offset.reshape((-1, 1)) * unit
            radial_motion = direction - axial_motion.reshape((-1, 1)) * unit
            qa = (radial_motion * radial_motion).sum(axis=1)
            qb = 2.0 * (radial_offset * radial_motion).sum(axis=1)
            qc = (radial_offset * radial_offset).sum(axis=1) - radius**2
            disc = qb * qb - 4.0 * qa * qc
            valid = (
                live & (qa > tol.zero) & (disc >= -_TOL_DISC * np.maximum(qb * qb, 1.0))
            )
            if not valid.any():
                continue
            t_hit = np.full(count, np.inf)
            t_hit[valid] = (-qb[valid] - np.sqrt(np.clip(disc[valid], 0.0, None))) / (
                2.0 * qa[valid]
            )
            # contact must lie within the edge segment axially
            along = np.full(count, np.inf)
            along[valid] = axial_offset[valid] + t_hit[valid] * axial_motion[valid]
            valid = valid & (along > -tol.merge) & (along < edge_length + tol.merge)
            if not valid.any():
                continue
            point = edge_start + np.clip(along, 0.0, edge_length).reshape((-1, 1)) * unit
            center = np.full((count, 3), np.nan)
            center[valid] = start + t_hit[valid].reshape((-1, 1)) * direction
            consider(valid, t_hit, point, _unitize_fallback(center - point, face_normal))

        # sphere versus the triangle plane: solve for the center
        # reaching distance radius from the plane on either side,
        # then check the contact point lies inside the triangle
        distance0 = ((start - tri[:, 0, :]) * face_normal).sum(axis=1)
        motion_normal = (direction * face_normal).sum(axis=1)
        moving = face_ok & (np.abs(motion_normal) > tol.zero)
        for side in (1.0, -1.0):
            t_hit = np.full(count, np.inf)
            t_hit[moving] = (side * radius - distance0[moving]) / motion_normal[moving]
            center = np.full((count, 3), np.nan)
            center[moving] = start + t_hit[moving].reshape((-1, 1)) * direction
            point = center - (side * radius) * face_normal
            barycentric = _triangles.points_to_barycentric(tri, point)
            inside = (barycentric > -_TOL_BARY).all(axis=1) & (
                barycentric < 1.0 + _TOL_BARY
            ).all(axis=1)
            consider(moving & inside, t_hit, point, face_normal * side)

    if not np.isfinite(best_t).any():
        return SweptSphereResult(
            hit=False,
            t=np.nan,
            point=nan3.copy(),
            normal=nan3.copy(),
            face=-1,
            faces=np.array([], dtype=np.int64),
            center=nan3.copy(),
        )

    # the earliest contact time is an exact minimum over every
    # candidate so it cannot depend on triangle input order
    t_min = best_t.min()
    tied = np.nonzero(best_t <= t_min + _TOL_TIME)[0]
    # candidates are sorted ascending so the first tied
    # entry is the reproducible lowest-index primary face
    primary = tied[0]
    t_final = float(t_min)
    return SweptSphereResult(
        hit=True,
        t=t_final,
        point=best_point[primary],
        normal=best_normal[primary],
        face=int(candidates[primary]),
        faces=candidates[tied],
        center=start + t_final * direction,
    )
