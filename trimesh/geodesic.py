"""
geodesic.py
-----------

Approximate geodesic distance on a triangular surface using the
"heat method":

  Crane, Weischedel, Wardetzky. "Geodesics in Heat:
  Computing the Distance on Meshes." 2013.

Rather than walking along mesh edges (which is visibly biased by
the meshing direction) this solves two sparse linear systems with
the cotangent Laplacian: a short heat diffusion step followed by
a Poisson reconstruction of a function whose gradient matches the
normalized heat flow.

Distances returned here are *approximate*: they converge to the
true geodesic distance as the mesh is refined, but they are not
exact shortest paths along the surface.

Boundary conditions are the natural (homogeneous Neumann, i.e.
zero heat flux through the boundary) conditions produced by the
finite element discretization: no special treatment is applied
to boundary edges.
"""

import numpy as np

from .constants import tol

try:
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import splu
except ImportError as E:
    from . import exceptions

    coo_matrix = exceptions.ExceptionWrapper(E)
    diags = exceptions.ExceptionWrapper(E)
    connected_components = exceptions.ExceptionWrapper(E)
    splu = exceptions.ExceptionWrapper(E)


class HeatGeodesic:
    """
    Precompute the discrete operators and factorizations needed
    for heat-method geodesic distances on one mesh, so distances
    from many different source sets can be evaluated without
    re-factorizing.

    The stored factorizations are only valid for the exact
    geometry they were built from: prefer the heat_geodesic
    function, which caches instances on mesh._cache and is
    therefore invalidated automatically when vertices or faces
    change.
    """

    def __init__(self, mesh, t=None):
        """
        Build and factorize the operators for a mesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
          Geometry to compute distances on. Must contain no
          degenerate (zero area) faces and no isolated vertices.
        t : None or float
          Heat diffusion time. Defaults to the squared mean edge
          length, a standard choice for the heat method.
        """
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)

        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError("mesh has no vertices or faces")
        if not np.isfinite(vertices).all():
            raise ValueError("mesh contains non-finite vertices")

        area = np.asarray(mesh.area_faces, dtype=np.float64)
        degenerate = np.nonzero(area <= tol.zero * mesh.scale**2)[0]
        if len(degenerate) > 0:
            raise ValueError(
                f"mesh has {len(degenerate)} degenerate (zero area) faces "
                + f"including index {degenerate[0]}: remove them before "
                + "computing geodesic distances"
            )

        referenced = np.zeros(len(vertices), dtype=bool)
        referenced[faces.ravel()] = True
        isolated = np.nonzero(~referenced)[0]
        if len(isolated) > 0:
            raise ValueError(
                f"mesh has {len(isolated)} vertices not referenced by any "
                + f"face including index {isolated[0]}: remove them with "
                + "remove_unreferenced_vertices"
            )

        # cotangent of the angle at each corner of each face
        cot = 1.0 / np.tan(np.asarray(mesh.face_angles, dtype=np.float64))
        if not np.isfinite(cot).all():
            raise ValueError("mesh face angles are not stably invertible")

        # the edge opposite corner k of a face connects corners k+1, k+2
        next_corner = faces[:, [1, 2, 0]]
        prev_corner = faces[:, [2, 0, 1]]
        weights = 0.5 * cot.ravel()
        # stiffness matrix of the cotangent Laplacian: positive
        # semidefinite with one constant null vector per component
        off_diagonal = coo_matrix(
            (
                np.concatenate([weights, weights]),
                (
                    np.concatenate([next_corner.ravel(), prev_corner.ravel()]),
                    np.concatenate([prev_corner.ravel(), next_corner.ravel()]),
                ),
            ),
            shape=(len(vertices), len(vertices)),
        ).tocsr()
        off_diagonal.sum_duplicates()
        laplacian = diags(np.asarray(off_diagonal.sum(axis=1)).ravel()) - off_diagonal

        # lumped (diagonal) mass matrix: a third of each adjacent face area
        mass = np.bincount(
            faces.ravel(), weights=np.repeat(area / 3.0, 3), minlength=len(vertices)
        )

        # one label per connected component of the surface
        count, components = connected_components(laplacian, directed=False)

        if t is None:
            t = float(np.mean(mesh.edges_unique_length) ** 2)
        t = float(t)
        if not np.isfinite(t) or t <= 0.0:
            raise ValueError("heat diffusion time t must be positive")

        try:
            # heat equation step: (M + t * L) u = delta
            heat_solve = splu((diags(mass) + t * laplacian).tocsc()).solve

            # the Poisson solve is only determined up to a constant per
            # connected component: pin the first vertex of each one
            _, pinned = np.unique(components, return_index=True)
            keep = np.setdiff1d(np.arange(len(vertices)), pinned)
            poisson_solve = splu(laplacian[keep][:, keep].tocsc()).solve
        except RuntimeError as error:
            raise RuntimeError(
                "could not factorize the discrete operators: "
                + "the mesh may be too degenerate to solve stably"
            ) from error

        # rotated edges used for per-face gradients: cross(normal, edge)
        # where the edge is the one opposite each corner
        triangles = vertices[faces]
        normals = np.asarray(mesh.face_normals, dtype=np.float64)
        rotated = [
            np.cross(normals, triangles[:, (k + 2) % 3] - triangles[:, (k + 1) % 3])
            for k in range(3)
        ]

        self.laplacian = laplacian
        self.mass = mass
        self.t = t
        self.components = components
        self.component_count = count

        self._faces = faces
        self._area = area
        self._rotated = rotated
        self._keep = keep
        self._heat_solve = heat_solve
        self._poisson_solve = poisson_solve

    def distance(self, sources):
        """
        Compute the approximate geodesic distance from the closest
        source to every vertex of the mesh.

        Parameters
        ----------
        sources : (m,) int or (face_index, barycentric)
          Either vertex indices, or a tuple of (m,) int face
          indices and (m, 3) float barycentric coordinates for
          sources located inside faces.

        Returns
        -------
        distance : (len(mesh.vertices),) float
          Approximate geodesic distance to the closest source.
          Vertices in connected components containing no source
          are unreachable and get np.inf, never a zero fill.
        """
        delta, ref_index, ref_weight = self._source_delta(sources)
        vertex_count = len(self.components)

        # diffuse the source impulses for time t
        heat = self._heat_solve(delta)
        if not np.isfinite(heat).all():
            raise RuntimeError("heat diffusion solve did not produce finite values")

        # gradient of the diffused heat on each face, normalized
        # to a unit vector field pointing back towards the sources
        face_heat = heat[self._faces]
        gradient = (
            sum(face_heat[:, k, None] * self._rotated[k] for k in range(3))
            / (2.0 * self._area)[:, None]
        )
        magnitude = np.linalg.norm(gradient, axis=1)
        nonzero = magnitude > 0.0
        field = np.zeros_like(gradient)
        field[nonzero] = -gradient[nonzero] / magnitude[nonzero, None]

        # integrated divergence of the unit field at each vertex
        divergence = sum(
            np.bincount(
                self._faces[:, k],
                weights=0.5 * (self._rotated[k] * field).sum(axis=1),
                minlength=vertex_count,
            )
            for k in range(3)
        )

        # reconstruct the distance function up to a constant per component
        distance = np.zeros(vertex_count)
        distance[self._keep] = self._poisson_solve(divergence[self._keep])

        # shift each component containing a source so that the
        # interpolated distance at the source location is zero
        component_min = np.full(self.component_count, np.inf)
        source_level = (distance[ref_index] * ref_weight).sum(axis=1)
        np.minimum.at(component_min, self.components[ref_index[:, 0]], source_level)
        has_source = np.isfinite(component_min)
        distance = distance - component_min[self.components]
        distance[~has_source[self.components]] = np.inf

        if not np.isfinite(distance[has_source[self.components]]).all():
            raise RuntimeError("Poisson solve did not produce finite values")

        return distance

    def _source_delta(self, sources):
        """
        Convert user source specification into a heat impulse vector.

        Returns
        -------
        delta : (n,) float
          Heat impulse per vertex.
        ref_index : (m, 3) int
          Vertex indices interpolating each source location.
        ref_weight : (m, 3) float
          Weights interpolating each source location.
        """
        vertex_count = len(self.components)

        if isinstance(sources, tuple) and len(sources) == 2:
            face_index, barycentric = sources
            face_index = np.asarray(face_index, dtype=np.int64).ravel()
            barycentric = np.asarray(barycentric, dtype=np.float64)
            if barycentric.shape == (3,):
                barycentric = barycentric.reshape(1, 3)
            if len(face_index) == 0:
                raise ValueError("no sources were passed")
            if barycentric.shape != (len(face_index), 3):
                raise ValueError("barycentric coordinates must be (m, 3) float")
            if face_index.min() < 0 or face_index.max() >= len(self._faces):
                raise ValueError("source face index out of range")
            if not np.allclose(barycentric.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError("barycentric coordinates must sum to 1")
            if (barycentric < -1e-8).any():
                raise ValueError("barycentric coordinates must be non-negative")
            source_faces = self._faces[face_index]
            delta = np.zeros(vertex_count)
            np.add.at(delta, source_faces.ravel(), barycentric.ravel())
            ref_index = source_faces
            ref_weight = barycentric
        else:
            index = np.asarray(sources)
            if index.ndim == 0:
                index = index.reshape(1)
            if len(index) == 0:
                raise ValueError("no sources were passed")
            if not np.issubdtype(index.dtype, np.integer):
                raise ValueError(
                    "sources must be vertex indices or a "
                    + "(face_index, barycentric) tuple"
                )
            if index.min() < 0 or index.max() >= vertex_count:
                raise ValueError("source vertex index out of range")
            delta = np.bincount(index.ravel(), minlength=vertex_count)
            ref_index = np.tile(index.ravel()[:, None], (1, 3))
            ref_weight = np.zeros_like(ref_index, dtype=np.float64)
            ref_weight[:, 0] = 1.0

        return delta, ref_index, ref_weight


def heat_geodesic(mesh, sources, t=None):
    """
    Compute approximate geodesic distances on a triangular surface
    from one or more source points using the heat method.

    Sources may be mesh vertices or points inside faces expressed
    with barycentric coordinates. Results are approximate surface
    distances which converge under mesh refinement; they are not
    exact shortest paths. Boundary edges use natural (zero-flux)
    boundary conditions.

    Factorizations are cached on mesh._cache: repeated calls on
    unchanged geometry reuse them, and any change to the mesh
    vertices or faces invalidates them automatically.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Geometry to compute distances on.
    sources : (m,) int or (face_index, barycentric)
      Either vertex indices, or a tuple of (m,) int face
      indices and (m, 3) float barycentric coordinates.
    t : None or float
      Heat diffusion time, defaults to squared mean edge length.

    Returns
    -------
    distance : (len(mesh.vertices),) float
      Approximate geodesic distance to the closest source, with
      np.inf for vertices in components containing no source.
    """
    cache = mesh._cache
    cache.verify()
    key = ("geodesic.HeatGeodesic", None if t is None else float(t))
    if key not in cache.cache:
        cache.cache[key] = HeatGeodesic(mesh, t=t)
    return cache.cache[key].distance(sources)
