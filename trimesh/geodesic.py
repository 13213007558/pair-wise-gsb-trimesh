"""
geodesic.py
-----------

Approximate geodesic (surface) distance on triangle meshes using
the heat method of Crane, Weischedel, and Wardetzky:

  "Geodesics in Heat: A New Approach to Computing Distance
   Based on Heat Flow" (2013).

The computation uses discrete operators on the triangulated
surface (cotangent Laplacian and lumped mass matrix) with
natural (Neumann) boundary conditions along mesh boundary
edges. The result is an *approximation* of the geodesic
distance function on the surface, not an exact shortest path
along mesh edges.
"""

import numpy as np

from . import graph, util
from .typed import ArrayLike, Number

try:
    from scipy.sparse import coo_matrix, csc_matrix, diags
    from scipy.sparse.linalg import factorized
except ImportError as E:
    from .exceptions import ExceptionWrapper

    wrapper = ExceptionWrapper(E)
    coo_matrix, csc_matrix, diags, factorized = (
        wrapper,
        wrapper,
        wrapper,
        wrapper,
    )


def cotangent_laplacian(mesh):
    """
    The cotangent Laplacian operator of a triangle mesh.

    The discrete operator is built from the cotangents of the
    face angles opposite each corner and encodes natural
    (Neumann) boundary conditions along boundary edges.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to compute the operator for.

    Returns
    --------
    laplacian : (n, n) scipy.sparse.csc_matrix
      Symmetric positive semidefinite cotangent Laplacian
      where `n == len(mesh.vertices)`.
    """
    faces = np.asarray(mesh.faces)
    vertices = np.asarray(mesh.vertices)

    if len(faces) == 0:
        raise ValueError("mesh has no faces!")

    # check for degenerate (zero area) faces which produce
    # non-finite cotangents and make the operator undefined
    bad = np.isclose(mesh.area_faces, 0.0)
    if bad.any():
        raise ValueError(
            f"mesh has {bad.sum()} degenerate (zero area) faces: {np.nonzero(bad)[0]}"
        )

    # cotangent of the angle at each corner of every face
    with np.errstate(divide="ignore", invalid="ignore"):
        cot = np.cos(mesh.face_angles) / np.sin(mesh.face_angles)
    if not np.isfinite(cot).all():
        raise ValueError("face angles produce non-finite cotangents!")

    # weight of the edge opposite each corner is half the
    # cotangent of the angle at that corner
    weights = 0.5 * cot

    # for corner k of face (i, j, k) the opposite edge is (i, j)
    row = np.concatenate((faces[:, 0], faces[:, 1], faces[:, 2]))
    col = np.concatenate((faces[:, 1], faces[:, 2], faces[:, 0]))
    data = np.concatenate((weights[:, 2], weights[:, 0], weights[:, 1]))

    n = len(vertices)
    # off-diagonal entries are the negated edge weights
    # accumulate both edge orientations
    off = coo_matrix(
        (
            -np.concatenate((data, data)),
            (np.concatenate((row, col)), np.concatenate((col, row))),
        ),
        shape=(n, n),
    ).tocsc()
    # diagonal is the negated row sum of the off-diagonal
    laplacian = off + diags(-np.asarray(off.sum(axis=1)).ravel(), 0, shape=(n, n))
    return laplacian


def lumped_mass(mesh):
    """
    The lumped (diagonal) mass matrix of a triangle mesh.

    Each vertex receives one third of the area of every
    incident face.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to compute the operator for.

    Returns
    --------
    mass : (n,) float
      Diagonal of the lumped mass matrix where
      `n == len(mesh.vertices)`.
    """
    mass = np.zeros(len(mesh.vertices), dtype=np.float64)
    np.add.at(mass, np.asarray(mesh.faces).ravel(), np.repeat(mesh.area_faces / 3.0, 3))
    return mass


def _source_delta(mesh, sources):
    """
    Convert source locations into a Dirac delta vector.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh the sources live on.
    sources : (k,) int or sequence of (int, (3,) float)
      Either vertex indices, or a sequence of
      `(face_index, barycentric)` pairs locating sources
      inside faces by barycentric coordinates.

    Returns
    --------
    delta : (n,) float
      Dirac delta vector over mesh vertices.
    """
    n = len(mesh.vertices)
    delta = np.zeros(n, dtype=np.float64)

    if isinstance(sources, int | np.integer):
        sources = [int(sources)]
    sources = np.asarray(sources, dtype=object)

    # vertex index sources
    if len(sources) > 0 and all(isinstance(s, int | np.integer) for s in sources):
        index = np.asarray(sources, dtype=np.int64)
        if index.min() < 0 or index.max() >= n:
            raise ValueError("source vertex index out of range!")
        delta[index] = 1.0
        return delta

    # barycentric sources inside faces
    for item in sources:
        face_index, barycentric = item
        barycentric = np.asarray(barycentric, dtype=np.float64)
        if not util.is_shape(barycentric, (3,)):
            raise ValueError("barycentric coordinates must be (3,)!")
        if not np.isclose(barycentric.sum(), 1.0):
            raise ValueError("barycentric coordinates must sum to 1.0!")
        if face_index < 0 or face_index >= len(mesh.faces):
            raise ValueError("source face index out of range!")
        delta[np.asarray(mesh.faces)[face_index]] += barycentric
    return delta


class HeatGeodesicSolver:
    """
    Compute approximate geodesic distances from a set of
    sources using the heat method.

    Factorizations of the discrete operators are cached and
    reused as long as the mesh geometry is unchanged; changing
    `mesh.vertices` or `mesh.faces` invalidates the cache
    and triggers a rebuild on the next :meth:`solve` call.
    """

    def __init__(self, mesh, t: Number = None):
        """
        Parameters
        ----------
        mesh : trimesh.Trimesh
          Mesh to compute distances on.
        t : float or None
          Heat diffusion time. If None, defaults to the mean
          edge length squared.
        """
        self.mesh = mesh
        self._t = t
        self._key = None
        self._build()

    def _build(self):
        """
        Build the discrete operators and factorizations for
        the current mesh geometry.
        """
        mesh = self.mesh
        n = len(mesh.vertices)

        # vertices not referenced by any face make the
        # system singular and cannot be handled
        degree = np.asarray(mesh.vertex_degree)
        isolated = np.nonzero(degree == 0)[0]
        if len(isolated) > 0:
            raise ValueError(f"mesh has {len(isolated)} isolated vertices: {isolated}")

        # heat diffusion time
        if self._t is None:
            t = float(np.mean(mesh.edges_unique_length) ** 2)
        else:
            t = float(self._t)
        if not np.isfinite(t) or t <= 0.0:
            raise ValueError("heat diffusion time must be positive and finite!")

        # discrete operators on the triangulated surface
        laplacian = cotangent_laplacian(mesh)
        mass = lumped_mass(mesh)

        # connected components of the surface
        components = graph.connected_components(
            edges=mesh.edges, nodes=np.arange(n), engine="scipy"
        )

        # the heat system (M + t * L) is symmetric positive
        # definite for t > 0 so it can be factored directly
        heat = csc_matrix(diags(mass, 0, shape=(n, n)) + t * laplacian)
        try:
            factor_heat = factorized(heat)
        except Exception as E:
            raise ArithmeticError(
                f"failed to factor heat system (M + t*L) with t={t}: {E}"
            ) from None

        # the Poisson system L is singular with one constant
        # null vector per connected component: pin one vertex
        # per component by replacing its row with an identity
        # row which fixes it at zero distance
        poisson = laplacian.tolil()
        pins = np.array([c[0] for c in components], dtype=np.int64)
        for p in pins:
            poisson.rows[p] = [p]
            poisson.data[p] = [1.0]
        poisson = poisson.tocsc()
        try:
            factor_poisson = factorized(poisson)
        except Exception as E:
            raise ArithmeticError(f"failed to factor Poisson system: {E}") from None

        self._t_used = t
        self._laplacian = laplacian
        self._mass = mass
        self._components = components
        self._pins = pins
        self._factor_heat = factor_heat
        self._factor_poisson = factor_poisson
        # geometry cache key: changes when vertices or faces change
        self._key = hash(mesh._data)

    @property
    def geometry_key(self):
        """
        Hash of the mesh geometry the current factorizations
        were built from.
        """
        return self._key

    def _check_geometry(self):
        """
        Rebuild operators and factorizations if the mesh
        geometry changed since they were built.
        """
        if hash(self.mesh._data) != self._key:
            self._build()

    def solve(self, sources: ArrayLike) -> np.ndarray:
        """
        Compute the approximate geodesic distance from the
        given sources to every vertex.

        Parameters
        ----------
        sources : (k,) int or sequence of (int, (3,) float)
          Either vertex indices, or a sequence of
          `(face_index, barycentric)` pairs locating sources
          inside faces by barycentric coordinates.

        Returns
        --------
        distance : (n,) float
          Approximate geodesic distance to every vertex.
          Vertices in connected components containing no
          source are unreachable and marked `np.inf`.
        """
        self._check_geometry()
        mesh = self.mesh
        n = len(mesh.vertices)

        delta = _source_delta(mesh, sources)
        if not delta.any():
            raise ValueError("no valid sources were provided!")

        # step 1: diffuse the Dirac delta for time t
        u = self._factor_heat(delta)
        if not np.isfinite(u).all():
            bad = np.logical_not(np.isfinite(u)).sum()
            raise ArithmeticError(f"heat solve produced {bad} non-finite values!")

        # step 2: evaluate the gradient of u on each face and
        # normalize it to get the unit vector field X
        faces = np.asarray(mesh.faces)
        triangles = np.asarray(mesh.triangles)
        area = np.asarray(mesh.area_faces)
        normal = np.asarray(mesh.face_normals)
        # edge opposite each corner: (v2 - v1, v0 - v2, v1 - v0)
        opposite = np.stack(
            (
                triangles[:, 2] - triangles[:, 1],
                triangles[:, 0] - triangles[:, 2],
                triangles[:, 1] - triangles[:, 0],
            ),
            axis=1,
        )
        # grad u = (1 / 2A) * sum_i u_i * (n x e_i)
        cross = np.cross(np.tile(normal[:, None, :], (1, 3, 1)), opposite)
        gradient = (cross * u[faces][:, :, None]).sum(axis=1) / (2.0 * area[:, None])
        norm = np.linalg.norm(gradient, axis=1)
        valid = norm > 1e-14
        field = np.zeros_like(gradient)
        field[valid] = -gradient[valid] / norm[valid][:, None]

        # step 3: solve the Poisson equation L d = -div(X)
        # note the discrete divergence is the negative adjoint
        # of the gradient so the right hand side picks up a
        # positive sign here
        # div X at vertex i = (1/2) * sum_f X_f . (n_f x e_i)
        per_corner = 0.5 * np.einsum(
            "fij,fij->fi",
            cross,
            np.tile(field[:, None, :], (1, 3, 1)),
        )
        divergence = np.zeros(n, dtype=np.float64)
        np.add.at(divergence, faces.ravel(), per_corner.ravel())
        # pinned vertices are fixed at zero distance
        divergence[self._pins] = 0.0

        distance = self._factor_poisson(divergence)
        if not np.isfinite(distance).all():
            bad = np.logical_not(np.isfinite(distance)).sum()
            raise ArithmeticError(f"Poisson solve produced {bad} non-finite values!")

        # shift each component so its minimum distance is zero
        # and mark components without sources as unreachable
        for component in self._components:
            component = np.asarray(component, dtype=np.int64)
            if delta[component].any():
                distance[component] -= distance[component].min()
            else:
                distance[component] = np.inf
        return distance


def heat_geodesic(mesh, sources: ArrayLike, t: Number = None) -> np.ndarray:
    """
    Compute the approximate geodesic distance from a set of
    sources to every vertex of a triangle mesh using the heat
    method.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Mesh to compute distances on.
    sources : (k,) int or sequence of (int, (3,) float)
      Either vertex indices, or a sequence of
      `(face_index, barycentric)` pairs locating sources
      inside faces by barycentric coordinates.
    t : float or None
      Heat diffusion time. If None, defaults to the mean
      edge length squared.

    Returns
    --------
    distance : (n,) float
      Approximate geodesic distance to every vertex.
      Vertices in connected components containing no
      source are unreachable and marked `np.inf`.
    """
    return HeatGeodesicSolver(mesh, t=t).solve(sources)
