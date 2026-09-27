"""
deformation.py
---------------

Deform meshes while trying to preserve local shape, currently
implementing the "As-Rigid-As-Possible" surface modeling scheme
from Sorkine and Alexa 2007.
"""

from dataclasses import dataclass, field

import numpy as np

from . import graph
from .typed import Integer, NDArray, float64, int64

try:
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
except ImportError as E:
    from .exceptions import ExceptionWrapper

    wrapper = ExceptionWrapper(E)
    coo_matrix, diags, spsolve = wrapper, wrapper, wrapper

__all__ = ["ARAPResult", "as_rigid_as_possible"]


@dataclass
class ARAPResult:
    """
    The result of an as-rigid-as-possible deformation.

    Attributes
    ----------
    vertices : (n, 3) float
      Deformed vertex positions, one per input vertex. No
      vertices are added or removed so faces and any
      per-face attributes still correspond to the original mesh.
    energy : float
      Final ARAP energy, i.e. the weighted sum over edges of
      "||p_i - p_j - R_i (v_i - v_j)|| ** 2".
    converged : bool
      True only if the relative energy change between the last
      two iterations dropped below the requested tolerance.
    iterations : int
      Number of local/global iterations actually run.
    stop_reason : str
      Why the solver stopped: "converged", "max_iterations",
      or "all_vertices_constrained".
    under_constrained : list of (m,) int
      Vertex indices of connected components which had no
      hard or soft constraint. These vertices were left at
      their original positions rather than being allowed to
      drift to an arbitrary location.
    degenerate_faces : (d,) int
      Indices of faces with a near-zero corner angle where
      cotangent weights are undefined; uniform edge weights
      were substituted for these faces.
    """

    vertices: NDArray[float64]
    energy: float
    converged: bool
    iterations: int
    stop_reason: str
    under_constrained: list = field(default_factory=list)
    degenerate_faces: NDArray[int64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )


def _edge_weights(mesh):
    """
    Compute cotangent edge weights for a mesh, substituting
    uniform weights on degenerate faces.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Source geometry.

    Returns
    ----------
    weights : (len(mesh.edges_unique),) float
      Positive edge weights.
    degenerate : (d,) int
      Indices of degenerate faces.
    """
    angles = mesh.face_angles
    # a face is degenerate if any corner angle is near zero
    # or the angles couldn't be computed at all
    degenerate = np.logical_or(
        ~np.isfinite(angles).all(axis=1), (angles < 1e-8).any(axis=1)
    )
    # uniform weights on degenerate faces, cotangent elsewhere
    cotangent = np.ones_like(angles)
    cotangent[~degenerate] = 1.0 / np.tan(angles[~degenerate])

    # edges are ordered (v0, v1), (v1, v2), (v2, v0) per face and the
    # angle opposite each edge is at corner (2, 0, 1) respectively
    opposite = cotangent[:, [2, 0, 1]].reshape(-1)
    weights = 0.5 * np.bincount(
        mesh.edges_unique_inverse,
        weights=opposite,
        minlength=len(mesh.edges_unique),
    )
    # clamp to a small positive value so the Laplacian stays
    # well conditioned even for obtuse or near-degenerate angles
    return np.maximum(weights, 1e-8), np.nonzero(degenerate)[0]


def _check_constraints(mesh, control_indices, control_points, anchors, control_weights):
    """
    Validate constraints and split them into hard targets
    and weighted soft targets.

    Returns
    ----------
    hard : dict
      Vertex index to (3,) float target position.
    soft_index : (s,) int
      Vertex indices of soft constraints.
    soft_target : (s, 3) float
      Target positions of soft constraints.
    soft_weight : (s,) float
      Positive weights of soft constraints.
    """
    hard = {}
    soft_index = []
    soft_target = []
    soft_weight = []

    def add_hard(index, target):
        index = int(index)
        target = np.asarray(target, dtype=np.float64)
        if index in hard:
            if not np.allclose(hard[index], target, atol=1e-10):
                conflict = f"{hard[index].tolist()} vs {target.tolist()}"
                raise ValueError(f"conflicting constraints on vertex {index}: {conflict}")
        else:
            hard[index] = target

    # anchors are hard constraints held at their current position
    if anchors is not None:
        anchors = np.asarray(anchors, dtype=np.int64).ravel()
        for index in anchors:
            add_hard(index, mesh.vertices[index])

    if control_indices is not None:
        control_indices = np.asarray(control_indices, dtype=np.int64).ravel()
        control_points = np.asanyarray(control_points, dtype=np.float64).reshape((-1, 3))
        if len(control_indices) != len(control_points):
            raise ValueError(
                "control_indices and control_points must have matching lengths"
            )
        if control_weights is None:
            control_weights = np.full(len(control_indices), np.inf)
        else:
            control_weights = np.asarray(control_weights, dtype=np.float64).ravel()
            if len(control_weights) != len(control_indices):
                raise ValueError(
                    "control_weights and control_indices must have matching lengths"
                )
            if (control_weights <= 0.0).any():
                raise ValueError("control_weights must be positive")
        for index, target, weight in zip(
            control_indices, control_points, control_weights
        ):
            if np.isinf(weight):
                add_hard(index, target)
            else:
                soft_index.append(index)
                soft_target.append(target)
                soft_weight.append(weight)

    if len(hard) == 0 and len(soft_index) == 0:
        raise ValueError("at least one control point or anchor is required")

    return (
        hard,
        np.asarray(soft_index, dtype=np.int64),
        np.asarray(soft_target, dtype=np.float64).reshape((-1, 3)),
        np.asarray(soft_weight, dtype=np.float64),
    )


def as_rigid_as_possible(
    mesh,
    control_indices=None,
    control_points=None,
    anchors=None,
    control_weights=None,
    max_iterations: Integer = 100,
    tolerance: float = 1e-6,
) -> ARAPResult:
    """
    Deform a mesh as-rigid-as-possible given target control
    points and anchors which stay in place.

    Minimizes the sum over edges of
    "w_ij ||p_i - p_j - R_i (v_i - v_j)|| ** 2" where "R_i"
    is a best-fit local rotation re-estimated each iteration so
    local rotation stays consistent with the current deformation.

    Only vertex positions are computed: no vertices are added
    or removed and faces are untouched, so per-face attributes
    still correspond to the original faces.

    Parameters
    ----------
    mesh : trimesh.Trimesh
      Source geometry, not modified in place.
    control_indices : (m,) int or None
      Indices of vertices with a target position.
    control_points : (m, 3) float or None
      Target position for each control vertex.
    anchors : (k,) int or None
      Indices of vertices held at their current position.
    control_weights : (m,) float or None
      Positive weight per control point. If None every control
      point is a hard constraint; use "numpy.inf" to make
      individual control points hard constraints.
    max_iterations : int
      Maximum number of local/global iterations.
    tolerance : float
      Relative energy change below which the solver is
      considered converged.

    Returns
    ----------
    result : ARAPResult
      Deformed vertices, final energy, convergence flag,
      stop reason, and any under-constrained components.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    vertices = mesh.vertices.view(np.ndarray)
    count = len(vertices)
    edges = mesh.edges_unique
    weights, degenerate = _edge_weights(mesh)

    hard, soft_index, soft_target, soft_weight = _check_constraints(
        mesh, control_indices, control_points, anchors, control_weights
    )

    # find connected components with no constraint at all: solving
    # for them would be singular and they would drift arbitrarily,
    # so hold them at their original position and report them
    labels = graph.connected_component_labels(edges, node_count=count)
    constrained = np.zeros(count, dtype=bool)
    if len(hard) > 0:
        constrained[list(hard.keys())] = True
    if len(soft_index) > 0:
        constrained[soft_index] = True
    under_constrained = [
        np.nonzero(labels == label)[0]
        for label in np.unique(labels)
        if not constrained[labels == label].any()
    ]
    if len(under_constrained) > 0:
        drifting = np.concatenate(under_constrained)
    else:
        drifting = np.zeros(0, dtype=np.int64)

    # fixed vertices are hard constraints plus under-constrained
    # vertices which are simply kept at their rest position
    fixed_index = np.concatenate([list(hard.keys()), drifting]).astype(np.int64)
    fixed_target = np.vstack(
        [
            np.asarray(list(hard.values())).reshape((-1, 3)),
            vertices[drifting].reshape((-1, 3)),
        ]
    )
    is_free = np.ones(count, dtype=bool)
    is_free[fixed_index] = False
    free = np.nonzero(is_free)[0]

    # weighted Laplacian of the rest mesh
    upper = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(count, count))
    adjacent = upper + upper.T
    laplacian = diags(np.asarray(adjacent.sum(axis=1)).ravel()) - adjacent

    # soft constraints add a weighted pull towards their target
    matrix = laplacian.tocsc()
    soft_rhs = np.zeros((count, 3))
    if len(soft_index) > 0:
        matrix = (
            matrix
            + coo_matrix(
                (soft_weight, (soft_index, soft_index)), shape=(count, count)
            ).tocsc()
        )
        soft_rhs[soft_index] += soft_weight[:, None] * soft_target

    matrix_free = matrix[free][:, free].tocsc()
    matrix_fixed = matrix[free][:, fixed_index]

    # rest edge vectors
    rest = vertices[edges[:, 0]] - vertices[edges[:, 1]]

    def local_rotations(current):
        # covariance of rest and current edge vectors per vertex;
        # each edge contributes the same outer product to both ends
        vector = current[edges[:, 0]] - current[edges[:, 1]]
        covariance = np.zeros((count, 3, 3))
        contribution = weights[:, None, None] * rest[:, :, None] * vector[:, None, :]
        np.add.at(covariance, edges[:, 0], contribution)
        np.add.at(covariance, edges[:, 1], contribution)
        u, _, vt = np.linalg.svd(covariance)
        # best fit rotation is V * diag(1, 1, det(V U**T)) * U**T
        rotation = vt.transpose(0, 2, 1)
        sign = np.sign(np.linalg.det(rotation @ u.transpose(0, 2, 1)))
        rotation[:, :, 2] *= sign[:, None]
        return rotation @ u.transpose(0, 2, 1)

    def energy(current, rotation):
        diff = (
            current[edges[:, 0]]
            - current[edges[:, 1]]
            - np.einsum("nij,nj->ni", rotation[edges[:, 0]], rest)
        )
        return float((weights * (diff**2).sum(axis=1)).sum())

    position = vertices.copy()
    previous = None
    current_energy = None
    converged = False
    iteration = 0

    if len(free) == 0:
        # nothing to solve for: report the energy of the only
        # possible configuration rather than iterating
        rotation = local_rotations(position)
        return ARAPResult(
            vertices=position,
            energy=energy(position, rotation),
            converged=True,
            iterations=0,
            stop_reason="all_vertices_constrained",
            under_constrained=under_constrained,
            degenerate_faces=degenerate,
        )

    while iteration < max_iterations and not converged:
        iteration += 1
        # local step: best fit rotation per vertex
        rotation = local_rotations(position)

        # global step: solve for positions given the rotations
        combined = rotation[edges[:, 0]] + rotation[edges[:, 1]]
        pull = 0.5 * weights[:, None] * np.einsum("eij,ej->ei", combined, rest)
        rhs = np.zeros((count, 3))
        np.add.at(rhs, edges[:, 0], pull)
        np.add.at(rhs, edges[:, 1], -pull)
        rhs += soft_rhs

        position[free] = spsolve(
            matrix_free, rhs[free] - matrix_fixed @ position[fixed_index]
        )
        position[fixed_index] = fixed_target

        current_energy = energy(position, rotation)
        if previous is not None:
            change = abs(previous - current_energy)
            if change <= tolerance * max(1.0, abs(previous)):
                converged = True
                break
        previous = current_energy

    return ARAPResult(
        vertices=position,
        energy=current_energy,
        converged=converged,
        iterations=iteration,
        stop_reason="converged" if converged else "max_iterations",
        under_constrained=under_constrained,
        degenerate_faces=degenerate,
    )
