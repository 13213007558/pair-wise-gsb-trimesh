"""
manifold.py
-----------

Split meshes with non-manifold topology into independently
processable manifold pieces.

Two kinds of non-manifold contact are handled:

- a vertex referenced by two or more disconnected face fans
  (for instance two triangles touching only at that vertex)
- an edge referenced by three or more faces

Only vertex *identities* are duplicated: coordinates never move
and faces are never altered, corners are simply reassigned to
per-fan vertex copies. UV coordinates, vertex normals and any
vertex or face attributes are propagated to every copy, and the
original index of every output vertex and face is returned.
"""

from dataclasses import dataclass

import numpy as np

from . import graph, grouping
from .typed import NDArray, int64


@dataclass
class EdgeAmbiguity:
    """
    A non-manifold edge whose incident faces could not be
    uniquely paired.

    Attributes
    ------------
    edge : (2,) int
      Vertex indices of the non-manifold edge.
    faces : (m,) int
      Indices of the incident faces left unpaired, which were
      treated as boundary when splitting.
    """

    edge: NDArray[int64]
    faces: NDArray[int64]


@dataclass
class ManifoldSplit:
    """
    Result of split_manifold.

    Attributes
    ------------
    meshes : list of Trimesh
      Manifold pieces of the input mesh.
    vertex_source : list of (p,) int
      For each piece, the original vertex index every output
      vertex was copied from.
    face_source : list of (q,) int
      For each piece, the original face index every output face
      came from.
    ambiguous : list of EdgeAmbiguity
      Non-manifold edges whose incident faces could not be paired
      uniquely and were left as boundary.
    """

    meshes: list
    vertex_source: list
    face_source: list
    ambiguous: list


@dataclass
class ManifoldReport:
    """
    Result of manifold_report.

    Attributes
    ------------
    is_manifold : bool
      True when no edge has more than two incident faces and the
      faces around every vertex form a single connected fan.
    nonmanifold_edges : (n, 2) int
      Vertex indices of edges shared by three or more faces.
    split_vertices : (m,) int
      Vertices referenced by multiple disconnected face fans.
    """

    is_manifold: bool
    nonmanifold_edges: NDArray[int64]
    split_vertices: NDArray[int64]


def _face_edge_instances(faces: NDArray[int64]) -> NDArray[int64]:
    """
    Directed edge instances, three per face.

    Instance 3 * f + k runs from corner 3 * f + k to
    corner 3 * f + (k + 1) % 3.
    """
    return faces[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 2))


def _link_corners(
    instance_a: int,
    instance_b: int,
    corner_vertex: NDArray[int64],
) -> list[tuple[int, int]]:
    """
    Corner pairs to union when two faces are adjacent across the
    edge of the two given edge instances. Corners referencing the
    same original vertex are paired.
    """
    face_a, edge_a = divmod(int(instance_a), 3)
    face_b, edge_b = divmod(int(instance_b), 3)
    corners_a = np.array([3 * face_a + edge_a, 3 * face_a + (edge_a + 1) % 3])
    corners_b = np.array([3 * face_b + edge_b, 3 * face_b + (edge_b + 1) % 3])
    order_a = np.argsort(corner_vertex[corners_a])
    order_b = np.argsort(corner_vertex[corners_b])
    return [
        (
            int(corners_a[order_a[0]]),
            int(corners_b[order_b[0]]),
        ),
        (
            int(corners_a[order_a[1]]),
            int(corners_b[order_b[1]]),
        ),
    ]


def _pair_faces(
    group_faces: NDArray[int64], partners: dict[int, set[int]]
) -> tuple[list[tuple[int, int]], NDArray[int64]]:
    """
    Pair up faces incident to a non-manifold edge using the
    caller-provided face adjacency constraints.

    Each face is paired at most once per edge; unpaired faces are
    returned as leftovers.
    """
    members = {int(face) for face in group_faces}
    paired: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for face in sorted(members):
        if face in paired:
            continue
        for other in partners.get(face, ()):
            if other != face and other in members and other not in paired:
                pairs.append((face, other))
                paired.update((face, other))
                break
    leftover = np.array(sorted(members - paired), dtype=int64)
    return pairs, leftover


def _build_piece(mesh, vertex_source, face_source, faces):
    """
    Construct one output Trimesh copying coordinates, visuals,
    normals and attributes from the source mesh without merging
    the intentionally duplicated vertex identities.
    """
    from .base import Trimesh
    from .visual.color import ColorVisuals
    from .visual.texture import TextureVisuals

    vertex_attributes = {
        key: np.asanyarray(value)[vertex_source]
        for key, value in mesh.vertex_attributes.items()
    }
    face_attributes = {
        key: np.asanyarray(value)[face_source]
        for key, value in mesh.face_attributes.items()
    }

    visual = None
    current = mesh.visual
    if current.kind == "texture":
        uv = current.uv
        face_materials = current.face_materials
        visual = TextureVisuals(
            uv=None if uv is None else np.asanyarray(uv)[vertex_source],
            material=current.material.copy(),
            face_materials=(
                None
                if face_materials is None
                else np.asanyarray(face_materials)[face_source]
            ),
        )
    elif current.kind in ("vertex", "face"):
        kwargs = {}
        if "vertex_colors" in current._data:
            kwargs["vertex_colors"] = current._data["vertex_colors"][vertex_source]
        if "face_colors" in current._data:
            kwargs["face_colors"] = current._data["face_colors"][face_source]
        visual = ColorVisuals(**kwargs)

    piece = Trimesh(
        vertices=mesh.vertices[vertex_source],
        faces=faces,
        visual=visual,
        vertex_attributes=vertex_attributes,
        face_attributes=face_attributes,
        process=False,
    )
    if "vertex_normals" in mesh._cache:
        piece.vertex_normals = np.asanyarray(mesh._cache["vertex_normals"])[vertex_source]
    return piece


def split_manifold(mesh, edge_adjacency=None) -> ManifoldSplit:
    """
    Split a mesh into manifold pieces by duplicating vertex
    identities at non-manifold contacts.

    Vertex fans are split automatically: when the faces around a
    vertex fall into several disconnected fan components the
    vertex is copied once per component and coordinates are left
    untouched.

    For an edge shared by three or more faces the caller may state
    which pairs of faces are adjacent across that edge via
    edge_adjacency. Faces that cannot be paired (no constraint or
    faces left over) are treated as boundary and the edge is
    returned in ManifoldSplit.ambiguous.

    Parameters
    ------------
    mesh : Trimesh
      Input mesh.
    edge_adjacency : iterable of (2,) int, optional
      Pairs of face indices that are adjacent across a shared
      non-manifold edge. Each face may be paired at most once
      per edge.

    Returns
    ----------
    result : ManifoldSplit
      Output pieces with original-identity mappings and any
      unresolved non-manifold edges.
    """
    faces = np.asanyarray(mesh.faces, dtype=int64)
    face_count = len(faces)
    corner_vertex = faces.reshape(-1)
    corner_count = len(corner_vertex)

    # face index -> faces the caller declares adjacent to it
    partners: dict[int, set[int]] = {}
    if edge_adjacency is not None:
        for pair in edge_adjacency:
            face_a, face_b = int(pair[0]), int(pair[1])
            if face_a == face_b or (
                len(set(faces[face_a].tolist()) & set(faces[face_b].tolist())) < 2
            ):
                raise ValueError(f"faces {face_a} and {face_b} do not share an edge")
            partners.setdefault(face_a, set()).add(face_b)
            partners.setdefault(face_b, set()).add(face_a)

    corner_links: list[NDArray[int64]] = []
    face_links: list[tuple[int, int]] = []
    ambiguous: list[EdgeAmbiguity] = []

    edge_instances = _face_edge_instances(faces)
    edge_groups = grouping.group_rows(np.sort(edge_instances, axis=1))

    for group in edge_groups:
        if len(group) == 1:
            # regular boundary edge: nothing to connect
            continue
        if len(group) == 2:
            instance_pairs = [(int(group[0]), int(group[1]))]
        else:
            # three or more faces on one edge: honor constraints
            group_faces = group // 3
            face_pairs, leftover = _pair_faces(group_faces, partners)
            if len(leftover) > 0:
                ambiguous.append(
                    EdgeAmbiguity(
                        edge=edge_instances[group[0]].copy(),
                        faces=leftover,
                    )
                )
            instance_of = {
                int(face): int(instance) for instance, face in zip(group, group_faces)
            }
            instance_pairs = [
                (instance_of[face_a], instance_of[face_b])
                for face_a, face_b in face_pairs
            ]

        for instance_a, instance_b in instance_pairs:
            corner_links.append(
                np.asarray(
                    _link_corners(instance_a, instance_b, corner_vertex),
                    dtype=int64,
                )
            )
            face_links.append((instance_a // 3, instance_b // 3))

    # fan components of corners -> new vertex identities
    corner_edges = (
        np.concatenate(corner_links)
        if len(corner_links) > 0
        else np.zeros((0, 2), dtype=int64)
    )
    corner_labels = graph.connected_component_labels(
        corner_edges, node_count=corner_count
    )
    _unique_labels, first_occurrence, inverse = np.unique(
        corner_labels, return_index=True, return_inverse=True
    )
    vertex_source_all = corner_vertex[first_occurrence]
    # order new vertices by their source index so meshes that
    # needed no splitting keep their original vertex numbering
    order = np.argsort(vertex_source_all, kind="stable")
    rank = np.zeros(len(order), dtype=int64)
    rank[order] = np.arange(len(order))
    vertex_source_all = vertex_source_all[order]
    faces_renumbered = rank[inverse].reshape(faces.shape)

    # connected components of faces across edge adjacency -> pieces
    if len(face_links) > 0:
        piece_labels = graph.connected_component_labels(
            np.asarray(face_links, dtype=int64), node_count=face_count
        )
    else:
        piece_labels = np.arange(face_count, dtype=int64)

    meshes = []
    vertex_sources = []
    face_sources = []
    for label in np.unique(piece_labels):
        face_source = np.nonzero(piece_labels == label)[0]
        piece_faces = faces_renumbered[face_source]
        used, compact = np.unique(piece_faces.reshape(-1), return_inverse=True)
        vertex_source = vertex_source_all[used]
        meshes.append(
            _build_piece(
                mesh,
                vertex_source=vertex_source,
                face_source=face_source,
                faces=compact.reshape((-1, 3)),
            )
        )
        vertex_sources.append(vertex_source)
        face_sources.append(face_source)

    return ManifoldSplit(
        meshes=meshes,
        vertex_source=vertex_sources,
        face_source=face_sources,
        ambiguous=ambiguous,
    )


def manifold_report(mesh) -> ManifoldReport:
    """
    Check both manifold conditions on a mesh:

    1. every edge is incident to at most two faces
    2. the faces around every vertex form a single connected fan

    The second condition is what rejects bowtie vertices even
    when every edge is only used once or twice.

    Parameters
    ------------
    mesh : Trimesh
      Mesh to inspect.

    Returns
    ----------
    report : ManifoldReport
      Which manifold conditions fail and where.
    """
    faces = np.asanyarray(mesh.faces, dtype=int64)
    corner_vertex = faces.reshape(-1)
    corner_count = len(corner_vertex)

    edge_instances = _face_edge_instances(faces)
    edge_sorted = np.sort(edge_instances, axis=1)
    edge_groups = grouping.group_rows(edge_sorted)

    corner_links: list[NDArray[int64]] = []
    nonmanifold = []
    for group in edge_groups:
        if len(group) == 2:
            corner_links.append(
                np.asarray(
                    _link_corners(group[0], group[1], corner_vertex),
                    dtype=int64,
                )
            )
        elif len(group) > 2:
            nonmanifold.append(edge_sorted[group[0]])

    if corner_count > 0:
        corner_edges = (
            np.concatenate(corner_links)
            if len(corner_links) > 0
            else np.zeros((0, 2), dtype=int64)
        )
        corner_labels = graph.connected_component_labels(
            corner_edges, node_count=corner_count
        )
        # one (vertex, fan-label) pair per fan component
        pairs = np.unique(np.column_stack([corner_vertex, corner_labels]), axis=0)
        fan_count = np.bincount(pairs[:, 0], minlength=len(mesh.vertices))
    else:
        fan_count = np.zeros(len(mesh.vertices), dtype=int64)

    nonmanifold_edges = np.asarray(nonmanifold, dtype=int64).reshape((-1, 2))
    split_vertices = np.nonzero(fan_count > 1)[0]

    return ManifoldReport(
        is_manifold=len(nonmanifold_edges) == 0 and len(split_vertices) == 0,
        nonmanifold_edges=nonmanifold_edges,
        split_vertices=split_vertices,
    )


def is_manifold(mesh) -> bool:
    """
    Check that a mesh satisfies both manifold conditions: no edge
    is shared by more than two faces and the faces around every
    vertex form a single connected fan.

    Parameters
    ------------
    mesh : Trimesh
      Mesh to inspect.

    Returns
    ----------
    is_manifold : bool
      True if the mesh is manifold.
    """
    return manifold_report(mesh).is_manifold
