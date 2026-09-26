"""
manifold.py
-----------

Split meshes with non-manifold topology into independent manifold
patches by duplicating vertex identity only: vertex positions are
never moved and faces are never added, removed, or reordered.

Two kinds of defects are handled:

- vertices whose incident faces form multiple disconnected fans
  (for example two cones touching only at their apex)
- edges shared by more than two faces

Every returned patch records the indexes of the source geometry in
its metadata under ``original_vertex_id`` and ``original_face_id``.
"""

from typing import NamedTuple

import numpy as np

from . import grouping, util
from .geometry import faces_to_edges
from .graph import connected_component_labels, face_adjacency
from .typed import NDArray, int64
from .visual.color import ColorVisuals
from .visual.texture import TextureVisuals


class AmbiguousEdge(NamedTuple):
    """
    An edge shared by more than two faces where no face
    pairing was provided by the caller.
    """

    #: (2,) int vertex indexes of the edge in the source mesh
    edge: tuple
    #: (k,) int indexes of the faces incident to the edge
    faces: tuple


class ManifoldSplit(NamedTuple):
    """
    Result of :func:`split_nonmanifold`.
    """

    #: list of Trimesh: patches satisfying the manifold conditions
    meshes: list
    #: list of AmbiguousEdge: non-manifold edges which still
    #: require a face pairing decision from the caller
    pending: list


def _corner_links(edges: NDArray[int64], row_group: NDArray[int64]) -> NDArray[int64]:
    """
    Get pairs of face corners which meet at a vertex through
    the same edge group.

    Parameters
    ------------
    edges : (n, 2) int
      Directed edges of every face as returned by `faces_to_edges`.
    row_group : (n,) int
      Group index for every edge row: rows in a group remain
      incident to each other after splitting.

    Returns
    ------------
    links : (m, 2) int
      Pairs of corner indexes in `range(len(edges))`.
    """
    row = np.arange(len(edges))
    slot = row % 3
    # corners are indexed by `3 * face + position`: the first
    # endpoint of an edge row is exactly its row index
    corner = np.column_stack((row, row - slot + (slot + 1) % 3))

    # two endpoint records per edge row: (vertex, group, corner)
    end_vertex = edges.reshape(-1)
    end_group = np.repeat(row_group, 2)
    end_corner = corner.reshape(-1)

    # sort so corners meeting at the same vertex through the
    # same edge group are adjacent
    order = np.lexsort((end_group, end_vertex))
    vertex_sorted = end_vertex[order]
    group_sorted = end_group[order]
    corner_sorted = end_corner[order]

    same = np.logical_and(
        vertex_sorted[1:] == vertex_sorted[:-1], group_sorted[1:] == group_sorted[:-1]
    )
    return np.column_stack((corner_sorted[:-1][same], corner_sorted[1:][same]))


def _fan_labels(faces: NDArray[int64]) -> NDArray[int64]:
    """
    Label every face corner with the fan it belongs to: corners
    sharing a vertex and connected through shared edges get the
    same label.
    """
    edges, _ = faces_to_edges(faces, return_index=True)
    groups = grouping.group_rows(np.sort(edges, axis=1))
    row_group = np.zeros(len(edges), dtype=np.int64)
    for index, rows in enumerate(groups):
        row_group[rows] = index
    links = _corner_links(edges, row_group)
    return connected_component_labels(links, node_count=len(edges))


def _remap_visual(visual, vertex_index: NDArray[int64], face_index: NDArray[int64]):
    """
    Carry visual information to duplicated vertices and
    re-indexed faces.
    """
    if isinstance(visual, TextureVisuals):
        uv = visual.uv
        face_materials = visual.face_materials
        if face_materials is not None:
            face_materials = np.asarray(face_materials)[face_index]
        return TextureVisuals(
            uv=None if uv is None else np.asarray(uv)[vertex_index],
            material=visual.material.copy(),
            face_materials=face_materials,
        )
    if isinstance(visual, ColorVisuals):
        copied = visual.copy()
        copied.update_vertices(vertex_index)
        copied.update_faces(face_index)
        return copied
    return visual.copy()


def split_nonmanifold(mesh, edge_constraints: dict | None = None) -> ManifoldSplit:
    """
    Split a mesh with non-manifold topology into manifold patches.

    Vertex fans are split by adjacency: every disconnected fan of
    faces around a vertex receives its own copy of the vertex.
    Edges shared by more than two faces require the caller to
    state which faces remain incident to each other; unpaired
    faces become boundary loops.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Source geometry, not altered.
    edge_constraints : dict or None
      Mapping of ``(vertex_a, vertex_b)`` edges to a list of
      ``(face_a, face_b)`` pairs which should remain incident
      to each other at that edge.

    Returns
    ------------
    result : ManifoldSplit
      `meshes` are Trimesh patches with `original_vertex_id` and
      `original_face_id` metadata, `pending` lists AmbiguousEdge
      for non-manifold edges with no provided pairing where the
      split was not applied.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(faces) == 0:
        return ManifoldSplit(meshes=[], pending=[])

    edges, edges_face = faces_to_edges(faces, return_index=True)
    edges_sorted = np.sort(edges, axis=1)
    groups = grouping.group_rows(edges_sorted)

    # normalize caller constraints to sorted integer edge keys
    constraints = {}
    if edge_constraints is not None:
        for edge, pairs in edge_constraints.items():
            key = tuple(int(i) for i in np.sort(np.asarray(edge, dtype=np.int64)))
            constraints[key] = [tuple(int(f) for f in pair) for pair in pairs]

    # assign every edge row to a group: rows in the same group
    # stay incident to each other in the split result
    row_group = np.zeros(len(edges), dtype=np.int64)
    pending = []
    group_index = 0
    for rows in groups:
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) <= 2:
            # manifold and boundary edges need no decision
            row_group[rows] = group_index
            group_index += 1
            continue
        key = tuple(int(i) for i in edges_sorted[rows[0]])
        faces_on_edge = edges_face[rows]
        pairs = constraints.pop(key, None)
        if pairs is None:
            # no caller preference: leave the edge whole and
            # report it as needing a pairing decision
            pending.append(
                AmbiguousEdge(edge=key, faces=tuple(int(f) for f in faces_on_edge))
            )
            row_group[rows] = group_index
            group_index += 1
            continue
        paired = np.asarray(pairs, dtype=np.int64).reshape(-1)
        if (
            paired.size % 2 != 0
            or not np.isin(paired, faces_on_edge).all()
            or len(np.unique(paired)) != len(paired)
        ):
            raise ValueError(f"invalid face pairing for edge {key}")
        # paired faces stay together, unpaired faces become boundary
        for pair in pairs:
            row_group[rows[np.isin(faces_on_edge, pair)]] = group_index
            group_index += 1
        for row in rows[~np.isin(faces_on_edge, paired)]:
            row_group[row] = group_index
            group_index += 1

    if len(constraints) > 0:
        raise ValueError(
            f"constraints reference edges not shared by 3+ faces: {sorted(constraints)}"
        )

    # label every corner with the fan it belongs to: each fan
    # becomes one vertex in the split mesh
    links = _corner_links(edges, row_group)
    labels = connected_component_labels(links, node_count=len(edges))

    # every corner of a fan references the same source vertex
    source = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    source[labels] = faces.reshape(-1)

    new_faces = labels.reshape(faces.shape)
    new_vertices = vertices[source]
    vertex_normals = np.asarray(mesh.vertex_normals)[source]
    face_normals = np.asarray(mesh.face_normals)
    vertex_attributes = {
        k: np.asarray(v)[source] for k, v in mesh.vertex_attributes.items()
    }
    face_attributes = {k: np.asarray(v) for k, v in mesh.face_attributes.items()}

    # connected components of the split topology are the patches
    adjacency = face_adjacency(faces=new_faces)
    face_labels = connected_component_labels(adjacency, node_count=len(faces))

    meshes = []
    trimesh_type = util.type_named(mesh, "Trimesh")
    for component in grouping.group(face_labels):
        face_index = np.asarray(component, dtype=np.int64)
        unique, inverse = np.unique(new_faces[face_index], return_inverse=True)
        current = trimesh_type(
            vertices=new_vertices[unique],
            faces=inverse.reshape((-1, 3)),
            face_normals=face_normals[face_index],
            vertex_normals=vertex_normals[unique],
            visual=_remap_visual(mesh.visual, source[unique], face_index),
            vertex_attributes={k: v[unique] for k, v in vertex_attributes.items()},
            face_attributes={k: v[face_index] for k, v in face_attributes.items()},
            metadata={
                "original_vertex_id": source[unique],
                "original_face_id": face_index,
            },
            process=False,
        )
        meshes.append(current)

    return ManifoldSplit(meshes=meshes, pending=pending)


def nonmanifold_edges(mesh) -> NDArray[int64]:
    """
    Get edges referenced by more than two faces.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Source geometry.

    Returns
    ------------
    edges : (n, 2) int
      Vertex indexes of non-manifold edges.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.sort(faces_to_edges(faces), axis=1)
    groups = grouping.group_rows(edges)
    over = [g[0] for g in groups if len(g) > 2]
    if len(over) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return edges[over]


def nonmanifold_vertices(mesh) -> NDArray[int64]:
    """
    Get vertices whose incident faces do not form a single
    connected fan, i.e. the vertex link is disconnected.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Source geometry.

    Returns
    ------------
    vertices : (n,) int
      Indexes of non-manifold vertices.
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    labels = _fan_labels(faces)
    # one entry per (vertex, fan) pair
    pairs = np.unique(np.column_stack((faces.reshape(-1), labels)), axis=0)
    vertex, count = np.unique(pairs[:, 0], return_counts=True)
    return vertex[count > 1]


def is_manifold(mesh) -> bool:
    """
    Check both manifold conditions: every edge is referenced by
    at most two faces and every vertex link is a single fan.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Source geometry.

    Returns
    ------------
    is_manifold : bool
      True if every edge has at most two incident faces and
      every vertex has a single connected fan of faces.
    """
    return len(nonmanifold_edges(mesh)) == 0 and len(nonmanifold_vertices(mesh)) == 0
