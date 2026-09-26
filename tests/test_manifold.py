try:
    from . import generic as g
except BaseException:
    import generic as g


def _bowtie():
    """
    Two closed cone fans sharing only their apex vertex.
    """
    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [-1.0, 0.0, -1.0],
    ]
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [0, 4, 5], [0, 5, 6], [0, 6, 4]]
    return g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _shared_edge():
    """
    Three faces all referencing the same edge.
    """
    vertices = [
        [10.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
        [10.0, 1.0, 0.0],
        [10.0, -1.0, 0.0],
        [10.0, 0.0, 1.0],
    ]
    faces = [[0, 1, 2], [0, 1, 3], [0, 1, 4]]
    return g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _seam_strip():
    """
    A cylindrical strip where the seam vertices are duplicated:
    same positions, different UV coordinates.
    """
    ring = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    bottom = [[x, y, 5.0] for x, y in ring]
    top = [[x, y, 6.0] for x, y in ring]
    # vertices 8 and 9 duplicate 0 and 4 exactly
    vertices = bottom + top + [bottom[0], top[0]]
    faces = [
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 8, 9],
        [3, 9, 7],
    ]
    uv = [[i / 4.0, 0.0] for i in range(4)]
    uv += [[i / 4.0, 1.0] for i in range(4)]
    uv += [[1.0, 0.0], [1.0, 1.0]]
    return vertices, faces, uv


class ManifoldTest(g.unittest.TestCase):
    def test_watertight(self):
        # a manifold mesh should pass the checks and be
        # returned as a single unchanged patch
        mesh = g.trimesh.creation.icosphere()
        assert g.trimesh.manifold.is_manifold(mesh)
        assert len(g.trimesh.manifold.nonmanifold_edges(mesh)) == 0
        assert len(g.trimesh.manifold.nonmanifold_vertices(mesh)) == 0

        result = g.trimesh.manifold.split_nonmanifold(mesh)
        assert len(result.pending) == 0
        assert len(result.meshes) == 1
        patch = result.meshes[0]
        assert len(patch.faces) == len(mesh.faces)
        assert g.np.allclose(
            g.np.sort(patch.vertices, axis=0), g.np.sort(mesh.vertices, axis=0)
        )

    def test_bowtie_vertex(self):
        mesh = _bowtie()
        # every edge has at most two faces: the defect is only
        # visible in the vertex link
        assert len(g.trimesh.manifold.nonmanifold_edges(mesh)) == 0
        assert g.trimesh.manifold.nonmanifold_vertices(mesh).tolist() == [0]
        assert not g.trimesh.manifold.is_manifold(mesh)

        # add per-vertex colors to check attribute propagation
        colors = g.np.arange(len(mesh.vertices) * 4, dtype=g.np.uint8).reshape(-1, 4)
        mesh.visual = g.trimesh.visual.ColorVisuals(vertex_colors=colors)

        result = g.trimesh.manifold.split_nonmanifold(mesh)
        assert len(result.pending) == 0
        assert len(result.meshes) == 2

        for patch in result.meshes:
            # both the edge count and the vertex link conditions
            assert g.trimesh.manifold.is_manifold(patch)
            # one vertex was duplicated: the apex
            assert len(patch.vertices) == 4
            ovid = patch.metadata["original_vertex_id"]
            ofid = patch.metadata["original_face_id"]
            # coordinates were not moved
            assert g.np.allclose(patch.vertices, mesh.vertices[ovid])
            # colors followed the vertex copies
            assert (patch.visual.vertex_colors == colors[ovid]).all()
            # each patch is exactly one of the two cones
            assert len(g.np.intersect1d(ofid, [0, 1, 2])) in (0, 3)

        # the apex appears once per patch, everything else once total
        ovid = g.np.concatenate([p.metadata["original_vertex_id"] for p in result.meshes])
        assert (g.np.sort(ovid) == [0, 0, 1, 2, 3, 4, 5, 6]).all()
        ofid = g.np.concatenate([p.metadata["original_face_id"] for p in result.meshes])
        assert (g.np.sort(ofid) == g.np.arange(6)).all()

    def test_shared_edge(self):
        mesh = _shared_edge()
        assert g.trimesh.manifold.nonmanifold_edges(mesh).tolist() == [[0, 1]]
        assert not g.trimesh.manifold.is_manifold(mesh)

        # without constraints the edge is reported as ambiguous
        result = g.trimesh.manifold.split_nonmanifold(mesh)
        assert len(result.pending) == 1
        assert result.pending[0].edge == (0, 1)
        assert result.pending[0].faces == (0, 1, 2)
        for patch in result.meshes:
            assert g.trimesh.manifold.is_manifold(patch)

        # pair the first two faces and leave the third as boundary
        result = g.trimesh.manifold.split_nonmanifold(
            mesh, edge_constraints={(0, 1): [(0, 1)]}
        )
        assert len(result.pending) == 0
        assert len(result.meshes) == 2
        sizes = sorted(len(p.faces) for p in result.meshes)
        assert sizes == [1, 2]
        for patch in result.meshes:
            assert g.trimesh.manifold.is_manifold(patch)
            ovid = patch.metadata["original_vertex_id"]
            assert g.np.allclose(patch.vertices, mesh.vertices[ovid])
        # the paired faces stayed in the same patch
        paired = next(p for p in result.meshes if len(p.faces) == 2)
        assert (g.np.sort(paired.metadata["original_face_id"]) == [0, 1]).all()

        # invalid pairings must be rejected
        with g.pytest.raises(ValueError):
            # face 9 is not incident to the edge
            g.trimesh.manifold.split_nonmanifold(
                mesh, edge_constraints={(0, 1): [(0, 9)]}
            )
        with g.pytest.raises(ValueError):
            # face 1 appears in two pairs
            g.trimesh.manifold.split_nonmanifold(
                mesh, edge_constraints={(0, 1): [(0, 1), (1, 2)]}
            )
        with g.pytest.raises(ValueError):
            # edge (1, 2) is a boundary edge, not a non-manifold one
            g.trimesh.manifold.split_nonmanifold(
                mesh, edge_constraints={(1, 2): [(0, 1)]}
            )

    def test_combo_textured(self):
        # combine a bowtie vertex, a three-face shared edge and a
        # textured seam strip into a single mesh
        bow = _bowtie()
        edge = _shared_edge()
        seam_vertices, seam_faces, seam_uv = _seam_strip()

        offset = len(bow.vertices) + len(edge.vertices)
        vertices = g.np.vstack([bow.vertices, edge.vertices, g.np.asarray(seam_vertices)])
        faces = g.np.vstack(
            [
                bow.faces,
                edge.faces + len(bow.vertices),
                g.np.asarray(seam_faces) + offset,
            ]
        )

        # per-vertex UV with a seam: duplicated positions differ
        uv = g.np.random.random((len(vertices), 2))
        uv[offset:] = seam_uv
        # custom vertex and face attributes
        temperature = g.np.arange(len(vertices), dtype=g.np.float64)
        face_tag = g.np.arange(len(faces), dtype=g.np.int64)

        mesh = g.trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=g.trimesh.visual.TextureVisuals(uv=uv),
            vertex_attributes={"temperature": temperature},
            face_attributes={"face_tag": face_tag},
            process=False,
        )
        assert not g.trimesh.manifold.is_manifold(mesh)

        # the shared edge of the second component is edge (7, 8)
        constraints = {(7, 8): [(6, 7)]}
        result = g.trimesh.manifold.split_nonmanifold(mesh, edge_constraints=constraints)
        assert len(result.pending) == 0
        # bowtie -> 2, shared edge -> 2, seam strip -> 1
        assert len(result.meshes) == 5

        vertex_normals = mesh.vertex_normals.copy()
        ofid_all = []
        for patch in result.meshes:
            # every patch satisfies edge and vertex-link conditions
            assert g.trimesh.manifold.is_manifold(patch)
            ovid = patch.metadata["original_vertex_id"]
            ofid = patch.metadata["original_face_id"]
            ofid_all.append(ofid)
            # coordinates were copied, never moved
            assert g.np.allclose(patch.vertices, vertices[ovid])
            # UV coordinates followed the vertex copies
            assert g.np.allclose(patch.visual.uv, uv[ovid])
            # normals and attributes followed the copies
            assert g.np.allclose(patch.vertex_normals, vertex_normals[ovid])
            assert g.np.allclose(
                patch.vertex_attributes["temperature"], temperature[ovid]
            )
            assert (patch.face_attributes["face_tag"] == ofid).all()

        # faces were partitioned, never duplicated or dropped
        assert (g.np.sort(g.np.concatenate(ofid_all)) == g.np.arange(len(faces))).all()

        # the texture seam survived: two vertices share a position
        # but keep different UV coordinates
        seam = [p for p in result.meshes if offset in p.metadata["original_vertex_id"]]
        assert len(seam) == 1
        seam = seam[0]
        ovid = seam.metadata["original_vertex_id"]
        assert offset in ovid and offset + 8 in ovid
        pos = vertices[offset]
        match = g.np.nonzero((seam.vertices == pos).all(axis=1))[0]
        assert len(match) == 2
        assert not g.np.allclose(seam.visual.uv[match[0]], seam.visual.uv[match[1]])

        # the source mesh was not altered
        assert len(mesh.faces) == len(faces)
        assert len(mesh.vertices) == len(vertices)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
