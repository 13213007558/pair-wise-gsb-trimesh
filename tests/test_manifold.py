try:
    from . import generic as g
except BaseException:
    import generic as g


class ManifoldTest(g.unittest.TestCase):
    def test_bowtie_vertex(self):
        # two triangles touching only at vertex 0
        vertices = g.np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        mesh = g.trimesh.Trimesh(
            vertices=vertices, faces=[[0, 1, 2], [0, 3, 4]], process=False
        )

        report = g.trimesh.manifold.manifold_report(mesh)
        # every edge is used at most twice: the naive edge check passes
        assert len(report.nonmanifold_edges) == 0
        # but vertex 0 has two disconnected fans in its link
        assert report.split_vertices.tolist() == [0]
        assert not report.is_manifold

        result = g.trimesh.manifold.split_manifold(mesh)
        assert len(result.meshes) == 2
        assert result.ambiguous == []
        assert sum(len(piece.vertices) for piece in result.meshes) == 6
        for piece, vsrc, fsrc in zip(
            result.meshes, result.vertex_source, result.face_source
        ):
            assert g.trimesh.manifold.is_manifold(piece)
            # coordinates are copied, never moved
            assert g.np.allclose(piece.vertices, mesh.vertices[vsrc])
            assert len(piece.faces) == 1
            # faces reference copies of the original corners
            assert g.np.array_equal(vsrc[piece.faces], mesh.faces[fsrc])

        # every original face appears exactly once
        assert sorted(g.np.concatenate(result.face_source).tolist()) == [0, 1]

    def test_three_faces_one_edge(self):
        # three triangles sharing edge (0, 1)
        vertices = g.np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
                [0.5, 0.0, 1.0],
            ]
        )
        mesh = g.trimesh.Trimesh(
            vertices=vertices,
            faces=[[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            process=False,
        )

        report = g.trimesh.manifold.manifold_report(mesh)
        assert report.nonmanifold_edges.tolist() == [[0, 1]]

        # no constraints: every face on the edge is ambiguous
        result = g.trimesh.manifold.split_manifold(mesh)
        assert len(result.meshes) == 3
        assert len(result.ambiguous) == 1
        assert result.ambiguous[0].edge.tolist() == [0, 1]
        assert result.ambiguous[0].faces.tolist() == [0, 1, 2]
        for piece in result.meshes:
            assert g.trimesh.manifold.is_manifold(piece)

        # pair two faces: the third stays pending
        result = g.trimesh.manifold.split_manifold(mesh, edge_adjacency=[(0, 1)])
        assert len(result.meshes) == 2
        assert len(result.ambiguous) == 1
        assert result.ambiguous[0].faces.tolist() == [2]
        sizes = sorted(len(piece.faces) for piece in result.meshes)
        assert sizes == [1, 2]
        for piece in result.meshes:
            assert g.trimesh.manifold.is_manifold(piece)

        # constraints must reference faces sharing an edge
        vertices_bad = g.np.vstack(
            [vertices, [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]]]
        )
        faces_bad = g.np.vstack([mesh.faces, [[5, 6, 7]]])
        mesh_bad = g.trimesh.Trimesh(
            vertices=vertices_bad, faces=faces_bad, process=False
        )
        with g.pytest.raises(ValueError):
            g.trimesh.manifold.split_manifold(mesh_bad, edge_adjacency=[(0, 3)])

    def test_shared_apex_tetrahedra(self):
        # two closed tetrahedra sharing exactly one apex vertex
        vertices_a = g.np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        faces_a = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
        vertices_b = g.np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        faces_b = [[0, 5, 4], [0, 6, 5], [0, 4, 6], [4, 5, 6]]
        mesh = g.trimesh.Trimesh(
            vertices=g.np.vstack([vertices_a, vertices_b[1:]]),
            faces=faces_a + faces_b,
            process=False,
        )

        report = g.trimesh.manifold.manifold_report(mesh)
        assert report.split_vertices.tolist() == [0]

        result = g.trimesh.manifold.split_manifold(mesh)
        assert len(result.meshes) == 2
        for piece in result.meshes:
            assert g.trimesh.manifold.is_manifold(piece)
            assert piece.is_watertight
            assert len(piece.faces) == 4
        assert result.ambiguous == []

    def test_textured_combined_attributes(self):
        # a bowtie fan contact and a three-face edge in one mesh
        bowtie_vertices = g.np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        bowtie_faces = [[0, 1, 2], [0, 3, 4]]
        edge_vertices = g.np.array(
            [
                [10.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [10.5, 1.0, 0.0],
                [10.5, -1.0, 0.0],
                [10.5, 0.0, 1.0],
            ]
        )
        edge_faces = [
            [5, 6, 7],
            [6, 5, 8],
            [5, 6, 9],
        ]
        vertices = g.np.vstack([bowtie_vertices, edge_vertices])
        faces = g.np.array(bowtie_faces + edge_faces)

        mesh = g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        vertex_count = len(vertices)
        face_count = len(faces)

        # custom vertex and face attributes
        mesh.vertex_attributes["temperature"] = g.np.arange(vertex_count) * 1.5
        mesh.face_attributes["face_id"] = g.np.arange(face_count) + 100

        # texture UVs: values are unique per original vertex so a
        # wrong copy is immediately visible
        uv = g.np.arange(vertex_count * 2, dtype=g.np.float64).reshape((vertex_count, 2))
        mesh.visual = g.trimesh.visual.TextureVisuals(uv=uv)

        # explicitly assigned vertex normals
        normals = g.trimesh.util.unitize(
            g.np.arange(vertex_count * 3, dtype=g.np.float64).reshape((vertex_count, 3))
            + 1.0
        )
        mesh.vertex_normals = normals

        # pair faces 2 and 3 across the non-manifold edge (5, 6)
        result = g.trimesh.manifold.split_manifold(mesh, edge_adjacency=[(2, 3)])

        # bowtie splits into two pieces, the edge into two more
        assert len(result.meshes) == 4
        assert len(result.ambiguous) == 1
        assert result.ambiguous[0].faces.tolist() == [4]
        assert result.ambiguous[0].edge.tolist() == [5, 6]

        all_faces = g.np.concatenate(result.face_source)
        assert sorted(all_faces.tolist()) == list(range(face_count))

        for piece, vsrc, fsrc in zip(
            result.meshes, result.vertex_source, result.face_source
        ):
            assert g.trimesh.manifold.is_manifold(piece)

            # only identities were copied: nothing was moved
            assert g.np.allclose(piece.vertices, mesh.vertices[vsrc])

            assert g.np.allclose(
                piece.vertex_attributes["temperature"],
                mesh.vertex_attributes["temperature"][vsrc],
            )
            assert g.np.allclose(
                piece.face_attributes["face_id"],
                mesh.face_attributes["face_id"][fsrc],
            )
            assert g.np.allclose(piece.visual.uv, uv[vsrc])
            assert g.np.allclose(piece.vertex_normals, normals[vsrc])

            # faces are unchanged apart from renumbering
            assert g.np.array_equal(vsrc[piece.faces], mesh.faces[fsrc])

    def test_already_manifold(self):
        sphere = g.trimesh.creation.icosphere(subdivisions=2)
        report = g.trimesh.manifold.manifold_report(sphere)
        assert report.is_manifold
        result = g.trimesh.manifold.split_manifold(sphere)
        assert len(result.meshes) == 1
        piece = result.meshes[0]
        assert g.trimesh.manifold.is_manifold(piece)
        assert g.np.array_equal(
            result.vertex_source[0], g.np.arange(len(sphere.vertices))
        )
        assert g.np.array_equal(result.face_source[0], g.np.arange(len(sphere.faces)))
        assert g.np.allclose(piece.vertices, sphere.vertices)
        assert g.np.array_equal(piece.faces, sphere.faces)
