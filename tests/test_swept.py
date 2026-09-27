try:
    from . import generic as g
except BaseException:
    import generic as g


def _triangle():
    # a single unit right triangle in the z=0 plane
    return g.trimesh.Trimesh(
        vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        faces=[[0, 1, 2]],
        process=False,
    )


def _square():
    # two coplanar triangles sharing a diagonal edge
    return g.trimesh.Trimesh(
        vertices=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )


class SweptSphereTest(g.unittest.TestCase):
    def test_face_interior_high_speed(self):
        # a fast sweep through a thin sliver triangle must not
        # be skipped the way fixed timestep sampling would
        mesh = g.trimesh.Trimesh(
            vertices=[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 1e-4, 0.0]],
            faces=[[0, 1, 2]],
            process=False,
        )
        result = g.trimesh.swept.swept_sphere(
            mesh, [5.0, 5e-5, 1.0], [5.0, 5e-5, -1.0], 0.01
        )
        assert result.hit
        # contact when the center is one radius above the plane
        assert g.np.isclose(result.t, (1.0 - 0.01) / 2.0)
        assert g.np.allclose(result.point, [5.0, 5e-5, 0.0], atol=1e-8)
        assert g.np.allclose(result.normal, [0.0, 0.0, 1.0], atol=1e-8)
        assert result.face == 0
        assert list(result.faces) == [0]
        assert g.np.allclose(result.center, [5.0, 5e-5, 0.01], atol=1e-8)

    def test_edge_graze(self):
        # sweep past an edge so the sphere is exactly tangent to it
        mesh = _triangle()
        result = g.trimesh.swept.swept_sphere(
            mesh, [0.5, -0.5, 1.0], [0.5, -0.5, -1.0], 0.5
        )
        assert result.hit
        # tangent to the edge at the midpoint of the motion
        assert g.np.isclose(result.t, 0.5)
        assert g.np.allclose(result.point, [0.5, 0.0, 0.0], atol=1e-8)
        assert g.np.allclose(result.normal, [0.0, -1.0, 0.0], atol=1e-8)

        # a slightly larger radius clips the edge twice: the
        # earliest contact is where the center is radius away
        radius = 0.5 * g.np.sqrt(2.0)
        result = g.trimesh.swept.swept_sphere(
            mesh, [0.5, -0.5, 1.0], [0.5, -0.5, -1.0], radius
        )
        assert result.hit
        assert g.np.isclose(result.t, 0.25)
        assert g.np.allclose(result.point, [0.5, 0.0, 0.0], atol=1e-8)
        assert g.np.allclose(result.normal, [0.0, -1.0, 1.0] / g.np.sqrt(2.0), atol=1e-8)

    def test_vertex_hit(self):
        # aim the sweep directly at a vertex of the triangle
        mesh = _triangle()
        radius = 0.1
        result = g.trimesh.swept.swept_sphere(
            mesh, [-1.0, -1.0, -1.0], [0.5, 0.5, 0.5], radius
        )
        assert result.hit
        # the center passes through the origin at t=2/3 and contact
        # is one radius of travel before that along the segment
        expected = (1.0 - radius / g.np.sqrt(3.0)) / 1.5
        assert g.np.isclose(result.t, expected)
        assert g.np.allclose(result.point, [0.0, 0.0, 0.0], atol=1e-8)
        assert g.np.allclose(
            result.normal,
            -g.np.ones(3) / g.np.sqrt(3.0),
            atol=1e-8,
        )

    def test_zero_displacement(self):
        # a zero length sweep is a static overlap query
        mesh = _triangle()
        result = g.trimesh.swept.swept_sphere(
            mesh, [0.1, 0.1, 0.05], [0.1, 0.1, 0.05], 0.1
        )
        assert result.hit
        assert result.t == 0.0
        assert g.np.allclose(result.point, [0.1, 0.1, 0.0], atol=1e-8)
        assert g.np.allclose(result.normal, [0.0, 0.0, 1.0], atol=1e-8)

        # a stationary sphere away from the mesh never contacts
        result = g.trimesh.swept.swept_sphere(mesh, [0.1, 0.1, 1.0], [0.1, 0.1, 1.0], 0.1)
        assert not result.hit
        assert g.np.isnan(result.t)
        assert result.face == -1
        assert len(result.faces) == 0

    def test_initial_overlap(self):
        # a sphere already intersecting the mesh contacts at t=0
        mesh = _triangle()
        result = g.trimesh.swept.swept_sphere(
            mesh, [0.25, 0.25, 0.05], [0.25, 0.25, 1.0], 0.1
        )
        assert result.hit
        assert result.t == 0.0
        assert g.np.allclose(result.point, [0.25, 0.25, 0.0], atol=1e-8)
        assert g.np.allclose(result.normal, [0.0, 0.0, 1.0], atol=1e-8)

    def test_exact_tangency(self):
        # ending exactly tangent to the surface counts as contact
        mesh = _triangle()
        result = g.trimesh.swept.swept_sphere(
            mesh, [0.25, 0.25, 1.0], [0.25, 0.25, 0.1], 0.1
        )
        assert result.hit
        assert g.np.isclose(result.t, 1.0)
        assert g.np.allclose(result.point, [0.25, 0.25, 0.0], atol=1e-8)

        # starting exactly tangent and moving away contacts at t=0
        result = g.trimesh.swept.swept_sphere(
            mesh, [0.25, 0.25, 0.1], [0.25, 0.25, 1.0], 0.1
        )
        assert result.hit
        assert result.t == 0.0

    def test_no_hit(self):
        mesh = _triangle()
        # passes to the side of the triangle
        result = g.trimesh.swept.swept_sphere(
            mesh, [2.0, 2.0, 1.0], [2.0, 2.0, -1.0], 0.1
        )
        assert not result.hit
        # sweep nowhere near any triangle: no broadphase candidates
        result = g.trimesh.swept.swept_sphere(
            mesh, [100.0, 100.0, 100.0], [200.0, 200.0, 200.0], 0.1
        )
        assert not result.hit

    def test_tie_reproducible(self):
        # dropping onto the shared edge of two coplanar triangles
        # contacts both faces at the same parameter
        mesh = _square()
        args = ([0.5, 0.5, 1.0], [0.5, 0.5, -1.0], 0.1)
        result = g.trimesh.swept.swept_sphere(mesh, *args)
        assert result.hit
        assert g.np.isclose(result.t, 0.45)
        assert list(result.faces) == [0, 1]
        assert result.face == 0
        assert g.np.allclose(result.point, [0.5, 0.5, 0.0], atol=1e-8)

        # shuffling the face order cannot change the first hit
        # time or the contact position, and ties are preserved
        for _ in range(10):
            perm = g.np.random.permutation(len(mesh.faces))
            shuffled = g.trimesh.Trimesh(
                vertices=mesh.vertices, faces=mesh.faces[perm], process=False
            )
            other = g.trimesh.swept.swept_sphere(shuffled, *args)
            assert other.hit
            assert other.t == result.t
            assert g.np.allclose(other.point, result.point, atol=1e-12)
            assert set(other.faces) == {
                int(g.np.nonzero(perm == f)[0][0]) for f in result.faces
            }

    def test_vertex_tie_on_mesh(self):
        # sweeping at a mesh vertex ties every face sharing it
        mesh = g.trimesh.creation.icosphere(subdivisions=3)
        vertex = mesh.vertices[0]
        radius = 0.1
        args = (vertex * 3.0, g.np.zeros(3), radius)
        result = g.trimesh.swept.swept_sphere(mesh, *args)
        assert result.hit
        # contact when the center is one radius from the vertex
        assert g.np.isclose(result.t, (2.0 - radius) / 3.0)
        assert g.np.allclose(result.point, vertex, atol=1e-8)
        expected = set(g.np.nonzero((mesh.faces == 0).any(axis=1))[0])
        assert set(result.faces) == expected
        assert result.face == min(expected)

        # face order permutations give a bitwise identical hit time
        for _ in range(5):
            perm = g.np.random.permutation(len(mesh.faces))
            shuffled = g.trimesh.Trimesh(
                vertices=mesh.vertices, faces=mesh.faces[perm], process=False
            )
            other = g.trimesh.swept.swept_sphere(shuffled, *args)
            assert other.hit
            assert other.t == result.t

    def test_validation(self):
        mesh = _triangle()
        with g.pytest.raises(ValueError):
            g.trimesh.swept.swept_sphere(mesh, [0.0, 0.0], [0.0, 0.0, 1.0], 0.1)
        with g.pytest.raises(ValueError):
            g.trimesh.swept.swept_sphere(mesh, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], -1.0)


if __name__ == "__main__":
    g.unittest.main()
