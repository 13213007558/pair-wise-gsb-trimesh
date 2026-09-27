try:
    from . import generic as g
except BaseException:
    import generic as g


class WindingTest(g.unittest.TestCase):
    def test_closed_sphere(self):
        # a closed outward-oriented sphere should give
        # winding numbers very close to 1 inside and 0 outside
        sphere = g.trimesh.creation.icosphere(subdivisions=3)
        points = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0]]

        result = sphere.winding_numbers(points)
        assert g.np.allclose(result.winding[:2], 1.0, atol=1e-8)
        assert g.np.abs(result.winding[2]) < 1e-8
        # none of these points are near the surface
        assert (result.boundary == g.trimesh.winding.BoundaryType.NONE).all()

    def test_open_sphere(self):
        # an open surface should produce continuous intermediate
        # values rather than being forced to 0 or 1
        sphere = g.trimesh.creation.icosphere(subdivisions=3)
        closed = sphere.winding_numbers([[0.0, 0.0, 0.0]]).winding[0]

        # remove all faces above a z plane to open a hole
        opened = sphere.copy()
        opened.update_faces(opened.triangles[:, :, 2].mean(axis=1) < 0.2)
        assert not opened.is_watertight

        result = opened.winding_numbers([[0.0, 0.0, 0.0]])
        value = result.winding[0]
        # strictly between 0 and 1 and different from the closed mesh
        assert 0.0 < value < 1.0
        assert not g.np.isclose(value, closed, atol=1e-3)

    def test_opposite_shells(self):
        # two nested shells with opposite orientation should cancel:
        # roughly 0 at the center, 1 between the shells, 0 outside
        outer = g.trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        inner = g.trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        inner.invert()
        shells = g.trimesh.util.concatenate([outer, inner])

        points = [[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [5.0, 0.0, 0.0]]
        winding = shells.winding_numbers(points).winding
        assert g.np.abs(winding[0]) < 1e-8
        assert g.np.isclose(winding[1], 1.0, atol=1e-8)
        assert g.np.abs(winding[2]) < 1e-8

        # a single closed shell gives different values at the center
        single = outer.winding_numbers([[0.0, 0.0, 0.0]]).winding[0]
        assert not g.np.isclose(winding[0], single, atol=1e-3)

    def test_invert_flips_sign(self):
        # inverting every face should negate the winding numbers
        sphere = g.trimesh.creation.icosphere(subdivisions=3)
        points = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 0.0, 0.0]]

        inverted = sphere.copy()
        inverted.invert()

        forward = sphere.winding_numbers(points).winding
        backward = inverted.winding_numbers(points).winding
        assert g.np.allclose(forward, -backward, atol=1e-8)

        # the hierarchical approximation should flip as well
        forward_a = sphere.winding_numbers(points, method="approximate").winding
        backward_a = inverted.winding_numbers(points, method="approximate").winding
        assert g.np.allclose(forward_a, -backward_a, atol=1e-8)

    def test_approximate_error(self):
        # the hierarchical approximation should track the exact sum
        # and improve as the opening angle is reduced
        sphere = g.trimesh.creation.icosphere(subdivisions=4)
        rng = g.np.random.default_rng(42)
        points = rng.normal(size=(100, 3))
        radii = g.np.linalg.norm(points, axis=1)
        # keep query points away from the surface
        points = points[(radii < 0.8) | (radii > 1.2)]

        exact = sphere.winding_numbers(points).winding
        errors = []
        for theta in [0.5, 0.3, 0.1]:
            approx = sphere.winding_numbers(
                points, method="approximate", theta=theta
            ).winding
            errors.append(g.np.abs(approx - exact).max())

        # reasonably accurate at the default scale of theta
        assert errors[0] < 0.1
        # and monotonically better as theta is reduced
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
        assert errors[2] < 1e-3

    def test_boundary_flags(self):
        # points near faces, edges, and vertices should be marked
        sphere = g.trimesh.creation.icosphere(subdivisions=3)
        boundary = g.trimesh.winding.BoundaryType

        face_point = sphere.triangles[0].mean(axis=0)
        edge_point = sphere.vertices[sphere.faces[0][:2]].mean(axis=0)
        vertex_point = sphere.vertices[0]
        far_point = [5.0, 0.0, 0.0]

        result = sphere.winding_numbers([face_point, edge_point, vertex_point, far_point])
        assert result.boundary[0] == boundary.FACE
        assert result.boundary[1] == boundary.EDGE
        assert result.boundary[2] == boundary.VERTEX
        assert result.boundary[3] == boundary.NONE

        # the approximate method should mark boundaries identically
        approx = sphere.winding_numbers(
            [face_point, edge_point, vertex_point, far_point],
            method="approximate",
        )
        assert (approx.boundary == result.boundary).all()

    def test_tree_invalidation(self):
        # modifying the geometry must invalidate the cached hierarchy
        sphere = g.trimesh.creation.icosphere(subdivisions=3)
        point = [[0.0, 0.0, 0.0]]

        tree_before = sphere.winding_tree
        assert g.np.isclose(
            sphere.winding_numbers(point, method="approximate").winding[0],
            1.0,
            atol=0.05,
        )

        # move the mesh so the query point is now outside
        sphere.vertices = sphere.vertices + [10.0, 0.0, 0.0]

        tree_after = sphere.winding_tree
        assert tree_after is not tree_before

        approx = sphere.winding_numbers(point, method="approximate").winding[0]
        exact = sphere.winding_numbers(point).winding[0]
        assert g.np.abs(exact) < 1e-8
        assert g.np.isclose(approx, exact, atol=0.05)

    def test_result_unpacking(self):
        # the result should unpack into (winding, boundary)
        sphere = g.trimesh.creation.icosphere(subdivisions=2)
        winding, boundary = sphere.winding_numbers([[0.0, 0.0, 0.0]])
        assert g.np.isclose(winding[0], 1.0)
        assert boundary[0] == g.trimesh.winding.BoundaryType.NONE


if __name__ == "__main__":
    g.unittest.main()
