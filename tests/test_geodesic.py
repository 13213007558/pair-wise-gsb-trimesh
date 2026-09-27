try:
    from . import generic as g
except BaseException:
    import generic as g


def plane_mesh(res, size=1.0):
    """
    A triangulated square grid in the z=0 plane with res cells
    per side, useful because geodesic distance on it is Euclidean.
    """
    grid = g.np.linspace(0.0, size, res + 1)
    x, y = g.np.meshgrid(grid, grid, indexing="ij")
    vertices = g.np.column_stack([x.ravel(), y.ravel(), g.np.zeros((res + 1) ** 2)])
    row = g.np.arange(res)[:, None] * (res + 1) + g.np.arange(res)[None, :]
    a = row.ravel()
    b = a + 1
    c = a + res + 1
    d = c + 1
    faces = g.np.column_stack(
        [g.np.stack([a, b, d], axis=1), g.np.stack([a, d, c], axis=1)]
    )
    faces = faces.reshape(-1, 3)
    return g.trimesh.Trimesh(vertices, faces, process=False)


class GeodesicTest(g.unittest.TestCase):
    def test_plane_distance(self):
        # on a flat plane geodesic distance is Euclidean distance
        m = plane_mesh(16)
        source = g.np.argmin(((m.vertices[:, :2] - 0.5) ** 2).sum(axis=1))
        distance = g.trimesh.geodesic.heat_geodesic(m, [source])
        truth = g.np.linalg.norm(m.vertices[:, :2] - m.vertices[source, :2], axis=1)
        # evaluate away from the source and the boundary
        band = (truth > 0.2) & (truth < 0.35)
        relative = g.np.abs(distance - truth)[band] / truth[band]
        assert relative.mean() < 0.05
        assert relative.max() < 0.1
        # the source itself should be (approximately) zero
        assert abs(distance[source]) < 1e-8

    def test_sphere_distance(self):
        # on a sphere geodesic distance from the source vertex is
        # radius times the central angle
        m = g.trimesh.creation.icosphere(subdivisions=3)
        distance = g.trimesh.geodesic.heat_geodesic(m, [0])
        truth = g.np.arccos(g.np.clip(m.vertices @ m.vertices[0], -1.0, 1.0))
        band = (truth > 0.3) & (truth < 2.6)
        relative = g.np.abs(distance - truth)[band] / truth[band]
        assert relative.mean() < 0.03
        assert relative.max() < 0.1

    def test_convergence(self):
        # the approximation should get better as the mesh is refined
        def sphere_error(subdivisions):
            m = g.trimesh.creation.icosphere(subdivisions=subdivisions)
            distance = g.trimesh.geodesic.heat_geodesic(m, [0])
            truth = g.np.arccos(g.np.clip(m.vertices @ m.vertices[0], -1.0, 1.0))
            band = (truth > 0.3) & (truth < 2.6)
            return (g.np.abs(distance - truth)[band] / truth[band]).mean()

        def plane_error(res):
            m = plane_mesh(res)
            source = g.np.argmin(((m.vertices[:, :2] - 0.5) ** 2).sum(axis=1))
            distance = g.trimesh.geodesic.heat_geodesic(m, [source])
            truth = g.np.linalg.norm(m.vertices[:, :2] - m.vertices[source, :2], axis=1)
            band = (truth > 0.2) & (truth < 0.35)
            return (g.np.abs(distance - truth)[band] / truth[band]).mean()

        sphere_errors = [sphere_error(s) for s in (1, 2, 3)]
        assert sphere_errors[0] > sphere_errors[1] > sphere_errors[2]

        plane_errors = [plane_error(r) for r in (8, 16, 32)]
        assert plane_errors[0] > plane_errors[1] > plane_errors[2]

    def test_multiple_sources(self):
        # multiple sources give the distance to the closest one
        m = plane_mesh(16)
        source_a = g.np.argmin(((m.vertices[:, :2] - [0.5, 0.5]) ** 2).sum(axis=1))
        source_b = g.np.argmin(((m.vertices[:, :2] - [0.25, 0.5]) ** 2).sum(axis=1))
        distance = g.trimesh.geodesic.heat_geodesic(m, [source_a, source_b])
        truth_a = g.np.linalg.norm(m.vertices[:, :2] - m.vertices[source_a, :2], axis=1)
        truth_b = g.np.linalg.norm(m.vertices[:, :2] - m.vertices[source_b, :2], axis=1)
        truth = g.np.minimum(truth_a, truth_b)
        band = (truth_a > 0.2) & (truth_b > 0.2) & (truth < 0.3)
        relative = g.np.abs(distance - truth)[band] / truth[band]
        assert relative.mean() < 0.08
        # both sources should be at approximately zero distance
        h = 1.0 / 16
        assert g.np.abs(distance[[source_a, source_b]]).max() < h

    def test_barycentric_source(self):
        # sources may be points inside faces given by barycentric
        # coordinates, here the centroid of a face
        m = plane_mesh(16)
        h = 1.0 / 16
        face_index = 100
        barycentric = g.np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        distance = g.trimesh.geodesic.heat_geodesic(m, ([face_index], barycentric))
        center = m.triangles[face_index].mean(axis=0)
        truth = g.np.linalg.norm(m.vertices - center, axis=1)
        # the corners of the source face should be close to the source
        assert g.np.abs(distance[m.faces[face_index]]).max() < h
        # and the far field should match the Euclidean distance
        band = (truth > 4 * h) & (truth < 0.35)
        relative = g.np.abs(distance - truth)[band] / truth[band]
        assert relative.mean() < 0.1

    def test_unreachable(self):
        # components without a source are unreachable: they must
        # get inf rather than a zero-filled distance
        boxes = [
            g.trimesh.creation.box(),
            g.trimesh.creation.box().apply_translation([5.0, 0.0, 0.0]),
        ]
        m = g.trimesh.util.concatenate(boxes)
        distance = g.trimesh.geodesic.heat_geodesic(m, [0])
        assert g.np.isinf(distance[8:]).all()
        reachable = distance[:8]
        assert g.np.isfinite(reachable).all()
        # no zero filling: only the source vertex is near zero
        assert (reachable[1:] > 0.1).all()

    def test_degenerate(self):
        m = plane_mesh(4)
        # a zero-area face cannot support a cotangent operator
        faces = g.np.vstack([m.faces, [[0, 1, 1]]])
        broken = g.trimesh.Trimesh(m.vertices.copy(), faces, process=False)
        with g.np.testing.assert_raises(ValueError):
            g.trimesh.geodesic.heat_geodesic(broken, [0])

    def test_isolated_vertex(self):
        m = plane_mesh(4)
        # a vertex referenced by no face has no mass to diffuse
        vertices = g.np.vstack([m.vertices, [[10.0, 10.0, 0.0]]])
        broken = g.trimesh.Trimesh(vertices, m.faces.copy(), process=False)
        with g.np.testing.assert_raises(ValueError):
            g.trimesh.geodesic.heat_geodesic(broken, [0])

    def test_invalid_sources(self):
        m = plane_mesh(4)
        # barycentric coordinates must sum to one
        with g.np.testing.assert_raises(ValueError):
            g.trimesh.geodesic.heat_geodesic(m, ([0], [[0.5, 0.5, 0.5]]))
        # vertex indices must be in range
        with g.np.testing.assert_raises(ValueError):
            g.trimesh.geodesic.heat_geodesic(m, [len(m.vertices) + 1])
        # at least one source is required
        with g.np.testing.assert_raises(ValueError):
            g.trimesh.geodesic.heat_geodesic(m, [])

    def test_factorization_cache(self):
        # repeated calls on the same geometry reuse the factorization
        m = plane_mesh(8)
        key = ("geodesic.HeatGeodesic", None)
        g.trimesh.geodesic.heat_geodesic(m, [0])
        first = m._cache.cache[key]
        g.trimesh.geodesic.heat_geodesic(m, [3])
        assert m._cache.cache[key] is first

        # changing the vertices must invalidate the cached factorization
        distance = g.trimesh.geodesic.heat_geodesic(m, [0])
        m.vertices = m.vertices * 2.0
        scaled = g.trimesh.geodesic.heat_geodesic(m, [0])
        assert m._cache.cache[key] is not first
        assert g.np.allclose(scaled, 2.0 * distance, rtol=1e-6)

        # changing the faces must also invalidate
        m.faces = m.faces[::-1]
        assert m._cache.cache[key] is not first

    def test_solver_reuse(self):
        # the solver object exposes the operators and can be
        # evaluated for many source sets without re-factorizing
        m = plane_mesh(8)
        solver = g.trimesh.geodesic.HeatGeodesic(m)
        assert solver.laplacian.shape == (len(m.vertices), len(m.vertices))
        assert solver.mass.shape == (len(m.vertices),)
        assert solver.t > 0.0
        assert solver.component_count == 1
        first = solver.distance([0])
        second = solver.distance([10])
        assert first.shape == second.shape == (len(m.vertices),)
        assert not g.np.allclose(first, second)


if __name__ == "__main__":
    g.unittest.main()
