try:
    from . import generic as g
except BaseException:
    import generic as g


def _grid_mesh(cells=12):
    """
    A triangulated grid on the unit square.
    """
    np = g.np
    x = np.linspace(0.0, 1.0, cells + 1)
    xx, yy = np.meshgrid(x, x)
    vertices = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    index = np.arange(xx.size).reshape(xx.shape)
    faces = []
    for i in range(cells):
        for j in range(cells):
            a = index[i, j]
            b = index[i, j + 1]
            c = index[i + 1, j]
            d = index[i + 1, j + 1]
            faces.append([a, b, d])
            faces.append([a, d, c])
    return g.trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces), process=False)


class GeodesicTest(g.unittest.TestCase):
    def test_plane(self):
        # on a flat grid the geodesic distance is the
        # euclidean distance and does not depend on the
        # direction of the mesh edges
        np = g.np
        mesh = _grid_mesh(cells=12)
        corner = int(np.argmin((mesh.vertices**2).sum(axis=1)))
        opposite = int(np.argmax((mesh.vertices**2).sum(axis=1)))
        distance = g.trimesh.geodesic.heat_geodesic(mesh, [corner])
        assert np.isfinite(distance).all()
        assert np.isclose(distance[corner], 0.0)
        # euclidean distance between opposite corners of the
        # unit square is sqrt(2)
        assert np.isclose(distance[opposite], np.sqrt(2.0), rtol=0.1)

    def test_sphere(self):
        # on a sphere the geodesic distance is the great
        # circle distance
        np = g.np
        mesh = g.trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        source = int(np.argmin(((mesh.vertices - [0.0, 0.0, 1.0]) ** 2).sum(axis=1)))
        target = int(np.argmin(((mesh.vertices - [1.0, 0.0, 0.0]) ** 2).sum(axis=1)))
        distance = g.trimesh.geodesic.heat_geodesic(mesh, [source])
        assert np.isfinite(distance).all()
        # a quarter of the equator is pi / 2
        assert np.isclose(distance[target], np.pi / 2.0, rtol=0.1)

    def test_convergence(self):
        # the error against the known great circle distance
        # should decrease as the mesh is refined
        np = g.np
        errors = []
        for subdivisions in (1, 2, 3):
            mesh = g.trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
            source = int(np.argmin(((mesh.vertices - [0.0, 0.0, 1.0]) ** 2).sum(axis=1)))
            target = int(np.argmin(((mesh.vertices - [1.0, 0.0, 0.0]) ** 2).sum(axis=1)))
            distance = g.trimesh.geodesic.heat_geodesic(mesh, [source])
            errors.append(abs(distance[target] - np.pi / 2.0))
        assert errors[0] > errors[1] > errors[2]

    def test_unreachable(self):
        # components without a source are unreachable and
        # must not be filled with zero distance
        np = g.np
        a = g.trimesh.creation.icosphere(subdivisions=2)
        b = g.trimesh.creation.icosphere(subdivisions=2)
        b.vertices += [5.0, 0.0, 0.0]
        mesh = g.trimesh.util.concatenate([a, b])
        distance = g.trimesh.geodesic.heat_geodesic(mesh, [0])
        assert np.isfinite(distance[: len(a.vertices)]).all()
        assert np.isinf(distance[len(a.vertices) :]).all()

    def test_degenerate_face(self):
        # zero area faces make the operator undefined and
        # must be diagnosed rather than silently accepted
        np = g.np
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2], [0, 1, 3]])
        mesh = g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        with self.assertRaises(ValueError):
            g.trimesh.geodesic.HeatGeodesicSolver(mesh)

    def test_isolated_vertex(self):
        # vertices not referenced by any face make the
        # system singular and must be diagnosed
        np = g.np
        mesh = g.trimesh.creation.icosphere(subdivisions=1)
        vertices = np.vstack((mesh.vertices, [[10.0, 10.0, 10.0]]))
        mesh = g.trimesh.Trimesh(vertices=vertices, faces=mesh.faces, process=False)
        with self.assertRaises(ValueError):
            g.trimesh.geodesic.HeatGeodesicSolver(mesh)

    def test_barycentric_source(self):
        # sources can be located inside a face with
        # barycentric coordinates
        np = g.np
        mesh = g.trimesh.creation.icosphere(subdivisions=2)
        sources = [(0, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])]
        distance = g.trimesh.geodesic.heat_geodesic(mesh, sources)
        assert np.isfinite(distance).all()
        # the vertices of the source face are closest
        face = mesh.faces[0]
        assert (distance[face] < 0.3).all()
        assert np.isclose(distance.min(), 0.0)

    def test_cache(self):
        # factorizations are reused while the geometry is
        # unchanged and invalidated when vertices or faces
        # change
        np = g.np
        mesh = g.trimesh.creation.icosphere(subdivisions=2)
        solver = g.trimesh.geodesic.HeatGeodesicSolver(mesh)
        first = solver.solve([0])
        factor = solver._factor_heat
        key = solver.geometry_key

        # solving again reuses the same factorization
        solver.solve([1])
        assert solver._factor_heat is factor
        assert solver.geometry_key == key

        # changing vertices invalidates the cache
        mesh.vertices *= 2.0
        second = solver.solve([0])
        assert solver._factor_heat is not factor
        assert solver.geometry_key != key
        # distances scale with the mesh
        assert np.isclose(second.max(), 2.0 * first.max(), rtol=0.05)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
