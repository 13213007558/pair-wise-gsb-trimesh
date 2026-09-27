"""
test_winding.py
---------------

Tests for generalized winding number queries.
"""

try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.winding import BoundaryType


def _sphere(subdivisions=3, radius=1.0):
    return g.trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


def test_closed_sphere():
    # a closed watertight sphere: ~1 inside, ~0 outside
    mesh = _sphere()
    points = g.np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 10.0]]
    )
    for method in ["exact", "tree"]:
        winding = mesh.winding_number(points, method=method).winding
        assert g.np.allclose(winding[:2], 1.0, atol=0.05)
        assert g.np.allclose(winding[2:], 0.0, atol=0.05)

    # values should transition continuously through the surface
    # rather than jumping 0/1
    radii = g.np.linspace(0.9, 1.1, 41)
    ray = g.np.column_stack((radii, g.np.zeros_like(radii), g.np.zeros_like(radii)))
    winding = mesh.winding_number(ray, method="exact", boundary_tol=0).winding
    # strictly decreasing through the surface crossing
    assert (g.np.diff(winding) < 1e-6).all()
    # and it passes through intermediate values near the surface
    assert ((winding > 0.05) & (winding < 0.95)).any()


def test_open_sphere():
    # an open hemisphere is not a closed solid: the winding number
    # at the sphere center is the fractional solid angle, not 0/1
    mesh = _sphere()
    mesh.update_faces(mesh.triangles[:, :, 2].mean(axis=1) > 0.0)
    mesh.remove_unreferenced_vertices()
    assert not mesh.is_watertight

    points = g.np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    for method in ["exact", "tree"]:
        winding = mesh.winding_number(points, method=method).winding
        # hemisphere subtends ~2*pi steradians -> winding ~0.5
        assert g.np.abs(winding[0] - 0.5) < 0.05
        # far away the surface subtends almost no solid angle
        assert g.np.abs(winding[1]) < 0.01
        # the value is continuous, not a pressed 0/1 boolean
        assert 0.1 < winding[0] < 0.9


def test_overlapping_shells():
    # two concentric shells with opposite orientations: the winding
    # numbers of the regions differ and are continuous
    outer = _sphere(radius=2.0)
    inner = _sphere(radius=1.0)
    inner.invert()
    mesh = g.trimesh.util.concatenate([outer, inner])

    points = g.np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]])
    winding = mesh.winding_number(points, method="exact").winding
    # inside both shells the contributions cancel: 1 + (-1) = 0
    assert g.np.abs(winding[0]) < 1e-8
    # between the shells only the outer one contributes
    assert g.np.abs(winding[1] - 1.0) < 1e-8
    # outside both
    assert g.np.abs(winding[2]) < 1e-8
    # the region between the shells is distinct from the regions
    # where the opposite orientations cancel out
    assert g.np.abs(winding[1] - winding[0]) > 0.5
    # the hierarchical approximation reproduces the same pattern
    approx = mesh.winding_number(points, method="tree").winding
    assert g.np.allclose(approx, winding, atol=0.15)


def test_orientation_flip():
    # reversing every face flips the sign of the winding number
    mesh = _sphere()
    points = g.np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 0.1], [3.0, 0.0, 0.0]])
    inverted = mesh.copy()
    inverted.invert()
    for method in ["exact", "tree"]:
        forward = mesh.winding_number(points, method=method).winding
        flipped = inverted.winding_number(points, method=method).winding
        assert g.np.allclose(forward, -flipped, atol=1e-8)


def test_tree_vs_exact():
    # the hierarchical approximation should track the direct sum,
    # and the error should shrink as beta is increased
    mesh = _sphere(subdivisions=4)
    # random points, kept away from the surface
    points = (
        g.trimesh.sample.sample_surface_sphere(200)
        * g.np.linspace(0.3, 3.0, 200)[:, None]
    )
    exact = mesh.winding_number(points, method="exact", boundary_tol=0).winding
    error = []
    for beta in [2.0, 8.0]:
        approx = mesh.winding_number(points, method="tree", beta=beta).winding
        error.append(g.np.abs(approx - exact).max())
    assert error[0] < 0.05
    assert error[1] < error[0]


def test_boundary_flags():
    # points near faces, edges and vertices get explicit markers
    mesh = _sphere(subdivisions=2)
    tri = mesh.triangles[0]
    points = g.np.array(
        [
            tri.mean(axis=0),  # face interior
            (tri[0] + tri[1]) / 2.0,  # edge midpoint
            tri[0],  # vertex
            [0.5, 0.2, 0.1],  # interior, away from surface
        ]
    )
    result = mesh.winding_number(points)
    assert result.boundary[0] == BoundaryType.FACE
    assert result.boundary[1] == BoundaryType.EDGE
    assert result.boundary[2] == BoundaryType.VERTEX
    assert result.boundary[3] == BoundaryType.NONE
    assert result.on_boundary[:3].all()
    assert not result.on_boundary[3]

    # boundary classification can be disabled
    result = mesh.winding_number(points, boundary_tol=0)
    assert (result.boundary == BoundaryType.NONE).all()


def test_cache_invalidation():
    # modifying the geometry must not return values from a stale
    # cached hierarchy
    mesh = _sphere(subdivisions=2)
    point = [[0.5, 0.0, 0.0]]
    before = mesh.winding_number(point).winding[0]
    assert g.np.abs(before - 1.0) < 0.05

    # shrink the mesh so the query point is now outside
    mesh.vertices *= 0.1
    after = mesh.winding_number(point).winding[0]
    assert g.np.abs(after) < 0.01
    # and it matches a fresh direct summation
    exact = mesh.winding_number(point, method="exact").winding[0]
    assert g.np.abs(after - exact) < 0.01


def test_no_repair():
    # a mesh with a hole must be used exactly as stored: no hole
    # filling and no re-orientation to force pretty 0/1 results
    mesh = _sphere(subdivisions=2)
    face_count = len(mesh.faces)
    mesh.update_faces(g.np.arange(face_count) > 3)
    assert not mesh.is_watertight

    point = [[0.5, 0.0, 0.0]]
    winding = mesh.winding_number(point).winding[0]
    exact = mesh.winding_number(point, method="exact").winding[0]
    # close to but not exactly 1.0: the hole is still there
    assert g.np.abs(winding - exact) < 0.01
    assert g.np.abs(exact - 1.0) > 1e-3
    # the mesh itself was not modified by querying
    assert len(mesh.faces) == face_count - 4
    assert not mesh.is_watertight


if __name__ == "__main__":
    g.unittest.main()
