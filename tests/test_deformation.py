try:
    from . import generic as g
except BaseException:
    import generic as g

from trimesh.deformation import as_rigid_as_possible


def _annulus():
    # a mesh with a hole through it
    mesh = g.trimesh.creation.annulus(r_min=0.5, r_max=1.0, height=0.2, sections=64)
    radius = g.np.linalg.norm(mesh.vertices[:, :2], axis=1)
    anchors = g.np.nonzero(radius < 0.6)[0]
    handles = g.np.nonzero(radius > 0.95)[0][:8]
    targets = mesh.vertices[handles] + [0.5, 0.0, 0.0]
    return mesh, anchors, handles, targets


def test_local_pull_preserves_shape():
    # pull a few control points on a mesh with a hole and
    # check the deformation stays local and smooth
    mesh, anchors, handles, targets = _annulus()
    result = as_rigid_as_possible(mesh, handles, targets, anchors=anchors)

    assert result.converged
    assert result.stop_reason == "converged"
    assert g.np.isfinite(result.energy)
    # only geometry moved: same vertices and faces
    assert result.vertices.shape == mesh.vertices.shape
    # hard constraints are exactly satisfied
    assert g.np.allclose(result.vertices[anchors], mesh.vertices[anchors])
    assert g.np.allclose(result.vertices[handles], targets)
    # nothing drifted that wasn't supposed to
    assert len(result.under_constrained) == 0

    # faces far from the pulled handle should barely rotate:
    # the mesh should bend smoothly rather than translating
    center = mesh.triangles_center
    far = g.np.linalg.norm(center - [1.0, 0.0, 0.0], axis=1) > 1.0
    assert far.any()
    deformed = mesh.copy()
    deformed.vertices = result.vertices
    dot = g.np.clip(
        (mesh.face_normals[far] * deformed.face_normals[far]).sum(axis=1),
        -1.0,
        1.0,
    )
    angle = g.np.arccos(dot)
    assert angle.max() < g.np.radians(5.0)


def test_global_rigid_transform_has_zero_energy():
    # constraining handles to a rigidly transformed copy of
    # themselves should be achievable with (nearly) zero energy
    mesh, _, _, _ = _annulus()
    matrix = g.trimesh.transformations.rotation_matrix(
        0.7, [0.0, 0.0, 1.0], [0.1, 0.2, 0.3]
    )
    index = g.np.arange(0, len(mesh.vertices), 7)
    targets = g.trimesh.transformations.transform_points(mesh.vertices[index], matrix)
    result = as_rigid_as_possible(mesh, index, targets, tolerance=1e-12)

    assert result.converged
    assert result.energy < 1e-9
    # and the whole mesh should follow the rigid transform
    expected = g.trimesh.transformations.transform_points(mesh.vertices, matrix)
    assert g.np.allclose(result.vertices, expected, atol=1e-4)


def test_soft_constraints_are_weighted():
    mesh, anchors, handles, targets = _annulus()

    # a weak soft constraint should not reach its target
    weak = as_rigid_as_possible(
        mesh,
        handles,
        targets,
        anchors=anchors,
        control_weights=g.np.full(len(handles), 0.01),
    )
    assert weak.converged
    weak_error = g.np.abs(weak.vertices[handles] - targets).max()
    assert weak_error > 0.1

    # an infinite weight is a hard constraint
    hard = as_rigid_as_possible(
        mesh,
        handles,
        targets,
        anchors=anchors,
        control_weights=g.np.full(len(handles), g.np.inf),
    )
    assert g.np.allclose(hard.vertices[handles], targets)

    # a strong soft constraint should get close to its target
    strong = as_rigid_as_possible(
        mesh,
        handles,
        targets,
        anchors=anchors,
        control_weights=g.np.full(len(handles), 1e6),
    )
    assert g.np.abs(strong.vertices[handles] - targets).max() < 1e-3


def test_under_constrained_component_reported():
    # a connected component with no constraint must be
    # reported and left in place rather than drifting
    mesh, anchors, handles, targets = _annulus()
    other = mesh.copy()
    other.apply_translation([5.0, 0.0, 0.0])
    both = g.trimesh.util.concatenate([mesh, other])

    result = as_rigid_as_possible(both, handles, targets, anchors=anchors)

    assert len(result.under_constrained) == 1
    moved = result.under_constrained[0]
    assert len(moved) == len(mesh.vertices)
    # the unconstrained component stayed exactly where it was
    assert g.np.allclose(result.vertices[moved], both.vertices[moved])
    # the constrained component still deformed
    assert g.np.allclose(result.vertices[handles], targets)


def test_conflicting_constraints_raise():
    mesh, anchors, handles, targets = _annulus()

    # the same vertex given two different hard targets
    with g.pytest.raises(ValueError):
        as_rigid_as_possible(mesh, [0, 0], [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    # an anchor which is also a control point somewhere else
    with g.pytest.raises(ValueError):
        as_rigid_as_possible(mesh, [anchors[0]], [[100.0, 0.0, 0.0]], anchors=anchors)

    # identical duplicates are fine
    result = as_rigid_as_possible(
        mesh,
        g.np.concatenate([handles, handles]),
        g.np.vstack([targets, targets]),
        anchors=anchors,
    )
    assert result.converged


def test_degenerate_faces_reported():
    mesh, anchors, handles, targets = _annulus()
    # append a zero-area (collinear) face with new vertices
    vertices = g.np.vstack(
        [mesh.vertices, [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]
    )
    faces = g.np.vstack(
        [mesh.faces, [[len(vertices) - 3, len(vertices) - 2, len(vertices) - 1]]]
    )
    broken = g.trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    result = as_rigid_as_possible(broken, handles, targets, anchors=anchors)
    # the degenerate face was detected and uniform weights used
    assert len(result.degenerate_faces) == 1
    assert result.degenerate_faces[0] == len(mesh.faces)
    assert result.converged
    assert g.np.isfinite(result.energy)


def test_non_convergence_reported():
    mesh, anchors, handles, targets = _annulus()
    # a single iteration can't claim convergence
    result = as_rigid_as_possible(
        mesh, handles, targets, anchors=anchors, max_iterations=1
    )
    assert not result.converged
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 1
    assert g.np.isfinite(result.energy)


def test_face_attributes_still_correspond():
    mesh, anchors, handles, targets = _annulus()
    colors = g.np.zeros((len(mesh.faces), 4), dtype=g.np.uint8)
    colors[:, 0] = g.np.arange(len(mesh.faces)) % 255
    colors[:, 3] = 255
    mesh.visual.face_colors = colors

    result = as_rigid_as_possible(mesh, handles, targets, anchors=anchors)
    deformed = mesh.copy()
    deformed.vertices = result.vertices

    # faces and their attributes are untouched by deformation
    assert g.np.array_equal(deformed.faces, mesh.faces)
    assert g.np.array_equal(deformed.visual.face_colors, mesh.visual.face_colors)
    # the input mesh was not modified in place
    assert not g.np.array_equal(deformed.vertices, mesh.vertices)


def test_no_constraints_raise():
    mesh, _, _, _ = _annulus()
    with g.pytest.raises(ValueError):
        as_rigid_as_possible(mesh)


def test_soft_only_and_anchors_only():
    mesh, anchors, handles, targets = _annulus()

    # with only soft constraints a rigid translation satisfies
    # them exactly so the deformation energy is near zero
    result = as_rigid_as_possible(
        mesh,
        handles,
        targets,
        control_weights=g.np.full(len(handles), 10.0),
    )
    assert result.converged
    assert result.energy < 1e-12
    assert len(result.under_constrained) == 0

    # with only anchors the mesh simply stays where it is
    result = as_rigid_as_possible(mesh, anchors=anchors)
    assert result.converged
    assert result.energy < 1e-12
    assert g.np.allclose(result.vertices, mesh.vertices)


if __name__ == "__main__":
    g.pytest.main([__file__])
