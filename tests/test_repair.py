try:
    from . import generic as g
except BaseException:
    import generic as g


def test_fill_holes():
    meshes = [
        "unit_cube.STL",
        "machinist.XAML",
        "round.stl",
        "sphere.ply",
        "teapot.stl",
        "soup.stl",
        "featuretype.STL",
        "angle_block.STL",
        "quadknot.obj",
    ]

    try:
        import cascadio  # noqa

        # if we have cascadio test against a STEP file
        meshes.append("featuretype.STEP")
    except ImportError:
        pass

    for name in meshes:
        mesh = g.get_mesh(name)
        # handle scene lazily
        if hasattr(mesh, "geometry"):
            mesh = next(iter(mesh.geometry.values()))

        if not mesh.is_watertight:
            # output of fill_holes should match watertight status
            returned = mesh.fill_holes()
            assert returned == mesh.is_watertight
            continue

        hashes = [{mesh._data.__hash__(), hash(mesh)}]

        # clip off the first and last few faces
        mask = g.np.ones(len(mesh.faces), dtype=bool)
        mask[[0, -1]] = False
        mesh.update_faces(mask)

        assert not mesh.is_watertight
        assert not mesh.is_volume

        # assert face attributes match faces
        for attrib in mesh.face_attributes.values():
            assert len(attrib) == len(mesh.faces)

        # color some faces
        g.trimesh.repair.broken_faces(mesh, color=[255, 0, 0, 255])

        hashes.append({mesh._data.__hash__(), hash(mesh)})

        assert hashes[0] != hashes[1]

        # put face_normals in cache then corrupt one
        # so we can verify extend_faces preserves cached normals
        _ = mesh.face_normals
        mesh._cache.cache["face_normals"].flags.writeable = True
        mesh._cache.cache["face_normals"][3, :] += 10
        mesh._cache.cache["face_normals"].flags.writeable = False
        cached_marker = mesh.face_normals[3].copy()
        assert (cached_marker >= 9).all()

        # run the fill holes operation should succeed
        assert mesh.fill_holes()
        # should be a superset of the last two
        assert mesh.is_volume
        assert mesh.is_watertight
        assert mesh.is_winding_consistent

        # the corrupted normal should have survived from cache
        assert g.np.allclose(mesh.face_normals[3], cached_marker)

        hashes.append({mesh._data.__hash__(), hash(mesh)})
        assert hashes[1] != hashes[2]

        # assert the filled holes padded attributes
        # this is also the test for `mesh.extend_faces`
        for attrib in mesh.face_attributes.values():
            assert len(attrib) == len(mesh.faces)
        # face colors should have been padded to match
        assert len(mesh.visual.face_colors) == len(mesh.faces)

        # try broken faces on a watertight mesh
        g.trimesh.repair.broken_faces(mesh, color=[255, 255, 0, 255])


def test_fix_normals():
    for mesh in g.get_meshes(5):
        mesh.fix_normals()


def test_winding():
    """
    Reverse some faces and make sure fix_face_winding flips
    them back.
    """

    meshes = [
        g.get_mesh(i)
        for i in [
            "unit_cube.STL",
            "machinist.XAML",
            "round.stl",
            "quadknot.obj",
            "soup.stl",
        ]
    ]

    for i, mesh in enumerate(meshes):
        # turn scenes into multibody meshes
        if g.trimesh.util.is_instance_named(mesh, "Scene"):
            meta = mesh.metadata
            meshes[i] = mesh.dump().sum()
            meshes[i].metadata = meta

    timing = {}
    for mesh in meshes:
        # save the initial state
        is_volume = mesh.is_volume
        winding = mesh.is_winding_consistent

        tic = g.time.time()
        # flip faces to break winding
        mesh.faces[:4] = g.np.fliplr(mesh.faces[:4])

        # run the operation
        mesh.fix_normals()

        # make sure mesh is repaired to former glory
        assert mesh.is_volume == is_volume
        assert mesh.is_winding_consistent == winding

        # save timings
        timing[mesh.source.file_name] = g.time.time() - tic
    # print timings as a warning
    g.log.warning(g.json.dumps(timing, indent=4))


def test_inversion():
    """Make sure fix_inversion switches all reversed faces back"""
    orig_mesh = g.get_mesh("unit_cube.STL")
    orig_verts = orig_mesh.vertices.copy()
    orig_faces = orig_mesh.faces.copy()

    mesh = g.Trimesh(orig_verts, orig_faces[:, ::-1])
    inv_faces = mesh.faces.copy()
    # check not fixed on the way in
    assert not g.np.allclose(inv_faces, orig_faces)

    g.trimesh.repair.fix_inversion(mesh)
    assert not g.np.allclose(mesh.faces, inv_faces)
    assert g.np.allclose(mesh.faces, orig_faces)


def test_multi():
    """
    Try repairing a multibody geometry
    """
    # create a multibody mesh with two cubes
    a = g.get_mesh("unit_cube.STL")
    b = a.copy()
    b.apply_translation([2, 0, 0])
    m = a + b
    # should be a volume: watertight, correct winding
    assert m.is_volume

    # flip one face of A
    a.faces[:1] = g.np.fliplr(a.faces[:1])
    # flip every face of A
    a.invert()
    # flip one face of B
    b.faces[:1] = g.np.fliplr(b.faces[:1])
    m = a + b

    # not a volume
    assert not m.is_volume

    m.fix_normals(multibody=False)

    # shouldn't fix inversion of one cube
    assert not m.is_volume

    # run fix normal with multibody mode
    m.fix_normals()

    # should be volume again
    assert m.is_volume

    # mesh should be volume of two boxes, and positive
    assert g.np.isclose(m.volume, 2.0)


def test_flip():
    # create two spheres
    a = g.trimesh.creation.icosphere()
    b = g.trimesh.creation.icosphere().apply_translation([2, 3, 0])
    # invert the second sphere
    b.faces = g.np.fliplr(b.faces)
    m = a + b
    # make sure normals are in cache
    assert m.face_normals.shape == m.faces.shape
    m.fix_normals(multibody=True)
    assert g.np.isclose(m.volume, a.volume * 2.0)


def test_nested_eccentric():
    # a solid box with an eccentric spherical cavity where the
    # cavity shell is incorrectly oriented as an independent solid
    box = g.trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    cavity = g.trimesh.creation.icosphere(radius=0.5)
    cavity.apply_translation([0.4, 0.3, -0.2])

    mesh = box + cavity
    # both shells start out oriented as solids
    assert g.np.isclose(mesh.volume, box.volume + cavity.volume)

    # add per-face attributes to check they survive the fix
    colors = g.np.arange(len(mesh.faces) * 4, dtype=g.np.uint8).reshape((-1, 4))
    mesh.visual.face_colors = colors
    vertices = mesh.vertices.copy()

    g.trimesh.repair.fix_inversion(mesh, nested=True)

    # signed volume and mass center should match a hollow box
    expect_volume = box.volume - cavity.volume
    assert g.np.isclose(mesh.volume, expect_volume)
    expect_com = (
        box.center_mass * box.volume - cavity.center_mass * cavity.volume
    ) / expect_volume
    assert g.np.allclose(mesh.center_mass, expect_com)

    # vertex coordinates and face attributes should be untouched
    assert g.np.allclose(mesh.vertices, vertices)
    assert (mesh.visual.face_colors == colors).all()


def test_nested_three_layers():
    # a solid sphere inside a spherical cavity inside a solid
    # box with all shell orientations scrambled
    outer = g.trimesh.creation.box(extents=[4.0, 4.0, 4.0])
    middle = g.trimesh.creation.icosphere(radius=1.2)
    middle.apply_translation([0.4, -0.3, 0.2])
    inner = g.trimesh.creation.icosphere(radius=0.5)
    inner.apply_translation([0.7, -0.1, 0.4])

    mesh = outer + middle + inner
    # invert every shell so all orientations are wrong
    mesh.invert()
    assert mesh.volume < 0.0

    mesh.fix_normals(nested=True)

    # layers alternate solid, cavity, solid
    expect_volume = outer.volume - middle.volume + inner.volume
    assert g.np.isclose(mesh.volume, expect_volume)
    expect_com = (
        outer.center_mass * outer.volume
        - middle.center_mass * middle.volume
        + inner.center_mass * inner.volume
    ) / expect_volume
    assert g.np.allclose(mesh.center_mass, expect_com)
    assert mesh.is_winding_consistent


def test_nested_independent():
    # disjoint solids should all be oriented outwards by nested
    # mode exactly like the existing multibody mode
    a = g.trimesh.creation.box()
    b = g.trimesh.creation.box()
    b.apply_translation([3.0, 0.0, 0.0])
    b.invert()
    mesh = a + b
    g.trimesh.repair.fix_inversion(mesh, nested=True)
    assert g.np.isclose(mesh.volume, 2.0)


def test_nested_diagnostics():
    # non-closed components should be diagnosed
    broken = g.trimesh.creation.box() + g.trimesh.Trimesh(
        vertices=[[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]],
        faces=[[0, 1, 2]],
    )
    with g.pytest.raises(ValueError):
        g.trimesh.repair.fix_inversion(broken, nested=True)

    # intersecting shells should be diagnosed
    a = g.trimesh.creation.box()
    b = g.trimesh.creation.box()
    b.apply_translation([0.7, 0.7, 0.3])
    with g.pytest.raises(ValueError):
        g.trimesh.repair.fix_inversion(a + b, nested=True)

    # shells sharing boundary geometry should be diagnosed
    inner = g.trimesh.creation.box(extents=[1.0, 0.5, 0.5])
    outer = g.trimesh.creation.box()
    with g.pytest.raises(ValueError):
        g.trimesh.repair.fix_inversion(inner + outer, nested=True)


def test_fan():
    # start by creating an icosphere and removing
    # all faces that include a single vertex to make
    # a nice hole in the mesh
    m = g.trimesh.creation.icosphere()
    clip = m.vertex_faces[0]
    clip = clip[clip >= 0]
    assert len(clip) > 4
    mask = g.np.ones(len(m.faces), dtype=bool)
    mask[clip] = False

    # should have been watertight
    assert m.is_watertight
    assert m.is_winding_consistent
    m.update_faces(mask)
    # now should not be watertight
    assert not m.is_watertight
    assert m.is_winding_consistent

    # create a triangle fan to cover the hole
    stitch = g.trimesh.repair.stitch(m)
    # should be an (n, 3) int
    assert len(stitch.shape) == 2
    assert stitch.shape[1] == 3
    assert stitch.dtype.kind == "i"

    # now check our stitch to see if it handled the hole
    repair = g.trimesh.Trimesh(
        vertices=m.vertices.copy(), faces=g.np.vstack((m.faces, stitch))
    )
    assert repair.is_watertight
    assert repair.is_winding_consistent


def test_fix_normals_mixed_multibox():
    from trimesh.creation import box
    # a test for multibody meshes with some watertight, some not

    # create a multibody mesh with two cubes, one of which is inverted
    a = box()
    b = box().apply_translation([4, 0, 0]).invert()
    c = a + b

    # inverted should sum to zero volume
    assert g.np.isclose(c.volume, 0.0)
    # area isn't signed so should be correct
    assert g.np.isclose(c.area, 12.0)

    # fix_normals should repair the disconnected meshes
    r = c.copy().fix_normals()
    assert g.np.allclose(r.volume, 2.0)

    # create a non-watertight cube
    frag = box().apply_translation([8, 0, 0])
    frag.faces = frag.faces[:-1]
    assert not frag.is_watertight

    # append on the non-watertight cube
    c += frag

    # exactly half of one 1x1 square face is missing
    assert g.np.isclose(c.area, 17.5), c.area

    # run the fix_normals call with:
    # - 1 good cube
    # - 1 inverted cube
    # - 1 cube missing a face
    # this should fix the inverted and ignore the one with holes in it
    c.fix_normals()

    # split without repairing the hole, but culling the non-watertight result
    split = c.split(only_watertight=True, repair=False)
    assert len(split) == 2, len(split)
    assert all(s.is_volume for s in split)
    assert all(g.np.isclose(s.volume, 1.0) for s in split)

    # split WITH repairing the hole and culling anything non-watertight
    split = c.split(only_watertight=True, repair=True)
    assert len(split) == 3, len(split)

    # everything should be a volume
    assert all(s.is_volume for s in split)
    assert all(g.np.isclose(s.volume, 1.0) for s in split)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    test_fill_holes()
