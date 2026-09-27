try:
    from . import generic as g
except BaseException:
    import generic as g


class SectionTest(g.unittest.TestCase):
    def test_section(self):
        mesh = g.get_mesh("featuretype.STL")
        # this hits many edge cases
        step = 0.125
        # step through mesh
        z_levels = g.np.arange(
            start=mesh.bounds[0][2], stop=mesh.bounds[1][2] + 2 * step, step=step
        )
        # randomly order Z so first level is probably not zero
        z_levels = g.np.random.default_rng(seed=0).permutation(z_levels)

        # rotate around so we're not just testing XY parallel planes
        for angle in [0.0, g.np.radians(1.0), g.np.radians(11.11232)]:
            # arbitrary test rotation axis
            axis = g.trimesh.unitize([1.0, 2.0, 0.32343])
            # rotate plane around axis to test along non- parallel planes
            base = g.trimesh.transformations.rotation_matrix(angle=angle, direction=axis)
            # to return to parallel to XY plane
            base_inv = g.np.linalg.inv(base)

            # transform normal to be slightly rotated
            plane_normal = g.trimesh.transform_points(
                [[0, 0, 1.0]], base, translate=False
            )[0]

            # store Path2D and Path3D results
            sections = [None] * len(z_levels)
            sections_3D = [None] * len(z_levels)

            for index, z in enumerate(z_levels):
                # move origin into rotated frame
                plane_origin = [0, 0, z]
                plane_origin = g.trimesh.transform_points([plane_origin], base)[0]

                section = mesh.section(
                    plane_origin=plane_origin, plane_normal=plane_normal
                )

                if section is None:
                    # section will return None if the plane doesn't
                    # intersect the mesh
                    # assert z > (mesh.bounds[1][2] -
                    #            g.trimesh.constants.tol.merge)
                    continue

                # move back
                section.apply_transform(base_inv)

                # Z should be in plane frame
                assert g.np.allclose(section.vertices[:, 2], z)
                assert len(section.centroid) == 3

                planar, _to_3D = section.to_planar()
                assert planar.is_closed
                assert len(planar.polygons_full) > 0
                assert len(planar.centroid) == 2

                sections[index] = planar
                sections_3D[index] = section

            # try getting the sections as Path2D through
            # the multiplane method
            paths = mesh.section_multiplane(
                plane_origin=[0, 0, 0], plane_normal=plane_normal, heights=z_levels
            )

            # call the multiplane method directly
            lines, _faces, _T = g.trimesh.intersections.mesh_multiplane(
                mesh=mesh,
                plane_origin=[0, 0, 0],
                plane_normal=plane_normal,
                heights=z_levels,
            )

            # make sure various methods return the same results
            for index in range(len(z_levels)):
                if sections[index] is None:
                    assert len(lines[index]) == 0
                    # make sure mesh.multipath_section is the same
                    assert paths[index] is None
                    continue
                # reconstruct path from line segments
                rc = g.trimesh.load_path(lines[index])

                assert g.np.isclose(rc.area, sections[index].area)
                assert g.np.isclose(rc.area, paths[index].area)

                # send Path2D back to 3D using the transform returned by
                # section
                back_3D = paths[index].to_3D(paths[index].metadata["to_3D"])
                # move to parallel test plane
                back_3D.apply_transform(base_inv)

                # make sure all vertices have constant Z
                assert g.np.ptp(back_3D.vertices[:, 2]) < 1e-8
                assert g.np.ptp(sections_3D[index].vertices[:, 2]) < 1e-8

                # make sure reconstructed 3D section is at right height
                assert g.np.isclose(
                    back_3D.vertices[:, 2].mean(),
                    sections_3D[index].vertices[:, 2].mean(),
                )

                # make sure reconstruction is at z of frame
                assert g.np.isclose(back_3D.vertices[:, 2].mean(), z_levels[index])

    def test_section_entity_count(self):
        """
        A cross-section of a convex shape should produce exactly
        one entity per connected component. Previously, starting
        DFS from an interior node fragmented single paths into
        multiple entities.
        """
        # box section at z=0 should be a single closed rectangle
        box = g.trimesh.creation.box()
        section = box.section(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
        )
        assert section is not None
        assert len(section.entities) == 1

        # cylinder section at z=0 should be a single closed ellipse
        cyl = g.trimesh.creation.cylinder(radius=1.0, height=2.0)
        section = cyl.section(
            plane_origin=[0, 0, 0],
            plane_normal=[0, 0, 1],
        )
        assert section is not None
        assert len(section.entities) == 1

    def test_multi_index(self):
        # make sure returned face indexes on a section are correct
        mesh = g.trimesh.creation.box()
        normal = g.np.array([0, 0, 1])
        # point for plane origin
        origin = mesh.center_mass
        # pick some offsets from the plane
        heights = g.np.linspace(0, 0.1, 2)
        # compute the multiple crosss sections with `section_multiplane`
        multi = mesh.section_multiplane(origin, normal, heights)
        # get the face indexes this should have hit by checking the normal
        idx = set(
            g.np.nonzero(g.np.isclose(g.np.dot(mesh.face_normals, normal), 0, atol=1e-4))[
                0
            ]
        )
        # make sure all indexes are set correctly
        assert all(set(m.metadata["face_index"] == idx) for m in multi)


class CoplanarSectionTest(g.unittest.TestCase):
    @staticmethod
    def _plate(flip=False, subdivide=False):
        """
        A flat plate at z=0: an outer square of size 4 with a
        centered square hole of size 1, triangulated in
        different ways.
        """
        outer = [[-2, -2], [2, -2], [2, 2], [-2, 2]]
        inner = [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
        vertices = [p + [0.0] for p in outer + inner]
        faces = []
        for i in range(4):
            o0, o1 = i, (i + 1) % 4
            i0, i1 = 4 + i, 4 + (i + 1) % 4
            if flip:
                # same quad strip with the other diagonal
                faces.append([o0, o1, i0])
                faces.append([o1, i1, i0])
            else:
                faces.append([o0, o1, i1])
                faces.append([o0, i1, i0])
        if subdivide:
            # replace the first quad with four triangles
            # meeting at a new interior vertex
            faces = faces[2:]
            c = len(vertices)
            vertices.append([-0.25, -1.25, 0.0])
            faces.extend([[0, 1, c], [1, 5, c], [5, 4, c], [4, 0, c]])
        return g.trimesh.Trimesh(
            vertices=g.np.array(vertices, dtype=g.np.float64),
            faces=g.np.array(faces),
            process=False,
        )

    @staticmethod
    def _signed_area(lines):
        # twice the signed area of a loop in the XY plane
        return (
            lines[:, 0, 0] * lines[:, 1, 1] - lines[:, 1, 0] * lines[:, 0, 1]
        ).sum()

    def test_box_top(self):
        # section plane exactly on the top face of a box
        box = g.trimesh.creation.box()
        normal = [0, 0, 1]
        origin = [0, 0, 0.5]

        # legacy behavior ignores coplanar faces entirely
        assert box.section(plane_origin=origin, plane_normal=normal) is None
        lines = g.trimesh.intersections.mesh_plane(
            mesh=box, plane_normal=normal, plane_origin=origin
        )
        assert len(lines) == 0

        # coplanar boundary handling returns the top square
        lines, face_index = g.trimesh.intersections.mesh_plane(
            mesh=box,
            plane_normal=normal,
            plane_origin=origin,
            return_faces=True,
            coplanar="boundary",
        )
        assert lines.shape == (4, 2, 3)
        assert face_index.shape == (4,)
        assert g.np.allclose(lines[:, :, 2], 0.5)

        # face source must be one of the two top faces
        top = set(g.np.nonzero(box.face_normals[:, 2] > 0.5)[0])
        assert set(face_index).issubset(top)

        # each segment must be a directed edge of its source face
        vid = {tuple(v): i for i, v in enumerate(box.vertices)}
        for seg, fi in zip(lines, face_index):
            edge = (vid[tuple(seg[0])], vid[tuple(seg[1])])
            f = box.faces[fi]
            assert edge in {(f[0], f[1]), (f[1], f[2]), (f[2], f[0])}

        # contour is oriented CCW seen from above like the
        # winding of the coplanar top faces
        assert self._signed_area(lines) > 0

        # as a section: a single closed loop with unit area
        section = box.section(
            plane_origin=origin, plane_normal=normal, coplanar="boundary"
        )
        assert section is not None
        assert section.is_closed
        assert len(section.entities) == 1
        planar, _to_3D = section.to_2D()
        assert len(planar.polygons_full) == 1
        assert g.np.isclose(planar.polygons_full[0].area, 1.0)

    def test_plate_with_hole(self):
        # the same plate with a hole triangulated three ways:
        # contour geometry must be identical and triangulation
        # diagonals must never show up in the result
        meshes = [
            self._plate(),
            self._plate(flip=True),
            self._plate(subdivide=True),
        ]
        contours = []
        for mesh in meshes:
            lines, face_index = g.trimesh.intersections.mesh_plane(
                mesh=mesh,
                plane_normal=[0, 0, 1],
                plane_origin=[0, 0, 0],
                return_faces=True,
                coplanar="boundary",
            )
            # 4 outer + 4 inner boundary segments
            assert lines.shape == (8, 2, 3)
            assert face_index.shape == (8,)

            # every segment endpoint is one of the 8 corners:
            # no triangulation diagonal or interior vertex
            corners = {tuple(p) for p in mesh.vertices[:8, :2]}
            for seg in lines:
                assert tuple(seg[0][:2]) in corners
                assert tuple(seg[1][:2]) in corners

            # normalize segments to compare contour geometry
            contours.append(
                sorted(tuple(sorted(map(tuple, seg[:, :2]))) for seg in lines)
            )

            # outer boundary follows face winding CCW and the
            # hole contour CW
            outer = lines[g.np.abs(lines[:, :, :2]).max(axis=(1, 2)) > 1.0]
            inner = lines[g.np.abs(lines[:, :, :2]).max(axis=(1, 2)) < 1.0]
            assert len(outer) == 4
            assert len(inner) == 4
            assert self._signed_area(outer) > 0
            assert self._signed_area(inner) < 0

            # as a section: outer boundary and hole contour
            section = mesh.section(
                plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], coplanar="boundary"
            )
            assert section is not None
            assert section.is_closed
            planar, _to_3D = section.to_2D()
            assert len(planar.polygons_full) == 1
            polygon = planar.polygons_full[0]
            assert g.np.isclose(polygon.area, 15.0)
            assert len(polygon.interiors) == 1

        # identical geometry regardless of triangulation
        assert contours[0] == contours[1]
        assert contours[0] == contours[2]

    def test_plane_through_edge(self):
        # two triangles sharing an edge lying on the plane
        # with both remaining vertices above the plane
        mesh = g.trimesh.Trimesh(
            vertices=g.np.array(
                [[0, 0, 0], [1, 0, 0], [0.5, 1, 1], [0.5, -1, 1]],
                dtype=g.np.float64,
            ),
            faces=g.np.array([[0, 1, 2], [1, 0, 3]]),
            process=False,
        )
        kwargs = {"plane_normal": [0, 0, 1], "plane_origin": [0, 0, 0]}

        # legacy behavior returns the shared edge twice
        lines = g.trimesh.intersections.mesh_plane(mesh=mesh, **kwargs)
        assert len(lines) == 2

        # coplanar boundary handling merges the duplicate
        lines, face_index = g.trimesh.intersections.mesh_plane(
            mesh=mesh, return_faces=True, coplanar="boundary", **kwargs
        )
        assert lines.shape == (1, 2, 3)
        assert face_index.shape == (1,)
        assert face_index[0] in {0, 1}
        assert g.np.allclose(lines[0][:, 2], 0.0)
        assert g.np.allclose(sorted(lines[0][:, 0]), [0.0, 1.0])

    def test_two_nearby_layers(self):
        # two thin layers separated by a small gap: the two
        # independent contours must not be welded together
        gap = 1e-3
        layers = []
        for x in [0.0, 1.0 + gap]:
            box = g.trimesh.creation.box(extents=[1.0, 1.0, 0.1])
            box.apply_translation([x, 0.0, 0.05])
            layers.append(box)
        mesh = g.trimesh.util.concatenate(layers)

        # plane on the coplanar top faces of both layers
        section = mesh.section(
            plane_origin=[0, 0, 0.1], plane_normal=[0, 0, 1], coplanar="boundary"
        )
        assert section is not None
        assert section.is_closed
        assert len(section.entities) == 2

        planar, _to_3D = section.to_2D()
        polygons = planar.polygons_closed
        assert len(polygons) == 2
        assert all(g.np.isclose(p.area, 1.0) for p in polygons)
        # the small gap between the layers is preserved
        assert g.np.isclose(polygons[0].distance(polygons[1]), gap)

    def test_open_and_branching(self):
        # three triangles sharing one edge on the plane: the
        # boundary of this coplanar region branches and the
        # result must be reported as open paths rather than
        # being stitched into a closed loop
        mesh = g.trimesh.Trimesh(
            vertices=g.np.array(
                [[0, 0, 0], [2, 0, 0], [1, 1, 0], [1, -1, 0], [3, 1, 0]],
                dtype=g.np.float64,
            ),
            faces=g.np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]),
            process=False,
        )
        kwargs = {"plane_normal": [0, 0, 1], "plane_origin": [0, 0, 0]}

        # the edge shared by three faces is discarded leaving
        # the six outer edges
        lines, face_index = g.trimesh.intersections.mesh_plane(
            mesh=mesh, return_faces=True, coplanar="boundary", **kwargs
        )
        assert lines.shape == (6, 2, 3)
        assert face_index.shape == (6,)

        section = mesh.section(coplanar="boundary", **kwargs)
        assert section is not None
        # the graph branches so the path is not closed and
        # the dangling chain is reported as an open entity
        # rather than being stitched into a closed loop
        assert not section.is_closed
        assert any(not e.closed for e in section.entities)
        # no segment was dropped or duplicated
        assert sum(len(e.points) - 1 for e in section.entities) == 6

    def test_scale(self):
        # the on-plane tolerance follows the model scale so
        # coplanar detection works for tiny and huge models
        for scale in [1e-9, 1e6]:
            box = g.trimesh.creation.box()
            box.apply_scale(scale)
            section = box.section(
                plane_origin=[0, 0, 0.5 * scale],
                plane_normal=[0, 0, 1],
                coplanar="boundary",
            )
            assert section is not None
            assert section.is_closed
            assert len(section.entities) == 1
            planar, _to_3D = section.to_2D()
            assert len(planar.polygons_full) == 1
            assert g.np.isclose(
                planar.polygons_full[0].area, scale**2, atol=0, rtol=1e-4
            )

    def test_invalid_option(self):
        box = g.trimesh.creation.box()
        with self.assertRaises(ValueError):
            g.trimesh.intersections.mesh_plane(
                mesh=box,
                plane_normal=[0, 0, 1],
                plane_origin=[0, 0, 0],
                coplanar="banana",
            )


class PlaneLine(g.unittest.TestCase):
    def test_planes(self):
        count = 10
        z = g.np.linspace(-1, 1, count)

        plane_origins = g.np.column_stack((g.random((count, 2)), z))
        plane_normals = g.np.tile([0, 0, -1], (count, 1))

        line_origins = g.np.tile([0, 0, 0], (count, 1))
        line_directions = g.random((count, 3))

        i, valid = g.trimesh.intersections.planes_lines(
            plane_origins=plane_origins,
            plane_normals=plane_normals,
            line_origins=line_origins,
            line_directions=line_directions,
        )
        assert valid.all()
        assert (g.np.abs(i[:, 2] - z) < g.tol.merge).all()


class SliceTest(g.unittest.TestCase):
    def test_slice(self):
        mesh = g.trimesh.creation.box()

        # Cut corner off of box and make sure the bounds and number of faces is correct
        # Tests new triangles, but not new quads or triangles contained
        # entirely
        plane_origin = mesh.bounds[1] - 0.05
        plane_normal = mesh.bounds[1]

        sliced = mesh.slice_plane(plane_origin=plane_origin, plane_normal=plane_normal)

        assert g.np.isclose(sliced.bounds[0], mesh.bounds[1] - 0.15).all()
        assert g.np.isclose(sliced.bounds[1], mesh.bounds[1]).all()
        assert len(sliced.faces) == 5

        # Cut top off of box and make sure bounds and number of faces is correct
        # Tests new quads and entirely contained triangles
        plane_origin = mesh.bounds[1] - 0.05
        plane_normal = g.np.array([0, 0, 1])

        sliced = mesh.slice_plane(plane_origin=plane_origin, plane_normal=plane_normal)

        assert g.np.isclose(
            sliced.bounds[0], mesh.bounds[0] + g.np.array([0, 0, 0.95])
        ).all()
        assert g.np.isclose(sliced.bounds[1], mesh.bounds[1]).all()
        assert len(sliced.faces) == 14

        # non- watertight more complex mesh
        bunny = g.get_mesh("bunny.ply")

        origin = bunny.bounds.mean(axis=0)
        normal = g.trimesh.unitize([1, 1, 2])

        sliced = bunny.slice_plane(plane_origin=origin, plane_normal=normal)
        assert len(sliced.faces) > 0

        # check the projections manually
        dot = g.np.dot(normal, (sliced.vertices - origin).T)
        # should be lots of stuff at the plane and nothing behind
        assert g.np.isclose(dot.min(), 0.0)

        # Cut part of top off of box with multiple planes at once
        # and make sure bounds and number of faces is correct
        plane_origins = [mesh.bounds[1] - 0.05, mesh.bounds[1] - 0.05]
        plane_normals = [g.np.array([0, 0, 1]), g.np.array([0, 1, 0])]

        sliced = mesh.slice_plane(plane_origin=plane_origins, plane_normal=plane_normals)

        assert g.np.isclose(
            sliced.bounds[0], mesh.bounds[0] + g.np.array([0, 0.95, 0.95])
        ).all()
        assert g.np.isclose(sliced.bounds[1], mesh.bounds[1]).all()
        assert len(sliced.faces) == 11

        # Try with more complicated mesh and make sure we get correct projections
        # and some faces
        origins = [
            bunny.bounds.mean(axis=0),
            bunny.bounds.mean(axis=0) + 0.01 * g.trimesh.unitize([1, 1, 2]),
        ]
        normals = [g.trimesh.unitize([1, 1, 2]), -g.trimesh.unitize([1, 1, 2])]

        sliced = bunny.slice_plane(plane_origin=origins, plane_normal=normals)
        assert len(sliced.faces) > 0

        for o, n in zip(origins, normals):
            # check the projections manually
            dot = g.np.dot(n, (sliced.vertices - o).T)
            # should be lots of stuff at the plane and nothing behind
            assert g.np.isclose(dot.min(), 0.0)

        # Test cap on more complicated watertight mesh to make sure the
        # resulting mesh is still watertight and slice is correct
        featuretype = g.get_mesh("featuretype.STL")

        origins = [
            featuretype.center_mass,
            featuretype.center_mass + 0.01 * g.trimesh.unitize([1, 0, 2]),
        ]
        normals = [g.trimesh.unitize([1, 1, 1]), g.trimesh.unitize([1, 2, 3])]

    def test_slice_onplane(self):
        m = g.get_mesh("featuretype.STL")
        # on a plane with a lot of coplanar triangles
        n = g.np.array([0, 0, 1.0])
        o = g.np.array([0, 0, 1.0])

        # slice the mesh in two pieces
        a = m.slice_plane(plane_origin=o, plane_normal=-n, cap=True)
        b = m.slice_plane(plane_origin=o, plane_normal=n, cap=True)

        # both slices should be watertiight
        assert a.is_watertight
        assert b.is_watertight
        # volume should match original
        assert g.np.isclose(m.volume, a.volume + b.volume)

    def test_slice_submesh(self):
        bunny = g.get_mesh("bunny.ply")

        # Find the faces on the body.
        neck_plane_origin = g.np.array([-0.0441905, 0.124347, 0.0235287])
        neck_plane_normal = g.np.array([0.35534835, -0.93424839, -0.03012456])

        dots = g.np.einsum(
            "i,ij->j", neck_plane_normal, (bunny.vertices - neck_plane_origin).T
        )
        signs = g.np.zeros(len(bunny.vertices), dtype=g.np.int8)
        signs[dots < -g.tol.merge] = 1
        signs[dots > g.tol.merge] = -1
        signs = signs[bunny.faces]

        signs_sum = signs.sum(axis=1, dtype=g.np.int8)
        signs_asum = g.np.abs(signs).sum(axis=1, dtype=g.np.int8)

        body_face_mask = signs_sum == -signs_asum
        body_face_index = body_face_mask.nonzero()[0]

        slicing_plane_origin = bunny.bounds.mean(axis=0)
        slicing_plane_normal = g.trimesh.unitize([1, 1, 2])

        sliced = bunny.slice_plane(
            plane_origin=slicing_plane_origin,
            plane_normal=slicing_plane_normal,
            face_index=body_face_index,
        )

        # Ideally we would assert that the triangles in `body_face_index` were
        # sliced if they are on in front of side of the slicing plane, and the
        # triangles not in `body_face_index` were preserved. This is easier to
        # verify visually.
        # sliced.show()

        assert len(sliced.faces) > 0

    def test_textured_mesh(self):
        # Create a plane mesh with UV == xy
        plane = g.trimesh.Trimesh(
            vertices=g.np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            ),
            faces=g.np.array([[0, 1, 2], [2, 3, 0]]),
        )
        plane.visual = g.trimesh.visual.TextureVisuals(
            uv=plane.vertices[:, :2], material=g.trimesh.visual.material.empty_material()
        )

        # We cut the plane and check that the new UV match the new vertices
        origin = g.np.array([0.5, 0.5, 0.0])
        normal = g.trimesh.unitize([2, 1, 2])
        sliced = plane.slice_plane(plane_origin=origin, plane_normal=normal)
        assert g.np.isclose(sliced.vertices[:, :2], sliced.visual.uv).all()

        # Test scenario when plane does not cut
        origin = g.np.array([-1.0, -1.0, 0.0])
        normal = g.trimesh.unitize([1, 1, 2])
        sliced = plane.slice_plane(plane_origin=origin, plane_normal=normal)
        assert g.np.isclose(sliced.vertices[:, :2], sliced.visual.uv).all()

        # Test cut with no new vertices
        origin = g.np.array([0.5, 0.5, 0.0])
        normal = g.trimesh.unitize([2, -2, 1])
        sliced = plane.slice_plane(plane_origin=origin, plane_normal=normal)
        assert g.np.isclose(sliced.vertices[:, :2], sliced.visual.uv).all()

    def test_cap_coplanar(self):
        # check to see if we handle capping with
        # existing coplanar faces correctly

        if not g.has_earcut:
            return

        s = g.get_mesh("cap.zip")
        mesh = next(iter(s.geometry.values()))

        plane_origin = [0, 0, 5000]
        plane_normal = [0, 0, -1]

        assert mesh.is_watertight
        newmesh = mesh.slice_plane(
            plane_origin=plane_origin, plane_normal=plane_normal, cap=True
        )
        assert newmesh.is_watertight

    def test_slice_exit(self):
        m = g.trimesh.creation.box()
        assert g.np.isclose(m.area, 6)

        # start with a slice plane at every vertex
        origins = m.vertices.copy()
        # box centered at origin so get a unit normal
        normals = g.trimesh.unitize(origins)

        # slice the tip of the box off
        origins -= normals * 0.1
        # make the first plane non-intersecting
        # to test the early exit case
        origins[0] += normals[0] * 10

        # reverse the normals to indicate positive volume
        normals *= -1

        # run the slices
        s = m.slice_plane(origins, normals)

        assert g.np.isclose(s.area, 5.685)

    def test_cap_nohit(self):
        # check to see if we handle capping with
        # non-intersecting planes well

        if not g.has_earcut:
            return

        # rotate-only as both boxes have to overlap to produce a cap
        transforms = list(g.random_transforms(200, translate=0.0))

        for first, second in zip(transforms[::2], transforms[1::2]):
            box1 = g.trimesh.primitives.Box(extents=[10, 20, 30], transform=first)
            box2 = g.trimesh.primitives.Box(extents=[10, 20, 30], transform=second)

            result = g.trimesh.intersections.slice_mesh_plane(
                mesh=box2,
                plane_normal=-box1.face_normals,
                plane_origin=box1.vertices[box1.faces[:, 1]],
                cap=True,
            )
            assert len(result.faces) > 0

    def test_cap(self):
        if not g.has_earcut:
            return
        mesh = g.trimesh.creation.box()

        # Cut corner off of box and make sure the bounds and number of faces is correct
        # Tests new triangles, but not new quads or triangles contained
        # entirely
        plane_origin = mesh.bounds[1] - 0.05
        plane_normal = mesh.bounds[1]

        # Same test with capping (should only add three more triangles)
        sliced_capped = mesh.slice_plane(
            plane_origin=plane_origin, plane_normal=plane_normal, cap=True
        )

        assert len(sliced_capped.faces) == 8
        assert g.np.isclose(sliced_capped.bounds[0], mesh.bounds[1] - 0.15).all()
        assert g.np.isclose(sliced_capped.bounds[1], mesh.bounds[1]).all()
        assert sliced_capped.is_watertight

        # Cut top off of box and make sure bounds and number of faces is correct
        # Tests new quads and entirely contained triangles
        plane_origin = mesh.bounds[1] - 0.05
        plane_normal = g.np.array([0, 0, 1])

        # Same test with capping (should only add six triangles)
        sliced_capped = mesh.slice_plane(
            plane_origin=plane_origin, plane_normal=plane_normal, cap=True
        )

        assert len(sliced_capped.faces) == 20
        assert g.np.isclose(
            sliced_capped.bounds[0], mesh.bounds[0] + g.np.array([0, 0, 0.95])
        ).all()
        assert g.np.isclose(sliced_capped.bounds[1], mesh.bounds[1]).all()
        assert sliced_capped.is_watertight

        # Cut part of top off of box with multiple planes at once
        # and make sure bounds and number of faces is correct
        plane_origins = [mesh.bounds[1] - 0.05, mesh.bounds[1] - 0.05]
        plane_normals = [g.np.array([0, 0, 1]), g.np.array([0, 1, 0])]

        # Test cap for multiple slices to check watertightness
        # (should add nine triangles)
        sliced_capped = mesh.slice_plane(
            plane_origin=plane_origins, plane_normal=plane_normals, cap=True
        )
        assert g.np.isclose(
            sliced_capped.bounds[0], mesh.bounds[0] + g.np.array([0, 0.95, 0.95])
        ).all()
        assert g.np.isclose(sliced_capped.bounds[1], mesh.bounds[1]).all()
        assert sliced_capped.is_watertight

        # Test cap on more complicated watertight mesh to make sure the
        # resulting mesh is still watertight and slice is correct
        featuretype = g.get_mesh("featuretype.STL")

        origins = [
            featuretype.center_mass,
            featuretype.center_mass + 0.01 * g.trimesh.unitize([1, 0, 2]),
        ]
        normals = [g.trimesh.unitize([1, 0, 1]), g.trimesh.unitize([1, 0, 0])]

        sliced_capped = featuretype.slice_plane(
            plane_origin=origins, plane_normal=normals, cap=True
        )
        assert len(sliced_capped.faces) > 0
        assert sliced_capped.is_winding_consistent

        for o, n in zip(origins, normals):
            # check the projections manually
            dot = g.np.dot(n, (sliced_capped.vertices - o).T)
            # should be lots of stuff at the plane and nothing behind
            assert g.np.isclose(dot.min(), 0.0)


if __name__ == "__main__":
    g.trimesh.util.attach_to_log()
    g.unittest.main()
