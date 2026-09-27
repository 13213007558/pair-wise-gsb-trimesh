import numpy as np

from .. import grouping, remesh, util
from .. import transformations as tr
from ..constants import log_time
from ..typed import ArrayLike, Integer, Number, VoxelizationMethodsType
from . import base
from . import encoding as enc


@log_time
def voxelize_subdivide(
    mesh, pitch: Number, max_iter: Integer | None = 10, edge_factor: Number = 2.0
) -> base.VoxelGrid:
    """
    Voxelize a surface by subdividing a mesh until every edge is
    shorter than: (pitch / edge_factor)

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Source mesh
    pitch
      Side length of a single voxel cube
    max_iter
      Cap maximum subdivisions or None for no limit.
    edge_factor
      Proportion of pitch maximum edge length.

    Returns
    -----------
    VoxelGrid instance representing the voxelized mesh.
    """
    max_edge = pitch / edge_factor

    if max_iter is None:
        longest_edge = np.linalg.norm(
            mesh.vertices[mesh.edges[:, 0]] - mesh.vertices[mesh.edges[:, 1]], axis=1
        ).max()
        max_iter = max(int(np.ceil(np.log2(longest_edge / max_edge))), 0)

    # get the same mesh sudivided so every edge is shorter
    # than a factor of our pitch
    v, _f, _idx = remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=max_edge, max_iter=max_iter, return_index=True
    )

    # convert the vertices to their voxel grid position
    # Provided edge_factor > 1 and max_iter is large enough, this is
    # sufficient to preserve 6-connectivity at the level of voxels.
    hit = np.round(v / pitch).astype(int)

    # remove duplicates
    unique, _inverse = grouping.unique_rows(hit)

    # get the voxel centers in model space
    occupied_index = hit[unique]

    origin_index = occupied_index.min(axis=0)
    origin_position = origin_index * pitch

    return base.VoxelGrid(
        enc.SparseBinaryEncoding(occupied_index - origin_index),
        transform=tr.scale_and_translate(scale=pitch, translate=origin_position),
    )


def _occupied_cells(triangle, lower, upper):
    """
    Return the integer indices of every unit grid cell in the inclusive
    index range lower..upper whose *closed* box is touched by a
    triangle, evaluated with the separating axis theorem.

    Cells are unit cubes centered on integer coordinates. Tangency
    counts as overlap, so a triangle lying exactly on a shared cell
    face, edge or corner marks all cells meeting there.

    Parameters
    ------------
    triangle : (3, 3) float
      Triangle vertices in grid coordinates.
    lower : (3,) int
      Lowest candidate cell index per axis.
    upper : (3,) int
      Highest candidate cell index per axis.

    Returns
    ------------
    cells : (n, 3) int
      Indices of the cells overlapped by the triangle.
    """
    grids = np.meshgrid(
        *(np.arange(lower[k], upper[k] + 1) for k in range(3)), indexing="ij"
    )
    centers = np.column_stack([g.ravel() for g in grids])

    # triangle vertices relative to each candidate cell center
    relative = triangle[np.newaxis, :, :] - centers[:, np.newaxis, :].astype(np.float64)

    edges = np.array(
        [
            triangle[1] - triangle[0],
            triangle[2] - triangle[1],
            triangle[0] - triangle[2],
        ]
    )
    # separating axes: the 9 edge/box-axis cross products and the
    # triangle normal; degenerate (zero) axes pass trivially
    axes = [np.cross(edge, axis) for edge in edges for axis in np.eye(3)]
    axes.append(np.cross(edges[0], edges[1]))

    keep = np.ones(len(centers), dtype=bool)
    for axis in axes:
        # projection radius of the unit cell onto the axis
        radius = 0.5 * np.abs(axis).sum()
        projection = np.einsum("nij,j->ni", relative, axis)
        # closed interval overlap: touching counts as occupied
        keep &= (projection.min(axis=1) <= radius) & (projection.max(axis=1) >= -radius)

    # the three box principal axes
    keep &= np.all(relative.min(axis=1) <= 0.5, axis=1)
    keep &= np.all(relative.max(axis=1) >= -0.5, axis=1)

    return centers[keep]


@log_time
def voxelize_surface(
    mesh, pitch: ArrayLike, origin: ArrayLike | None = None
) -> base.VoxelGrid:
    """
    Conservatively voxelize the surface of a mesh by testing each
    triangle against the voxel boxes it overlaps, rather than by
    point sampling.

    Every voxel whose closed cell is touched by any triangle is marked
    occupied, so thin or sliver geometry cannot fall between samples
    and the result does not depend on how the surface was triangulated.
    Only surface voxels are produced: no interior filling is done.

    Boundary tangency counts as occupied: cells are treated as closed
    boxes, hence a triangle lying exactly on a shared voxel face, edge
    or corner marks all cells sharing that boundary element.

    Candidate cells are enumerated locally inside each triangle's own
    bounding box, so widely separated small faces never require a
    dense array spanning the whole scene.

    Parameters
    ------------
    mesh : trimesh.Trimesh
      Source geometry.
    pitch : float or (3,) float
      Side length of a voxel, either uniform or per-axis.
    origin : None or (3,) float
      Position of the center of cell (0, 0, 0) in model space,
      anchoring the grid. If None the global origin (0, 0, 0) is
      used, matching voxelize_subdivide.

    Returns
    ------------
    grid : VoxelGrid
      Surface voxels; the transform carries the per-axis pitch and
      the translation implied by origin.
    """
    pitch = np.asarray(pitch, dtype=np.float64)
    if pitch.ndim == 0:
        pitch = np.full(3, float(pitch))
    if pitch.shape != (3,) or np.any(pitch <= 0.0):
        raise ValueError("pitch must be a positive float or (3,) float")

    if origin is None:
        origin = np.zeros(3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)

    # work in grid coordinates where cell i spans [i - 0.5, i + 0.5]
    triangles = (np.asarray(mesh.triangles, dtype=np.float64) - origin) / pitch

    occupied = []
    for triangle in triangles:
        # closed boxes: a cell is a candidate whenever its box touches
        # the triangle's own bounding box, tangency included
        lower = np.ceil(triangle.min(axis=0) - 0.5).astype(np.int64)
        upper = np.floor(triangle.max(axis=0) + 0.5).astype(np.int64)
        if np.any(upper < lower):
            continue
        occupied.append(_occupied_cells(triangle, lower, upper))

    if len(occupied) == 0:
        indices = np.zeros((0, 3), dtype=np.int64)
    else:
        indices = np.vstack(occupied)
        unique, _ = grouping.unique_rows(indices)
        indices = indices[unique]

    if len(indices) == 0:
        origin_index = np.zeros(3, dtype=np.int64)
        encoding = enc.SparseBinaryEncoding(indices, shape=(0, 0, 0))
    else:
        origin_index = indices.min(axis=0)
        encoding = enc.SparseBinaryEncoding(indices - origin_index)

    return base.VoxelGrid(
        encoding,
        transform=tr.scale_and_translate(
            scale=pitch, translate=origin + origin_index * pitch
        ),
    )


def local_voxelize(
    mesh,
    point: ArrayLike,
    pitch: Number,
    radius: Number,
    fill: bool = True,
    **kwargs,
) -> base.VoxelGrid | None:
    """
    Voxelize a mesh in the region of a cube around a point. When fill=True,
    uses proximity.contains to fill the resulting voxels so may be meaningless
    for non-watertight meshes. Useful to reduce memory cost for small values of
    pitch as opposed to global voxelization.

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Source geometry
    point : (3, ) float
      Point in space to voxelize around
    pitch
      Side length of a single voxel cube
    radius
      Number of voxel cubes to return in each direction.
    kwargs
      Parameters to pass to voxelize_subdivide

    Returns
    -----------
    voxels : VoxelGrid instance with resolution (m, m, m) where m=2*radius+1
        or None if the volume is empty
    """
    from scipy import ndimage

    # make sure point is correct type/shape
    point = np.asanyarray(point, dtype=np.float64).reshape(3)
    # this is a gotcha- radius sounds a lot like it should be in
    # float model space, not int voxel space so check
    if not isinstance(radius, int):
        raise ValueError("radius needs to be an integer number of cubes!")

    # Bounds of region
    bounds = np.concatenate(
        (point - (radius + 0.5) * pitch, point + (radius + 0.5) * pitch)
    )

    # faces that intersect axis aligned bounding box
    faces = list(mesh.triangles_tree.intersection(bounds))

    # didn't hit anything so exit
    if len(faces) == 0:
        return None

    local = mesh.submesh([[f] for f in faces], append=True)

    # Translate mesh so point is at 0,0,0
    local.apply_translation(-point)

    # sparse, origin = voxelize_subdivide(local, pitch, **kwargs)
    vox = voxelize_subdivide(local, pitch, **kwargs)
    origin = vox.transform[:3, 3]
    matrix = vox.encoding.dense

    # Find voxel index for point
    center = np.round(-origin / pitch).astype(np.int64)

    # pad matrix if necessary
    prepad = np.maximum(radius - center, 0)
    postpad = np.maximum(center + radius + 1 - matrix.shape, 0)

    matrix = np.pad(matrix, np.stack((prepad, postpad), axis=-1), mode="constant")
    center += prepad

    # Extract voxels within the bounding box
    voxels = matrix[
        center[0] - radius : center[0] + radius + 1,
        center[1] - radius : center[1] + radius + 1,
        center[2] - radius : center[2] + radius + 1,
    ]
    local_origin = point - radius * pitch  # origin of local voxels

    # Fill internal regions
    if fill:
        regions, n = ndimage.label(~voxels)
        distance = ndimage.distance_transform_cdt(~voxels)
        representatives = [
            np.unravel_index((distance * (regions == i)).argmax(), distance.shape)
            for i in range(1, n + 1)
        ]
        contains = mesh.contains(np.asarray(representatives) * pitch + local_origin)
        where = np.where(contains)[0] + 1
        internal = np.isin(regions.flatten(), where).reshape(regions.shape)
        voxels = np.logical_or(voxels, internal)

    return base.VoxelGrid(voxels, tr.translation_matrix(local_origin))


@log_time
def voxelize_ray(
    mesh, pitch: Number, per_cell: ArrayLike | None = None
) -> base.VoxelGrid:
    """
    Voxelize a mesh using ray queries.

    Parameters
    -------------
    mesh
      Mesh to be voxelized
    pitch
      Length of voxel cube
    per_cell : (2,) int
      How many ray queries to make per cell

    Returns
    -------------
    grid
      VoxelGrid instance representing the voxelized mesh.
    """
    if per_cell is None:
        # how many rays per cell
        per_cell = np.array([2, 2], dtype=np.int64)
    else:
        per_cell = np.array(per_cell, dtype=np.int64).reshape(2)

    # edge length of cube voxels
    pitch = float(pitch)

    # create the ray origins in a grid
    bounds = mesh.bounds[:, :2].copy()
    # offset start so we get the requested number per cell
    bounds[0] += pitch / (1.0 + per_cell)
    # offset end so arange doesn't short us
    bounds[1] += pitch
    # on X we are doing multiple rays per voxel step
    step = pitch / per_cell
    # 2D grid
    ray_ori = util.grid_arange(bounds, step=step)
    # a Z position below the mesh
    z = np.ones(len(ray_ori)) * (mesh.bounds[0][2] - pitch)
    ray_ori = np.column_stack((ray_ori, z))
    # all rays are along positive Z
    ray_dir = np.ones_like(ray_ori) * [0, 0, 1]

    # if you have pyembree this should be decently fast
    hits = mesh.ray.intersects_location(ray_ori, ray_dir)[0]

    # just convert hit locations to integer positions
    voxels = np.round(hits / pitch).astype(np.int64)

    # offset voxels by min, so matrix isn't huge
    origin_index = voxels.min(axis=0)
    voxels -= origin_index
    encoding = enc.SparseBinaryEncoding(voxels)
    origin_position = origin_index * pitch
    return base.VoxelGrid(
        encoding, tr.scale_and_translate(scale=pitch, translate=origin_position)
    )


@log_time
def voxelize_binvox(
    mesh,
    pitch: Number | None = None,
    dimension: Integer | None = None,
    bounds: ArrayLike | None = None,
    **binvoxer_kwargs,
) -> base.VoxelGrid:
    """
    Voxelize via binvox tool.

    Parameters
    --------------
    mesh : trimesh.Trimesh
      Mesh to voxelize
    pitch : float
      Side length of each voxel. Ignored if dimension is provided
    dimension: int
      Number of voxels along each dimension. If not provided, this is
        calculated based on pitch and bounds/mesh extents
    bounds: (2, 3) float
      min/max values of the returned `VoxelGrid` in each instance. Uses
      `mesh.bounds` if not provided.
    **binvoxer_kwargs:
      Passed to `trimesh.exchange.binvox.Binvoxer`.
      Should not contain `bounding_box` if bounds is not None.

    Returns
    --------------
    grid
      `VoxelGrid` instance

    Raises
    --------------
    `ValueError` if `bounds is not None and 'bounding_box' in binvoxer_kwargs`.
    """
    from trimesh.exchange import binvox

    if dimension is None:
        # pitch must be provided
        if bounds is None:
            extents = mesh.extents
        else:
            mins, maxs = bounds
            extents = maxs - mins
        dimension = int(np.ceil(np.max(extents) / pitch))
    if bounds is not None:
        if "bounding_box" in binvoxer_kwargs:
            raise ValueError("Cannot provide both bounds and bounding_box")
        binvoxer_kwargs["bounding_box"] = np.asanyarray(bounds).flatten()

    binvoxer = binvox.Binvoxer(dimension=dimension, **binvoxer_kwargs)
    return binvox.voxelize_mesh(mesh, binvoxer)


voxelizers = util.FunctionRegistry(
    ray=voxelize_ray,
    subdivide=voxelize_subdivide,
    binvox=voxelize_binvox,
    surface=voxelize_surface,
)


def voxelize(
    mesh,
    pitch: Number | None,
    method: VoxelizationMethodsType = "subdivide",
    **kwargs,
) -> base.VoxelGrid | None:
    """
    Voxelize the given mesh using the specified implementation.

    See `voxelizers` for available implementations or to add your own, e.g. via
    `voxelizers['custom_key'] = custom_fn`.

    `custom_fn` should have signature `(mesh, pitch, **kwargs) -> VoxelGrid`
    and should not modify encoding.

    Parameters
    --------------
    mesh
      Geometry to voxelize
    pitch
      Side length of each voxel.
    method
      Which voxelization method to use.
    kwargs
      Passed through to the specified implementation.

    Returns
    --------------
    grid
      A VoxelGrid instance.
    """
    return voxelizers(method, mesh=mesh, pitch=pitch, **kwargs)
