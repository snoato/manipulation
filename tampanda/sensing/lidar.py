"""Lidar sensor simulation via MuJoCo raycasting.

Uses ``mujoco.mj_multiRay`` to cast all rays in a single batched call per
scan, making it efficient enough for real-time use even with 360+ rays.

Supports 2D (single horizontal ring) and 3D (multi-layer) scan patterns.
The lidar is attached to a named **site** in the model so it automatically
follows the robot body it is mounted on.

Typical usage::

    from tampanda.sensing import Lidar

    # 2D laser scanner (360 rays, full circle, 10 m range)
    lidar = Lidar(env, site="lidar_site", num_rays=360, range_max=10.0,
                  body_exclude="base_link")
    distances = lidar.scan()          # np.ndarray (360,), metres
    pts = lidar.to_pointcloud(distances)  # np.ndarray (360, 3), world frame

    # 3D multi-layer (32 layers × 360 rays, ±15° vertical)
    lidar3d = Lidar(env, site="lidar_site", num_rays=360,
                    num_layers=32, fov_v=30.0, range_max=50.0,
                    body_exclude="base_link")
    distances = lidar3d.scan()        # np.ndarray (32, 360)
    pts = lidar3d.to_pointcloud(distances)  # np.ndarray (32*360, 3)

    # With geom IDs for object identification
    result = lidar.scan(return_geom_ids=True)
    distances, geom_ids = result["distances"], result["geom_ids"]
"""

from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

import mujoco
import numpy as np

if TYPE_CHECKING:
    from tampanda.core.base_env import BaseEnvironment


class Lidar:
    """Simulated lidar sensor attached to a model site.

    Rays sweep in the **XY plane** of the site's local frame (i.e., the site's
    Z axis is the rotation axis for the horizontal scan).  For a mobile robot,
    mount the site with Z pointing up.

    Args:
        env:          Environment with ``get_model()`` / ``get_data()``.
        site:         Name of the site the lidar is attached to.
        num_rays:     Number of rays per layer (horizontal resolution).
        fov_h:        Horizontal field of view in degrees.  360 = full circle.
        num_layers:   Number of vertical layers.  1 = 2D scan.
        fov_v:        Vertical field of view in degrees (ignored when
                      ``num_layers == 1``).  Layers are distributed
                      symmetrically around the horizontal plane.
        range_min:    Minimum valid range in metres.  Hits closer than this
                      are reported as ``range_min``.
        range_max:    Maximum range in metres.  No-hit rays are reported as
                      ``range_max``.
        body_exclude: Name of a body whose geoms are excluded from raycasts
                      (typically the robot chassis).  Pass ``None`` to include
                      all bodies.
    """

    def __init__(
        self,
        env: "BaseEnvironment",
        *,
        site: str,
        num_rays: int = 360,
        fov_h: float = 360.0,
        num_layers: int = 1,
        fov_v: float = 0.0,
        range_min: float = 0.05,
        range_max: float = 10.0,
        body_exclude: Optional[str] = None,
    ):
        self._env = env
        self._num_rays = num_rays
        self._num_layers = num_layers
        self._range_min = range_min
        self._range_max = range_max

        model = env.get_model()

        # Resolve site
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self._site_id == -1:
            raise ValueError(f"Site '{site}' not found in model.")

        # Resolve body exclusion
        self._body_exclude_id = -1
        if body_exclude is not None:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_exclude)
            if bid == -1:
                raise ValueError(f"Body '{body_exclude}' not found in model.")
            self._body_exclude_id = bid

        # Pre-compute ray directions in site local frame: (N, 3)
        self._local_dirs = _make_ray_directions(num_rays, fov_h, num_layers, fov_v)
        # Flatten to (num_layers*num_rays, 3) for mj_multiRay
        self._local_dirs_flat = self._local_dirs.reshape(-1, 3)
        self._n_total = self._local_dirs_flat.shape[0]

        # Persistent buffers for mj_multiRay (avoids per-call allocation)
        self._geomid_buf = np.full(self._n_total, -1, dtype=np.int32)
        self._dist_buf   = np.full(self._n_total, -1.0, dtype=np.float64)
        self._geomgroup  = np.ones(6, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_rays(self) -> int:
        return self._num_rays

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def range_max(self) -> float:
        return self._range_max

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def scan(
        self,
        *,
        return_geom_ids: bool = False,
    ) -> Union[np.ndarray, dict]:
        """Cast all rays and return distance measurements.

        Args:
            return_geom_ids: When True, also return the intersected geom ID
                             per ray (-1 = no hit).

        Returns:
            If ``return_geom_ids`` is False: distance array.
            If ``return_geom_ids`` is True: dict with keys
            ``"distances"`` and ``"geom_ids"``.

            Shape for 2D scan (num_layers == 1): ``(num_rays,)``.
            Shape for 3D scan: ``(num_layers, num_rays)``.
            Units: metres.  No-hit rays → ``range_max``.
        """
        model = self._env.get_model()
        data  = self._env.get_data()

        # Site pose in world frame
        pos = data.site_xpos[self._site_id].copy()          # (3,)
        rot = data.site_xmat[self._site_id].reshape(3, 3)   # (3, 3)

        # Rotate pre-computed local directions to world frame
        world_dirs = (rot @ self._local_dirs_flat.T).T      # (N, 3)
        world_dirs = np.ascontiguousarray(world_dirs)

        # Reset buffers
        self._dist_buf[:] = -1.0
        self._geomid_buf[:] = -1

        # Batch raycast — single call for all rays
        mujoco.mj_multiRay(
            model, data,
            pos, world_dirs,
            self._geomgroup,
            1,                       # flg_static: include static geoms
            self._body_exclude_id,
            self._geomid_buf,
            self._dist_buf,
            self._n_total,
            self._range_max,         # cutoff distance
        )

        # Clamp: no-hit → range_max, too-close → range_min
        distances = np.where(
            (self._dist_buf < 0) | (self._dist_buf > self._range_max),
            self._range_max,
            np.maximum(self._dist_buf, self._range_min),
        )

        # Reshape output
        if self._num_layers == 1:
            distances = distances  # (num_rays,)
            geom_ids  = self._geomid_buf.copy()
        else:
            distances = distances.reshape(self._num_layers, self._num_rays)
            geom_ids  = self._geomid_buf.reshape(self._num_layers, self._num_rays).copy()

        if return_geom_ids:
            return {"distances": distances, "geom_ids": geom_ids}
        return distances

    # ------------------------------------------------------------------
    # Pointcloud conversion
    # ------------------------------------------------------------------

    def to_pointcloud(self, distances: np.ndarray) -> np.ndarray:
        """Convert a distance scan to 3D world-frame points.

        Args:
            distances: Output of :meth:`scan`.  Shape ``(num_rays,)`` for 2D
                       or ``(num_layers, num_rays)`` for 3D.

        Returns:
            Points array of shape ``(N, 3)`` in world frame, where N equals
            the number of rays.  Points at ``range_max`` correspond to no-hit
            rays.
        """
        data = self._env.get_data()
        pos  = data.site_xpos[self._site_id].copy()
        rot  = data.site_xmat[self._site_id].reshape(3, 3)

        world_dirs = (rot @ self._local_dirs_flat.T).T  # (N, 3)

        flat_dist = np.asarray(distances).ravel()       # (N,)
        points = pos + world_dirs * flat_dist[:, np.newaxis]
        return points.astype(np.float32)


# ---------------------------------------------------------------------------
# Ray direction factory
# ---------------------------------------------------------------------------

def _make_ray_directions(
    num_rays: int,
    fov_h: float,
    num_layers: int,
    fov_v: float,
) -> np.ndarray:
    """Build (num_layers, num_rays, 3) unit ray directions in site local frame.

    The scan plane is XY; Z is the vertical rotation axis.  For a single-layer
    scan all rays have z=0 (horizontal plane).
    """
    # Azimuth angles (horizontal)
    if fov_h >= 360.0:
        azimuths = np.linspace(0.0, 2 * np.pi, num_rays, endpoint=False)
    else:
        half_h = np.radians(fov_h) / 2.0
        azimuths = np.linspace(-half_h, half_h, num_rays)

    # Elevation angles (vertical)
    if num_layers == 1:
        elevations = np.array([0.0])
    else:
        half_v = np.radians(fov_v) / 2.0
        elevations = np.linspace(-half_v, half_v, num_layers)

    # Build direction array
    el_grid, az_grid = np.meshgrid(elevations, azimuths, indexing="ij")  # (L, R)
    dirs = np.stack([
        np.cos(el_grid) * np.cos(az_grid),   # x
        np.cos(el_grid) * np.sin(az_grid),   # y
        np.sin(el_grid),                      # z
    ], axis=-1)  # (num_layers, num_rays, 3)

    return dirs.astype(np.float64)
