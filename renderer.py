"""Fuse 60 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np

import fusion


if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 164 #164
  start_no = 1710835728
  depth_intr = np.loadtxt("my_data/depth_intrinsics.txt", delimiter=' ')
  color_intr = np.loadtxt("my_data/color_intrinsics.txt", delimiter=' ')

  vol_bnds = np.zeros((3,2))
  for i in range(start_no, start_no + n_imgs):
    # Read depth image and camera pose
    depth_im = cv2.imread("my_data/%10d.depth.jpg"%(i),-1).astype(float)

    depth_im /= 20.  # depth is saved in 16-bit PNG in 5cm: change value in meter
    depth_im[depth_im > 3.0] = 0  # set invalid depth to 0
    cam_pose = np.loadtxt("my_data/%10d.pose.txt"%(i))  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, depth_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1) - 1)
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1) + 1)
  # ======================================================================================================== #

  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.05)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()
  for i in range(start_no, start_no + n_imgs):
    print("Fusing frame %d/%d"%(i, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread("my_data/%10d.color.jpg"%(i)), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread("my_data/%10d.depth.jpg"%(i),-1).astype(float)
    depth_im /= 20.  # depth is saved in 16-bit PNG in 5cm: change value in meter
    depth_im[depth_im > 3.0] = 0  # set invalid depth to 0
    cam_pose = np.loadtxt("my_data/%10d.pose.txt"%(i))

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, depth_intr, cam_pose, obs_weight=1., color_intr=color_intr)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))
  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)


