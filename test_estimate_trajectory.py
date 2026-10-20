"""Geometry and failure-handling checks; run: python -m unittest -v."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import estimate_trajectory as et


def render_frame(T, ns=0):
    """Raycast a plane and two spheres with a world-fixed analytic texture."""
    K = np.array([[210.,0,160],[0,210.,120],[0,0,1]])
    v,u = np.indices((240,320))
    rays = np.stack(((u-160)/210,(v-120)/210,np.ones_like(u)),axis=-1)
    direction = rays@T[:3,:3].T
    origin = T[:3,3]
    depth = (3.-origin[2])/direction[...,2]
    for center,radius in [(np.array([-.65,-.25,2.3]),.42),(np.array([.7,.3,2.1]),.37)]:
        offset = origin-center
        a = (direction**2).sum(axis=-1)
        b = 2*(direction*offset).sum(axis=-1)
        c = offset@offset-radius**2
        disc = b*b-4*a*c
        distance = (-b-np.sqrt(np.maximum(disc,0)))/(2*a)
        depth = np.where((disc>0)&(distance>0),np.minimum(depth,distance),depth)
    world = origin+direction*depth[...,None]
    gray = np.clip(128+35*np.sin(world[...,0]*28)+35*np.cos(world[...,1]*32)+35*np.sin(world[...,2]*37),0,255).astype(np.uint8)
    points = (rays*depth[...,None])[::3,::3].reshape(-1,3)
    mask = np.full(depth.shape,255,np.uint8)
    return et.Frame(ns,0,depth.astype(np.float32),mask,gray,mask.copy(),None,points,cKDTree(points)),K


class GeometryTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1)
        cv2.setRNGSeed(7)
        self.T = np.eye(4)
        self.T[:3,:3] = Rotation.from_euler("xyz",[.4,.8,-.2],degrees=True).as_matrix()
        self.T[:3,3] = [.015,-.005,.02]

    def test_timestamp_association_uses_time_not_frame_index(self):
        depth = [{"frame_unix_ns":str(x),"frame_index":i} for i,x in enumerate([10,110,210,310])]
        color = [{"frame_unix_ns":str(x),"frame_index":i+99} for i,x in enumerate([112,212])]
        pairs = et.associate(depth,color,5)
        self.assertEqual([(d["frame_index"],c["frame_index"]) for d,c in pairs],[(1,99),(2,100)])
        self.assertEqual(et.timestamp_ns({"frame_timestamp_ms":"1792411858363.1892"}),1792411858363189200)

    def test_depth_point_icp_recovers_known_rigid_motion(self):
        rng = np.random.default_rng(14)
        cloud = rng.normal(size=(1500,3))*[.4,.3,.2]+[0,0,2]
        source = et.transform(cloud,et.inverse(self.T))
        a = et.Frame(0,0,None,None,None,None,None,source,cKDTree(source))
        b = et.Frame(0,0,None,None,None,None,None,cloud,cKDTree(cloud))
        ok,T,_ = et.point_icp(a,b,np.eye(4),.1)
        self.assertTrue(ok)
        np.testing.assert_allclose(T,self.T,atol=1e-6)

    def test_all_camera_methods_recover_pose_direction_and_metric_scale(self):
        previous,K = render_frame(np.eye(4),0)
        current,_ = render_frame(self.T,33_333_333)
        for name in ["rgbd-feature","rgbd-direct","rgbd-hybrid","depth-icp"]:
            with self.subTest(method=name):
                args = et.parse_args(["--methods",name])
                est = et.Estimator(name,K,args)
                est.process(previous,0)
                result = est.process(current,0)
                self.assertEqual(result["status"],"ok",result["reason"])
                error = et.inverse(self.T)@result["pose"]
                self.assertLess(np.linalg.norm(error[:3,3]),.012)
                self.assertLess(np.degrees(Rotation.from_matrix(error[:3,:3]).magnitude()),.4)

    def test_lost_frame_not_reported_as_stationary_and_reset_is_separate(self):
        frame,K = render_frame(np.eye(4),0)
        args = et.parse_args(["--methods","depth-point"])
        est = et.Estimator("depth-point",K,args)
        est.process(frame,0)
        frame2,_ = render_frame(np.eye(4),33_333_333)
        with patch.object(et,"point_icp",return_value=(False,np.eye(4),{})):
            row = est.process(frame2,0)
        self.assertEqual(row["status"],"lost")
        self.assertIsNone(row["pose"])
        frame3,_ = render_frame(np.eye(4),2_000_000_000)
        row = est.process(frame3,0)
        self.assertEqual((row["status"],row["segment"]),("reset",1))
        with tempfile.TemporaryDirectory() as tmp:
            et.save_poses(Path(tmp),"depth-point",est.rows)
            self.assertEqual(len(list(Path(tmp).glob("*.tum"))),2)
            z = np.load(Path(tmp)/"depth-point.npz")
            self.assertTrue(np.isnan(z["T_segment_camera"][1]).all())

    def test_depth_only_can_load_without_any_rgb_files_or_color_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = {"width":8,"height":8,"fx":10.,"fy":10.,"ppx":4.,"ppy":4.,"distortion_coefficients":[0]*5}
            (root/"metadata.json").write_text(json.dumps({"streams":{"depth":info},"depth_scale_meters":.001}))
            (root/"timestamps.csv").write_text("stream,frame_index,image_path,frame_unix_ns,timestamp_domain\ndepth,0,0.png,1000000000,hardware_clock\ndepth,1,1.png,1033333333,hardware_clock\n")
            for i in range(2):
                cv2.imwrite(str(root/f"{i}.png"),np.full((8,8),2000,np.uint16))
            args = et.parse_args([str(root),"--methods","depth-point"])
            rec = et.Recording(args)
            frame = rec.load(rec.pairs[0])
            self.assertIsNone(frame.gray)
            np.testing.assert_allclose(frame.depth,2.)

    def test_color_warp_uses_column_major_extrinsic_and_metric_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = {"width":8,"height":8,"fx":10.,"fy":10.,"ppx":4.,"ppy":4.,"distortion_coefficients":[0]*5}
            # 90-degree rotation and 20 cm offset; at 2 m this adds one pixel.
            R = np.array([[0,-1,0],[1,0,0],[0,0,1]])
            meta = {"streams":{"depth":info,"color":info},"depth_scale_meters":.001,
                    "depth_to_color":{"rotation_column_major":R.flatten(order="F").tolist(),"translation_meters":[.2,0,0]}}
            (root/"metadata.json").write_text(json.dumps(meta))
            (root/"timestamps.csv").write_text("stream,frame_index,image_path,frame_unix_ns,timestamp_domain\ndepth,0,d.png,1000000000,global_time\ncolor,0,c.png,1000000000,global_time\ndepth,1,d.png,1033333333,global_time\ncolor,1,c.png,1033333333,global_time\n")
            cv2.imwrite(str(root/"d.png"),np.full((8,8),2000,np.uint16))
            v,u = np.indices((8,8),dtype=np.uint8)
            cv2.imwrite(str(root/"c.png"),np.stack((u*20,v*20,np.zeros_like(u)),axis=-1))
            rec = et.Recording(et.parse_args([str(root),"--methods","rgbd-feature","--downsample","1"]))
            frame = rec.load(rec.pairs[0])
            np.testing.assert_array_equal(frame.color[4,4],[0,80,100])
            np.testing.assert_array_equal(frame.color[4,3],[0,60,100])


if __name__=="__main__":
    unittest.main()
