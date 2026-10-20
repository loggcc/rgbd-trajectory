#!/usr/bin/env python3
"""Compare calibrated RGB-D and depth-only odometry on exported RealSense PNGs.

All poses map the current depth optical frame into the first depth optical frame
of their segment: +X right, +Y down, +Z forward, translation in meters.
This is local odometry, not loop-closing SLAM or robot-base localization.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
import html
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "trajectory-mpl"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

METHODS = {
    "rgbd-hybrid": "RGB-D: intensity + point-to-plane ICP",
    "rgbd-direct": "RGB-D: direct photometric odometry",
    "rgbd-feature": "RGB-D: optical flow + PnP RANSAC",
    "depth-icp": "Depth: projective point-to-plane ICP",
    "depth-point": "Depth: trimmed point-to-point ICP",
}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def transform(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def camera_matrix(info, factor=1):
    return np.array([[info["fx"] / factor, 0, info["ppx"] / factor],
                     [0, info["fy"] / factor, info["ppy"] / factor],
                     [0, 0, 1]], dtype=np.float64)


def timestamp_ns(row):
    if row.get("frame_unix_ns"):
        return int(row["frame_unix_ns"])
    return int(Decimal(row["frame_timestamp_ms"]) * 1_000_000)


def associate(depth_rows, color_rows, tolerance_ns):
    """Monotonic nearest-time association without reusing color frames."""
    ct = np.array([timestamp_ns(r) for r in color_rows], dtype=np.int64)
    pairs, last = [], -1
    for d in depth_rows:
        t = timestamp_ns(d)
        j = int(np.searchsorted(ct, t))
        candidates = [i for i in (j - 1, j) if last < i < len(ct)]
        if not candidates:
            continue
        i = min(candidates, key=lambda x: abs(int(ct[x]) - t))
        if abs(int(ct[i]) - t) <= tolerance_ns:
            pairs.append((d, color_rows[i]))
            last = i
    return pairs


@dataclass
class Frame:
    ns: int
    index: int
    depth: np.ndarray
    mask: np.ndarray
    gray: np.ndarray | None
    rgb_mask: np.ndarray | None
    color: np.ndarray | None
    points: np.ndarray
    tree: cKDTree


class Recording:
    def __init__(self, args):
        self.root = args.recording.resolve()
        self.meta = json.loads((self.root / "metadata.json").read_text())
        self.args = args
        self.use_rgb = any(x.startswith("rgbd") for x in args.methods)
        streams = self.meta["streams"]
        self.aligned = self.meta.get("storage", {}).get("depth", {}).get("aligned_to_rgb", False)
        self.info = streams["color" if self.aligned else "depth"]
        for name in (["depth", "color"] if self.use_rgb or self.aligned else ["depth"]):
            if np.any(np.abs(streams[name].get("distortion_coefficients", [])) > 1e-9):
                raise ValueError("Nonzero distortion is not supported: rectify images/calibration first.")
        self.K = camera_matrix(self.info, args.downsample)
        self.scale = float(self.meta["depth_scale_meters"])
        if not 0 < self.scale < 1:
            raise ValueError("Invalid depth_scale_meters")
        with (self.root / "timestamps.csv").open() as f:
            rows = list(csv.DictReader(f))
        depth = sorted((r for r in rows if r["stream"] == "depth"), key=timestamp_ns)
        color = sorted((r for r in rows if r["stream"] == "color"), key=timestamp_ns)
        if not depth:
            raise ValueError("No depth rows in timestamps.csv")
        for stream_rows in ([depth, color] if self.use_rgb else [depth]):
            stamps = [timestamp_ns(r) for r in stream_rows]
            if any(b <= a for a, b in zip(stamps, stamps[1:])):
                raise ValueError("Frame timestamps must be strictly increasing within a stream")
        if self.use_rgb:
            domains = {r["timestamp_domain"] for r in depth + color}
            if len(domains) != 1 or not domains <= {"global_time", "system_time"}:
                raise ValueError("RGB/depth association needs a shared global_time/system_time clock")
            pairs = associate(depth, color, int(args.sync_ms * 1e6))
        else:
            pairs = [(d, None) for d in depth]
        if len(pairs) < 2:
            raise ValueError("Fewer than two synchronized frames")
        self.start_ns = timestamp_ns(pairs[0][0])
        self.pairs = [p for p in pairs if (timestamp_ns(p[0]) - self.start_ns) / 1e9 >= args.start][::args.stride]
        if args.max_frames:
            self.pairs = self.pairs[:args.max_frames]
        if len(self.pairs) < 2:
            raise ValueError("Selection has fewer than two frames")
        self.start_ns = timestamp_ns(self.pairs[0][0])
        self.association_summary = {
            "raw_depth_frames": len(depth), "raw_color_frames": len(color),
            "synchronized_pairs": len(pairs) if self.use_rgb else None,
            "unmatched_depth_frames": len(depth) - len(pairs),
            "processed_frames": len(self.pairs),
            "max_sync_difference_ms": max(abs(timestamp_ns(d)-timestamp_ns(c))/1e6 for d,c in pairs) if self.use_rgb else None,
        }
        h = len(range(0, self.info["height"], args.downsample))
        w = len(range(0, self.info["width"], args.downsample))
        v, u = np.indices((h, w), dtype=np.float32)
        self.rays = np.stack(((u-self.K[0,2])/self.K[0,0], (v-self.K[1,2])/self.K[1,1], np.ones_like(u)), axis=-1)
        if self.use_rgb:
            self.Kc = camera_matrix(streams["color"])
            self.extrinsic = np.eye(4)
            if not self.aligned:
                ext = self.meta["depth_to_color"]
                self.extrinsic[:3,:3] = np.array(ext["rotation_column_major"]).reshape(3,3,order="F")
                self.extrinsic[:3,3] = ext["translation_meters"]

    def load(self, pair):
        d, c = pair
        raw = cv2.imread(str(self.root / d["image_path"]), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.shape != (self.info["height"], self.info["width"]):
            raise ValueError(f"Missing or invalid uint16 depth image: {d['image_path']}")
        # Subsampling at integer pixels keeps K/factor exact (no half-pixel shift).
        z = np.ascontiguousarray(raw[::self.args.downsample, ::self.args.downsample], dtype=np.float32) * self.scale
        valid = (z >= self.args.min_depth) & (z <= self.args.max_depth)
        xyz = self.rays * z[...,None]
        gray = rgb_mask = color = None
        if self.use_rgb:
            bgr = cv2.imread(str(self.root / c["image_path"]))
            if bgr is None:
                raise ValueError(f"Missing RGB PNG: {c['image_path']}")
            pc = transform(xyz.reshape(-1,3), self.extrinsic).reshape(xyz.shape)
            den = np.maximum(pc[...,2], 1e-6)
            u = (self.Kc[0,0]*pc[...,0]/den + self.Kc[0,2]).astype(np.float32)
            v = (self.Kc[1,1]*pc[...,1]/den + self.Kc[1,2]).astype(np.float32)
            visible = valid & (pc[...,2]>0) & (u>=0) & (v>=0) & (u<bgr.shape[1]-1) & (v<bgr.shape[0]-1)
            # Z-buffer rejects samples occluded from the color camera.
            uu = np.clip(np.rint(u).astype(int),0,bgr.shape[1]-1)
            vv = np.clip(np.rint(v).astype(int),0,bgr.shape[0]-1)
            pixel = vv*bgr.shape[1]+uu
            zbuf = np.full(bgr.shape[0]*bgr.shape[1], np.inf, dtype=np.float32)
            np.minimum.at(zbuf,pixel[visible],pc[...,2][visible])
            visible &= pc[...,2] <= zbuf[pixel]+0.03
            color = cv2.cvtColor(cv2.remap(bgr,u,v,cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
            color[~visible] = 0
            gray = cv2.cvtColor(color,cv2.COLOR_RGB2GRAY)
            rgb_mask = cv2.erode(visible.astype(np.uint8)*255,np.ones((3,3),np.uint8))
        mask = valid.astype(np.uint8)*255
        # Preserve invalids as NaN: filling holes with zero can create false surfaces.
        z[~valid] = np.nan
        cloud = xyz[::3,::3][valid[::3,::3]].astype(np.float64)
        return Frame(timestamp_ns(d),int(d["frame_index"]),z,mask,gray,rgb_mask,color,cloud,cKDTree(cloud))


def geometry_quality(current, reference, T, K, threshold):
    """Projective depth agreement; a consistency score, not pose accuracy."""
    p = transform(current.points, T)
    positive = p[:,2] > 0
    p = p[positive]
    if not len(p):
        return 0., float("inf")
    u = np.rint(K[0,0]*p[:,0]/p[:,2]+K[0,2]).astype(int)
    v = np.rint(K[1,1]*p[:,1]/p[:,2]+K[1,2]).astype(int)
    h,w = reference.depth.shape
    inside = (u>=0)&(v>=0)&(u<w)&(v<h)
    z = reference.depth[v[inside],u[inside]]
    delta = np.abs(z-p[inside,2])
    good = np.isfinite(delta)&(delta<threshold)
    ratio = float(good.sum()/max(len(current.points),1))
    rmse = float(np.sqrt(np.mean(delta[good]**2))) if np.any(good) else float("inf")
    return ratio,rmse


def point_icp(source, target, initial, max_distance):
    """Depth only: multiscale distance gates, trimmed least-squares rigid fit."""
    T = initial.copy()
    if min(len(source.points),len(target.points)) < 40:
        return False,T,{}
    for distance,iterations in [(max_distance*2,10),(max_distance,10),(max_distance*.5,8)]:
        for _ in range(iterations):
            p = transform(source.points,T)
            dist,idx = target.tree.query(p,k=1,workers=1)
            good = dist<distance
            if good.sum()<40:
                return False,T,{}
            cutoff = np.quantile(dist[good],.85)
            good &= dist<=cutoff
            a,b = p[good],target.points[idx[good]]
            ac,bc = a.mean(axis=0),b.mean(axis=0)
            U,_,Vt = np.linalg.svd((a-ac).T@(b-bc))
            R = Vt.T@np.diag([1.,1.,np.linalg.det(Vt.T@U.T)])@U.T
            delta = np.eye(4)
            delta[:3,:3],delta[:3,3] = R,bc-R@ac
            T = delta@T
            if np.linalg.norm(delta[:3,3])<1e-5 and Rotation.from_matrix(R).magnitude()<1e-5:
                break
    return True,T,{}


def feature_odometry(current, previous, K):
    mask = previous.rgb_mask.copy()
    # Exclude depth discontinuities where a feature may borrow the wrong surface.
    z = np.nan_to_num(previous.depth)
    spread = cv2.dilate(z,np.ones((3,3),np.uint8))-cv2.erode(z,np.ones((3,3),np.uint8))
    mask[spread>.08] = 0
    p0 = cv2.goodFeaturesToTrack(previous.gray,1200,.008,5,mask=mask,blockSize=5)
    if p0 is None or len(p0)<20:
        return False,np.eye(4),{"inliers":0}
    p1,s1,_ = cv2.calcOpticalFlowPyrLK(previous.gray,current.gray,p0,None,winSize=(21,21),maxLevel=3)
    pback,s2,_ = cv2.calcOpticalFlowPyrLK(current.gray,previous.gray,p1,None,winSize=(21,21),maxLevel=3)
    a,b = p0[:,0],p1[:,0]
    good = s1[:,0].astype(bool)&s2[:,0].astype(bool)&(np.linalg.norm(pback[:,0]-a,axis=1)<1.)
    h,w = current.depth.shape
    good &= (b[:,0]>=1)&(b[:,0]<w-1)&(b[:,1]>=1)&(b[:,1]<h-1)
    a,b = a[good],b[good]
    if len(a)<20:
        return False,np.eye(4),{"inliers":0}
    ai = np.rint(a).astype(int)
    bi = np.rint(b).astype(int)
    depth = previous.depth[ai[:,1],ai[:,0]]
    good = np.isfinite(depth)&(current.rgb_mask[bi[:,1],bi[:,0]]>0)
    a,b,depth = a[good],b[good],depth[good]
    points = np.column_stack(((a[:,0]-K[0,2])*depth/K[0,0],(a[:,1]-K[1,2])*depth/K[1,1],depth)).astype(np.float64)
    if len(points)<20:
        return False,np.eye(4),{"inliers":0}
    ok,r,t,inliers = cv2.solvePnPRansac(points,b.astype(np.float64),K,None,iterationsCount=150,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    n = 0 if inliers is None else len(inliers)
    metrics = {"inliers":n,"feature_inlier_ratio":n/len(points)}
    if not ok or n<20 or n/len(points)<.3:
        return False,np.eye(4),metrics
    ids = inliers[:,0]
    r,t = cv2.solvePnPRefineLM(points[ids],b[ids].astype(np.float64),K,None,r,t)
    T = np.eye(4)
    T[:3,:3],T[:3,3] = cv2.Rodrigues(r)[0],t[:,0]
    # PnP maps previous -> current; all estimators return current -> previous.
    return True,inverse(T),metrics


class Estimator:
    def __init__(self,name,K,args):
        self.name,self.K,self.args = name,K,args
        self.odometry = None
        if name in ("rgbd-direct","rgbd-hybrid"):
            maker = cv2.rgbd.RgbdOdometry_create if name=="rgbd-direct" else cv2.rgbd.RgbdICPOdometry_create
            self.odometry = maker(cameraMatrix=K,minDepth=args.min_depth,maxDepth=args.max_depth,maxDepthDiff=args.correspondence_distance,iterCounts=(7,7,7),minGradientMagnitudes=(5.,5.,5.),maxPointsPart=.3)
            self.odometry.setMaxTranslation(1.)
            self.odometry.setMaxRotation(45.)
        elif name=="depth-icp":
            self.odometry = cv2.rgbd.FastICPOdometry_create(K,maxDistDiff=args.correspondence_distance,angleThreshold=np.deg2rad(30.),iterCounts=(7,7,7))
        self.reference = None
        self.pose = np.eye(4)
        self.segment = 0
        self.rows = []
        self.last_step = np.eye(4)
        self.last_dt = 0.

    def process(self,frame,start_ns):
        row = {"frame_index":frame.index,"timestamp_ns":frame.ns,"timestamp_s":frame.ns/1e9,
               "elapsed_s":(frame.ns-start_ns)/1e9,"segment":self.segment,
               "status":"ok","reason":"","overlap":None,"depth_rmse_m":None,"inliers":None}
        if self.reference is None:
            self.reference = frame
            row["status"] = "origin"
            row["pose"] = self.pose.copy()
        elif (frame.ns-self.reference.ns)/1e9>self.args.reset_gap:
            self.segment+=1
            self.pose = np.eye(4)
            self.reference = frame
            self.last_step = np.eye(4)
            self.last_dt = 0.
            row.update(segment=self.segment,status="reset",reason="Tracking gap; NEW LOCAL coordinate frame",pose=self.pose.copy())
        else:
            dt = (frame.ns-self.reference.ns)/1e9
            initial = np.eye(4)
            if self.last_dt>0 and dt/self.last_dt<3:
                ratio = dt/self.last_dt
                initial[:3,:3] = Rotation.from_rotvec(Rotation.from_matrix(self.last_step[:3,:3]).as_rotvec()*ratio).as_matrix()
                initial[:3,3] = self.last_step[:3,3]*ratio
            try:
                extra = {}
                if self.name=="depth-point":
                    ok,T,extra = point_icp(frame,self.reference,initial,self.args.correspondence_distance)
                elif self.name=="rgbd-feature":
                    ok,T,extra = feature_odometry(frame,self.reference,self.K)
                else:
                    rgb = self.name.startswith("rgbd")
                    ok,T = self.odometry.compute(frame.gray if rgb else None,frame.depth,frame.rgb_mask if rgb else frame.mask,
                                                 self.reference.gray if rgb else None,self.reference.depth,self.reference.rgb_mask if rgb else self.reference.mask,
                                                 initRt=initial)
                row.update(extra)
                reason = "solver_failed" if not ok else ""
                if ok:
                    if T is None or not np.all(np.isfinite(T)) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4) or np.linalg.det(T[:3,:3])<.99:
                        reason = "invalid_transform"
                    else:
                        overlap,rmse = geometry_quality(frame,self.reference,T,self.K,self.args.correspondence_distance)
                        row.update(overlap=overlap,depth_rmse_m=rmse if np.isfinite(rmse) else None)
                        if overlap<self.args.min_overlap:
                            reason = "low_depth_overlap"
                        elif np.linalg.norm(T[:3,3])>self.args.max_speed*dt+.02:
                            reason = "translation_limit"
                        elif np.degrees(Rotation.from_matrix(T[:3,:3]).magnitude())>self.args.max_angular_speed*dt+2.:
                            reason = "rotation_limit"
                if reason:
                    row.update(status="lost",reason=reason,pose=None)
                else:
                    self.pose = self.pose@T
                    self.last_step,self.last_dt = T,dt
                    self.reference = frame
                    row["pose"] = self.pose.copy()
            except cv2.error as e:
                row.update(status="lost",reason="opencv_error: "+str(e).splitlines()[-1],pose=None)
        self.rows.append(row)
        return row


def segments(rows):
    return sorted({r["segment"] for r in rows if r["pose"] is not None})


def trajectory_stats(rows):
    per_segment = []
    for seg in segments(rows):
        good = [r for r in rows if r["segment"]==seg and r["pose"] is not None]
        xyz = np.array([r["pose"][:3,3] for r in good])
        per_segment.append({"segment":seg,"poses":len(good),"start_s":good[0]["elapsed_s"],"end_s":good[-1]["elapsed_s"],
                            "path_length_m":float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
                            "end_displacement_m":float(np.linalg.norm(xyz[-1]-xyz[0]))})
    rejected = [r for r in rows if r["status"]=="lost"]
    reasons = {k:sum(r["reason"]==k for r in rejected) for k in sorted({r["reason"] for r in rejected})}
    return {"accepted_poses":len(rows)-len(rejected),"rejected_frames":len(rejected),"rejection_reasons":reasons,
            "segments":per_segment,"accepted_fraction":(len(rows)-len(rejected))/len(rows)}


def save_poses(out,name,rows):
    columns = ["frame_index","timestamp_ns","timestamp_s","elapsed_s","segment","status","reason",
               "tx","ty","tz","qx","qy","qz","qw","overlap","depth_rmse_m","inliers"]
    with (out/f"{name}.csv").open("w",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        for row in rows:
            data = {k:row.get(k,"") for k in columns}
            if row["pose"] is not None:
                data.update(zip(["tx","ty","tz"],row["pose"][:3,3]))
                data.update(zip(["qx","qy","qz","qw"],Rotation.from_matrix(row["pose"][:3,:3]).as_quat()))
            writer.writerow(data)
    for seg in segments(rows):
        with (out/f"{name}_segment{seg:02d}.tum").open("w") as f:
            f.write("# timestamp tx ty tz qx qy qz qw; T_segment_depth_camera; meters\n")
            for row in rows:
                if row["pose"] is None or row["segment"]!=seg:
                    continue
                T = row["pose"]
                values = [row["timestamp_s"],*T[:3,3],*Rotation.from_matrix(T[:3,:3]).as_quat()]
                f.write(" ".join(f"{v:.9f}" for v in values)+"\n")
    np.savez_compressed(out/f"{name}.npz",timestamp_ns=np.array([r["timestamp_ns"] for r in rows],dtype=np.int64),
                        T_segment_camera=np.array([r["pose"] if r["pose"] is not None else np.full((4,4),np.nan) for r in rows]),
                        segment=np.array([r["segment"] for r in rows]),status=np.array([r["status"] for r in rows]))


def make_plots(out,estimators,title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(15,11),layout="constrained")
    ax3 = fig.add_subplot(221,projection="3d")
    axes = [fig.add_subplot(222),fig.add_subplot(223),fig.add_subplot(224)]
    ax3.set(xlabel="X right (m)",ylabel="Z forward (m)",zlabel="−Y up (m)",title="3D camera trajectory")
    for ax,(i,j) in zip(axes,[(0,2),(0,1),(2,1)]):
        ax.set(xlabel=["X (m)","Y (m)","Z (m)"][i],ylabel=["X (m)","Y (m)","Z (m)"][j])
        ax.grid(alpha=.25)
        ax.set_aspect("equal",adjustable="datalim")
        if j==1:
            ax.invert_yaxis()
    all_points = []
    for color,est in zip(COLORS,estimators):
        for seg in segments(est.rows):
            xyz = np.array([r["pose"][:3,3] if r["pose"] is not None and r["segment"]==seg else [np.nan]*3 for r in est.rows])
            label = est.name if seg==0 else "_nolegend_"
            style = "-" if seg==0 else "--"
            ax3.plot(xyz[:,0],xyz[:,2],-xyz[:,1],style,color=color,label=label,lw=1.5)
            all_points.extend(xyz[np.isfinite(xyz[:,0])][:,[0,2,1]]*[1,1,-1])
            for ax,(i,j) in zip(axes,[(0,2),(0,1),(2,1)]):
                ax.plot(xyz[:,i],xyz[:,j],style,color=color,lw=1.5)
    if all_points:
        p = np.array(all_points)
        center = (p.min(axis=0)+p.max(axis=0))/2
        radius = max(np.ptp(p,axis=0).max()/2,.1)*1.1
        ax3.set_xlim(center[0]-radius,center[0]+radius)
        ax3.set_ylim(center[1]-radius,center[1]+radius)
        ax3.set_zlim(center[2]-radius,center[2]+radius)
        ax3.set_box_aspect((1,1,1))
    ax3.legend(fontsize=7,loc="upper left")
    fig.suptitle(title+"\nLocal odometry · meters · no ground truth / loop closure · dashed segments have independent origins",fontsize=12)
    fig.savefig(out/"trajectory_comparison.png",dpi=150,bbox_inches="tight")
    plt.close(fig)
    fig,axs = plt.subplots(3,1,figsize=(14,10),sharex=True,layout="constrained")
    for color,est in zip(COLORS,estimators):
        t = [r["elapsed_s"] for r in est.rows]
        axs[0].plot(t,[r["overlap"] if r["overlap"] is not None else np.nan for r in est.rows],color=color,label=est.name,alpha=.8)
        axs[1].plot(t,[r["depth_rmse_m"] if r["depth_rmse_m"] is not None else np.nan for r in est.rows],color=color,alpha=.8)
        y = [r["pose"][:3,3] if r["pose"] is not None else [np.nan]*3 for r in est.rows]
        axs[2].plot(t,np.linalg.norm(y,axis=1),color=color,alpha=.8)
        lost = [r["elapsed_s"] for r in est.rows if r["status"]=="lost"]
        axs[0].scatter(lost,np.zeros(len(lost)),marker="x",s=8,color=color)
    axs[0].set_ylabel("Depth agreement fraction")
    axs[1].set_ylabel("Inlier depth residual (m)")
    axs[2].set(ylabel="Distance from segment origin (m)",xlabel="Elapsed time (s)")
    axs[0].legend(loc="lower left",ncol=3,fontsize=8)
    for ax in axs:
        ax.grid(alpha=.2)
    fig.suptitle("Tracking diagnostics — agreement/residuals are NOT absolute trajectory accuracy; × = rejected")
    fig.savefig(out/"tracking_diagnostics.png",dpi=150,bbox_inches="tight")
    plt.close(fig)


def make_html(out,estimators,title):
    data = []
    for color,est in zip(COLORS,estimators):
        data.append({"name":est.name,"description":METHODS[est.name],"color":color,
                     "rows":[{"t":round(r["elapsed_s"],4),"segment":r["segment"],"status":r["status"],
                              "p":r["pose"][:3,3].round(6).tolist() if r["pose"] is not None else None,
                              "R":r["pose"][:3,:3].round(6).tolist() if r["pose"] is not None else None} for r in est.rows]})
    template = Path(__file__).with_name("trajectory_viewer.html").read_text()
    template = template.replace("__TITLE__",html.escape(title)).replace("__DATA__",json.dumps(data,separators=(",",":")))
    (out/"trajectory.html").write_text(template)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("recording",type=Path,nargs="?",default=Path(__file__).resolve().parent / "data" / "20261019T121058_156205Z")
    p.add_argument("--methods",nargs="+",choices=list(METHODS)+["all"],default=["all"])
    p.add_argument("--output",type=Path)
    p.add_argument("--stride",type=int,default=1,help="Process every Nth frame; 1 is safer during fast motion")
    p.add_argument("--downsample",type=int,default=2,help="Image subsampling factor")
    p.add_argument("--start",type=float,default=0,help="Start offset in seconds")
    p.add_argument("--max-frames",type=int,default=0,help="0 = complete selected recording")
    p.add_argument("--sync-ms",type=float,default=20,help="Maximum RGB/depth timestamp difference")
    p.add_argument("--min-depth",type=float,default=.35)
    p.add_argument("--max-depth",type=float,default=5.)
    p.add_argument("--correspondence-distance",type=float,default=.10,help="Depth correspondence gate in meters")
    p.add_argument("--min-overlap",type=float,default=.25)
    p.add_argument("--max-speed",type=float,default=3.,help="Reject translation exceeding this m/s plus 2 cm tolerance")
    p.add_argument("--max-angular-speed",type=float,default=240.,help="Degrees per second, plus 2 degree tolerance")
    p.add_argument("--reset-gap",type=float,default=1.,help="After this tracking gap, start an explicitly separate local trajectory")
    p.add_argument("--threads",type=int,default=2)
    args = p.parse_args(argv)
    if "all" in args.methods:
        args.methods = list(METHODS)
    args.methods = list(dict.fromkeys(args.methods))
    if args.stride<1 or args.downsample<1 or args.max_frames<0 or args.start<0 or args.threads<1:
        p.error("stride/downsample/threads must be positive, start/max-frames nonnegative")
    if not 0<args.min_depth<args.max_depth or not 0<args.min_overlap<=1 or min(args.sync_ms,args.correspondence_distance,args.max_speed,args.max_angular_speed,args.reset_gap)<=0:
        p.error("Invalid depth, overlap, timing, or motion limits")
    return args


def main(argv=None):
    args = parse_args(argv)
    if any(m in ("rgbd-hybrid","rgbd-direct","depth-icp") for m in args.methods) and not hasattr(cv2,"rgbd"):
        raise SystemExit("cv2.rgbd is missing. Install opencv-contrib-python (not opencv-python), or use --methods rgbd-feature depth-point.")
    cv2.setNumThreads(args.threads)
    cv2.setRNGSeed(7)
    recording = Recording(args)
    out = (args.output or Path(__file__).resolve().parent / "runs" / recording.root.name).resolve()
    # Each run is immutable: avoid stale segment exports or silently overwritten results.
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out}. Choose a new --output directory.")
    estimators = [Estimator(name,recording.K,args) for name in args.methods]
    print(json.dumps(recording.association_summary),flush=True)
    start = time.perf_counter()
    for i,pair in enumerate(recording.pairs):
        frame = recording.load(pair)
        for est in estimators:
            est.process(frame,recording.start_ns)
        if i%100==0 or i==len(recording.pairs)-1:
            status = " | ".join(f"{e.name}: {sum(r['status']=='lost' for r in e.rows)} lost, {e.segment+1} segments" for e in estimators)
            print(f"{i+1}/{len(recording.pairs)} frames, {time.perf_counter()-start:.1f}s | {status}",flush=True)
    summary = {"recording":str(recording.root),"pose_convention":"T_segment_depth_camera; meters; optical X-right Y-down Z-forward",
               "warning":"Local odometry, no loop closure, no ground truth. Resets start unrelated local coordinates. Dynamic objects can bias estimates even if quality checks pass.",
               "robot_base":"Camera poses only. A camera-to-base extrinsic is required for robot-base poses.",
               "parameters":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               "association":recording.association_summary,"versions":{"opencv":cv2.__version__,"numpy":np.__version__},
               "runtime_seconds":time.perf_counter()-start,"methods":{e.name:trajectory_stats(e.rows) for e in estimators}}
    for est in estimators:
        save_poses(out,est.name,est.rows)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    make_plots(out,estimators,recording.root.name)
    make_html(out,estimators,recording.root.name)
    print(f"Saved results to {out}",flush=True)


if __name__=="__main__":
    main()
