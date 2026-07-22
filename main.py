import os
import sys
import subprocess
import argparse
import joblib
import numpy as np
import cv2
import ffmpeg
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

# ==========================================
# CONFIGURATION
# ==========================================
PATH_4D_HUMANS = "./4D-Humans"
PATH_SMPL2BVH = "./smpl2bvh"

def get_video_info(video_path):
    """Reads the actual FPS of the video."""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream is None:
            raise ValueError("No video stream found.")
        
        fps_string = video_stream.get('avg_frame_rate', '30/1')
        num, den = map(int, fps_string.split('/'))
        fps = round(num / den)
        return fps
    except Exception as e:
        print(f"[Warning] Could not read FPS ({e}). Using default: 30")
        return 30

def ensure_mp4(video_path):
    """Checks the format. If not mp4, it is converted via ffmpeg."""
    base, ext = os.path.splitext(video_path)
    
    if ext == '.mp4':
        return video_path
    
    output_mp4 = base + ".mp4"
    
    print(f"[Step 1] Converting {video_path} to lowercase .mp4...")
    
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_mp4, vcodec='libx264', acodec='aac', loglevel="quiet")
            .overwrite_output()
            .run()
        )
        
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return output_mp4
        
    except Exception as e:
        print(f"[Error] Video conversion failed: {e}")
        if os.path.exists(output_mp4):
            os.remove(output_mp4)
        sys.exit(1)

def run_4d_humans(video_mp4):
    """Runs 4D-Humans tracking."""
    print(f"[Step 2] Starting 4D-Humans tracking for {video_mp4}...")
    
    cmd = [
        "python", "track.py",
        f"video.source={os.path.abspath(video_mp4)}"
    ]
    
    result = subprocess.run(cmd, cwd=PATH_4D_HUMANS)
    if result.returncode != 0:
        print("[Error] 4D-Humans failed.")
        sys.exit(1)
    
    video_name = os.path.splitext(os.path.basename(video_mp4))[0]
    pkl_path = os.path.join(PATH_4D_HUMANS, "outputs", "results", f"demo_{video_name}.pkl")
    return pkl_path

def rotmat_to_axis_angle(rotmats):
    out = np.zeros((rotmats.shape[0], 3), dtype=np.float32)
    for i in range(rotmats.shape[0]):
        aa, _ = cv2.Rodrigues(rotmats[i].astype(np.float64))
        out[i] = aa.squeeze()
    return out

def find_auto_target_tid(data):
    """Finds the track ID (TID) that appears most frequently in the video (main person)."""
    all_tids = []
    for fk in data.keys():
        all_tids.extend(data[fk].get("tid", []))
    if not all_tids:
        return None
    counts = np.bincount(all_tids)
    return int(np.argmax(counts))

def convert_pkl_structure(input_pkl, output_pkl, target_tid=None):
    print(f"[Step 3] Converting PKL structure...")
    data = joblib.load(input_pkl)
    frame_keys = sorted(data.keys())

    if target_tid is None:
        target_tid = find_auto_target_tid(data)
        if target_tid is None:
            print("[Error] No track IDs found in the PKL file.")
            sys.exit(1)
        print(f"Automatically selected main track ID (TID): {target_tid}")

    poses_seq = []
    trans_seq = []
    last_pose = None
    last_trans = None

    R_flip = R.from_euler('x', 180, degrees=True).as_matrix()

    for fk in frame_keys:
        frame = data[fk]
        tids = frame.get("tid", [])
        
        if target_tid in tids:
            idx = tids.index(target_tid)
            smpl = frame["smpl"][idx]

            global_orient = smpl["global_orient"]
            body_pose = smpl["body_pose"]
            
            global_orient_corrected = np.dot(R_flip, global_orient[0])[np.newaxis, ...]
            rotmats = np.concatenate([global_orient_corrected, body_pose], axis=0)

            aa = rotmat_to_axis_angle(rotmats)
            pose_flat = aa.reshape(-1)
            last_pose = pose_flat

            pelvis = frame["3d_joints"][idx][0]
            pelvis_corrected = np.dot(R_flip, pelvis)
            
            last_trans = pelvis_corrected.astype(np.float32)
            trans = last_trans
        else:
            if last_pose is None:
                continue
            pose_flat = last_pose
            trans = last_trans

        poses_seq.append(pose_flat)
        trans_seq.append(trans)

    if not poses_seq:
        print(f"[Error] Track ID {target_tid} was not found in any frame.")
        sys.exit(1)

    smpl_poses = np.stack(poses_seq, axis=0).astype(np.float32)
    smpl_trans = np.stack(trans_seq, axis=0).astype(np.float32)
    smpl_trans = smpl_trans - smpl_trans[0]
    smpl_scaling = np.array([1.0], dtype=np.float32)

    out_data = {
        "smpl_poses": smpl_poses,
        "smpl_trans": smpl_trans,
        "smpl_scaling": smpl_scaling,
    }

    joblib.dump(out_data, output_pkl)
    print(f"Successfully exported {smpl_poses.shape[0]} frames to {output_pkl}.")

def smooth_smpl_data(data, fps, polyorder=2):
    """
    Smooth SMPL poses and translations.

    Args:
        data (dict): Loaded PKL data.
        fps (float): Video FPS.
        polyorder (int): Savitzky-Golay polynomial order.

    Returns:
        dict: Smoothed data.
    """
    print(f"[Step 4] Smoothing SMPL Data...")

    window = max(5, int(round(fps * 0.25)))
    if window % 2 == 0:
        window += 1

    def valid_window(length):
        w = min(window, length)
        if w % 2 == 0:
            w -= 1
        return max(w, polyorder + 2)

    if "smpl_poses" in data:
        poses = data["smpl_poses"]

        if len(poses) > polyorder + 2:
            w = valid_window(len(poses))

            poses = median_filter(poses, size=(3, 1))

            root = savgol_filter(
                poses[:, :3],
                window_length=w,
                polyorder=polyorder,
                axis=0,
                mode="interp"
            )

            body = savgol_filter(
                poses[:, 3:],
                window_length=w,
                polyorder=polyorder,
                axis=0,
                mode="interp"
            )

            data["smpl_poses"] = np.concatenate([root, body], axis=1)

    if "smpl_trans" in data:
        trans = data["smpl_trans"]

        if len(trans) > polyorder + 2:
            w = valid_window(len(trans))

            trans = median_filter(trans, size=(3, 1))

            trans = savgol_filter(
                trans,
                window_length=w,
                polyorder=polyorder,
                axis=0,
                mode="interp"
            )

            data["smpl_trans"] = trans

    return data

def run_smpl2bvh(converted_pkl, fps, gender, output_bvh):
    """Calls the smpl2bvh tool."""
    print(f"[Step 5] Generating BVH file with {fps} FPS...")
    
    cmd = [
        "python", "smpl2bvh.py",
        "--gender", gender,
        "--poses", os.path.abspath(converted_pkl),
        "--fps", str(fps),
        "--output", os.path.abspath(output_bvh)
    ]
    
    result = subprocess.run(cmd, cwd=PATH_SMPL2BVH)
    if result.returncode != 0:
        print("[Error] smpl2bvh failed.")
        sys.exit(1)
    print(f"[Done] BVH successfully saved to: {output_bvh}")

def main():
    parser = argparse.ArgumentParser(description="Video -> 4D-Humans -> PKL-Fix -> BVH Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video (any format)")
    parser.add_argument("--gender", default="NEUTRAL", choices=["MALE", "FEMALE", "NEUTRAL"], help="Gender for SMPL")
    parser.add_argument("--tid", type=int, default=None, help="Optional fixed track ID (if omitted, the most active one is taken automatically)")
    parser.add_argument("--output", default="output.bvh", help="Path for the final .bvh file")
    
    args = parser.parse_args()

    # 1. Check & convert video if needed
    video_mp4 = ensure_mp4(args.video)
    
    # 2. Get FPS info
    fps = get_video_info(video_mp4)

    # 3. Run 4D-Humans
    raw_pkl = run_4d_humans(video_mp4)
    
    # 4. Convert and fix PKL structure
    temp_converted_pkl = "temp.pkl"
    convert_pkl_structure(raw_pkl, temp_converted_pkl, target_tid=args.tid)

    # 5. Smooth SMPL Data
    with open(temp_converted_pkl, "rb") as f:
        smooth_data = joblib.load(f)

    smooth_data = smooth_smpl_data(smooth_data, fps)

    with open(temp_converted_pkl, "wb") as f:
        joblib.dump(smooth_data, f)

    # 6. Convert to BVH
    run_smpl2bvh(temp_converted_pkl, fps, args.gender, args.output)

    # Clean up (optional)
    if os.path.exists(temp_converted_pkl):
        os.remove(temp_converted_pkl)

if __name__ == "__main__":
    main()