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
from scipy.spatial.transform import Slerp

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

def convert_pkl_structure(input_pkl, output_pkl, fps, target_tid=None, outlier_thresh=2.5):
    print(f"[Step 3] Converting PKL structure & Smoothing Rotations...")
    data = joblib.load(input_pkl)
    frame_keys = sorted(data.keys())

    if target_tid is None:
        target_tid = find_auto_target_tid(data)
        if target_tid is None:
            print("[Error] No track IDs found in the PKL file.")
            sys.exit(1)
        print(f"Automatically selected main track ID (TID): {target_tid}")

    quats_seq = []
    trans_seq = []
    last_rotmats = None
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

            pelvis = frame["3d_joints"][idx][0]
            pelvis_corrected = np.dot(R_flip, pelvis)
            last_rotmats = rotmats
            last_trans = pelvis_corrected.astype(np.float32)
        else:
            if last_rotmats is None:
                continue
            rotmats = last_rotmats
            pelvis_corrected = last_trans

        quats = R.from_matrix(rotmats).as_quat()
        quats_seq.append(quats)
        trans_seq.append(pelvis_corrected)

    if not quats_seq:
        print(f"[Error] Track ID {target_tid} was not found in any frame.")
        sys.exit(1)

    quats_seq = np.array(quats_seq)
    trans_seq = np.array(trans_seq)
    T, J, _ = quats_seq.shape

    # ------------------------------------------------------------------
    # 1. SIGN-FLIP FIX (+q und -q sind dieselbe Rotation)
    # ------------------------------------------------------------------
    for t in range(1, T):
        dots = np.einsum('ij,ij->i', quats_seq[t - 1], quats_seq[t])
        flip = dots < 0
        quats_seq[t, flip] = -quats_seq[t, flip]

    # ------------------------------------------------------------------
    # 2. OUTLIER-ERKENNUNG & -ERSATZ (pro Gelenk, per Winkel-Sprung)
    #    Frames mit untypisch großem Rotations-Sprung werden durch
    #    Slerp-Interpolation der Nachbarn ersetzt, BEVOR geglättet wird.
    # ------------------------------------------------------------------
    if T > 4:
        for j in range(J):
            rots_j = R.from_quat(quats_seq[:, j])
            # Winkel-Distanz zwischen aufeinanderfolgenden Frames
            rel = rots_j[:-1].inv() * rots_j[1:]
            angles = rel.magnitude()

            med = np.median(angles)
            mad = np.median(np.abs(angles - med)) + 1e-8
            # robuster z-Score
            z = np.abs(angles - med) / (1.4826 * mad)

            # Frame t gilt als Outlier, wenn sowohl der Sprung IN t
            # als auch der Sprung AUS t auffällig groß ist
            bad = np.zeros(T, dtype=bool)
            jump_in = np.concatenate([[False], z > outlier_thresh])
            jump_out = np.concatenate([z > outlier_thresh, [False]])
            bad = jump_in & jump_out

            bad_idx = np.where(bad)[0]
            if len(bad_idx) == 0:
                continue

            good_idx = np.where(~bad)[0]
            if len(good_idx) < 2:
                continue

            quats_fixed = quats_seq[:, j].copy()
            for bi in bad_idx:
                # nächsten guten Frame davor/danach suchen
                prev_candidates = good_idx[good_idx < bi]
                next_candidates = good_idx[good_idx > bi]
                if len(prev_candidates) == 0 or len(next_candidates) == 0:
                    continue
                p, n = prev_candidates[-1], next_candidates[0]
                alpha = (bi - p) / (n - p)
                slerp = Slerp([0, 1], R.from_quat([quats_fixed[p], quats_fixed[n]]))
                quats_fixed[bi] = slerp(alpha).as_quat()

            quats_seq[:, j] = quats_fixed

    # ------------------------------------------------------------------
    # 3. FENSTERGRÖSSE ABSICHERN (kurze Clips)
    # ------------------------------------------------------------------
    window = max(5, int(round(fps * 0.25)))
    if window % 2 == 0:
        window += 1
    max_window = T if T % 2 == 1 else T - 1
    window = min(window, max_window)
    window = max(window, 3)
    polyorder = min(2, window - 1)

    if T <= polyorder:
        print(f"[Warning] Too little frames ({T}) for smoothing. Skip smoothing.")
        smpl_poses = R.from_quat(quats_seq.reshape(-1, 4)).as_rotvec().reshape(T, J * 3).astype(np.float32)
        smpl_trans = (trans_seq - trans_seq[0]).astype(np.float32)
        smpl_scaling = np.array([1.0], dtype=np.float32)
        out_data = {"smpl_poses": smpl_poses, "smpl_trans": smpl_trans, "smpl_scaling": smpl_scaling}
        joblib.dump(out_data, output_pkl)
        print(f"Successfully exported {smpl_poses.shape[0]} frames to {output_pkl}.")
        return

    # ------------------------------------------------------------------
    # 4. LOG-SPACE SMOOTHING (statt naivem Komponenten-Glätten)
    #    Wir glätten die RELATIVEN Rotationen zwischen Frames (kleine
    #    Winkel -> Axis-Angle ist dort stabil und ohne Manifold-Probleme),
    #    statt x/y/z/w einzeln zu glätten und danach zu renormalisieren.
    # ------------------------------------------------------------------
    quats_smoothed = np.zeros_like(quats_seq)
    for j in range(J):
        rots_j = R.from_quat(quats_seq[:, j])

        # Relative Rotation jedes Frames zum Vorframe, als Rotvec (T-1, 3)
        rel = (rots_j[:-1].inv() * rots_j[1:]).as_rotvec()

        # Diese "Rotations-Geschwindigkeit" glätten
        rel_smoothed = np.zeros_like(rel)
        for c in range(3):
            rel_smoothed[:, c] = savgol_filter(rel[:, c], window_length=window, polyorder=polyorder)

        # Rekonstruktion durch Aufakkumulieren ab dem ersten Frame
        rec = [rots_j[0]]
        for t in range(rel_smoothed.shape[0]):
            rec.append(rec[-1] * R.from_rotvec(rel_smoothed[t]))

        quats_smoothed[:, j] = R.concatenate(rec).as_quat()

    # Translation glätten (kein Manifold-Problem, klassisches Savgol reicht)
    trans_smoothed = np.zeros_like(trans_seq)
    for c in range(3):
        trans_smoothed[:, c] = savgol_filter(trans_seq[:, c], window_length=window, polyorder=polyorder)

    # ------------------------------------------------------------------
    # 5. ZURÜCK ZU AXIS-ANGLE FÜR SMPL
    # ------------------------------------------------------------------
    poses_seq = []
    for t in range(T):
        aa = R.from_quat(quats_smoothed[t]).as_rotvec()
        poses_seq.append(aa.reshape(-1))

    smpl_poses = np.stack(poses_seq, axis=0).astype(np.float32)
    smpl_trans = trans_smoothed.astype(np.float32)
    smpl_trans = smpl_trans - smpl_trans[0]
    smpl_scaling = np.array([1.0], dtype=np.float32)

    out_data = {
        "smpl_poses": smpl_poses,
        "smpl_trans": smpl_trans,
        "smpl_scaling": smpl_scaling,
    }

    joblib.dump(out_data, output_pkl)
    print(f"Successfully exported {smpl_poses.shape[0]} frames to {output_pkl}.")

def run_smpl2bvh(converted_pkl, fps, gender, output_bvh):
    """Calls the smpl2bvh tool."""
    print(f"[Step 4] Generating BVH file with {fps} FPS...")
    
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
    convert_pkl_structure(raw_pkl, temp_converted_pkl, fps, target_tid=args.tid)

    # 5. Convert to BVH
    run_smpl2bvh(temp_converted_pkl, fps, args.gender, args.output)

    # Clean up (optional)
    if os.path.exists(temp_converted_pkl):
        os.remove(temp_converted_pkl)

if __name__ == "__main__":
    main()