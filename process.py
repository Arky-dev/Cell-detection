import cv2
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import tqdm



def visualize_detections(frame, detections, title="Détections de l'image"):

    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(frame, cmap="gray")

    for det in detections:
        x, y, w, h = det["bbox"]
        cx, cy = det["centroid"]
        area = int(det["area"])

        ax.add_patch(Rectangle((x, y), w, h, linewidth=1.2, edgecolor="lime", facecolor="none"))
        ax.plot(cx, cy, "ro", markersize=3)
        ax.text(x, y - 4, f"{area}px", color="yellow", fontsize=7)

    ax.set_title(f"{title} — {len(detections)} cellules trouvées")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def load_stack(filepath, channel=0, n_channels=1):

    with tiff.TiffFile(filepath, is_ome=False, is_mmstack=False) as tif:
        pages = tif.pages[channel::n_channels]
        raw_stack = np.stack([p.asarray() for p in pages], axis=0)

    p_low, p_high = np.percentile(raw_stack[:, ::4, ::4], (1, 99.8))

    vals = np.arange(65536, dtype=np.float32)
    lut = np.clip((vals - p_low) / (p_high - p_low + 1e-8) * 255, 0, 255).astype(np.uint8)

    return lut[raw_stack]


def binarize(frame, N=8):

    L = len(frame)

    for i in range(N):
        for j in range(N):
            subimg = frame[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N]
            ret, otsu_thresh = cv2.threshold(
                subimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.corrcoef(otsu_thresh[:-1,:].flatten(), otsu_thresh[1:,:].flatten())[0, 1] >= 0.5:
                frame[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N] = otsu_thresh
            else:
                frame[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N] = np.zeros_like(frame[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N])

    return frame


def binarize_glissante(image, N=4):

    L = len(image)
    vote1 = np.zeros_like(image)
    vote2 = np.zeros_like(image)
    vote3 = np.zeros_like(image)
    vote4 = np.zeros_like(image)

    for i in range(N):
        for j in range(N):
                subimg = image[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N]
                ret, otsu_thresh = cv2.threshold(
                    subimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                if np.corrcoef(otsu_thresh[:-1,:].flatten(), otsu_thresh[1:,:].flatten())[0, 1] >= 0.5:
                    vote1[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N] = otsu_thresh

    for i in range(N - 1):
        for j in range(N - 1):
            subimg = image[L//(2*N) + i*L//N:L//(2*N) +(i+1)*L//N, L//(2*N) + j*L//N:L//(2*N) +(j+1)*L//N]
            ret, otsu_thresh = cv2.threshold(
                subimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.corrcoef(otsu_thresh[:-1,:].flatten(), otsu_thresh[1:,:].flatten())[0, 1] >= 0.5:
                vote2[L//(2*N) + i*L//N:L//(2*N)+(i+1)*L//N, L//(2*N) + j*L//N:L//(2*N)+(j+1)*L//N] = otsu_thresh

    for i in range(N):
        for j in range(N - 1):
            subimg = image[ i*L//N: (i+1)*L//N, L//(2*N) + j*L//N:L//(2*N) +(j+1)*L//N]
            ret, otsu_thresh = cv2.threshold(
                subimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.corrcoef(otsu_thresh[:-1,:].flatten(), otsu_thresh[1:,:].flatten())[0, 1] >= 0.5:
                vote3[ i*L//N:(i+1)*L//N, L//(2*N) + j*L//N:L//(2*N)+(j+1)*L//N] = otsu_thresh

    for i in range(N - 1):
        for j in range(N):
            subimg = image[L//(2*N) + i*L//N:L//(2*N) +(i+1)*L//N,  j*L//N:(j+1)*L//N]
            ret, otsu_thresh = cv2.threshold(
                subimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.corrcoef(otsu_thresh[:-1,:].flatten(), otsu_thresh[1:,:].flatten())[0, 1] >= 0.5:
                vote4[L//(2*N) + i*L//N:L//(2*N)+(i+1)*L//N, j*L//N:(j+1)*L//N] = otsu_thresh

    
    votes = ((vote1 == 255).astype(int) +
            (vote2 == 255).astype(int) +
            (vote3 == 255).astype(int) +
            (vote4 == 255).astype(int))

    largeur = L // (2*N)

    mask_bord = np.zeros_like(votes, dtype=bool)
    mask_bord[:, :largeur] = True
    mask_bord[:, -largeur:] = True
    mask_bord[:largeur, :] = True
    mask_bord[-largeur:, :] = True

    mask = np.where(mask_bord, votes >= 2, votes >= 3)

    image = np.zeros_like(image)
    image[mask] = 255

    #les coins
    image[:largeur, :largeur] = vote1[:largeur, :largeur]
    image[:largeur, -largeur:] = vote1[:largeur, -largeur:]
    image[-largeur:, :largeur] = vote1[-largeur:, :largeur]
    image[-largeur:, -largeur:] = vote1[-largeur:, -largeur:]

    return image


def detect_cells(frame, background, min_area=400, max_area=12000, min_circularity=0.45, etapes=False):

    diff = np.abs(frame.astype(np.float32) - background).astype(np.uint8)

    sobelx = cv2.Sobel(diff, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(diff, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    grad_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    binary = binarize_glissante(grad_norm)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    filled = binary_fill_holes(closed > 0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (min_area <= area <= max_area):
            perimeter = cv2.arcLength(cnt, True)
            if perimeter != 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity >= min_circularity:
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    bbox = cv2.boundingRect(cnt)
                    detections.append({"centroid": (cx, cy), "bbox": bbox, "area": area})

    if etapes:
        stages = {
            "diff": diff,
            "grad_norm": grad_norm,
            "mask": mask,
        }
        return detections, stages

    return detections


def visualize_debug_stages(frame, detections, stages):
    _, axes = plt.subplots(2, 2, figsize=(14, 14))

    axes[0, 0].imshow(frame, cmap="gray")
    
    for det in detections:
        x, y, w, h = det["bbox"]
        cx, cy = det["centroid"]
        axes[0, 0].add_patch(Rectangle((x, y), w, h, linewidth=1.2, edgecolor="lime", facecolor="none"))
        axes[0, 0].plot(cx, cy, "ro", markersize=3)
        axes[0, 0].text(x, y - 4, f"{int(det['area'])}px", color="yellow", fontsize=7)

    axes[0, 0].set_title(f"1. Final Detections ({len(detections)} cells)")

    axes[0, 1].imshow(stages["diff"], cmap="gray")
    axes[0, 1].set_title("2. Background Subtraction (|Frame - Median|)")

    axes[1, 0].imshow(stages["grad_norm"], cmap="gray")
    axes[1, 0].set_title("3. Normalized Sobel Gradient")

    axes[1, 1].imshow(stages["mask"], cmap="gray")
    axes[1, 1].set_title("4. Binary Mask (After Morphology & Hole Fill)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def track_cells(detections_per_frame, max_dist=35.0, max_missed=2, min_track_len=5):

    active_tracks = []
    finished_tracks = []
    next_id = 0

    for frame_idx, detections in enumerate(detections_per_frame):
        det_coords = np.array([d["centroid"] for d in detections]) if detections else np.empty((0, 2))

        if len(active_tracks) > 0 and len(det_coords) > 0:
            track_coords = np.array([t["positions"][-1][1:] for t in active_tracks])
            cost_matrix = cdist(track_coords, det_coords)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_tracks, matched_dets = set(), set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= max_dist:
                    cx, cy = det_coords[c]
                    active_tracks[r]["positions"].append((frame_idx, cx, cy))
                    active_tracks[r]["missed"] = 0
                    matched_tracks.add(r)
                    matched_dets.add(c)

            still_active = []
            for idx, track in enumerate(active_tracks):
                if idx not in matched_tracks:
                    track["missed"] += 1
                if track["missed"] <= max_missed:
                    still_active.append(track)
                else:
                    finished_tracks.append(track)
            active_tracks = still_active

            for idx, det in enumerate(detections):
                if idx not in matched_dets:
                    cx, cy = det["centroid"]
                    active_tracks.append({
                        "id": next_id,
                        "positions": [(frame_idx, cx, cy)],
                        "missed": 0
                    })
                    next_id += 1

        elif len(det_coords) > 0:
            for det in detections:
                cx, cy = det["centroid"]
                active_tracks.append({
                    "id": next_id,
                    "positions": [(frame_idx, cx, cy)],
                    "missed": 0
                })
                next_id += 1
        else:
            for track in active_tracks:
                track["missed"] += 1
            finished_tracks.extend([t for t in active_tracks if t["missed"] > max_missed])
            active_tracks = [t for t in active_tracks if t["missed"] <= max_missed]

    all_tracks = finished_tracks + active_tracks
    return [t for t in all_tracks if len(t["positions"]) >= min_track_len]


def save_tracking_video(stack, tracks, output_path="tracked_cells.mp4", fps=15, tail_length=25):
    """Render an MP4 video showing cell centroids, IDs, and trailing trajectories."""
    n_frames, h, w = stack.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Assign a distinct BGR color to each track ID
    rng = np.random.default_rng(42)
    colors = {tr["id"]: tuple(int(c) for c in rng.integers(80, 255, size=3)) for tr in tracks}

    # Index track history by frame for fast lookup
    frame_lookup = {f: [] for f in range(n_frames)}
    for tr in tracks:
        positions = tr["positions"]
        for idx, (f, x, y) in enumerate(positions):
            start = max(0, idx - tail_length)
            trail = [(int(round(px)), int(round(py))) for _, px, py in positions[start : idx + 1]]
            frame_lookup[f].append((tr["id"], trail))

    for f in range(n_frames):
        # Convert grayscale frame to BGR color so we can draw colored trails
        canvas = cv2.cvtColor(stack[f], cv2.COLOR_GRAY2BGR)

        for track_id, trail in frame_lookup[f]:
            color = colors[track_id]
            if len(trail) >= 2:
                pts = np.array(trail, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2)

            cx, cy = trail[-1]
            cv2.circle(canvas, (cx, cy), radius=4, color=color, thickness=-1)
            cv2.putText(
                canvas,
                f"ID {track_id}",
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"Frame {f + 1}/{n_frames}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        writer.write(canvas)

    writer.release()
    print(f"Saved tracking video to {output_path}")



if __name__ == "__main__":
    tif_path = "../data/_1_MMStack_A1_1.ome.tif"

    stack = load_stack(tif_path, channel=0, n_channels=1)
    background = np.median(stack, axis=0).astype(np.float32)

    dets, stages = detect_cells(
        stack[41],
        background,
        min_area=400,
        max_area=12000,
        min_circularity=0.45,
        etapes=True,
    )
    visualize_detections(stack[41], dets)

    visualize_debug_stages(stack[41], dets, stages)

    all_detections = [detect_cells(frame, background) for frame in tqdm.tqdm(stack)]
    
    #tracks = track_cells(all_detections, max_dist=35.0, max_missed=2, min_track_len=5)
    #tracks = track_cells(all_detections, max_dist=35.0, max_missed=2, min_track_len=5)
    #save_tracking_video(stack, tracks, output_path="../data/tracked.mp4", fps=5, tail_length=25)