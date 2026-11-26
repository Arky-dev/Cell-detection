
import numpy as np
import ot
from scipy.spatial.distance import cdist
from process import * 
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
import cv2
from collections import defaultdict
import imageio.v2 as imageio
import os




def trajectoire_transport_optimal(cells_t, cells_next,
                max_distance=50,
                alpha=0.85,
                beta=0.15,
                reg=0.015,
                reg_m=0.8,
                min_fraction=0.2,
                frame_gaps=None):

    n, m = len(cells_t), len(cells_next)

    if n == 0 or m == 0:
        return [], np.zeros((n, m))

    pos_t = np.array([cell["centroid"] for cell in cells_t])
    pos_next = np.array([cell["centroid"] for cell in cells_next])
    area_t = np.array([cell["area"] for cell in cells_t])
    area_next = np.array([cell["area"] for cell in cells_next])

    if frame_gaps is None:
        frame_gaps = np.ones(n)

    max_distances = max_distance * np.asarray(frame_gaps)[:, None]

    a = area_t / area_t.sum()
    b = area_next / area_next.sum()

    distances = cdist(pos_t, pos_next)

    spatial_cost = distances / max_distances

    area_t_exp = area_t[:, None]
    area_next_exp = area_next[None, :]

    area_cost = np.abs(area_t_exp - area_next_exp) / (
        area_t_exp + area_next_exp
    )

    total_cost = alpha * spatial_cost + beta * area_cost

    plan = ot.unbalanced.sinkhorn_unbalanced(
        a, b, total_cost, reg, reg_m
    )

    plan[distances > max_distances] = 0

    row_sum = plan.sum(axis=1, keepdims=True)
    col_sum = plan.sum(axis=0, keepdims=True)

    row_fraction = plan / np.maximum(row_sum, 1e-12)
    col_fraction = plan / np.maximum(col_sum, 1e-12)

    matches = []

    for i in range(n):
        for j in range(m):
            if (row_fraction[i, j] >= min_fraction
                and col_fraction[i, j] >= min_fraction):

                matches.append((i, j))

    return matches, plan




def detect_fusion_division(matches, n, m):

    children = {i: [] for i in range(n)}
    parents = {j: [] for j in range(m)}

    for i, j in matches:
        children[i].append(j)
        parents[j].append(i)

    events = []

    for i, children_i in children.items():
        if len(children_i) > 1:
            events.append({
                "type": "division",
                "parent": i,
                "children": children_i
            })

    for j, parents_j in parents.items():
        if len(parents_j) > 1:
            events.append({
                "type": "fusion",
                "parents": parents_j,
                "child": j
            })

    return events



def visualize_matching(frame_t, frame_next, cells_t, cells_next, matches):

    fig, ax = plt.subplots(figsize=(16, 8))

    h, w = frame_t.shape[:2]
    ax.imshow(frame_t, cmap="gray", extent=[0, w, h, 0])
    ax.imshow(frame_next, cmap="gray", extent=[w, 2*w, h, 0])

    ax.axvline(w, color="white", linewidth=2)
    ax.text(w/2, -10, "Frame 0", ha="center")
    ax.text(1.5*w, -10, "Frame 1", ha="center")

    colors = plt.cm.tab20(
        np.linspace(0, 1, max(len(matches), 1))
    )

    matched_t = set()
    matched_next = set()

    for k, (i, j) in enumerate(matches):

        color = colors[k]
        x1, y1 = cells_t[i]["centroid"]
        x2, y2 = cells_next[j]["centroid"]
        # Cellules sur les deux images
        ax.add_patch(Circle((x1, y1), 8, fill=False,
                            color=color, linewidth=2))

        ax.add_patch(Circle((x2+w, y2), 8, fill=False,
                            color=color, linewidth=2))

        # Trait reliant les deux cellules
        ax.plot([x1, x2+w], [y1, y2],
                color=color, linewidth=1.5, alpha=0.8)

        matched_t.add(i)
        matched_next.add(j)

    # Cellules non matchées
    for i, cell in enumerate(cells_t):
        if i not in matched_t:
            x, y = cell["centroid"]
            ax.plot(x, y, "o", color="red", markersize=4)

    for j, cell in enumerate(cells_next):
        if j not in matched_next:
            x, y = cell["centroid"]
            ax.plot(x+w, y, "o", color="red", markersize=4)

    ax.set_xlim(0, 2*w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    plt.tight_layout()
    plt.show()



def full_trajectory(all_detections, max_missing=2):

    trajectories = []
    next_id = 0

    for i, cell in enumerate(all_detections[0]):
        traj = {
            "id": next_id,
            "parent_id": None,
            "positions": [cell["centroid"]],
            "frames": [0],
            "cells": [(0, i)]
        }
        trajectories.append(traj)
        next_id += 1

    for t in range(1, len(all_detections)):
        print("Frame", t)
        cells_next = all_detections[t]
        # Trajectoires pouvant encore être reconnectées
        active_trajectories = [
            traj for traj in trajectories
            if t - traj["frames"][-1] <= max_missing + 1
        ]
        candidate_cells = []
        frame_gaps = []
        for traj in active_trajectories:
            last_frame, last_index = traj["cells"][-1]
            candidate_cells.append(
                all_detections[last_frame][last_index]
            )
            frame_gaps.append(t - last_frame)

        matches, plan = trajectoire_transport_optimal(
            candidate_cells,
            cells_next,
            frame_gaps=frame_gaps
        )

        children = {i: [] for i in range(len(candidate_cells))}
        parents = {j: [] for j in range(len(cells_next))}
        for i, j in matches:
            children[i].append(j)
            parents[j].append(i)
        best_parent = {}

        for j, parents_j in parents.items():
            if parents_j:
                best_parent[j] = max(
                    parents_j,
                    key=lambda i: plan[i, j]
                )

        best_child = {}
        for i, children_i in children.items():
            selected_children = [
                j for j in children_i
                if best_parent.get(j) == i
            ]
            if selected_children:
                best_child[i] = max(
                    selected_children,
                    key=lambda j: plan[i, j]
                )

        for j, cell in enumerate(cells_next):
            if j in best_parent:
                i = best_parent[j]
                parent_traj = active_trajectories[i]
                if best_child.get(i) == j:
                    traj = parent_traj
                    traj["positions"].append(cell["centroid"])
                    traj["frames"].append(t)
                    traj["cells"].append((t, j))

                else:
                    traj = {
                        "id": next_id,
                        "parent_id": parent_traj["id"],
                        "positions": [cell["centroid"]],
                        "frames": [t],
                        "cells": [(t, j)]
                    }
                    trajectories.append(traj)
                    next_id += 1

            else:
                traj = {
                    "id": next_id,
                    "parent_id": None,
                    "positions": [cell["centroid"]],
                    "frames": [t],
                    "cells": [(t, j)]
                }
                trajectories.append(traj)
                next_id += 1

    return trajectories




def visualize_trajectories(stack, trajectories, min_length=5, output="trajectories.mp4", fps=10, line_width=2, cell_radius=3):
    trajectories = [traj for traj in trajectories if len(traj["frames"]) >= min_length]
    n = len(trajectories)
    colors = {}
    rng = np.random.default_rng(42)
    for traj in trajectories:
        color = rng.integers(50, 256, size=3)
        colors[traj["id"]] = tuple(map(int, color))

    events = defaultdict(list)
    for traj in trajectories:
        frames = traj["frames"]
        positions = np.array(traj["positions"], dtype=int)
        color = colors[traj["id"]]
        events[frames[0]].append(("point", positions[0], color, traj["id"]))
        for k in range(1, len(frames)):
            events[frames[k]].append(("segment", positions[k-1], positions[k], color))

    children_by_parent = defaultdict(list)
    for traj in trajectories:
        if traj.get("parent_id") is not None:
            children_by_parent[traj["parent_id"]].append(traj)

    print("\nDivisions détectées")
    division_count = 0
    for parent_id, children in children_by_parent.items():
        division_count += 1
        print(f"\nDivision de la trajectoire {parent_id}")
        for child in children:
            x, y = child["positions"][0]
            print(f"  -> Trajectoire fille {child['id']} : frame {child['frames'][0]}, coordonnées ({x:.1f}, {y:.1f})")
    if division_count == 0:
        print("Aucune division détectée.")
    print(f"\nNombre total de divisions : {division_count}")

    height, width = stack[0].shape[:2]
    if stack.dtype != np.uint8:
        vmin, vmax = np.min(stack), np.max(stack)
        stack_8bit = np.clip((stack.astype(np.float32) - vmin) * (255.0 / (vmax - vmin)), 0, 255).astype(np.uint8) if vmax > vmin else np.zeros(stack.shape, dtype=np.uint8)
    else:
        stack_8bit = stack

    extension = os.path.splitext(output)[1].lower()
    is_folder = extension == ""
    is_gif = extension == ".gif"

    if is_folder:
        os.makedirs(output, exist_ok=True)
    elif is_gif:
        writer = imageio.get_writer(output, mode="I", fps=fps, loop=0)
    else:
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    try:
        for t in range(len(stack)):
            frame = stack_8bit[t]
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            overlay = np.zeros((height, width, 3), dtype=np.uint8)

            for traj in trajectories:
                frames = np.array(traj["frames"])
                positions = np.array(traj["positions"], dtype=int)
                mask = frames <= t
                if not np.any(mask):
                    continue
                last_idx = np.where(mask)[0][-1]
                last_frame = frames[last_idx]
                if t - last_frame > 2:
                    continue
                visible_positions = positions[mask]
                color = colors[traj["id"]]
                for k in range(1, len(visible_positions)):
                    cv2.line(overlay, tuple(visible_positions[k-1]), tuple(visible_positions[k]), color, line_width, cv2.LINE_AA)
                for pos in visible_positions:
                    cv2.circle(overlay, tuple(pos), cell_radius, color, -1, cv2.LINE_AA)
                x, y = visible_positions[0]
                cv2.putText(overlay, str(traj["id"]), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            result = cv2.addWeighted(frame, 1.0, overlay, 0.85, 0)
            cv2.putText(result, f"Frame {t}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if is_folder:
                filename = os.path.join(output, f"frame_{t:04d}.png")
                cv2.imwrite(filename, result)
            elif is_gif:
                writer.append_data(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                writer.write(result)

            if t % 20 == 0:
                print(f"Export : {t}/{len(stack)} frames")

    finally:
        if not is_folder:
            if is_gif:
                writer.close()
            else:
                writer.release()

    if is_folder:
        print(f"\nImages sauvegardées dans : {output}")
    else:
        print(f"\nVidéo sauvegardée : {output}")



"""


if __name__ == "__main__":
    tif_path = "../data/_1_MMStack_A1_1.ome.tif"
    stack = load_stack(tif_path, channel=0, n_channels=1)
    with open("all_detection.pkl", "rb") as f:
        all_detections = pickle.load(f)
    cells_t = all_detections[39]
    cells_next = all_detections[40]
    matches, plan = trajectoire_transport_optimal(cells_t, cells_next)
    visualize_matching(
        stack[39],
        stack[40],
        cells_t,
        cells_next,
        matches
    )
    print (matches, plan)
    

"""
if __name__ == "__main__":
    tif_path = "../data/_1_MMStack_A1_1.ome.tif"
    stack = load_stack(tif_path, channel=0, n_channels=1)
    with open("all_detection.pkl", "rb") as f:
        all_detections = pickle.load(f)
    trajectories = full_trajectory(all_detections, max_missing=2)
    trajectories = [t for t in trajectories if len(t["frames"])> 20]
    visualize_trajectories(
        stack,
        trajectories,
        min_length=5,
        output="trajectories",
        fps=1
    )
    





