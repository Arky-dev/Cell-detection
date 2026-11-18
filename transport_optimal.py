
import numpy as np
import ot
from scipy.spatial.distance import cdist
from process import * 
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm

# Avec les cellules données comme ça :  [x, y, area]

def trajectoire_transport_optimal(cells_t, cells_next,
                max_distance=50,
                alpha=0.8,
                beta=0.2,
                reg=0.01,
                reg_m=1.0,
                min_fraction=0.15):

    pos_t = np.array([cell["centroid"] for cell in cells_t])
    pos_next = np.array([cell["centroid"] for cell in cells_next])
    area_t = np.array([cell["area"] for cell in cells_t])
    area_next = np.array([cell["area"] for cell in cells_next])

    n, m = len(cells_t), len(cells_next)
    if n == 0 or m == 0:
        return []

    a = area_t / area_t.sum()
    b = area_next / area_next.sum()

    distances = cdist(pos_t, pos_next)

    spatial_cost = distances / max_distance
    area_t = np.expand_dims(area_t, axis=1)
    area_next = np.expand_dims(area_next, axis=0)
    area_cost = np.abs(area_t - area_next) / (area_t + area_next)
    total_cost = alpha * spatial_cost + beta * area_cost

    plan = ot.unbalanced.sinkhorn_unbalanced(
        a, b, total_cost, reg, reg_m
    )
    plan[distances > max_distance] = 0

    
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


def full_trajectory(all_detections):

    trajectories = []
    next_id = 0

    # Initialisation de la première frame
    cell_to_trajectory = {}

    for i, cell in enumerate(all_detections[0]):

        traj = {
            "id": next_id,
            "parent_id": None,
            "positions": [cell["centroid"]],
            "frames": [0],
            "cells": [(0, i)]
        }

        trajectories.append(traj)
        cell_to_trajectory[i] = traj
        next_id += 1

    # Parcours des frames
    for t in range(len(all_detections) - 1):

        print("Frame", t + 1)

        cells_t = all_detections[t]
        cells_next = all_detections[t + 1]

        matches, plan = trajectoire_transport_optimal(
            cells_t, cells_next
        )

        # Associations parent -> enfants et enfant -> parents
        children = {i: [] for i in range(len(cells_t))}
        parents = {j: [] for j in range(len(cells_next))}

        for i, j in matches:
            children[i].append(j)
            parents[j].append(i)

        # Pour chaque enfant, choisir son meilleur parent
        best_parent = {}

        for j, parents_j in parents.items():

            if parents_j:
                best_parent[j] = max(
                    parents_j,
                    key=lambda i: plan[i, j]
                )

        # Pour chaque parent, choisir son meilleur enfant
        # parmi ceux qui l'ont choisi comme meilleur parent
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

        # Construire les trajectoires de la frame suivante
        next_cell_to_trajectory = {}

        for j, cell in enumerate(cells_next):

            if j in best_parent:

                i = best_parent[j]

                if best_child.get(i) == j:
                    # Continuité de trajectoire
                    traj = cell_to_trajectory[i]

                    traj["positions"].append(cell["centroid"])
                    traj["frames"].append(t + 1)
                    traj["cells"].append((t + 1, j))

                else:
                    # Division ou association secondaire :
                    # nouvelle trajectoire
                    parent_traj = cell_to_trajectory[i]

                    traj = {
                        "id": next_id,
                        "parent_id": parent_traj["id"],
                        "positions": [cell["centroid"]],
                        "frames": [t + 1],
                        "cells": [(t + 1, j)]
                    }

                    trajectories.append(traj)
                    next_id += 1

            else:
                # Cellule sans parent détecté
                traj = {
                    "id": next_id,
                    "parent_id": None,
                    "positions": [cell["centroid"]],
                    "frames": [t + 1],
                    "cells": [(t + 1, j)]
                }

                trajectories.append(traj)
                next_id += 1

            next_cell_to_trajectory[j] = traj

        cell_to_trajectory = next_cell_to_trajectory

    return trajectories


def visualize_trajectories(frame, trajectories, min_length=5):

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(frame, cmap="gray")

    # Filtrer les trajectoires
    trajectories = [
        traj for traj in trajectories
        if len(traj["frames"]) > min_length and traj["frames"][0] == 0
    ]

    colors = plt.cm.nipy_spectral(
        np.linspace(0, 1, len(trajectories))
    )

    for traj, color in zip(trajectories, colors):

        positions = np.array(traj["positions"])

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            "-o",
            color=color,
            markersize=2,
            linewidth=1.2,
            alpha=0.8
        )

        # Identifiant au début de la trajectoire
        ax.text(
            positions[0, 0],
            positions[0, 1],
            str(traj["id"]),
            color=color,
            fontsize=7
        )

    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tif_path = "../data/_1_MMStack_A1_1.ome.tif"
    stack = load_stack(tif_path, channel=0, n_channels=1)
    with open("all_detection.pkl", "rb") as f:
        all_detections = pickle.load(f)
    cells_t = all_detections[14]
    cells_next = all_detections[15]
    matches, plan = trajectoire_transport_optimal(cells_t, cells_next)
    visualize_matching(
        stack[14],
        stack[15],
        cells_t,
        cells_next,
        matches
    )
    

"""
if __name__ == "__main__":
    tif_path = "../data/_1_MMStack_A1_1.ome.tif"
    stack = load_stack(tif_path, channel=0, n_channels=1)
    with open("all_detection.pkl", "rb") as f:
        all_detections = pickle.load(f)
    trajectories = full_trajectory(all_detections)
    print( [
            traj for traj in trajectories
            if len(traj["frames"]) > 5 and traj["frames"][0] == 0
        ])
    visualize_trajectories(stack[0], trajectories, 5)

 """   




