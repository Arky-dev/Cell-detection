import os
import cv2
import pandas as pd

def extract_tracking_with_drops(mdf_path, image_folder, output_folder, track_to_keep=1, window_size=100, modified = False):
    """
    On traite un fichier .mdf généré par MTrackJ et on coupe les images autour des coordonnées dans la BDD
    """
    os.makedirs(output_folder, exist_ok=True)

    with open(mdf_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = [line.strip().split() for line in lines if line.startswith("Point")]

    if not data_lines:
        print("Erreur dans la lecture du fichier")
        return

    df = pd.DataFrame(data_lines)
    
    n_cols = len(df.columns)
    default_names = ["Type", "Frame_ID", "X", "Y", "Z", "T", "Track_ID", "Extra_1"]
    df.columns = default_names[:n_cols]
    
    for col in df.columns:
        if col != "Type":
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # On veut séparer les cellules en regardant quand la Frame_ID diminue
    track_ids = []
    current_track = 1
    prev_frame = -1
    
    for _, row in df.iterrows():
        current_frame = row['Frame_ID']
        
        if current_frame < prev_frame and prev_frame != -1:
            current_track += 1
            
        track_ids.append(current_track)
        prev_frame = current_frame

    df['Custom_Track_ID'] = track_ids
    
    original_count = len(df)
    df_filtered = df[df["Custom_Track_ID"] == track_to_keep]
    
    print(f"{original_count} point détectés.")
    print(f"{len(track_ids)} trajectoires détectées.")
    print(f"Conservation de la trajectoire {track_to_keep}: {len(df_filtered)} points conservés.")

    if df_filtered.empty:
        print(f"Aucun point pour la trajectoire {track_to_keep}.")
        return

    half_window = window_size // 2

    for index, row in df_filtered.iterrows():
        frame_idx = int(row['T'])
        print(row)
        
        center_x = int(round(row['X']))
        center_y = int(round(row['Y']))

        if modified : 
            img_filename = f"page_{frame_idx}_modified.jpeg"  
        else : 
            img_filename = f"page_{frame_idx}.jpeg" 
        img_path = os.path.join(image_folder, img_filename)

        if not os.path.exists(img_path):
            print(f"Impossible de trouver l'image {frame_idx} en {img_path}. Image sautée.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Erreur dans la lecture de l'image {img_path}")
            continue

        height, width = img.shape[:2]

        x_min = max(0, center_x - half_window)
        x_max = min(width, center_x + half_window)
        y_min = max(0, center_y - half_window)
        y_max = min(height, center_y + half_window)

        patch = img[y_min:y_max, x_min:x_max]

        if modified : 
            out_filename = f"modified_track_{track_to_keep}_frame_{frame_idx}.png"
        else : 
            out_filename = f"track_{track_to_keep}_frame_{frame_idx}.png"
        out_path = os.path.join(output_folder, out_filename)
        cv2.imwrite(out_path, patch)

if __name__ == "__main__":
    MDF_PATH = "../data/Analysis and stats/_1_MMStack_A1_1.ome.tif.mdf"
    IMAGE_FOLDER = "../data/_1_MMStack_A1_1"
    OUTPUT_FOLDER = "../data/tracked_crops_modified"    
    
    extract_tracking_with_drops(
        mdf_path=MDF_PATH, 
        image_folder=IMAGE_FOLDER, 
        output_folder=OUTPUT_FOLDER, 
        track_to_keep=2,
        window_size=150, 
        modified = True
    )