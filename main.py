import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tifffile as tiff


def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def modify_image(path_in, path_out, N = 4) : 
    image = cv2.imdecode(np.fromfile(path_in, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    print(path_in)
    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    image = cv2.magnitude(sobelx, sobely).astype(np.uint8)
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
             vote1[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N] = otsu_thresh

    for i in range(N - 1):
        for j in range(N - 1):
            subimg = image[L//(2*N) + i*L//N:L//(2*N) +(i+1)*L//N, L//(2*N) + j*L//N:L//(2*N) +(j+1)*L//N]
            ret, otsu_thresh = cv2.threshold(
                subimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            vote2[L//(2*N) + i*L//N:L//(2*N)+(i+1)*L//N, L//(2*N) + j*L//N:L//(2*N)+(j+1)*L//N] = otsu_thresh

    for i in range(N):
            for j in range(N - 1):
                subimg = image[ i*L//N: (i+1)*L//N, L//(2*N) + j*L//N:L//(2*N) +(j+1)*L//N]
                ret, otsu_thresh = cv2.threshold(
                    subimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                vote3[ i*L//N:(i+1)*L//N, L//(2*N) + j*L//N:L//(2*N)+(j+1)*L//N] = otsu_thresh

    for i in range(N - 1):
            for j in range(N):
                subimg = image[L//(2*N) + i*L//N:L//(2*N) +(i+1)*L//N,  j*L//N:(j+1)*L//N]
                ret, otsu_thresh = cv2.threshold(
                    subimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    image[:largeur, :largeur] = vote1[:largeur, :largeur]
    image[:largeur, -largeur:] = vote1[:largeur, -largeur:]
    image[-largeur:, :largeur] = vote1[-largeur:, :largeur]
    image[-largeur:, -largeur:] = vote1[-largeur:, -largeur:]

    cv2.imwrite(path_out, image)



        
    # _, thresh_img190 = cv2.threshold(sobel, 190, 255, cv2.THRESH_BINARY)
    # show_image(thresh_img190, "Otsu’s Thresholding")


# plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

# plt.show()



def convertToJpeg(filepath, outfolder = None, modify = True):
    if outfolder is None : 
        outfolder = filepath.split('.')[0]
    os.makedirs(outfolder, exist_ok=True)

    im = Image.open(filepath)

    with tiff.TiffFile(filepath) as tif:
        for i, page in enumerate(tif.pages):
            # Extract the page data as a 16-bit numpy array
            data = page.asarray()
            
            # Save as an individual 16-bit TIFF
            output_name = f"page_{i}.jpeg"
            output_name2 = f"page_{i}_modified.jpeg"
            tiff.imwrite(os.path.join(outfolder, output_name2), data, photometric='minisblack')
            if modify : 
                modify_image(os.path.join(outfolder, output_name2), os.path.join(outfolder, output_name2), 4)
            print(f"Saved {output_name} with shape {data.shape} and dtype {data.dtype}")




if __name__ == "__main__":
    #modify_image("../data/_1_MMStack_A1_1/page_1.jpeg", "../data/modified_1_MMStack_A1_1/page_1.jpeg")
    convertToJpeg("../data/_1_MMStack_A1_1.ome.tif", "../data/modified_1_MMStack_A1_1", True)