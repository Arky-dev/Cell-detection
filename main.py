import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
import tifffile as tiff
from scipy.ndimage.measurements import label
from scipy.ndimage import gaussian_filter

image = cv2.imread(r'C:\Users\paula\Documents\[1] School\[0] X\[3] 3A\Computer Vision\Cell detection\page_2.jpeg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread(r'C:\Users\paula\Documents\[1] School\[0] X\[3] 3A\Computer Vision\Cell detection\page_2.jpeg', cv2.IMREAD_GRAYSCALE)


def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


def full_process(image, name, out_folder):

    show_image(image, '')

    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    mag = cv2.magnitude(sobelx, sobely)
    image2 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    show_image(image2, '')

    L = len(image2)
    N = 8

    for i in range(N):
        for j in range(N):
            subimg = image2[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N]
            ret, otsu_thresh = cv2.threshold(
                subimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.corrcoef(otsu_thresh[:-1,:].flatten(), otsu_thresh[1:,:].flatten())[0, 1] >= 0.5:
                image2[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N] = otsu_thresh
            else:
                image2[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N] = np.zeros_like(image2[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N])
    show_image(image2, '')

    blurred = gaussian_filter(image2, sigma=4)
    show_image(image2, '')

    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image2 = otsu_thresh
    show_image(image2, '')

    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, ncomponents = label(image2, structure)

    indices = np.indices(image2.shape).T[:,:,[1, 0]]

    plt.imshow(image, cmap='gray')
    for index in range(1, np.max(labeled)+1):
        indexes = indices[labeled == index]
        mini_y = min([e[0] for e in indexes])
        mini_x = min([e[1] for e in indexes])
        maxi_y = max([e[0] for e in indexes])
        maxi_x = max([e[1] for e in indexes])

        width = maxi_x-mini_x
        height = maxi_y-mini_y

        if 30 <= width < 150 and 30 <= height < 150:
            plt.gca().add_patch(Rectangle((mini_x,mini_y),width,height,linewidth=1,edgecolor='red',facecolor='none'))

    plt.show()
    # plt.savefig(os.path.join(out_folder, name+'.png'))
    plt.cla()

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

    #les coins
    image[:largeur, :largeur] = vote1[:largeur, :largeur]
    image[:largeur, -largeur:] = vote1[:largeur, -largeur:]
    image[-largeur:, :largeur] = vote1[-largeur:, :largeur]
    image[-largeur:, -largeur:] = vote1[-largeur:, -largeur:]


    image = gaussian_filter(image, sigma = 4)
    ret, image = cv2.threshold(
                        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


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
    # FOLDER_PATH = r"C:\Users\paula\Documents\[1] School\[0] X\[3] 3A\Computer Vision\data\_1_MMStack_A1_1"
    # OUT_FOLDER = r"C:\Users\paula\Documents\[1] School\[0] X\[3] 3A\Computer Vision\data\BB_detected"

    # os.makedirs(OUT_FOLDER, exist_ok=True)

    # for filename in os.listdir(FOLDER_PATH):
    #     print(filename)
    #     image = cv2.imread(os.path.join(FOLDER_PATH, filename), cv2.IMREAD_GRAYSCALE)
    #     full_process(image, filename.split('.')[0], OUT_FOLDER)

    full_process(image, 'test', 'efgz')

    
    # convertToJpeg(r"C:\Users\paula\Documents\[1] School\[0] X\[3] 3A\Computer Vision\Cell detection\data\_1_MMStack_A1_1.ome.tif")