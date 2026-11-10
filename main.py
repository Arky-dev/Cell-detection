import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import tifffile as tiff

# image = cv2.imread(r'C:\Users\paula\Documents\[1] School\[0] X\[3] 3A\Computer Vision\Cell detection\data\test.jpeg', cv2.IMREAD_GRAYSCALE)


def show_image(img, title):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()


# laplacian = cv2.Laplacian(image,cv2.CV_64F)
# sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
# image = cv2.magnitude(sobelx, sobely).astype(np.uint8)

# show_image(image, 'coucou Nadouge')

# L = len(image)
# N = 8

# for i in range(N):
#     for j in range(N):
#         subimg = image[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N]
#         ret, otsu_thresh = cv2.threshold(
#             subimg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         if np.sum(otsu_thresh) < 255*0.25*(L//N)**2:
#             image[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N] = otsu_thresh
#         else:
#             image[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N] = np.zeros_like(image[i*L//N:(i+1)*L//N, j*L//N:(j+1)*L//N])

# show_image(image, 'coucou Nadouge encore')


	
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



def convertToJpeg(filepath):
    outfolder = filepath.split('.')[0]
    os.makedirs(outfolder, exist_ok=True)

    im = Image.open(filepath)

    with tiff.TiffFile(filepath) as tif:
        for i, page in enumerate(tif.pages):
            # Extract the page data as a 16-bit numpy array
            data = page.asarray()
            
            # Save as an individual 16-bit TIFF
            output_name = f"page_{i}.jpeg"
            tiff.imwrite(os.path.join(outfolder, output_name), data, photometric='minisblack')
            print(f"Saved {output_name} with shape {data.shape} and dtype {data.dtype}")




if __name__ == "__main__":
    convertToJpeg("../data/_1_MMStack_A1_1.ome.tif")