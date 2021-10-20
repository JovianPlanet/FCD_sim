import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import gaussian_filter

def plot_slice(prueba_img, test_mask_img, predictions_img):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=1 ,wspace=1)

    ax1 = fig.add_subplot(1, 3, 1)
    #ax1.title.set_text('imagen')
    ax1.axis("off")
    ax1.imshow(prueba_img, cmap="gray")

    ax2 = fig.add_subplot(1, 3, 2)
    #ax2.title.set_text('mask')
    ax2.axis("off")
    ax2.imshow(test_mask_img, cmap="gray")

    ax3 = fig.add_subplot(1, 3, 3)
    #ax3.title.set_text('prediccion')
    ax3.axis("off")
    ax3.imshow(predictions_img, cmap="gray")

    plt.show()


#np.random.seed(1)
tamano = 5
intensidad = 1.3
img_path = 'temp_ds'
original = nib.load(os.path.join(img_path, 'T1.nii')).get_fdata()
orig_flair = nib.load(os.path.join(img_path, 'T2_FLAIR.nii')).get_fdata()
gm = nib.load(os.path.join(img_path, 'unet2D_predictions_GM.nii')).get_fdata() # Mascara de la materia gris
wm = nib.load(os.path.join(img_path, 'unet2D_predictions_WM.nii')).get_fdata() # Mascara de la materia blanca

ug, gm_ind = np.unique(gm, return_index=True) # Devuelve los indices de los pixeles pertenecientes a materia gris
uw, wm_ind = np.unique(wm, return_index=True) # Devuelve los indices de los pixeles pertenecientes a materia blanca

#plot_slice(original[:,:,24], original[:,:,24]*gm[:,:,24], original[:,:,24]*wm[:,:,24])
#plot_slice(original[:,:,24], np.where(gm[:,:,24]==1, original[:,:,24], 0), np.where(wm[:,:,24]==1, original[:,:,24], 0))

# coger al azar un punto de wm, senalarlo y aumentarle el brillo a este y a la region 

x = np.copy(wm[:,:,24]) # Un slice de materia blanca

x_ind = np.argwhere(x==1) # Arreglo con los indices donde los pixeles pertenecen a la materia blanca

point_ = x_ind[np.random.randint(0, x_ind.shape[0]),:] # Punto aleatorio de la materia blanca

# Con base en l punto seleccionado se extrae una region de materia blanca (cuadro de area = 2tamano x 2tamano)
reg_wm_msk = x[point_[0]-tamano:point_[0]+tamano, point_[1]-tamano:point_[1]+tamano] 

# Region de la img original
reg_orig = np.copy(original[point_[0]-tamano:point_[0]+tamano, point_[1]-tamano:point_[1]+tamano, 24])
#print(reg_wm_msk)

reg_fcd = gaussian_filter(reg_orig, sigma=1)*reg_wm_msk # Region con filtro gaussiano

print(np.mean(reg_orig))
print(np.mean(np.copy(orig_flair[point_[0]-tamano:point_[0]+tamano, point_[1]-tamano:point_[1]+tamano, 24])))
#print(reg_wm_msk*reg_orig*intensidad)

# Region con FCD simulada por aumento de la intensidad en la materia blanca
wm_intensity = np.where(reg_wm_msk*reg_orig*intensidad>0, reg_wm_msk*reg_orig*intensidad, reg_orig) 
#print(wm_intensity)

fcd_sim = np.copy(original) # Copia del volumen original
fcd_sim_flair = np.copy(orig_flair)

# Sobre la region del volumen original se lleva la region con FCD simulada
fcd_sim[point_[0]-tamano:point_[0]+tamano, point_[1]-tamano:point_[1]+tamano, 24] = wm_intensity#reg_orig*reg_wm_msk*intensidad#reg_fcd #np.copy(reg_wm_msk)
#print(fcd_sim[point_[0]-tamano:point_[0]+tamano, point_[1]-tamano:point_[1]+tamano, 24])

# Proyeccion de la region simulada sobre la secuencia FLAIR
fcd_sim_flair[point_[0]-tamano:point_[0]+tamano, point_[1]-tamano:point_[1]+tamano, 24] = wm_intensity
#print(fcd_sim_flair[point_[0]-tamano:point_[0]+tamano, point_[1]-tamano:point_[1]+tamano, 24])

plot_slice(original[:,:,24], fcd_sim[:,:,24], fcd_sim_flair[:,:,24])

## Cargar una FLAIR

## Revisar en la literatura las manifestaciones de FCD en MRI (especialmente aumento de intensidad de materia blanca)