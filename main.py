import os
import numpy as np
import nibabel as nib
from utils import generate_fcd, plot_slice, plt_fcd_bbox
from matplotlib.patches import Rectangle

#np.random.seed(1)

fcd_num = np.random.randint(1, 4) # Cantidad de malformaciones a simular en el volumen (aleatorio entre 1 y 4)
print('Num FCDs = {}\n'.format(fcd_num))

img_path = 'temp_ds'
#save_path = '/media/david/datos1/Coding/maestria/3_semestre/va/3ra_entrega/sim4_1.2.png'

t1 = nib.load(os.path.join(img_path, 'T1.nii')).get_fdata()
print('t1 dtype = {}\n'.format(t1.dtype))
t2_flair = nib.load(os.path.join(img_path, 'T2_FLAIR.nii')).get_fdata()
#gm = nib.load(os.path.join(img_path, 'unet2D_predictions_GM.nii')).get_fdata() # Mascara de la materia gris
wm = nib.load(os.path.join(img_path, 'unet2D_predictions_WM.nii')).get_fdata() # Mascara de la materia blanca

# seleccionar al azar un punto de wm, senalarlo y aumentarle el brillo a este y a la region 
n_slice = np.random.randint(0, t1.shape[2])
x = np.copy(wm[:,:,n_slice]) # Un slice de materia blanca
x_ind = np.argwhere(x==1) # Arreglo con los indices donde los pixeles pertenecen a la materia blanca
print('x_ind shape = {}\n'.format(x_ind.shape))

while x_ind.shape[0] == 0:
    n_slice = np.random.randint(0, t1.shape[2])
    x = np.copy(wm[:,:,n_slice]) # Un slice de materia blanca
    x_ind = np.argwhere(x==1) # Arreglo con los indices donde los pixeles pertenecen a la materia blanca

patches = [1]
fcd_t1, fcd_flair, patches[0] = generate_fcd(x, x_ind, t1, t2_flair, n_slice)
for fcd in range(fcd_num - 1):
    fcd_t1, fcd_flair, patch = generate_fcd(x, x_ind, fcd_t1, fcd_flair, n_slice)
    patches.append(patch)
    
print('fcd_t1 main dtype = {}\n'.format(fcd_t1.dtype))
print('fcd_flair main dtype = {}\n'.format(fcd_flair.dtype))

#t1_fcd_nii = nib.Nifti1Image(fcd_t1, nib.load(os.path.join(img_path, 'T1.nii')).affine)
#nib.save(t1_fcd_nii, 't1-fcd.nii')
plt_fcd_bbox(t1[:,:,n_slice], fcd_t1[:,:,n_slice], fcd_flair[:,:,n_slice], patches)

# OJO: NO necesariamente todas las FCD estan en el mismo slice, se deben generar en puntos aleatorios del volumen
# es decir, el slice debe ser aleatorio para cada FCD simulado

## Revisar en la literatura las manifestaciones de FCD en MRI (especialmente aumento de k de materia blanca)
#plot_slice(t1[:,:,24], np.where(gm[:,:,24]==1, t1[:,:,24], 0), np.where(wm[:,:,24]==1, t1[:,:,24], 0))
#ug, gm_ind = np.unique(gm, return_index=True) # Devuelve los indices de los pixeles pertenecientes a materia gris
#uw, wm_ind = np.unique(wm, return_index=True) # Devuelve los indices de los pixeles pertenecientes a materia blanca


# implementar un clasificador de prueba sin optimuzar par'ametros para probrar la sim