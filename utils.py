import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import grey_erosion
from matplotlib.patches import Rectangle

def generate_fcd(x, x_ind, t1, t2_flair, slice_):
    #np.random.seed(1)
    area = np.random.randint(5, 10) # Tamano de la malformacion
    k = np.random.uniform(1.3, 1.6) # Factor que multiplica la intensidad de los pixeles
    fcd_depth = np.random.randint(4, 10) # Profundidad en slices de la malformacion
    print('Intensidad FCD = {}\n'.format(k))

    point_ = x_ind[np.random.randint(0, x_ind.shape[0]),:] # Elegir un punto aleatorio de la materia blanca
    subreg = np.s_[point_[0]-area:point_[0]+area, point_[1]-area:point_[1]+area]
    region = np.s_[point_[0]-area:point_[0]+area, point_[1]-area:point_[1]+area, slice_]

    # Con base en l punto seleccionado se extrae una region de materia blanca (cuadro de area = 2area x 2area)
    reg_wm = x[subreg] 

    # Region de la img t1 y flair
    reg_t1 = np.copy(t1[region])
    reg_flair = np.copy(t2_flair[region])

    #reg_fcd = np.random.normal(1.4, 1.1, reg_t1.shape) #
    reg_fcd = grey_erosion(gaussian_filter(reg_t1, sigma=1, mode='mirror')/np.mean(reg_t1), (3, 3)) # Region con filtro gaussiano

    fcd_t1 = np.copy(t1) # Copia del volumen t1
    fcd_flair = np.copy(t2_flair)

    e = np.where(reg_wm>0, np.multiply(reg_wm*reg_t1,reg_fcd), reg_t1)

    # Sobre la region del volumen t1 se lleva la region con FCD simulada
    fcd_t1[region] = e #np.where(reg_wm>0, np.multiply(reg_wm*reg_t1,reg_fcd), reg_t1)

    #print('fcd_t1 func shape = {}\n'.format(fcd_t1.shape))
    #print('fcd_t1 func dtype = {}\n'.format(fcd_t1.dtype))

    # Proyeccion de la region simulada sobre la secuencia FLAIR
    fcd_flair[region] = np.where(reg_wm>0, np.multiply(reg_wm*reg_t1,reg_fcd), reg_flair) #np.where(reg_wm>0, reg_wm*reg_flair*k, reg_flair) 
    plot_slice(reg_wm, fcd_flair[region], e)
    patch = Rectangle((point_[1]-2*area, point_[0]-2*area), 4*area, 4*area, edgecolor='red', fill=False)

    return fcd_t1, fcd_flair, patch

def plt_fcd_bbox(t1, t1_sim, flair_sim, patches):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=1 ,wspace=1)

    ax1 = fig.add_subplot(1, 3, 1)
    #ax1.title.set_text('imagen')
    ax1.axis("off")
    #ax1.add_patch(patches[0])
    ax1.imshow(t1, cmap="gray")

    ax2 = fig.add_subplot(1, 3, 2)
    #ax2.title.set_text('mask')
    ax2.axis("off")
    #ax2.imshow(t1_sim, cmap="gray")

    ax3 = fig.add_subplot(1, 3, 3)
    #ax3.title.set_text('prediccion')
    ax3.axis("off")

    for patch in patches:
        #ax2.add_patch(patch)
        ax3.add_patch(patch)

    ax2.imshow(t1_sim, cmap="gray")
    ax3.imshow(flair_sim, cmap="gray")
    #plt.savefig(save_path, dpi=300)

    plt.show()

def plot_slice(t1, t1_sim, flair_sim):
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(hspace=1 ,wspace=1)

    ax1 = fig.add_subplot(1, 3, 1)
    #ax1.title.set_text('imagen')
    ax1.axis("off")
    ax1.imshow(t1, cmap="gray")

    ax2 = fig.add_subplot(1, 3, 2)
    #ax2.title.set_text('mask')
    ax2.axis("off")
    ax2.imshow(t1_sim, cmap="gray")

    ax3 = fig.add_subplot(1, 3, 3)
    #ax3.title.set_text('prediccion')
    ax3.axis("off")
    ax3.imshow(flair_sim, cmap="gray")
    #plt.savefig(save_path, dpi=300)

    plt.show()