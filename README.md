![alt text](img/ITM_rect.png "ITM")

## FCD in MRI: Features simulator

This is an attempt at simulating Focal Cortical Dysplasias features on MRI volumes of healthy patients.

	Clone this repository
	Make directory named `temp_ds' inside this project's folder
	Put the T1, FLAIR and White matter's mask in it
	Run `main.py' 

### Simulation by brightness increment of white matter on T2-FLAIR sequence

Simulates brightness augmentation of the white matter on T2-FLAIR sequences. By extracting the white matter mask and selecting a random point in it, a squared region is extracted with center on such point. Thus, the simulated region, as of right now, is a square placed on a single slice of the sequence. The intensity of the pixel is increased by a factor of the pixel value, only the pixels that belong to white matter and are inside the square are considered for brightness increase.
