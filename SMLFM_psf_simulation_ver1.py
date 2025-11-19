#PSF simulation script for SMLFM
#Adapted from the 'getPSF' function in MATLAB from Bin Fu (bf341) and Ezra (eb758)

"""
Ver 1.0 2025-11-06 
- Made a tweak on the MLA centres generation function. 
- Adjusted the system class and a few corresponding lines accordingly.
- Made a tweak on the MLA phase plotting section.
- Otherwise similar to the MATLAB version.



"""


#/-- import section --/

import numpy as np 
import matplotlib.pyplot as plt

from matplotlib.patches import RegularPolygon
from tqdm import tqdm

#/-- self-defined classes --/
class system:
    """
    Class for the SMLFM system parameters.
    NOTE: This is used for version 1.0.  Might be changed to a dictionary instead in future versions.

    """
    def __init__(self, 
                 NA = 1.49, # Numerical aperture of the objective lens.
                 f_obj = 2e-3, # Focal length of the objective lens. in mm
                 f_tube = 200e-3, # Focal length of the tube lens. in mm.
                 f_fourier = 125e-3, # Focal length of the Fourier lens. in mm.
                 cam_pix_size = 4.86e-6, # Camera pixel size. in meters.
                 mla_pitch = 1.1e-3, # Microlens array pitch. in meters.
                 f_mla = 50e-3,      # Focal length of the microlens array. in mm.
                 wavelength = 600e-9, # Emission wavelength. in meters.
                 z_range = 8e-6, # Range of z positions to simulate PSF. in meters.
                 z_step = 0.2e-6, # Step size in z positions. in meters.
                 xscale = 1, # Scaling factor in x direction (pixels per unit length at BFP)
                 yscale = 1, # Scaling factor in y direction (pixels per unit length at BFP)
                 image_size = [1000, 1000], # Size of the output PSF image in pixels [y,x]
                 mla_rotation = 0.0, # Rotation angle of the microlens array in radians. #TODO: check the unit later.
                 mla_direction = 'horizontal',# Direction of the microlens array ('horizontal' or 'vertical')
                 n_medium = 1.52 # Refractive index of the medium
                 ):
        
        self.NA = NA  # Numerical aperture of the objective
        self.f_obj = f_obj  # Focal length of the objective lens 
        self.f_tube = f_tube  # Focal length of the tube lens
        self.f_fourier = f_fourier  # Focal length of the Fourier lens
        self.cam_pix_size = cam_pix_size  # Camera pixel size
        self.mla_pitch = mla_pitch  # Microlens array pitch
        self.f_mla = f_mla  # Focal length of the microlens array
        self.wavelength = wavelength  # Emission wavelength
        self.z_range = z_range  # Range of z positions to simulate PSF
        self.z_step = z_step  # Step size in z positions
        self.mla_rotation = mla_rotation # Rotation angle of the microlens array in radians. 
        self.mla_direction = mla_direction # Direction of the microlens array ('horizontal' or 'vertical')
        self.n_medium = n_medium # Refractive index of the sample medium. 

        self.xscale = xscale # Scaling factor in x direction (pixels per unit length at BFP)
        self.yscale = yscale # Scaling factor in y direction (pixels per unit length at BFP)

        self.image_size = image_size # Size of the output PSF image in pixels [y,x]
        self.bfp_radius = NA * f_obj * (f_fourier / f_tube)  # Radius of the back focal plane (BFP) image in meters.



        





#/-- self-defined functions --/

def getPSF(img_size,system,array_centre,plot_result):
    """
    Core function to simulate system PSF.
    
    Input args:
    - img_size: [y,x] size of the output PSF image in pixels
    - system : system class containing system parameters

    - array_centre: Centre position of the microlens array at the back focal plane #NOTE: this is set to mandatory input in this version.
    - plot_result: Boolean flag to plot the PSF result or not. renamed from the 'flag' arg in the MATLAB version.
    
    Output args:
    - PSF: Simulated system psf 
    """

    #system.bfp_radius = system.NA * system.f_obj * (system.f_fourier / system.f_tube) # Radius of the back focal plane (BFP) image #NOTE: commented out as this is initialised in the system class.

    print("System BFP radius (in meters):", system.bfp_radius) #debug print

    bfp_diameter = (2 * system.bfp_radius) / system.cam_pix_size# Diameter of the BFP image
    print("BFP diameter (in camera pixels):", bfp_diameter) #debug print
    
    system.image_size = img_size
    system.xscale = img_size[1] / bfp_diameter # Scaling factor in x direction (pixels per unit length at BFP)

    print("System xscale:", system.xscale) #debug print
    # how much the image is larger compared to back focal plane (bfp) size in x direction (e.g., image is 1000x1000, bfp is 800x400, then xscale is 1.25)

    system.yscale = img_size[0] / bfp_diameter # Scaling factor in y direction (pixels per unit length at BFP)
    print("System yscale:", system.yscale) #debug print
    # how much the image is larger compared to back focal plane (bfp) size in y direction (e.g., image is 1000x1000, bfp is 800x400, then yscale is 2.5)

    array_centre = (array_centre - np.flip(np.array(img_size)) / 2) / (system.bfp_radius / system.cam_pix_size) # convert the unit to m #TODO: check this later. 

    #-- core section --
    z_positions = np.arange( - (system.z_range / 2), (system.z_range / 2) + system.z_step, system.z_step ) # z positions to simulate PSF at different depths
    PSF = np.zeros( (img_size[0], img_size[1], len(z_positions)) ) # Pre-allocate PSF array # TODO: rearrange the array to [z, y, x] ?

    phase_MLA =  get_phase_MLA(system, array_centre, plot_result) # Calculate the phase of the microlens array

    # if plot_result:
    #    plt.show()

    for i in tqdm(range(len(z_positions))):
        # get the electric field in pupil due to an isotropic emitter at point (x,y,z)
        E_bfp = get_Field_BFP(z_positions[i], system)

        # apply the MLA phase
        E_bfp_MLA = E_bfp * np.exp(1j * phase_MLA)

        # propagate to image plane.
        E_img = propagate_Fresnel_TF(E_bfp_MLA, system)

        # get the PSF intensity
        I_img = np.abs(E_img) ** 2
        PSF[:, :, i] = I_img
    
    # normalize PSF
    PSF = (PSF - PSF.min()) / (PSF.max() - PSF.min())

    if plot_result:
        # plot the coloured PSFs at different z positions 
        colourPSF = get_coloured_hyperstack(PSF)

        fig_psf, ax_psf = plt.subplots(1,1)
        ax_psf.imshow(colourPSF)
        ax_psf.set_title('Coloured PSF Hyperstack')
        ax_psf.axis('off')
        plt.show() 

    return PSF


def get_phase_MLA(system, array_centre, plot_result):
    """
    Calculate the phase of the microlens array
    
    Input args:
    - system : system class containing system parameters
    - array_centre: Centre position of the microlens array
    - plot_result: Boolean flag to plot the result

    Output args:
    - phase_MLA: Phase of the microlens array

    """
    

    array_pitch = system.mla_pitch
    f_mla = system.f_mla
    bfp_radius = system.bfp_radius
    wavelength = system.wavelength
    cam_xscale = system.xscale
    cam_yscale = system.yscale
    cam_size = system.image_size

    num_microlens = (2 * bfp_radius) / array_pitch # Number of microlenses across the diameter of the BFP.

    MLA_centres = get_centres_mla(system) # Get the centres of the microlens array

    if plot_result:
        # Create figure with 3 subplots
        fig, ax1 = plt.subplots()
        

        # Convert to mm for plotting
        x_mm = MLA_centres[:, 0] * 1e3
        y_mm = MLA_centres[:, 1] * 1e3
        
        # Calculate hexagon radius (for hexagonal packing)
        # For a hexagonal grid with pitch p, the hexagon side length is p/sqrt(3)
        hexagon_radius = (array_pitch * 1e3) /2 
        
        # Plot hexagons around each MLA centre
        for i in range(len(x_mm)):
            hexagon = RegularPolygon((x_mm[i], y_mm[i]), 
                                    numVertices=6, 
                                    radius=hexagon_radius,
                                    orientation=np.pi/6,  # Rotate 30 degrees for flat-top hexagons
                                    facecolor='lightblue', 
                                    edgecolor='blue', 
                                    linewidth=0.5,
                                    alpha=0.3)
            ax1.add_patch(hexagon)
        
        # Plot microlens centers
        ax1.plot(x_mm, y_mm, 'b.', markersize=4, label='microlens centers')
        
        # Plot BFP outline
        theta = np.arange(0, 2.1 * np.pi, 0.1)
        x_outline_bfp = bfp_radius * 1e3 * np.cos(theta)
        y_outline_bfp = bfp_radius * 1e3 * np.sin(theta)
        ax1.plot(x_outline_bfp, y_outline_bfp, 'r-', linewidth=2, label='conjugate bfp outline')
        
        # Set axis properties
        n_ulenses = num_microlens
        ax1.set_xlim([-n_ulenses * array_pitch * 1e3, n_ulenses * array_pitch * 1e3])
        ax1.set_ylim([-n_ulenses * array_pitch * 1e3, n_ulenses * array_pitch * 1e3])
        ax1.set_aspect('equal')
        ax1.set_xlabel('x (mm)')
        ax1.set_ylabel('y (mm)')
        ax1.set_title('BFP plane with respect to Microlens array', fontweight='normal')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.tick_params(labelsize=10)
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show() 

    # Normalise such that the bfp radius is 1 unit
    #x_c = MLA_centres[:, 0] / bfp_radius
    #y_c = MLA_centres[:, 1] / bfp_radius
    MLA_centres_normed = MLA_centres / bfp_radius

    Nx = cam_size[1] # number of pixels on x
    Ny = cam_size[0] # number of pixels on y

    xrange = np.linspace( -cam_xscale, cam_xscale, Nx ) # x range in normalised unit
    yrange = np.linspace( -cam_yscale, cam_yscale, Ny ) # y range in normalised unit

    

    lrf, lrf_ids = local_Radius_Field(xrange, yrange, MLA_centres_normed) # Calculate local radius field

    local_radius = bfp_radius * lrf
    k0 = 2 * np.pi / wavelength
    phase_MLA = (- k0 / (2 * f_mla)) * (local_radius ** 2)

    return phase_MLA
    

def get_centres_mla(system):
    """
    Calculate the centres of the microlens array

    Adapted for new MLA designs.


    """

    # Generate a grid of hexagonal microlens centres assuming pitch to be unit 1. Currently only 7HEX supported.  #TODO: add 3HEX and maybe a more generalised version later. 
    # NOTE: pitch is the 'short axis' pitch.

    MLA_centre = [0,0] # Set the MLA centre at the origin for simplicity. #TODO   : make this an input later.

    if system.mla_direction == 'horizontal': # TODO: rename the orientation later.
        #7-HEX array set up.
        MLA_centres = np.zeros((7,2))

        # Top centre lens
        MLA_centres[0, 0] = MLA_centre[0] 
        MLA_centres[0, 1] = MLA_centre[1] + 1
        # 2nd row left lens
        MLA_centres[1, 0] = MLA_centre[0] - 1*(np.sqrt(3)/2)
        MLA_centres[1, 1] = MLA_centre[1] + 1*(1/2)
        # 2nd row right lens
        MLA_centres[2, 0] = MLA_centre[0] + 1*(np.sqrt(3)/2)
        MLA_centres[2, 1] = MLA_centre[1] + 1*(1/2)
        # Centre lens
        MLA_centres[3, 0] = MLA_centre[0]
        MLA_centres[3, 1] = MLA_centre[1]
        # 3rd row left lens
        MLA_centres[4, 0] = MLA_centre[0] - 1*(np.sqrt(3)/2)
        MLA_centres[4, 1] = MLA_centre[1] - 1*(1/2)
        # 3rd row right lens
        MLA_centres[5, 0] = MLA_centre[0] + 1*(np.sqrt(3)/2)
        MLA_centres[5, 1] = MLA_centre[1] - 1*(1/2)
        # Bottom centre lens
        MLA_centres[6, 0] = MLA_centre[0] 
        MLA_centres[6, 1] = MLA_centre[1] - 1
    
    elif system.mla_direction == 'vertical': # rotated 30 degerees from the horizontal case.
        #7-HEX array set up.
        MLA_centres = np.zeros((7,2))

        # Top left lens 
        MLA_centres[0, 0] = MLA_centre[0] - 1*(1/2)
        MLA_centres[0, 1] = MLA_centre[1] + 1*(np.sqrt(3)/2)
        # Top right lens
        MLA_centres[1, 0] = MLA_centre[0] + 1*(1/2)
        MLA_centres[1, 1] = MLA_centre[1] + 1*(np.sqrt(3)/2)
        # 2nd row left lens
        MLA_centres[2, 0] = MLA_centre[0] - 1
        MLA_centres[2, 1] = MLA_centre[1]
        # Centre lens
        MLA_centres[3, 0] = MLA_centre[0]
        MLA_centres[3, 1] = MLA_centre[1]
        # 2nd row right lens
        MLA_centres[4, 0] = MLA_centre[0] + 1
        MLA_centres[4, 1] = MLA_centre[1] 
        # Bottom left lens
        MLA_centres[5, 0] = MLA_centre[0] - 1*(1/2)
        MLA_centres[5, 1] = MLA_centre[1] - 1*(np.sqrt(3)/2)
        # Bottom right lens
        MLA_centres[6, 0] = MLA_centre[0] + 1*(1/2)
        MLA_centres[6, 1] = MLA_centre[1] - 1*(np.sqrt(3)/2)
    
    # shift is skipped. 

    # Scale to physical units in bfp space 
    MLA_centres = MLA_centres *  ( system.mla_pitch * (np.sqrt(3)/2) )  # scale the centres according to pitch. #NOTE : mla_pitch input is the 'long axis' pitch. therefore the scaling factor (sqrt(3)/2).

    # Rotation skipped. and may not be required.

    # COV also skipped. 

    return MLA_centres


    

    


def local_Radius_Field(xrange, yrange, axes):
    """
    Adapated function from MATLAB version to calculate local radius field.
    orginal author: Kevin O'Holleran
    
    Input args:
    - xrange: Range of x positions
    - yrange: Range of y positions
    - axes: Centre position of the microlens array
    
    Output args:
    - lrf : Local radius field
    - ids : array with axes index linking to axes coords list (can be deleted?)
    """

    xx, yy = np.meshgrid(xrange, yrange)
    nr = len(yrange) # number of rows
    nc = len(xrange) # number of columns
    num_axes = axes.shape[0] # number of axes
    print("Number of axes (microlenses):", num_axes) #debug print

    lrf = np.full((nr, nc), np.inf) # Pre-allocate lrf array
    ids = np.zeros((nr, nc)) # Pre-allocate ids array

    for i in range(num_axes):
        xi = axes[i, 0]
        yi = axes[i, 1]
        r_temp = np.sqrt((xx - xi)**2 + (yy - yi)**2)
        id_temp = r_temp < lrf
        lrf[id_temp] = r_temp[id_temp]
        ids[id_temp] = i + 1  # +1 to match MATLAB's 1-based indexing
    
    return lrf, ids


def get_Field_BFP(z_position, system):
    """
    Calculate the electric field at the back focal plane (BFP) due to an isotropic emitter.

    Adapted from Bin Fu's MATLAB function 'getFieldBFP.m'

    Input args:
    - z_position: z position of the emitter. 
    - system: system class containing system parameters

    Output args:
    - E_bfp: Electric field at the back focal plane
    
    """

    NA = system.NA
    wavelength = system.wavelength
    n_medium = system.n_medium
    cam_xscale = system.xscale
    cam_yscale = system.yscale
    cam_size = system.image_size

    # get field for isotropic emitter at the origin (x,y,z) = (0,0,0)
    Nx = cam_size[1]  # number of pixels on x
    Ny = cam_size[0]  # number of pixels on y
    E_bfp = np.ones((Ny, Nx),dtype=complex)  # Pre-allocate electric field array 

    # get polar coordinates in the pupil plane and the aperture mask
    rho_max = NA / n_medium  # Maximum normalized radius in the pupil
    xx,yy = np.meshgrid( np.linspace(-rho_max*cam_xscale, rho_max*cam_xscale, Nx),
                         np.linspace(-rho_max*cam_yscale, rho_max*cam_yscale, Ny) )
    
    """ cart2pol(xx,yy) function translated from MATLAB """
    rho = np.hypot(xx, yy)
    phi = np.arctan2(yy, xx) 
    aperture_mask = rho <= rho_max

    # move the emitter in object space in z direction by adding defocus
    k0 = 2 * np.pi / wavelength
    phase_z = n_medium * k0 * z_position * np.sqrt( 1 - (rho ** 2) )  # Defocus phase term
    total_phase = phase_z # psf is spatially invariant. x and y phase is not necessary.
    E_bfp = E_bfp * np.exp(1j * total_phase) * aperture_mask  # Apply phase and aperture mask

    # set field outside the aperture to zero
    E_bfp[~aperture_mask] = 0

    return E_bfp


def propagate_Fresnel_TF(E_input, system):
    """
    Propagate the electric field from the BFP to the image plane using Fresnel transfer function method.
    
    adapted from Bin Fu's MATLAB function 'propagateFresnelTF.m'

    Input args:
    - E_input: Electric field at the input plane (BFP)  named as u1 in the MATLAB version.
    - system: system class containing system parameters

    Output args:
    - E_output: Electric field at the output plane (image plane)
    
    """

    wavelength = system.wavelength #NOTE: named as lambda in MATLAB version.
    z = system.f_mla  # Propagation distance (focal length of the microlens array) #NOTE: same as the MATLAB version. not a good name??
    bfp_radius = system.bfp_radius
    yscale = system.yscale
    xscale = system.xscale

    r,c = E_input.shape  # Get size of input field
    Lx = xscale * 2 * bfp_radius  # width
    Ly = yscale * 2 * bfp_radius  # length ?

    dx = Lx / c  # pixel size in x
    dy = Ly / r  # pixel size in y

    fx = np.arange(-1/(2*dx), 1/(2*dx), 1/Lx)  # frequency coordinates in x
    fy = np.arange(-1/(2*dy), 1/(2*dy), 1/Ly)  # frequency coordinates in y
    FX, FY = np.meshgrid(fx, fy)

    H = np.exp(-1j *np.pi * wavelength * z * (FX**2 + FY**2))  # Transfer function
    H = np.fft.fftshift(H)  # Shift zero frequency to center
    U1 = np.fft.fft2(np.fft.fftshift(E_input))  # Fourier transform of input field
    U2 = U1 * H  # Apply transfer function
    E_output = np.fft.ifftshift(np.fft.ifft2(U2))  # Inverse Fourier transform to get output field

    return E_output


def get_coloured_hyperstack(PSF):
    """
    Generate a coloured hyperstack from the PSF stack.

    adapted from Bin Fu's MATLAB function 'getColourcodedHyperstack.m'
    
    Input args:
    - PSF: 3D array of PSF images (y, x, z)
    Output args:
    - colourPSF: Coloured hyperstack of PSF images
    
    """

    num_frames = PSF.shape[2]
    colours = plt.cm.jet(np.linspace(0, 1, num_frames))[:, :3]  # Get colours from jet colormap
    colourPSF = np.zeros((PSF.shape[0], PSF.shape[1], 3), dtype=float)  # Pre-allocate coloured PSF array

    for i in range(num_frames):
        R = PSF[:, :, i] * colours[i, 0]
        G = PSF[:, :, i] * colours[i, 1]
        B = PSF[:, :, i] * colours[i, 2]

        temp_rgb = np.stack((R, G, B), axis=2)
        colourPSF += temp_rgb
    
    # Normalize the coloured PSF
    colourPSF = colourPSF / np.max(colourPSF)

    return colourPSF


#//-- main thread --//
if __name__ == "__main__":
    # Example usage of the PSF simulation script

    # Define system parameters
    sys_params = system(
        NA = 1.49,
        f_obj = 2e-3,
        f_tube = 200e-3,
        f_fourier = 125e-3,
        cam_pix_size = 4.86e-6,
        mla_pitch = 1.2e-3,
        f_mla = 50e-3,
        wavelength = 600e-9,
        z_range = 8e-6,
        z_step = 0.05e-6,
        image_size = [512, 512],
        mla_rotation = 0.0,
        mla_direction = 'horizontal',
        n_medium = 1.518
    )

    # Define image size and microlens array centre
    img_size = [512,512]  # [y, x]
    array_centre = np.array([0, 0])  # Centre position of the microlens array in pixels

    # Simulate PSF
    PSF_result = getPSF(img_size, sys_params, array_centre, plot_result=True)








