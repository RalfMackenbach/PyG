import re
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import pandas as pd




class gist_sim:
    """
    Class that accesses the GIST data in a more pythonic way.
    Invoke an element of the class as follows:
    gist_sim = G_functions.gist_sim(filename='.../GIST.txt')
    This creates a simulation object. This object various attributes:
    sim.
    - s0            : Normalized psi coordinate
    - alpha0        : Alpha coordinate
    - minor_radius  : Minor radius
    - major_radius  : Major radius
    - my_dpdx       : Radial pressure gradient
    - q0            : rotational transform
    - shat          : shear
    - gridpoints    : Number of gridpoints
    - n_pol         : Number of poloidal turns
    - z_coord       : z (theta) coordinate of the various functions
    - l_coord       : The l coordinate of various functions
    - functions     : A matrix of all functions outputted by GIST,
                      columns correspond to a different quantities
    - B_arr         : Array containing magnetic field data,
                      the indexing convention is
                            B[0] = B_ref
                            B[1] = B_00
                            B[2] = B_01
                            B[3] = B_10
                            B[4] = B_11
                            .....

    Subclasses:
    !! Dimensionfull parameters !!
    After creating a simulation object, one can invoke a subclass
    with dimensionfull parameters by doing
    gist_sim_dim = gist_sim.get_dimfull()
    This subclass has dimfull functions stored as
    various attributes, which can be accessed by
    sim_dim.
    - g_11          : The 11 metric tensor component,
                      expressed as: (nabla psi) in (nabla psi)
    - g_12          : The 12 metric tensor component,
                      expressed as: (nabla psi) in (nabla alpha)
    - g_22          : The 22 metric tensor component,
                      expressed as: (nabla alpha) in (nabla alpha)
    - B             : The magnetic field strength stored in Tesla
    - jac           : The jacobian expressed as:
                      ((nabla psi) cross (nabla theta)) in (nabla phi)
    - dBdpsi        : Variation of B along psi
    - dBdalpha      : Variation of B along alpha
    - dBdl          : Variation of magnetic field along arclength
    - l_coord       : (signed) arclength coordinates of the various quantities
    """
    # Entire init class contains all information in txt file,
    # only extra is a z-coordinate
    def __init__(self, filename='GIST.txt'):
        """

        """
        # Import entire txt file and assign to self
        txt_RGX       = (open(filename, 'r')).read()
        self.txt_file = txt_RGX


        # Import s and alpha coordinates
        pattern_s_alpha= re.compile(r"""(?<=!s0, alpha0 =)(.*$)""",re.MULTILINE)
        x = (re.findall(pattern_s_alpha, self.txt_file)[0]).split()
        x = [float(i) for i in x]

        # Assign to self
        self.s0 = x[0]
        self.alpha0 = x[1]


        # Import major and minor radius
        pattern_R_a = re.compile(r"""(?<=!major, minor radius\[m\]=)(.*$)""",re.MULTILINE)
        x = (re.findall(pattern_R_a, self.txt_file)[0]).split()
        x = [float(i) for i in x]

        # Assign to self
        self.major_radius = x[0]
        self.minor_radius = x[1]


        # Import B
        pattern_B = re.compile(r"""(?<=B_\d\d =)(.*$)|(?<=Bref =)(.*$)""",re.MULTILINE)
        x = re.findall(pattern_B, self.txt_file)
        x_flattened = [item for subl in x for item in subl]
        x_filtered = (list(filter(None, x_flattened))[0]).split()
        B_container = [float(i) for i in x_filtered]

        # Assign to self
        self.B_arr = B_container



        # Import my_dpdx
        pattern_mydpdx = re.compile(r"""(?<=my_dpdx =)(.*$)""",re.MULTILINE)
        x = (re.findall(pattern_mydpdx, self.txt_file)[0]).split()
        x = [float(i) for i in x]

        # Assign to self
        self.my_dpdx = x[0]


        # Import q0
        pattern_q0 = re.compile(r"""(?<=q0 =)(.*$)""",re.MULTILINE)
        x = (re.findall(pattern_q0, self.txt_file)[0]).split()
        x = [float(i) for i in x]

        # Assign q0 to self
        self.q0 = x[0]


        # Import s-hat
        pattern_shat = re.compile(r"""(?<=shat =)(.*$)""",re.MULTILINE)
        x = (re.findall(pattern_shat, self.txt_file)[0]).split()
        x = [float(i) for i in x]

        # Assign s-hat to self
        self.shat = x[0]


        # Import gridpoints
        pattern_gridpoints = re.compile(r"""(?<=gridpoints =)(.*$)""",re.MULTILINE)
        x = (re.findall(pattern_gridpoints, self.txt_file)[0]).split()
        x = [float(i) for i in x]

        # Assign gridpoints to self
        self.gridpoints = int(x[0])


        # Import n_pol
        pattern_gridpoints = re.compile(r"""(?<=n_pol =)(.*$)""",re.MULTILINE)
        x = (re.findall(pattern_gridpoints, self.txt_file)[0]).split()
        x = [float(i) for i in x]

        # Assign s-hat to self
        self.n_pol = x[0]


        # Import all columns
        pattern_functions  = re.compile(r"""(?<=\/)(?s)(.*$)""",flags=re.MULTILINE|re.DOTALL)
        x = (re.findall(pattern_functions, self.txt_file)[0])
        x_split = x.split('\n ')

        l = []
        for item in x_split:
            subl = []
            for num in item.split():
                subl.append(float(num))
            l.append(subl)
        l_new = list2 = [x for x in l if x != []]
        self.functions = np.asarray(l_new)


        # Create z_coordinate
        self.z_coord   = np.linspace(-np.pi*self.n_pol,np.pi*self.n_pol,num=self.gridpoints)
















    # Fetch dimful quantities if requested
    def get_dimfull(self):
        return self.dimfull_functions(self)















    class dimfull_functions:
    # Class with dimensionfull arrays, because I hate nothing more
    # then converting between units - so let's abstract that away
    # Needs to be manually updated if columns mapping is changed
        def __init__(self,sim):

            self.sim = sim
            # First set up some handy constants
            Bref             =  sim.B_arr[0]
            phi_edge         =  Bref * sim.minor_radius**2. / 2.
            # This cost me some headache - psi = phi/2*pi
            psi_edge         =  phi_edge / (2.0*np.pi)
            # calculate (\nabla \psi)**2
            self.g_11        =  sim.functions[:,0] * 4. * \
                                sim.s0 * psi_edge**2.0 / (sim.minor_radius**2.)
            # calculate (\nabla \psi) * (\nabla \alpha)
            self.g_12        =  sim.functions[:,1] * 2. * sim.q0 * \
                                psi_edge / (sim.minor_radius**2.)
            # calculate (\nabla \alpha)**2
            self.g_22        =  sim.functions[:,2] * sim.q0**2. / \
                                (sim.minor_radius**2. * sim.s0)
            # calculate B of z in tesla
            B_of_z           =  Bref * sim.functions[:,3]
            self.B           =  B_of_z
            # calculate inverse Boozer jacobian
            # (\nabla \psi \cross \nabla theta) \cdot \nabla \varphi
            jac              =  sim.functions[:,4] * 2. * sim.q0 * psi_edge / (sim.minor_radius**(3.))
            self.jac         =  jac
            # calculate dBdpsi
            self.dBdpsi      =  Bref * sim.functions[:,5] / ( 2 *np.sqrt(sim.s0) * \
                                psi_edge )
            # calculate dBdalpha
            self.dBdalpha    =  Bref * sim.functions[:,6] * np.sqrt(sim.s0)/ sim.q0
            # calculate dBdarclength
            self.dBdl        =  Bref * sim.functions[:,7] * jac / \
                                (sim.q0 * B_of_z)

            # Create l-coordinates by solving ODE, set up integrand first
            integrand       = np.asarray((sim.q0*B_of_z)/jac)
            # Straightforward cumtrapz for integral
            l_arr           = cumtrapz(integrand,sim.z_coord,initial=0)
            # Set z=0 => l=0
            l_interp = interp1d(sim.z_coord, l_arr,kind='linear')
            l_offset = l_interp(0)
            self.l_coord = l_arr - l_offset






class GENE_nrg:
    """
    Class that accesses the GENE nrg data in a more pythonic way.
    Invoke an element of the class as follows:
    GENE_nrg = G_functions.GENE_nrg(filename='.../nrg.txt')
    This creates an nrg object. This object various attributes:
    GENE_nrg.
    - n_species         : Number of species
    - times             : Times of corresponding quantities
    - n1                : [n_species,:] array with n1
    - u1                : [n_species,:] array with u1
    - T1par             : [n_species,:] array with T1par
    - T1perp            : [n_species,:] array with T1perp
    - GammaES           : [n_species,:] array with GammaES
    - GammaEM           : [n_species,:] array with GammaEM
    - QES               : [n_species,:] array with QES
    - QEM               : [n_species,:] array with QEM
    - PiES              : [n_species,:] array with PiES
    - PiEM              : [n_species,:] array with PiEM
    """
    def __init__(self, filename='nrg.txt'):
        # We use pandas to easily import all data
        col_names   =  ["n1","u1","T1par","T1perp","GammaES",
                        "GammaEM","QES","QEM","PiES","PiEM"]
        file_full   = pd.read_csv(filename, names=col_names, sep="\s+")
        # Separate out all columns with no NaN
        file_nrg    = file_full.dropna()
        file_times  = file_full.drop(file_nrg.index)
        # Calculate number of rows in both files
        n_times         = len(file_times)
        n_nrgs          = len(file_nrg)
        # We can now calculate the number of species
        n_species       = int(n_nrgs/n_times)
        self.n_species  = n_species
        # Let's assign times to self
        self.times      = file_times.loc[:,"n1"].to_numpy()
        # And now we import all variables
        # n1
        arr_full        = file_nrg.loc[:,"n1"].to_numpy()
        arr_split       = np.transpose(np.asarray(np.split(arr_full, n_nrgs/n_species)))
        self.n1         = arr_split
        # u1
        arr_full        = file_nrg.loc[:,"u1"].to_numpy()
        arr_split       = np.transpose(np.asarray(np.split(arr_full, n_nrgs/n_species)))
        self.u1         = arr_split
        # T1par
        arr_full        = file_nrg.loc[:,"T1par"].to_numpy()
        arr_split       = np.transpose(np.asarray(np.split(arr_full, n_nrgs/n_species)))
        self.T1par      = arr_split
        # T1perp
        arr_full        = file_nrg.loc[:,"T1perp"].to_numpy()
        arr_split       = np.transpose(np.asarray(np.split(arr_full, n_nrgs/n_species)))
        self.T1perp     = arr_split
        # GammaES
        arr_full        = file_nrg.loc[:,"GammaES"].to_numpy()
        arr_split       = np.transpose(np.asarray(np.split(arr_full, n_nrgs/n_species)))
        self.GammaES    = arr_split
        # GammaEM
        arr_full        = file_nrg.loc[:,"GammaEM"].to_numpy()
        arr_split       = np.transpose(np.asarray(np.split(arr_full, n_nrgs/n_species)))
        self.GammaEM    = arr_split
        # QES
        arr_full        = file_nrg.loc[:,"QES"].to_numpy()
        arr_split       = np.transpose(np.asarray(np.split(arr_full, n_nrgs/n_species)))
        self.QES        = arr_split
        # QEM
        arr_full        = file_nrg.loc[:,"QEM"].to_numpy()
        arr_split       = np.transpose(np.asarray(np.split(arr_full, n_nrgs/n_species)))
        self.QEM        = arr_split
        # PiES
        arr_full        = file_nrg.loc[:,"PiES"].to_numpy()
        arr_split       = np.transpose(np.asarray(np.split(arr_full, n_nrgs/n_species)))
        self.PiES       = arr_split
        # PiEM
        arr_full        = file_nrg.loc[:,"PiEM"].to_numpy()
        arr_split       = np.transpose(np.asarray(np.split(arr_full, n_nrgs/n_species)))
        self.PiEM       = arr_split
