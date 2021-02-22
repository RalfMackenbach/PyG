import re






class sim:
    """
    Class that accesses the GIST data in a more pythonic way.
    Invoke an element of the class as follows:
    sim = GIST_functions.sim(filename='../GIST.txt')
    This creates a simulation object. This object various attributes:

    - s0            : Normalized psi coordinate
    - alpha0        : Alpha coordinate
    - minor_radius  : Minor radius
    - major_radius  : Major radius
    - my_dpdx       : Radial pressure gradient
    - q0            : rotational transform
    - shat          : Not a clue what this is
    - gridpoints    : Number of gridpoints
    - n_pol         : Number of poloidal turns
    - functions     : A matrix of all functions outputted by GIST,
                      columns correspond to a different quantities
    - B             : Array containing magnetic field data,
                      the indexing convention is
                            B[0] = B_ref
                            B[1] = B_00
                            B[2] = B_01
                            B[3] = B_10
                            B[4] = B_11
                            .....
    """

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
        self.B = B_container



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

        # Assign s-hat to self
        self.gridpoints = x[0]


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

        self.functions = l
