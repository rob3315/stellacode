from os import sep
from subprocess import run


def generate_bnorm(config, input="128\n128\n24\n14\n24\n20"):
    """
    Calls the BNORM Fortran code to generate bnorm and nescin files.

    Adds the path to the bnorm file generated and the curpol constant in the configuration file.

    :param config: Configuration used
    :type config: configparser.ConfigParser

    :param input: Input given to BNORM (resolutions, number of Fourier modes...), optional
    :type input: str

    :return: None
    :rtype: NoneType

    .. seealso::

       `STELLOPT BNORM documentation <https://princetonuniversity.github.io/STELLOPT/BNORM.html>`_

    .. warning::

       The input argument needs to have a specific format :
       see the form of the optionnal argument.
       See the BNORM documentation for the meaning of the argument.
    """
    # Get the path to the wout file
    path_folder_wout, _, wout = config["geometry"]["path_plasma"].rpartition(
        sep)

    # Generate the bnorm and nescin files
    command = f"cd {path_folder_wout} && xbnorm {wout} 0.1 0 0 0 0 0 0"
    run(command, input=input.encode(), shell=True)

    # Write the path to bnorm file in config
    config["other"]["path_bnorm"] = f"{path_folder_wout}{sep}bnorm.{wout[5:-3:]}"

    # Get curpol from nescin
    path_nescin = f"{path_folder_wout}{sep}nescin.{wout[5:-3:]}"
    with open(path_nescin, "r") as f:
        # Find the line containing curpol
        line = f.readline()
        while "curpol" not in line:
            line = f.readline()

        # Get the value of curpol
        line = f.readline()
        curpol = line.split()[3]

    # Write curpol in config
    config["other"]["curpol"] = curpol
