from subprocess import run
from os import sep


def generate_bnorm(config, input="128\n128\n24\n14\n24\n20"):
    """This function calls the BNORM Fortran code to generate bnorm and nescin files.
    It adds the path to the bnorm file generated in config.
    It adds the curpol constant in config.

    :param config: configuration used
    :type config: configparser.ConfigParser

    :param input: input given to BNORM (resolutions, number of Fourier modes...), optionnal
    :type input: str

    :return: None
    :rtype: NoneType

    .. seealso::

       `STELLOPT BNORM documentation <https://princetonuniversity.github.io/STELLOPT/BNORM.html>`_ 

    .. warning::

       The input argument needs to have a specific format :
       see the form of the optionnal argument
    """
    path_folder_wout, _, wout = config["geometry"]["path_plasma"].rpartition(
        sep)
    command = "cd " + path_folder_wout + " && xbnorm " + wout + " 0.1 0 0 0 0 0 0"
    run(command, input=input.encode(), shell=True)

    # Write the path to bnorm file in config
    config["other"]["path_bnorm"] = path_folder_wout + \
        sep + "bnorm." + wout[5:-3:]

    # Get curpol from nescin
    path_nescin = path_folder_wout + sep + "nescin." + wout[5:-3:]
    with open(path_nescin, "r") as f:
        line = f.readline()
        while "curpol" not in line:
            line = f.readline()
        line = f.readline()
        curpol = line.split()[3]

    # Write curpol in config
    config["other"]["curpol"] = curpol
