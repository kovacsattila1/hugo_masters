import subprocess
import os

# Define the path to the CoppeliaSim executable
coppelia_sim_path = "/home/kovacs/Downloads/CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu20_04/coppeliaSim.sh"  # Adjust this path based on your OS and installation

arguments = [
    # "/home/kovacs/Downloads/CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu20_04/coppeliaSim.sh",
    "-h",
    "-gparam1=0",
    "-GzmqRemoteApi.rpcPort=23000",
    "-GwsRemoteApi.port=23050",
    "-GROSInterface.nodeName=MyNodeName0",
    "/home/kovacs/Documents/disszertacio/hugo_python_control_coppeliasim_v4/asti.ttt"
]

# # Ensure the path is correct and executable
# if not os.path.isfile(coppelia_sim_path):
#     raise FileNotFoundError(f"The specified path does not exist: {coppelia_sim_path}")
# if not os.access(coppelia_sim_path, os.X_OK):
#     raise PermissionError(f"The specified file is not executable: {coppelia_sim_path}")

# Start CoppeliaSim as a subprocess
process = subprocess.Popen([coppelia_sim_path] + arguments)

# Optional: Wait for the process to complete
process.wait()