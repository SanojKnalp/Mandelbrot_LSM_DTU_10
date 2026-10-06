import sys
import numpy as np
import math
from mpi4py import MPI


def calculate_local_size_and_xlim(size, x_lim, mpi_rank, mpi_size):
    local_batch_size = size[0]/mpi_size;    
    if ((math.floor(local_batch_size) != local_batch_size)):
        local_batch_size = math.floor(local_batch_size);
        if((mpi_rank+1) == mpi_size):
            local_batch_size += 1
    new_size = int(local_batch_size), 1000;
    length_of_x = x_lim[1]-x_lim[0];
    local_length = length_of_x/mpi_size;
    x_lower_lim = mpi_rank*local_length + x_lim[0];
    x_upper_lim = x_lower_lim + local_length;
    new_x_lim = x_lower_lim, x_upper_lim;
    return new_size, new_x_lim;

def calculate_local_size_and_ylim(size, y_lim, mpi_rank, mpi_size):
    local_batch_size = size[0]/mpi_size;    
    if ((math.floor(local_batch_size) != local_batch_size)):
        local_batch_size = math.floor(local_batch_size);
        if((mpi_rank+1) == mpi_size):
            local_batch_size += 1
    new_size = 1000, int(local_batch_size);
    length_of_x = y_lim[1]-y_lim[0];
    local_length = length_of_x/mpi_size;
    x_lower_lim = mpi_rank*local_length + y_lim[0];
    x_upper_lim = x_lower_lim + local_length;
    new_x_lim = x_lower_lim, x_upper_lim;
    return new_size, new_x_lim;
# Initialize MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()


_help = f"""\
{sys.argv[0]} [chunk-size] [size widthXheight] [limits xmin:xmax ymin:ymax]

Here are some examples:

Call it with a chunk-size of 10
$ {sys.argv[0]} 10

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
$ {sys.argv[0]} 10 100x500

Call it with a chunk-size of 10 and image size of 100 by 500 pixels
spanning the coordinates x \\in 0.1-0.3 and y \\in 0.2-0.3
$ {sys.argv[0]} 10 100x500 0.1:0.3 0.2:0.3
"""

for h in ("help", "-h", "-help", "--help"):
    if h in sys.argv:
        print(_help)
        sys.exit(0)

# First we define all the defaults, then we let the arguments overwrite
# them.
chunk_size = mpi_size;
size = 1000, 1000
xlim = -2.2, 0.75
ylim = -1.3, 1.3

# Now grab the arguments
argv = sys.argv[1:]
if argv:
    chunk_size = int(argv.pop(0))
if argv:
    size = tuple(map(int, argv.pop(0).split("x")))
if argv:
    xlim = tuple(map(float, argv.pop(0).split(":")))
if argv:
    ylim = tuple(map(float, argv.pop(0).split(":")))

print(f"""\
Calculating the Mandelbrot set with these arguments:

{chunk_size = }
{size = }
{xlim = }
{ylim = }
""")

x_lim_global = xlim;
y_lim_global = ylim;
size_global = size;
size, xlim = calculate_local_size_and_xlim(size, xlim, mpi_rank, mpi_size);
# Convert to numpy arrays, not really needed...
size = np.asarray(size)
xlim = np.asarray(xlim)
ylim = np.asarray(ylim)

# Dimensions of the image
image = np.zeros(size, dtype =np.int32);

xconst = np.diff(xlim)[0] / size[0]
yconst = np.diff(ylim)[0] / size[1]

for x in range(size[0]):
    cx = complex(xlim[0] + x * xconst, 0)
    for y in range(size[1]):
        # process (x, y)
        c = cx + complex(0, ylim[0] + y * yconst)
        z = 0
        for i in range(100):
            z = z*z + c
            if np.abs(z) > 2:
                image[x, y] = i
                break

# Now we gather all information at rank 0
if(mpi_rank == 0):
   receive_buffer = np.zeros(size_global, dtype=np.int32);
else:
   receive_buffer = None;


comm.Gather(image, receive_buffer, root=0); # does not work for mpirun = 3, 6
if(mpi_rank == 0):
   image = receive_buffer;
import matplotlib.pyplot as plt
if (mpi_rank == 0):
    # Increase font-size
    plt.rcParams.update({
        "font.size": 10,
    })
    plt.imshow(image.T, extent=np.concatenate([x_lim_global, y_lim_global]))
    plt.xlabel(r"x / Re(p_0)")
    plt.ylabel(r"y / Im(p_0)")

    # Just minimize white-space around the actual plot...
    plt.margins(0, 0)
    plt.savefig("Figure_1.png", bbox_inches="tight", pad_inches=0)
    plt.show()
