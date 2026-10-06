import sys
import numpy as np
import math
from mpi4py import MPI


def calculate_local_size_and_xlim(size, x_lim, mpi_rank, mpi_size):
    local_batch_size = size[0]/mpi_size;    
    if ((math.floor(local_batch_size) != local_batch_size)):
        local_batch_size = math.floor(local_batch_size);
        remainder = size[0] - mpi_size*local_batch_size;
        if (mpi_rank < remainder):
            local_batch_size+=1;
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

def compute_image(size, xlim, ylim):
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
    return image;

def split_by_index_and_chunk(xlim, size, index, chunk_size):
    # start by computing the local size
    divisor = size[0]/chunk_size;
    if ((size[0]-chunk_size*index) >=0):
        length_of_x = xlim[1]-xlim[0];
        local_length = length_of_x/divisor;
        x_lower_lim = index*local_length + xlim[0];
        if((size[0]-chunk_size*(index+1)) >=0):
            x_upper_lim = x_lower_lim + local_length;
            new_upper_size_lim = chunk_size;
        else:
            local_length = size[1]-local_length*index;
            x_upper_lim = x_lower_lim + local_length;
            new_upper_size_lim = size[1]-index*chunk_size;
        new_x_lim = x_lower_lim, x_upper_lim;
        new_size = int(new_upper_size_lim), 1000;
    return new_size, new_x_lim;

def communicate_rank_0(comm, index):
    receive_buffer = np.empty(1, type=np.int32);
    comm.Recv(receive_buffer);
    send_buffer = np.array([index + 1], type=np.int32);
    comm.Send(send_buffer, dest=receive_buffer[0]);
    return index+1


def communicate_index(comm):
    # ask rank 0 what index is free
    send_buffer = np.array([mpi_rank], type=np.int32);
    comm.Send(send_buffer, dest=0);
    receive_buffer = np.empty(1, type=np.int32);
    comm.Recv(receive_buffer, source=0);
    return receive_buffer[0];

def communicate_the_image_back(index, image, comm):
    # send the index
    comm.Send(np.array([index], np.int32), dest=0);
    #send the image
    comm.Send(image, dest=0);

def receive_image(comm):
    index_receive_buffer = np.empty(1, type=np.int32);
    comm.Recv(index_receive_buffer);
    image_receive_buffer = np.empty([chunk_size, 1000],type=np.int32);
    comm.Recv(image_receive_buffer,status=status);
    num_received_elements = int(status.Get_elements(MPI.INT)/1000);
    return index_receive_buffer[0], image_receive_buffer, num_received_elements;

def compute_max_image_number(chunk_size, size):
    local = int(np.floor(size[0]/chunk_size));
    if(local != np.floor(size[0]/chunk_size)):
        local +=1;
    return local;

def assemble_image(local_image, image, local_index, chunk_size, num_received_elements):
    image[local_index*chunk_size:(local_index*chunk_size+num_received_elements), :] = local_image[0:num_received_elements, :];

# Initialize MPI
comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()
status = MPI.Status();


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
chunk_size = 10;
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
max_images = compute_max_image_number(chunk_size, size);


if (mpi_rank ==0):
    image = np.zeros(size_global, dtype =np.int32);
    index =0;
    num_received_images = 0;
    # send out all the images
    for i in range(mpi_size):
        index = communicate_rank_0(comm, index);
    while(True):
        local_index, local_image, num_received_elements = receive_image(comm);
        num_received_images +=1;

        
        if(index < max_images):
            index = communicate_rank_0(comm, index);
        assemble_image(local_image, image, local_index, chunk_size, num_received_elements);
        if(num_received_images == max_images): break;
else:
    while(True):
        index = communicate_index(comm);
        size, xlim = split_by_index_and_chunk(xlim, size, index, chunk_size);
        # Convert to numpy arrays, not really needed...
        size = np.asarray(size)
        xlim = np.asarray(xlim)
        ylim = np.asarray(ylim)

        image = compute_image(size, xlim, ylim);
        communicate_the_image_back(index, image, comm);   


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
