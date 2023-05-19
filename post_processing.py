import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def concat_files(grid, variables):
    """
    Function to concat data files from different processors into a single file for each timestep.
    
    Parameters:
    grid (list of list of int): processor grid
    variables (list of str): variables to combine
    """

    # loop over the variables
    for var in variables:
        # loop over the timesteps
        for t in range(0,101):
            grid_rows = []

            # loop over the rows of the processor grid
            for row in grid:

                grid_row = []

                # loop over the processors in this row
                for i in row:
                    filename = f'C:\\Users\\zy622\\Desktop\\MPI\\out\\{var}_{t}_{i}.dat'
                    grid_i = np.loadtxt(filename)
                    grid_row.append(grid_i)

                grid_row_combined = np.concatenate(grid_row, axis=1)  # concatenate along columns
                grid_rows.append(grid_row_combined)

            # concatenate all data rows for a timestep
            grid_tall = np.concatenate(grid_rows, axis=0)  # concatenate along rows

            # write to files
            filename_t = f'{var}_{t}.dat'
            np.savetxt(filename_t, grid_tall)

def animate_files(variables):
    """
    Function to create an animation from the combined data files.
    
    Parameters:
    variables (list of str): variables to animate
    """

    # Create a figure and a set of subplots
    fig, ax = plt.subplots()

    u_ = np.loadtxt('C:\\Users\\zy622\\Desktop\\MPI\\out\\u_0.dat')
    v_ = np.loadtxt('C:\\Users\\zy622\\Desktop\\MPI\\out\\v_0.dat')
    p_ = np.loadtxt('C:\\Users\\zy622\\Desktop\\MPI\\out\\p_0.dat')


    # Create heatmap for pressure
    im = ax.imshow(p_, cmap='viridis', animated=True)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.05)

    # Create vectors for velocity
    X, Y = np.meshgrid(np.arange(0, u_.shape[1], 1), np.arange(0, u_.shape[0], 1))
    Q = ax.quiver(X, Y, u_, v_)

    #function for animation updating with time
    def update_with_time(i):
        u_ = np.loadtxt(f'u_{i}.dat')
        v_ = np.loadtxt(f'v_{i}.dat')
        p_ = np.loadtxt(f'p_{i}.dat')
        im.set_data(p_)
        Q.set_UVC(u_, v_)
        return Q, im

    ani = animation.FuncAnimation(fig, update_with_time, frames=100, interval=200, blit=True)
    ani.save('assessment_2.mp4', writer='ffmpeg')
    plt.show()

def main():

    variables = ['p', 'u', 'v']  # variables to concat.
    grid = [[2, 3], [0, 1]] # my processor rank structure.
    concat_files(grid, variables)
    animate_files(variables)

if __name__ == "__main__":
    main()
