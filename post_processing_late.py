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
        for t in range(101):
            grid_cols = []

            # loop over the columns of the processor grid
            for col in zip(*grid):

                grid_col = []

                # loop over the processors in this column
                for i in col:
                    filename = f'C:\\Users\\zy622\\Desktop\\MPI\\x64\\release\\out\\{var}_{t}_{i}.dat'
                    grid_i = np.loadtxt(filename)
                    grid_col.append(grid_i)

                grid_col_combined = np.concatenate(grid_col, axis=0)  # concatenate along rows
                grid_cols.append(grid_col_combined)

            # concatenate all data columns for a timestep
            grid_wide = np.concatenate(grid_cols, axis=1)  # concatenate along columns

            # write to files
            filename_t = f'C:\\Users\\zy622\\Desktop\\MPI\\x64\\release\\out_con\\{var}_{t}.dat'
            np.savetxt(filename_t, grid_wide)


def animate_files(variables):
    """
    Function to create an animation from the combined data files.
    
    Parameters:
    variables (list of str): variables to animate
    """

    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 7))

    # Load initial data
    u_ = np.loadtxt(f'C:\\Users\\zy622\\Desktop\\MPI\\x64\\release\\out_con\\u_0.dat')
    v_ = np.loadtxt(f'C:\\Users\\zy622\\Desktop\\MPI\\x64\\release\\out_con\\v_0.dat')
    p_ = np.loadtxt(f'C:\\Users\\zy622\\Desktop\\MPI\\x64\\release\\out_con\\P_0.dat')
    
    X, Y = np.meshgrid(np.arange(0, u_.shape[1], 1), np.arange(0, u_.shape[0], 1))

    # Initial plots
    cont1 = ax1.contourf(X, Y, p_, cmap='coolwarm')
    quiv1 = ax1.quiver(X[::20, ::5], Y[::20, ::5], u_[::20, ::5], v_[::20, ::5])
    ax1.set_xlabel('$x$', fontsize=16)
    ax1.set_ylabel('$y$', fontsize=16)
    ax1.set_title('Pressure driven problem - pressure and velocity vectors', fontsize=16)
    fig.colorbar(cont1, ax=ax1)

    cont2 = ax2.contourf(X, Y, np.sqrt(u_*u_ + v_*v_), cmap='coolwarm')
    ax2.set_xlabel('$x$', fontsize=16)
    ax2.set_ylabel('$y$', fontsize=16)
    ax2.set_title('Pressure driven problem - speed', fontsize=16)
    fig.colorbar(cont2, ax=ax2)

    # Function for updating plots
    def update_plots(i):
        ax1.clear()
        ax2.clear()
    
        u_ = np.loadtxt(f'C:\\Users\\zy622\\Desktop\\MPI\\x64\\Release\\out_con\\u_{i}.dat')
        v_ = np.loadtxt(f'C:\\Users\\zy622\\Desktop\\MPI\\x64\\Release\\out_con\\v_{i}.dat')
        p_ = np.loadtxt(f'C:\\Users\\zy622\\Desktop\\MPI\\x64\\Release\\out_con\\P_{i}.dat')

        cont1 = ax1.contourf(X, Y, p_, cmap='coolwarm')
        quiv1 = ax1.quiver(X[::20, ::5], Y[::20, ::5], u_[::20, ::5], v_[::20, ::5])
        ax1.set_xlabel('$x$', fontsize=16)
        ax1.set_ylabel('$y$', fontsize=16)
        ax1.set_title('Pressure driven problem - pressure and velocity vectors', fontsize=16)
        fig.colorbar(cont1, ax=ax1)

        cont2 = ax2.contourf(X, Y, np.sqrt(u_*u_ + v_*v_), cmap='coolwarm')
        ax2.set_xlabel('$x$', fontsize=16)
        ax2.set_ylabel('$y$', fontsize=16)
        ax2.set_title('Pressure driven problem - speed', fontsize=16)
        fig.colorbar(cont2, ax=ax2)




    ani = animation.FuncAnimation(fig, update_plots, frames=100, interval=200, blit=False)
    ani.save('C:\\Users\\zy622\\Desktop\\MPI\\x64\\release\\animation\\assessment_2.gif', writer='pillow')


def main():

    variables = ['P', 'u', 'v']  # variables to concat.
    grid = [[0,1], [2,3],[4,5],[6,7]] # my processor rank structure.
    concat_files(grid, variables)
    # data = np.loadtxt('u_32.dat')
    # print(data.shape)
    animate_files(variables)

if __name__ == "__main__":
    main()
