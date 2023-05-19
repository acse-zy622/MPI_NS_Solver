#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cmath>
#include <mpi.h>

using namespace std;

const int Nx = 201;
const int Ny = 101;

const double Lx = 0.1, Ly = 0.05;
const double rho = 1000, nu = 1e-6;
const double P_max = 0.5;
const double t_end = 50.0;
const double dt_min = 1.e-3;
const double courant = 0.01;
const double dt_out = 0.5;

vector<vector<double>> P, P_old, u, u_old, v, v_old, PPrhs;
double dx, dy, dt, t;
int id, p;

void grids_to_file(int out)
{
	//Write the output for a single time step to file for each processor.
	stringstream fname;
	fstream f1;

	fname << "./out/P" << "_" << out << "_" << id << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i < Nx; i++)
	{
		for (int j = 0; j < Ny; j++)
			f1 << P[i][j] << "\t";
		f1 << endl;
	}
	f1.close();
	fname.str("");
	fname << "./out/u" << "_" << out << "_" << id << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i < Nx; i++)
	{
		for (int j = 0; j < Ny; j++)
			f1 << u[i][j] << "\t";
		f1 << endl;
	}
	f1.close();
	fname.str("");
	fname << "./out/v" << "_" << out << "_" << id << ".dat";
	f1.open(fname.str().c_str(), ios_base::out);
	for (int i = 0; i < Nx; i++)
	{
		for (int j = 0; j < Ny; j++)
			f1 << v[i][j] << "\t";
		f1 << endl;
	}
	f1.close();
}

void setup(int local_Nx, int local_Ny, int id_x) {

	// Resize vectors to match local domain size, plus 2 for padding to receive data from neighbouring processors at the most case.
	P.resize(local_Nx + 2, vector<double>(local_Ny + 2, 0.0));
	P_old.resize(local_Nx + 2, vector<double>(local_Ny + 2, 0.0));
	u.resize(local_Nx + 2, vector<double>(local_Ny + 2, 0.0));
	u_old.resize(local_Nx + 2, vector<double>(local_Ny + 2, 0.0));
	v.resize(local_Nx + 2, vector<double>(local_Ny + 2, 0.0));
	v_old.resize(local_Nx + 2, vector<double>(local_Ny + 2, 0.0));
	PPrhs.resize(local_Nx + 2, vector<double>(local_Ny + 2, 0.0));

	// Compute dx and dy 
	dx = Lx / (local_Nx - 1);
	dy = Ly / (local_Ny - 1);

	// Initialize pressure at boundary
	if (id_x == 0) {
		for (int j = 1; j < local_Ny+1; j++) {
			P[1][j] = P_max;
		}
	}

	P_old = P;
	t = 0.0;
}

void set_pressure_BCs(int local_Nx, int local_Ny, int id_y, int p_y, int id_x, int p_x)
{
	// left boundary
	if (id_y == 0)
		for (int i = 1; i < local_Nx+1; i++)
			P[i][1] = P[i][2];
	// right boundary
	if (id_y == p_y - 1)
		for (int i = 1; i < local_Nx+1; i++)
			P[i][local_Ny] = P[i][local_Ny - 1];

	//somewhere in the middle
	if (id_x == p_x - 1)
		for (int j = local_Ny / 2; j < local_Ny+1; j++)
			P[local_Nx][j] = P[local_Nx -1][j];
}

void set_velocity_BCs(int local_Nx, int local_Ny, int id_y, int p_y, int id_x, int p_x)
{
	// at the bottom boundary
	if (id_x == 0)
		for (int j = 1; j < local_Ny+1; j++)
			u[1][j] = u[2][j];

	// at the top boudary
	if (id_x == p_x - 1)
		for (int j = 1; j < local_Ny / 2; j++)
			u[local_Nx][j] = u[local_Nx-1][j];
}

void calculate_intermediate_velocity(int local_Nx, int local_Ny, int id_y, int p_y, int id_x, int p_x)
{
	MPI_Request send_request[8], recv_request[8];
	int num_reqs = 0;
	int tag = 0;

	// Initialize all MPI_Request objects to MPI_REQUEST_NULL to handle corner conditions.
	for (int i = 0; i < 8; i++) {
		send_request[i] = MPI_REQUEST_NULL;
		recv_request[i] = MPI_REQUEST_NULL;
	}

	MPI_Datatype row_type, column_type;

	// Create a new MPI data type for a row
	MPI_Type_contiguous(local_Ny, MPI_DOUBLE, &row_type);
	MPI_Type_commit(&row_type);

	// Create a new MPI data type for a column
	MPI_Type_vector(local_Nx, 1, local_Ny + 2, MPI_DOUBLE, &column_type);
	MPI_Type_commit(&column_type);

	for (int i = 2; i < local_Nx; i++)
		for (int j = 2; j < local_Ny; j++)
		{
			//viscous diffusion
			u[i][j] = u_old[i][j] + dt * nu * ((u_old[i + 1][j] + u_old[i - 1][j] - 2.0 * u_old[i][j]) / (dx * dx) + (u_old[i][j + 1] + u_old[i][j - 1] - 2.0 * u_old[i][j]) / (dy * dy));
			v[i][j] = v_old[i][j] + dt * nu * ((v_old[i + 1][j] + v_old[i - 1][j] - 2.0 * v_old[i][j]) / (dx * dx) + (v_old[i][j + 1] + v_old[i][j - 1] - 2.0 * v_old[i][j]) / (dy * dy));

			//advection - upwinding
			if (u[i][j] > 0.0)
			{
				u[i][j] -= dt * u_old[i][j] * (u_old[i][j] - u_old[i - 1][j]) / dx;
				v[i][j] -= dt * u_old[i][j] * (v_old[i][j] - v_old[i - 1][j]) / dx;
			}
			else
			{
				u[i][j] -= dt * u_old[i][j] * (u_old[i + 1][j] - u_old[i][j]) / dx;
				v[i][j] -= dt * u_old[i][j] * (v_old[i + 1][j] - v_old[i][j]) / dx;
			}
			if (v[i][j] > 0.0)
			{
				u[i][j] -= dt * v_old[i][j] * (u_old[i][j] - u_old[i][j - 1]) / dy;
				v[i][j] -= dt * v_old[i][j] * (v_old[i][j] - v_old[i][j - 1]) / dy;
			}
			else
			{
				u[i][j] -= dt * v_old[i][j] * (u_old[i][j + 1] - u_old[i][j]) / dy;
				v[i][j] -= dt * v_old[i][j] * (v_old[i][j + 1] - v_old[i][j]) / dy;
			}
		}

	//communication for u
	if (id_x < p_x - 1) {  // right
		MPI_Isend(&u[local_Nx][1], 1, column_type, id + 1, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&u[local_Nx + 1][1], 1, column_type, id + 1, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_x > 0) {  // left
		MPI_Isend(&u[1][1], 1, column_type, id - 1, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&u[0][1], 1, column_type, id - 1, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_y < p_y - 1) {  // Up
		MPI_Isend(&u[1][local_Ny], 1, row_type, id + p_x, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&u[1][local_Ny + 1], 1, row_type, id + p_x, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_y > 0) {  // Down
		MPI_Isend(&u[1][1], 1, row_type, id - p_x, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&u[1][0], 1, row_type, id - p_x, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}

	//communication for v
	if (id_x < p_x - 1) {  // right
		MPI_Isend(&v[local_Nx][1], 1, column_type, id + 1, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&v[local_Nx + 1][1], 1, column_type, id + 1, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_x > 0) {  // left
		MPI_Isend(&v[1][1], 1, column_type, id - 1, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&v[0][1], 1, column_type, id - 1, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_y < p_y - 1) {  // Up
		MPI_Isend(&v[1][local_Ny], 1, row_type, id + p_x, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&v[1][local_Ny + 1], 1, row_type, id + p_x, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_y > 0) {  // Down
		MPI_Isend(&v[1][1], 1, row_type, id - p_x, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&v[1][0], 1, row_type, id - p_x, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}

	MPI_Waitall(num_reqs, send_request, MPI_STATUSES_IGNORE);
	MPI_Waitall(num_reqs, recv_request, MPI_STATUSES_IGNORE);

	// Free the custom MPI data types after we're done
	MPI_Type_free(&row_type);
	MPI_Type_free(&column_type);
}

void calculate_ppm_RHS_central(int local_Nx, int local_Ny)
{
	for (int i = 2; i < local_Nx ; i++)
		for (int j = 2; j < local_Ny ; j++)
		{
			PPrhs[i][j] = rho / dt * ((u[i + 1][j] - u[i - 1][j]) / (2. * dx) + (v[i][j + 1] - v[i][j - 1]) / (2. * dy));
		}
}

int pressure_poisson_jacobi(int local_Nx, int local_Ny, int id_y, int p_y, int id_x, int p_x, double rtol = 1.e-5)
{
	double tol = 10. * rtol;
	int it = 0;
	swap(P, P_old);
	// MPI request arrays
	MPI_Request send_request[8], recv_request[8];
	int num_reqs;
	int tag = 1;
	MPI_Datatype row_type, column_type;

	// Create a new MPI data type for a row
	MPI_Type_contiguous(local_Ny, MPI_DOUBLE, &row_type);
	MPI_Type_commit(&row_type);

	// Create a new MPI data type for a column
	MPI_Type_vector(local_Nx, 1, local_Ny + 2, MPI_DOUBLE, &column_type);
	MPI_Type_commit(&column_type);


	while (tol > rtol)
	{
		double sum_val = 0.0;
		tol = 0.0;
		it++;

		//Jacobi iteration
		for (int i = 2; i < local_Nx ; i++)
			for (int j = 2; j < local_Ny ; j++)
			{
				P[i][j] = 1.0 / (2.0 + 2.0 * (dx * dx) / (dy * dy)) * (P_old[i + 1][j] + P_old[i - 1][j] +
					(P_old[i][j + 1] + P_old[i][j - 1]) * (dx * dx) / (dy * dy)
					- (dx * dx) * PPrhs[i][j]);
				sum_val += fabs(P[i][j]);
				tol += fabs(P[i][j] - P_old[i][j]);
			}

		num_reqs = 0; // reset num_reqs to zero before making requests for each iteration.

		// Initialize all MPI_Request objects to MPI_REQUEST_NULL to handle corner conditions.
		for (int i = 0; i < 8; i++) {
			send_request[i] = MPI_REQUEST_NULL;
			recv_request[i] = MPI_REQUEST_NULL;
		}

		if (id_x < p_x - 1) {  // right
			MPI_Isend(&P[local_Nx][1], 1, column_type, id + 1, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
			MPI_Irecv(&P[local_Nx + 1][1], 1, column_type, id + 1, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
			num_reqs++;
		}
		if (id_x > 0) {  // left
			MPI_Isend(&P[1][1], 1, column_type, id - 1, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
			MPI_Irecv(&P[0][1], 1, column_type, id - 1, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
			num_reqs++;
		}
		if (id_y < p_y - 1) {  // Up

			MPI_Isend(&P[1][local_Ny], 1, row_type, id + p_x, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
			MPI_Irecv(&P[1][local_Ny + 1], 1, row_type, id + p_x, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
			num_reqs++;
		}
		if (id_y > 0) {  // Down

			MPI_Isend(&P[1][1], 1, row_type, id - p_x, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
			MPI_Irecv(&P[1][0], 1, row_type, id - p_x, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
			num_reqs++;
		}

		MPI_Waitall(num_reqs, send_request, MPI_STATUSES_IGNORE);
		MPI_Waitall(num_reqs, recv_request, MPI_STATUSES_IGNORE);
		set_pressure_BCs(local_Nx, local_Ny, id_y, p_y, id_x, p_x);
		double global_tol, global_sum_val;

		MPI_Allreduce(&tol, &global_tol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&sum_val, &global_sum_val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		tol = global_tol / max(1.e-10, global_sum_val);

		swap(P, P_old);
	}

	MPI_Type_free(&row_type);
	MPI_Type_free(&column_type);
	return it;
}

double project_velocity(int local_Nx, int local_Ny, int id_y, int p_y, int id_x, int p_x)
{
	double vmax = 0.0;
	// MPI request arrays
	MPI_Request send_request[8], recv_request[8];
	int num_reqs = 0;
	int tag = 2;
	// Initialize all MPI_Request objects to MPI_REQUEST_NULL to handle corner conditions.
	for (int i = 0; i < 8; i++) {
		send_request[i] = MPI_REQUEST_NULL;
		recv_request[i] = MPI_REQUEST_NULL;
	}

	MPI_Datatype row_type, column_type;

	// Create a new MPI data type for a row
	MPI_Type_contiguous(local_Ny, MPI_DOUBLE, &row_type);
	MPI_Type_commit(&row_type);

	// Create a new MPI data type for a column
	MPI_Type_vector(local_Nx, 1, local_Ny + 2, MPI_DOUBLE, &column_type);
	MPI_Type_commit(&column_type);

	for (int i = 2; i < local_Nx ; i++)
		for (int j = 2; j < local_Ny; j++)
		{
			u[i][j] = u[i][j] - dt * (1. / rho) * (P[i + 1][j] - P[i - 1][j]) / (2. * dx);
			v[i][j] = v[i][j] - dt * (1. / rho) * (P[i][j + 1] - P[i][j - 1]) / (2. * dy);

			double vel = sqrt(u[i][j] * u[i][j] + v[i][j] * v[i][j]);

			vmax = max(vmax, vel);
		}

	if (id_x < p_x - 1) {  //right
		MPI_Isend(&u[local_Nx][1], 1, column_type, id + 1, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&u[local_Nx + 1][1], 1, column_type, id + 1, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_x > 0) {  // left
		MPI_Isend(&u[1][1], 1, column_type, id - 1, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&u[0][1], 1, column_type, id - 1, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_y < p_y - 1) {  // Up
		MPI_Isend(&u[1][local_Ny], 1, row_type, id + p_x, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&u[1][local_Ny + 1], 1, row_type, id + p_x, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_y > 0) {  // Down
		MPI_Isend(&u[1][1], 1, row_type, id - p_x, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&u[1][0], 1, row_type, id - p_x, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}

	//communication for v
	if (id_x < p_x - 1) {  // right
		MPI_Isend(&v[local_Nx][1], 1, column_type, id + 1, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&v[local_Nx + 1][1], 1, column_type, id + 1, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_x > 0) {  // left
		MPI_Isend(&v[1][1], 1, column_type, id - 1, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&v[0][1], 1, column_type, id - 1, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_y < p_y - 1) {  // Up
		MPI_Isend(&v[1][local_Ny], 1, row_type, id + p_x, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&v[1][local_Ny + 1], 1, row_type, id + p_x, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}
	if (id_y > 0) {  // Down
		MPI_Isend(&v[1][1], 1, row_type, id - p_x, tag, MPI_COMM_WORLD, &send_request[num_reqs]);
		MPI_Irecv(&v[1][0], 1, row_type, id - p_x, tag, MPI_COMM_WORLD, &recv_request[num_reqs]);
		num_reqs++;
	}

	set_velocity_BCs(local_Nx, local_Ny, id_y,p_y,id_x,p_x);

	// Wait for all MPI communications to complete
	MPI_Waitall(num_reqs, send_request, MPI_STATUSES_IGNORE);
	MPI_Waitall(num_reqs, recv_request, MPI_STATUSES_IGNORE);

	// Free the custom MPI data types after we're done
	MPI_Type_free(&row_type);
	MPI_Type_free(&column_type);

	return vmax;
}

void solve_NS(int local_Nx, int local_Ny, int id_y, int p_y, int id_x, int p_x)
{
	double vel_max = 0.0;
	int time_it = 0;
	int its;
	int out_it = 0;
	double t_out = dt_out;
	grids_to_file(out_it);

	while (t < t_end)
	{
		if (vel_max > 0.0)
		{
			dt = min(courant * min(dx, dy) / vel_max, dt_min);
		}
		else dt = dt_min;

		double dt_global;
		// Collectively communicate with the minimum time step.
		MPI_Allreduce(&dt, &dt_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		dt = dt_global;

		t += dt;
		time_it++;

		swap(u, u_old);
		swap(v, v_old);


		calculate_intermediate_velocity(local_Nx, local_Ny, id_y, p_y, id_x, p_x);
		/*cout << "PASSV" << endl;*/
		calculate_ppm_RHS_central(local_Nx, local_Ny);
		/*cout << "PASSR" << endl;*/
		its = pressure_poisson_jacobi(local_Nx, local_Ny, id_y, p_y, id_x, p_x, 1.e-5);
		/*cout << "PASSP" << endl;*/
		vel_max = project_velocity(local_Nx, local_Ny, id_y, p_y, id_x, p_x);
		/*cout << "PASSVfinal" << endl;*/

		if (t >= t_out)
		{
			out_it++;
			t_out += dt_out;
			std::cout << time_it << ": " << t << " Jacobi iterations: " << its << " vel_max: " << vel_max << endl;
			grids_to_file(out_it);
		}
	}
}

int main(int argc, char** argv) {

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	// Compute the number of processors in each direction.
	int p_x, p_y;
	for (int i = 1; i <= sqrt(p); i++) {
		if (p % i == 0) {
			p_x = i;
			p_y = p / i;
		}
	}

	// Compute the coordinates of this process in a 2D grid .
	int id_x = id % p_x;
	int id_y = id / p_x;

	// Compute the size of the local domain for this process.

	int local_Nx = Nx / p_x;
	int local_Ny = Ny / p_y;

	setup(local_Nx, local_Ny, id_x);

	solve_NS(local_Nx, local_Ny, id_y, p_y, id_x, p_x);

	MPI_Finalize();
	return 0;
}