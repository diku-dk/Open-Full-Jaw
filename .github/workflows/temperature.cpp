#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <string>
#include <limits>

using std::cin;
using std::cout;
using std::cerr;
using std::endl;

static inline int idx(int x, int y, int z, int N) {
    return (z * N + y) * N + x;
}

static void print_slice(const std::vector<double>& T, int N, int z_slice, int step) {
    // Print a simple ASCII heatmap-like slice at z=z_slice
    // Map temperature to characters
    const std::string ramp = " .:-=+*#%@"; // low -> high

    double tmin = std::numeric_limits<double>::infinity();
    double tmax = -std::numeric_limits<double>::infinity();

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            double v = T[idx(x,y,z_slice,N)];
            tmin = std::min(tmin, v);
            tmax = std::max(tmax, v);
        }
    }

    cout << "\n=== Step " << step << " | z = " << z_slice
         << " | slice Tmin=" << tmin << " Tmax=" << tmax << " ===\n";

    // Avoid divide by zero
    double denom = (tmax - tmin);
    if (denom < 1e-12) denom = 1.0;

    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            double v = T[idx(x,y,z_slice,N)];
            double a = (v - tmin) / denom;           // 0..1
            int k = (int)std::round(a * (ramp.size() - 1));
            k = std::clamp(k, 0, (int)ramp.size() - 1);
            cout << ramp[k] << ramp[k]; // double-width for readability
        }
        cout << "\n";
    }
}

int main() {
    // 3D heat diffusion (explicit finite difference) in a cube N x N x N
    // Boundary condition: fixed at 0 (Dirichlet) on all boundaries
    // Initial condition: a point source at (x0,y0,z0) with temperature T0

    int N = 0;
    double alpha = 0.0; // diffusion coefficient
    int steps = 0;
    double dt = 0.0;
    int x0 = 0, y0 = 0, z0 = 0;
    double T0 = 0.0;

    cout << "3D Heat Diffusion in a Cube (Explicit FD)\n";
    cout << "Enter grid size N (e.g., 30 or 40): ";
    cin >> N;
    cout << "Enter diffusion coefficient alpha (e.g., 1.0): ";
    cin >> alpha;
    cout << "Enter time step dt (e.g., 0.1): ";
    cin >> dt;
    cout << "Enter number of steps (e.g., 200): ";
    cin >> steps;
    cout << "Enter source coordinate x0 y0 z0 (0..N-1): ";
    cin >> x0 >> y0 >> z0;
    cout << "Enter source temperature T0 (e.g., 100.0): ";
    cin >> T0;

    if (N < 3) {
        cerr << "N must be >= 3.\n";
        return 1;
    }
    if (x0 < 0 || x0 >= N || y0 < 0 || y0 >= N || z0 < 0 || z0 >= N) {
        cerr << "Source coordinate out of range.\n";
        return 1;
    }
    if (alpha <= 0 || dt <= 0 || steps <= 0) {
        cerr << "alpha, dt, steps must be positive.\n";
        return 1;
    }

    // Grid spacing (assume unit cube with dx = 1)
    // Stability for explicit 3D: alpha*dt/dx^2 <= 1/6
    const double dx = 1.0;
    const double r = alpha * dt / (dx * dx);

    if (r > (1.0 / 6.0)) {
        cerr << "Warning: explicit scheme may be unstable.\n";
        cerr << "Current r = alpha*dt/dx^2 = " << r << " > 1/6 (~0.1667).\n";
        cerr << "Consider smaller dt or alpha.\n";
        // Not exiting; user may still proceed.
    }

    const int NN = N * N * N;
    std::vector<double> T(NN, 0.0), Tnew(NN, 0.0);

    // initial: point source
    T[idx(x0,y0,z0,N)] = T0;

    // Choose a slice to visualize
    int z_slice = z0; // show the slice containing the source

    // Print step 0
    print_slice(T, N, z_slice, 0);

    for (int s = 1; s <= steps; ++s) {
        // Apply update only for interior nodes; boundaries fixed at 0
        // Tnew = T + r*(sum of 6 neighbors - 6*T)
        std::fill(Tnew.begin(), Tnew.end(), 0.0);

        for (int z = 1; z < N-1; ++z) {
            for (int y = 1; y < N-1; ++y) {
                for (int x = 1; x < N-1; ++x) {
                    double center = T[idx(x,y,z,N)];
                    double lap =
                        T[idx(x+1,y,z,N)] + T[idx(x-1,y,z,N)] +
                        T[idx(x,y+1,z,N)] + T[idx(x,y-1,z,N)] +
                        T[idx(x,y,z+1,N)] + T[idx(x,y,z-1,N)] -
                        6.0 * center;

                    Tnew[idx(x,y,z,N)] = center + r * lap;
                }
            }
        }

        T.swap(Tnew);

        // Print every k steps to look like animation (adjust as you like)
        int every = std::max(1, steps / 50); // about 50 frames
        if (s % every == 0 || s == steps) {
            print_slice(T, N, z_slice, s);
        }
    }

    cout << "\nDone.\n";
    cout << "Tip: You can rerun with different dt/steps/N for smoother animation.\n";
    return 0;
}
