#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>

// ====== CUDA libraries ======
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ====== OpenGL / CUDA-OpenGL Interop ======
#include <GL/glew.h>       // if using GLEW
#include <GL/freeglut.h>   // if using freeGLUT
#include <cuda_gl_interop.h>

// Time info
int currentStep = 0;
std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
float updatesPerSecond = 0.0f;
float mlups = 0.0f;
float framesPerSecond = 0.0f;

// Force the use of the NVIDIA GPU
#ifdef _WIN32
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

//-----------------------------------------------------
// Lattice parameters and simulation constants
//-----------------------------------------------------
const int nx = 1024;
const int ny = 1024;
const int numDirs = 9;
typedef float DTYPE;

__constant__ int cx_const[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
__constant__ int cy_const[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
__constant__ DTYPE w_const[9] = {
    4.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

const int cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
const int cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
const DTYPE w[9] = {
    4.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

DTYPE U = 0.3;  // Theoretically should be less than 0.577 (1/sqrt(3))
DTYPE Re = 35000.0;
DTYPE nu, tao, omega;  // nu = U*nx/Re; tao = 3*nu+0.5; omega = 1/tao

// Simulation arrays (device pointers)
DTYPE* d_f = nullptr, * d_f_new = nullptr;
char* d_mask = nullptr;

size_t simSize = nx * ny * numDirs * sizeof(DTYPE);
size_t maskSize = nx * ny * sizeof(char);

//-----------------------------------------------------
// Helper device and host inline for indexing
//-----------------------------------------------------
__device__ inline int idx(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}
inline int idx_h(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}

//-----------------------------------------------------
// CUDA kernels for the LBM solver
//-----------------------------------------------------
__global__ void collision_kernel(DTYPE* f, DTYPE omega, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        DTYPE rho = 0.0;
        DTYPE u_x = 0.0;
        DTYPE u_y = 0.0;
        for (int k = 0; k < numDirs; k++) {
            DTYPE val = f[idx(i, j, k, nx, ny)];
            rho += val;
            u_x += val * cx_const[k];
            u_y += val * cy_const[k];
        }
        if (rho > DTYPE(0.0)) {
            u_x /= rho;
            u_y /= rho;
        }
        DTYPE usqr = u_x * u_x + u_y * u_y;
        for (int k = 0; k < numDirs; k++) {
            DTYPE cu = DTYPE(3.0) * (cx_const[k] * u_x + cy_const[k] * u_y);
            DTYPE feq = w_const[k] * rho * (DTYPE(1.0) + cu + DTYPE(0.5) * cu * cu - DTYPE(1.5) * usqr);
            f[idx(i, j, k, nx, ny)] = (DTYPE(1.0) - omega) * f[idx(i, j, k, nx, ny)] + omega * feq;
        }
    }
}

__global__ void streaming_kernel(DTYPE* f_in, DTYPE* f_out, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
            for (int k = 0; k < numDirs; k++) {
                int ip = i - cx_const[k];
                int jp = j - cy_const[k];
                f_out[idx(i, j, k, nx, ny)] = f_in[idx(ip, jp, k, nx, ny)];
            }
        }
        else {
            // For simplicity, no wrap-around. Just copy as-is at borders
            for (int k = 0; k < numDirs; k++) {
                f_out[idx(i, j, k, nx, ny)] = f_in[idx(i, j, k, nx, ny)];
            }
        }
    }
}

__global__ void bounce_back_kernel(DTYPE* f, char* mask, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        if (mask[i + j * nx] == 1) {
            // Swap east (1) and west (3)
            int idx1 = idx(i, j, 1, nx, ny);
            int idx3 = idx(i, j, 3, nx, ny);
            DTYPE tmp = f[idx1];
            f[idx1] = f[idx3];
            f[idx3] = tmp;
            // Swap north (2) and south (4)
            int idx2 = idx(i, j, 2, nx, ny);
            int idx4 = idx(i, j, 4, nx, ny);
            tmp = f[idx2];
            f[idx2] = f[idx4];
            f[idx4] = tmp;
            // Swap NE (5) and SW (7)
            int idx5 = idx(i, j, 5, nx, ny);
            int idx7 = idx(i, j, 7, nx, ny);
            tmp = f[idx5];
            f[idx5] = f[idx7];
            f[idx7] = tmp;
            // Swap NW (6) and SE (8)
            int idx6 = idx(i, j, 6, nx, ny);
            int idx8 = idx(i, j, 8, nx, ny);
            tmp = f[idx6];
            f[idx6] = f[idx8];
            f[idx8] = tmp;
        }
    }
}

__global__ void moving_lid_kernel(DTYPE* f, int nx, int ny, DTYPE U) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        int j = ny - 2;
        // Zou-He velocity BC
        //DTYPE rho = f[idx(i, j, 0, nx, ny)] + f[idx(i, j, 1, nx, ny)] + f[idx(i, j, 3, nx, ny)]
        //    + 2.0f * (f[idx(i, j, 2, nx, ny)] + f[idx(i, j, 5, nx, ny)] + f[idx(i, j, 6, nx, ny)]);
        //    f[idx(i, j, 4, nx, ny)] = f[idx(i, j, 2, nx, ny)];
        //    f[idx(i, j, 7, nx, ny)] = f[idx(i, j, 5, nx, ny)]
        //        + 0.5f * (f[idx(i, j, 1, nx, ny)] - f[idx(i, j, 3, nx, ny)])
        //            - 0.5f * rho * U;
        //        f[idx(i, j, 8, nx, ny)] = f[idx(i, j, 6, nx, ny)]
        //            - 0.5f * (f[idx(i, j, 1, nx, ny)] - f[idx(i, j, 3, nx, ny)])
        //                + 0.5f * rho * U;

		// Mid-grid velocity BC
		f[idx(i, j, 4, nx, ny)] = f[idx(i, j, 2, nx, ny)];
		f[idx(i, j, 7, nx, ny)] = f[idx(i, j, 5, nx, ny)] - DTYPE(1.0) / DTYPE(6.0) * U;
		f[idx(i, j, 8, nx, ny)] = f[idx(i, j, 6, nx, ny)] + DTYPE(1.0) / DTYPE(6.0) * U;
    }
}

// Helper device function for sqrt
template<typename T>
__device__ __forceinline__ T device_sqrt(T x) {
    if constexpr (std::is_same<T, float>::value) {
        return sqrtf(x);
    }
    else {
        return sqrt(x);
    }
}

//-----------------------------------------------------
// Kernel to compute velocity magnitude into a float array
//-----------------------------------------------------
__global__ void compute_velocity_field_kernel(const DTYPE* f, DTYPE* velocity_mag, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        DTYPE rho = 0.0;
        DTYPE u_x = 0.0;
        DTYPE u_y = 0.0;
        for (int k = 0; k < numDirs; k++) {
            DTYPE val = f[idx(i, j, k, nx, ny)];
            rho += val;
            u_x += val * cx_const[k];
            u_y += val * cy_const[k];
        }
        if (rho > DTYPE(1e-12)) {
            u_x /= rho;
            u_y /= rho;
        }
        DTYPE vel = device_sqrt(u_x * u_x + u_y * u_y);
        velocity_mag[i + j * nx] = vel;
    }
}

//-----------------------------------------------------
// Kernel that copies the velocity magnitudes into RGBA
// for display, writing directly into a CUDA-mapped buffer
// (pbo) which has size (nx*ny*4 bytes).

// Blue to red color map
__global__ void fill_pbo_kernel(unsigned char* pbo_ptr,
    const DTYPE* velocity_mag,
    int nx, int ny,
    float U)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        int idx_out = 4 * (i + j * nx); // RGBA
        float v = velocity_mag[i + j * nx];

        // Normalize velocity to [0, 1] based on U
        float t = fminf(v / float(U), 1.0f); // Clamp to [0, 1]

        // Jet color bar (Blue to yellow to red color map)
        unsigned char r, g, b;
        if (t < 0.5f) {
            // Blue to yellow (0,0,255) -> (255,255,0)
            float s = t * 2.0f; // Map [0, 0.5] to [0, 1]
            r = (unsigned char)(s * 255.0f);         // 0 to 255
            g = (unsigned char)(s * 255.0f);         // 0 to 255
            b = (unsigned char)((1.0f - s) * 255.0f); // 255 to 0
        }
        else {
            // Yellow to red (255,255,0) -> (255,0,0)
            float s = (t - 0.5f) * 2.0f; // Map [0.5, 1] to [0, 1]
            r = 255;                                 // Stays 255
            g = (unsigned char)((1.0f - s) * 255.0f); // 255 to 0
            b = 0;                                   // Stays 0
        }

        pbo_ptr[idx_out + 0] = r;   // R
        pbo_ptr[idx_out + 1] = g;   // G
        pbo_ptr[idx_out + 2] = b;   // B
        pbo_ptr[idx_out + 3] = 255; // A (fully opaque)
    }
}

//-----------------------------------------------------
// Host routines for initialization and simulation
//-----------------------------------------------------
void initialize_simulation(DTYPE rho0, DTYPE ux0, DTYPE uy0) {
    // Allocate device memory
    cudaMalloc(&d_f, simSize);
    cudaMalloc(&d_f_new, simSize);
    cudaMalloc(&d_mask, maskSize);

    // Initialize f on host (uniform density=1, velocity=0)
    DTYPE* h_f = new DTYPE[nx * ny * numDirs];
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            DTYPE usq = ux0 * ux0 + uy0 * uy0;
            for (int k = 0; k < numDirs; k++) {
                DTYPE cu = DTYPE(3.0) * (cx[k] * ux0 + cy[k] * uy0);
                h_f[idx_h(i, j, k, nx, ny)] = w[k] * rho0 * (DTYPE(1.0) + cu + DTYPE(0.5) * cu * cu - DTYPE(1.5) * usq);
            }
        }
    }
    cudaMemcpy(d_f, h_f, simSize, cudaMemcpyHostToDevice);
    delete[] h_f;

    // Initialize mask (solid boundary)
    char* h_mask = new char[nx * ny];
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (i == 1 || i == nx - 2 || j == 1)
                h_mask[i + j * nx] = 1;
            else
                h_mask[i + j * nx] = 0;
        }
    }
    cudaMemcpy(d_mask, h_mask, maskSize, cudaMemcpyHostToDevice);
    delete[] h_mask;
}

//--------------------------------------------------------------------------------------------
// Runs one simulation step
// Since all kernel launches use the default stream, removing all the cudaDeviceSynchronize()
// calls will not affect the correctness of the LBM calculation, but increase the performance
//--------------------------------------------------------------------------------------------
void simulation_step() {
    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    collision_kernel <<<gridDim, blockDim>>> (d_f, omega, nx, ny);
    //cudaDeviceSynchronize();

    streaming_kernel <<<gridDim, blockDim>>> (d_f, d_f_new, nx, ny);
    //cudaDeviceSynchronize();

    bounce_back_kernel <<<gridDim, blockDim>>> (d_f_new, d_mask, nx, ny);
    //cudaDeviceSynchronize();

    // Apply moving lid
    dim3 blockDim1(256);
    dim3 gridDim1((nx + blockDim1.x - 1) / blockDim1.x);
    moving_lid_kernel <<<gridDim1, blockDim1>>> (d_f_new, nx, ny, U);
    //cudaDeviceSynchronize();

    // Swap pointers
    DTYPE* temp = d_f;
    d_f = d_f_new;
    d_f_new = temp;
}

//-----------------------------------------------------
// Global (static) OpenGL/CUDA variables
//-----------------------------------------------------
static GLuint pbo = 0;                             // OpenGL pixel buffer object
static struct cudaGraphicsResource* cuda_pbo = nullptr;
static DTYPE* d_velocity = nullptr;                // device array for velocity magnitude
static const int WIN_WIDTH = nx;                  // match your lattice dims
static const int WIN_HEIGHT = ny;
static int stepsPerFrame = 100;                  // how many LBM steps per OpenGL frame?

//-----------------------------------------------------
// Create the PBO and register it with CUDA
//-----------------------------------------------------
void create_pbo() {
    // Generate a buffer ID for the PBO
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    // Allocate the buffer (RGBA, 1 byte each, total 4 bytes/pixel)
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIN_WIDTH * WIN_HEIGHT * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Register this buffer object with CUDA
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

    // Also allocate d_velocity
    cudaMalloc((void**)&d_velocity, nx * ny * sizeof(DTYPE));
}

//-----------------------------------------------------
// Cleanup
//-----------------------------------------------------
void cleanup() {
    if (cuda_pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo);
        cuda_pbo = nullptr;
    }
    if (pbo) {
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    if (d_velocity) {
        cudaFree(d_velocity);
        d_velocity = nullptr;
    }
    // Free LBM arrays
    if (d_f)       cudaFree(d_f);
    if (d_f_new)   cudaFree(d_f_new);
    if (d_mask)    cudaFree(d_mask);
}

//-----------------------------------------------------
// Render callback: called by GLUT whenever we want to
// redraw the screen.
//-----------------------------------------------------
void display() {
    // 0) Record start time
    if (currentStep == 0) {
        startTime = std::chrono::high_resolution_clock::now();
    }

    // 1) Run LBM time steps
    for (int s = 0; s < stepsPerFrame; s++) {
        simulation_step();
		currentStep++;
    }

    // 2) Compute velocity magnitude on GPU
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    compute_velocity_field_kernel <<<grid, block>>> (d_f, d_velocity, nx, ny);
    cudaDeviceSynchronize();

    // 3) Map the PBO so we can write into it from CUDA
    cudaGraphicsMapResources(1, &cuda_pbo, 0);
    unsigned char* d_pbo_ptr = nullptr;
    size_t num_bytes = 0;
    cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_ptr, &num_bytes, cuda_pbo);

    // 4) Fill PBO with color from velocity
    fill_pbo_kernel <<<grid, block>>> (d_pbo_ptr, d_velocity, nx, ny, U);
    cudaDeviceSynchronize();

    // 5) Unmap
    cudaGraphicsUnmapResources(1, &cuda_pbo, 0);

    // 6) Clear screen and draw the pixel buffer
    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2f(-1, -1); // draw from bottom-left
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    // The data we have is 8-bit RGBA
    glDrawPixels(WIN_WIDTH, WIN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // 7) Compute and display time step, update per second, and MLUPS
    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsedTime = currentTime - startTime;
    updatesPerSecond = float(currentStep) / elapsedTime.count();
    framesPerSecond = updatesPerSecond / float(stepsPerFrame);
    mlups = (float(currentStep) * float(nx) * float(ny)) / (elapsedTime.count() * float(1e6));

    std::ostringstream oss;
    oss << "Time Step: " << currentStep 
        << "  UPS: " << std::fixed << std::setprecision(1) << updatesPerSecond 
        << "  MLUPS: " << std::fixed << std::setprecision(2) << mlups
        << "  FPS: " << std::fixed << std::setprecision(1) << framesPerSecond;
    std::string info = oss.str();

	glColor3f(1.0f, 1.0f, 1.0f); // white text
	glRasterPos2f(-0.95f, 0.95f); // upper-left corner
    for (char c : info) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
    }

    // 8) Swap buffers
    glutSwapBuffers();
}

//-----------------------------------------------------
// Optional: Idle callback �C just request a new display.
// Could also use glutTimerFunc for fixed framerate.
//-----------------------------------------------------
void idle() {
    glutPostRedisplay();
}

//-----------------------------------------------------
// OpenGL init
//-----------------------------------------------------
void initGL(int* argc, char** argv) {
    // Initialize freeGLUT
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);
    glutCreateWindow("LBM + OpenGL Visualization");

    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("OpenGL vendor: %s\n", glGetString(GL_VENDOR));
    printf("OpenGL renderer: %s\n", glGetString(GL_RENDERER));

    // Initialize GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(err));
        exit(1);
    }

    // Create the PBO
    create_pbo();

    // Set callbacks
    glutDisplayFunc(display);
    glutIdleFunc(idle);

    // Basic GL state
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    // Enable blend to display text
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

//-----------------------------------------------------
// Main
//-----------------------------------------------------
int main(int argc, char** argv) {
    // 1) Compute relaxation parameter
	nu = U * DTYPE(nx) / Re;
    tao = DTYPE(3.0) * nu + DTYPE(0.5);
    omega = DTYPE(1.0) / tao;
	DTYPE rho0 = 1.0;
    DTYPE ux0 = 0.0;
	DTYPE uy0 = 0.0;
	// Print nu and tao and omega
	printf("Viscosity = %f, Relaxation time = %f, Omega = %f\n", nu, tao, omega);
	// Print U and Re
	printf("U = %f, Re = %f\n", U, Re);

    // 2) Init the LBM arrays on GPU
    initialize_simulation(rho0, ux0, uy0);

    // 3) Init OpenGL and enter GLUT main loop
    initGL(&argc, argv);
    glutMainLoop();

    // 4) Cleanup (won't usually reach here unless you close the window)
    cleanup();
    return 0;
}
