#include <chrono>
#include <iomanip>
#include <string>
#include <sstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>

int currentStep = 0;
std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
float updatesPerSecond = 0.0f;
float mlups = 0.0f;
float framesPerSecond = 0.0f;

#ifdef _WIN32
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
}
#endif

const int nx = 512;
const int ny = 512;
typedef float DTYPE;

DTYPE U = 0.3;
DTYPE Re = 17500.0;
DTYPE nu;
const DTYPE L = 1.0;
const DTYPE h = L / (nx - 1);
const DTYPE dt = 0.001;
const DTYPE rho = 1.0;
const int num_jacobi_iters = 20;

DTYPE* d_u = nullptr, * d_v = nullptr, * d_p = nullptr;
DTYPE* d_u_star = nullptr, * d_v_star = nullptr;
DTYPE* d_div = nullptr, * d_p_new = nullptr;
DTYPE* d_vorticity = nullptr;

size_t velSize = nx * ny * sizeof(DTYPE);

// **FDM Kernels**

__global__ void initialize_velocity(DTYPE* u, DTYPE* v, DTYPE U, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        if (j == ny - 1) u[i + j * nx] = U;
        else u[i + j * nx] = 0;
        v[i + j * nx] = 0;
    }
}

__global__ void set_boundary_conditions(DTYPE* u_star, DTYPE* v_star, DTYPE U, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        // Bottom wall
        u_star[i + 0 * nx] = 0;
        v_star[i + 0 * nx] = 0;
        // Top lid
        u_star[i + (ny - 1) * nx] = U;
        v_star[i + (ny - 1) * nx] = 0;
    }
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < ny) {
        // Left wall
        u_star[0 + j * nx] = 0;
        v_star[0 + j * nx] = 0;
        // Right wall
        u_star[(nx - 1) + j * nx] = 0;
        v_star[(nx - 1) + j * nx] = 0;
    }
}

__global__ void compute_intermediate_velocity(DTYPE* u, DTYPE* v, DTYPE* u_star, DTYPE* v_star, DTYPE dt, DTYPE nu, DTYPE h, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        // Advection term for u
        DTYPE du_dx, du_dy;
        if (u[i + j * nx] > 0) du_dx = (u[i + j * nx] - u[(i - 1) + j * nx]) / h;
        else du_dx = (u[(i + 1) + j * nx] - u[i + j * nx]) / h;
        if (v[i + j * nx] > 0) du_dy = (u[i + j * nx] - u[i + (j - 1) * nx]) / h;
        else du_dy = (u[i + (j + 1) * nx] - u[i + j * nx]) / h;
        DTYPE advection_u = u[i + j * nx] * du_dx + v[i + j * nx] * du_dy;

        // Diffusion term for u
        DTYPE laplacian_u = (u[(i + 1) + j * nx] + u[(i - 1) + j * nx] + u[i + (j + 1) * nx] + u[i + (j - 1) * nx] - 4 * u[i + j * nx]) / (h * h);
        u_star[i + j * nx] = u[i + j * nx] + dt * (-advection_u + nu * laplacian_u);

        // Advection term for v
        DTYPE dv_dx, dv_dy;
        if (u[i + j * nx] > 0) dv_dx = (v[i + j * nx] - v[(i - 1) + j * nx]) / h;
        else dv_dx = (v[(i + 1) + j * nx] - v[i + j * nx]) / h;
        if (v[i + j * nx] > 0) dv_dy = (v[i + j * nx] - v[i + (j - 1) * nx]) / h;
        else dv_dy = (v[i + (j + 1) * nx] - v[i + j * nx]) / h;
        DTYPE advection_v = u[i + j * nx] * dv_dx + v[i + j * nx] * dv_dy;

        // Diffusion term for v
        DTYPE laplacian_v = (v[(i + 1) + j * nx] + v[(i - 1) + j * nx] + v[i + (j + 1) * nx] + v[i + (j - 1) * nx] - 4 * v[i + j * nx]) / (h * h);
        v_star[i + j * nx] = v[i + j * nx] + dt * (-advection_v + nu * laplacian_v);
    }
}

__global__ void compute_divergence(DTYPE* u_star, DTYPE* v_star, DTYPE* div, DTYPE h, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        DTYPE du_dx, dv_dy;
        if (i == 0) du_dx = (u_star[i + 1 + j * nx] - u_star[i + j * nx]) / h;
        else if (i == nx - 1) du_dx = (u_star[i + j * nx] - u_star[i - 1 + j * nx]) / h;
        else du_dx = (u_star[i + 1 + j * nx] - u_star[i - 1 + j * nx]) / (2 * h);
        if (j == 0) dv_dy = (v_star[i + (j + 1) * nx] - v_star[i + j * nx]) / h;
        else if (j == ny - 1) dv_dy = (v_star[i + j * nx] - v_star[i + (j - 1) * nx]) / h;
        else dv_dy = (v_star[i + (j + 1) * nx] - v_star[i + (j - 1) * nx]) / (2 * h);
        div[i + j * nx] = du_dx + dv_dy;
    }
}

__global__ void jacobi_iteration(DTYPE* p, DTYPE* p_new, DTYPE* div, DTYPE dt, DTYPE rho, DTYPE h, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        DTYPE b = (rho / dt) * div[i + j * nx];
        p_new[i + j * nx] = (p[(i + 1) + j * nx] + p[(i - 1) + j * nx] + p[i + (j + 1) * nx] + p[i + (j - 1) * nx] - h * h * b) / 4.0;
    }
}

__global__ void set_pressure_bc(DTYPE* p_new, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        p_new[i + 0 * nx] = p_new[i + 1 * nx];         // Bottom
        p_new[i + (ny - 1) * nx] = p_new[i + (ny - 2) * nx]; // Top
    }
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < ny) {
        p_new[0 + j * nx] = p_new[1 + j * nx];         // Left
        p_new[(nx - 1) + j * nx] = p_new[(nx - 2) + j * nx]; // Right
    }
}

__global__ void correct_velocity(DTYPE* u, DTYPE* v, DTYPE* u_star, DTYPE* v_star, DTYPE* p, DTYPE dt, DTYPE rho, DTYPE h, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (i < nx - 1 && j < ny - 1) {
        DTYPE dp_dx = (p[i + 1 + j * nx] - p[i - 1 + j * nx]) / (2 * h);
        DTYPE dp_dy = (p[i + (j + 1) * nx] - p[i + (j - 1) * nx]) / (2 * h);
        u[i + j * nx] = u_star[i + j * nx] - (dt / rho) * dp_dx;
        v[i + j * nx] = v_star[i + j * nx] - (dt / rho) * dp_dy;
    }
}

// **Visualization Kernels**

template<typename T>
__device__ __forceinline__ T device_sqrt(T x) {
    return sqrtf(x);
}

__global__ void compute_velocity_field_kernel(const DTYPE* u, const DTYPE* v, DTYPE* velocity_mag, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        DTYPE u_val = u[i + j * nx];
        DTYPE v_val = v[i + j * nx];
        velocity_mag[i + j * nx] = device_sqrt(u_val * u_val + v_val * v_val);
    }
}

__global__ void compute_vorticity_kernel(const DTYPE* ux, const DTYPE* uy, DTYPE* vorticity, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
        DTYPE d_uy_dx = (uy[(i + 1) + j * nx] - uy[(i - 1) + j * nx]) / DTYPE(2.0);
        DTYPE d_ux_dy = (ux[i + (j + 1) * nx] - ux[i + (j - 1) * nx]) / DTYPE(2.0);
        vorticity[i + j * nx] = d_uy_dx - d_ux_dy;
    }
    else {
        vorticity[i + j * nx] = DTYPE(0.0);
    }
}

__global__ void fill_pbo_kernel(unsigned char* pbo_ptr, const DTYPE* data, int nx, int ny, float U, int mode, float min_vort, float max_vort) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        int idx_out = 4 * (i + j * nx);
        float val = data[i + j * nx];
        unsigned char r, g, b;

        if (mode == 0) {
            float t = fminf(val / U, 1.0f);
            if (t < 0.5f) {
                float s = t * 2.0f;
                r = (unsigned char)(s * 255.0f);
                g = (unsigned char)(s * 255.0f);
                b = (unsigned char)((1.0f - s) * 255.0f);
            }
            else {
                float s = (t - 0.5f) * 2.0f;
                r = 255;
                g = (unsigned char)((1.0f - s) * 255.0f);
                b = 0;
            }
        }
        else {
            float vort_range = max_vort - min_vort;
            if (vort_range > 1e-6f) {
                float t = (val - min_vort) / vort_range;
                if (t < 0.5f) {
                    float s = t * 2.0f;
                    r = (unsigned char)(s * 255.0f);
                    g = (unsigned char)(s * 255.0f);
                    b = (unsigned char)((1.0f - s) * 255.0f);
                }
                else {
                    float s = (t - 0.5f) * 2.0f;
                    r = 255;
                    g = (unsigned char)((1.0f - s) * 255.0f);
                    b = 0;
                }
            }
        }
        pbo_ptr[idx_out + 0] = r;
        pbo_ptr[idx_out + 1] = g;
        pbo_ptr[idx_out + 2] = b;
        pbo_ptr[idx_out + 3] = 255;
    }
}

// **Simulation Functions**

void initialize_simulation() {
    cudaMalloc(&d_u, velSize);
    cudaMalloc(&d_v, velSize);
    cudaMalloc(&d_p, velSize);
    cudaMalloc(&d_u_star, velSize);
    cudaMalloc(&d_v_star, velSize);
    cudaMalloc(&d_div, velSize);
    cudaMalloc(&d_p_new, velSize);
    cudaMalloc(&d_vorticity, velSize);

    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    initialize_velocity << <gridDim, blockDim >> > (d_u, d_v, U, nx, ny);
    cudaMemset(d_p, 0, velSize);
}

void simulation_step() {
    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
    dim3 blockDim1(256);
    dim3 gridDim1((nx + blockDim1.x - 1) / blockDim1.x);

    // Set boundary conditions
    set_boundary_conditions << <gridDim, blockDim >> > (d_u_star, d_v_star, U, nx, ny);
    //cudaDeviceSynchronize();

    // Compute intermediate velocity
    compute_intermediate_velocity << <gridDim, blockDim >> > (d_u, d_v, d_u_star, d_v_star, dt, nu, h, nx, ny);
    //cudaDeviceSynchronize();

    // Compute divergence
    compute_divergence << <gridDim, blockDim >> > (d_u_star, d_v_star, d_div, h, nx, ny);
    //cudaDeviceSynchronize();

    // Jacobi iterations for pressure
    for (int iter = 0; iter < num_jacobi_iters; iter++) {
        jacobi_iteration << <gridDim, blockDim >> > (d_p, d_p_new, d_div, dt, rho, h, nx, ny);
        //cudaDeviceSynchronize();
        set_pressure_bc << <gridDim, blockDim >> > (d_p_new, nx, ny);
        //cudaDeviceSynchronize();
        DTYPE* temp = d_p;
        d_p = d_p_new;
        d_p_new = temp;
    }

    // Correct velocity
    correct_velocity << <gridDim, blockDim >> > (d_u, d_v, d_u_star, d_v_star, d_p, dt, rho, h, nx, ny);
    //cudaDeviceSynchronize();

}

// **OpenGL Functions**

static GLuint pbo = 0;
static struct cudaGraphicsResource* cuda_pbo = nullptr;
static DTYPE* d_velocity = nullptr;
static const int WIN_WIDTH = nx;
static const int WIN_HEIGHT = ny;
static int stepsPerFrame = 100;
static bool displayVorticity = false;

void create_pbo() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIN_WIDTH * WIN_HEIGHT * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    cudaMalloc((void**)&d_velocity, nx * ny * sizeof(DTYPE));
}

void cleanup() {
    if (cuda_pbo) {
        cudaGraphicsUnregisterResource(cuda_pbo);
        cuda_pbo = nullptr;
    }
    if (pbo) {
        glDeleteBuffers(1, &pbo);
        pbo = 0;
    }
    if (d_velocity) cudaFree(d_velocity);
    if (d_u) cudaFree(d_u);
    if (d_v) cudaFree(d_v);
    if (d_p) cudaFree(d_p);
    if (d_u_star) cudaFree(d_u_star);
    if (d_v_star) cudaFree(d_v_star);
    if (d_div) cudaFree(d_div);
    if (d_p_new) cudaFree(d_p_new);
    if (d_vorticity) cudaFree(d_vorticity);
}

void display() {
    if (currentStep == 0) startTime = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < stepsPerFrame; s++) {
        simulation_step();
        currentStep++;
    }
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    compute_velocity_field_kernel << <grid, block >> > (d_u, d_v, d_velocity, nx, ny);
    cudaDeviceSynchronize();

    compute_vorticity_kernel << <grid, block >> > (d_u, d_v, d_vorticity, nx, ny);
    cudaDeviceSynchronize();

    cudaGraphicsMapResources(1, &cuda_pbo, 0);
    unsigned char* d_pbo_ptr = nullptr;
    size_t num_bytes = 0;
    cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_ptr, &num_bytes, cuda_pbo);

    if (displayVorticity) {
        thrust::device_ptr<float> dev_vort_ptr(d_vorticity);
        auto min_max = thrust::minmax_element(dev_vort_ptr, dev_vort_ptr + nx * ny);
        float min_vort = *min_max.first;
        float max_vort = *min_max.second;
        fill_pbo_kernel << <grid, block >> > (d_pbo_ptr, d_vorticity, nx, ny, U, 1, min_vort, max_vort);
    }
    else {
        fill_pbo_kernel << <grid, block >> > (d_pbo_ptr, d_velocity, nx, ny, U, 0, 0.0, 0.0);
    }
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &cuda_pbo, 0);

    glClear(GL_COLOR_BUFFER_BIT);
    glRasterPos2f(-1, -1);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(WIN_WIDTH, WIN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    auto currentTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> elapsedTime = currentTime - startTime;
    updatesPerSecond = float(currentStep) / elapsedTime.count();
    framesPerSecond = updatesPerSecond / float(stepsPerFrame);
    mlups = (float(currentStep) * float(nx) * float(ny)) / (elapsedTime.count() * float(1e6));

    std::ostringstream oss;
    oss << "Time Step: " << currentStep
        << "  UPS: " << std::fixed << std::setprecision(1) << updatesPerSecond
        << "  MLUPS: " << std::fixed << std::setprecision(2) << mlups
        << "  FPS: " << std::fixed << std::setprecision(1) << framesPerSecond
        << "  Vis: " << (displayVorticity ? "Vorticity" : "Velocity");
    std::string info = oss.str();

    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos2f(-0.95f, 0.95f);
    for (char c : info) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);

    glutSwapBuffers();
}

void keyboard(unsigned char key, int x, int y) {
    if (key == 'v') displayVorticity = !displayVorticity;
}

void idle() {
    glutPostRedisplay();
}

void initGL(int* argc, char** argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);
    glutCreateWindow("FDM + OpenGL Visualization");

    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("OpenGL vendor: %s\n", glGetString(GL_VENDOR));
    printf("OpenGL renderer: %s\n", glGetString(GL_RENDERER));

    GLenum err = glewInit();
    if (GLEW_OK != err) {
        fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(err));
        exit(1);
    }

    create_pbo();
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);

    glDisable(GL_DEPTH_TEST);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

int main(int argc, char** argv) {
    nu = U / Re;
    printf("U = %f, Re = %f, nu = %f, h = %f, dt = %f\n", U, Re, nu, h, dt);

    initialize_simulation();
    initGL(&argc, argv);
    glutMainLoop();

    cleanup();
    return 0;
}