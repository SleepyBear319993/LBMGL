//#include <chrono>
//#include <iomanip>
//#include <string>
//#include <sstream>
//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <thrust/device_vector.h>
//#include <thrust/extrema.h>
//
//#include <GL/glew.h>
//#include <GL/freeglut.h>
//#include <cuda_gl_interop.h>
//
//int currentStep = 0;
//std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
//float updatesPerSecond = 0.0f;
//float mlups = 0.0f;
//float framesPerSecond = 0.0f;
//
//#ifdef _WIN32
//extern "C" {
//    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
//}
//#endif
//
//const int nx = 1024;
//const int ny = 1024;
//const int numDirs = 9;
//typedef float DTYPE;
//
//__constant__ int cx_const[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
//__constant__ int cy_const[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
//__constant__ DTYPE w_const[9] = {
//    4.0 / 9.0,
//    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
//    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
//};
//
//const int cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
//const int cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
//const DTYPE w[9] = {
//    4.0 / 9.0,
//    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
//    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
//};
//
//DTYPE U = 0.3;
//DTYPE Re = 35000.0;
//DTYPE nu, tao, omega;
//
//DTYPE* d_f = nullptr, * d_f_new = nullptr;
//char* d_mask = nullptr;
//DTYPE* d_ux = nullptr, * d_uy = nullptr, * d_vorticity = nullptr;
//
//size_t simSize = nx * ny * numDirs * sizeof(DTYPE);
//size_t maskSize = nx * ny * sizeof(char);
//size_t velSize = nx * ny * sizeof(DTYPE);
//
//__device__ inline int idx(int i, int j, int k, int nx, int ny) {
//    return i + j * nx + k * nx * ny;
//}
//inline int idx_h(int i, int j, int k, int nx, int ny) {
//    return i + j * nx + k * nx * ny;
//}
//
//__global__ void collision_kernel(DTYPE* f, DTYPE omega, int nx, int ny) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//    if (i < nx && j < ny) {
//        DTYPE rho = 0.0;
//        DTYPE u_x = 0.0;
//        DTYPE u_y = 0.0;
//        for (int k = 0; k < numDirs; k++) {
//            DTYPE val = f[idx(i, j, k, nx, ny)];
//            rho += val;
//            u_x += val * cx_const[k];
//            u_y += val * cy_const[k];
//        }
//        if (rho > DTYPE(0.0)) {
//            u_x /= rho;
//            u_y /= rho;
//        }
//        DTYPE usqr = u_x * u_x + u_y * u_y;
//        for (int k = 0; k < numDirs; k++) {
//            DTYPE cu = DTYPE(3.0) * (cx_const[k] * u_x + cy_const[k] * u_y);
//            DTYPE feq = w_const[k] * rho * (DTYPE(1.0) + cu + DTYPE(0.5) * cu * cu - DTYPE(1.5) * usqr);
//            f[idx(i, j, k, nx, ny)] = (DTYPE(1.0) - omega) * f[idx(i, j, k, nx, ny)] + omega * feq;
//        }
//    }
//}
//
//__global__ void streaming_kernel(DTYPE* f_in, DTYPE* f_out, int nx, int ny) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//    if (i < nx && j < ny) {
//        if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
//            for (int k = 0; k < numDirs; k++) {
//                int ip = i - cx_const[k];
//                int jp = j - cy_const[k];
//                f_out[idx(i, j, k, nx, ny)] = f_in[idx(ip, jp, k, nx, ny)];
//            }
//        }
//        else {
//            for (int k = 0; k < numDirs; k++) {
//                f_out[idx(i, j, k, nx, ny)] = f_in[idx(i, j, k, nx, ny)];
//            }
//        }
//    }
//}
//
//__global__ void bounce_back_kernel(DTYPE* f, char* mask, int nx, int ny) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//    if (i < nx && j < ny) {
//        if (mask[i + j * nx] == 1) {
//            int idx1 = idx(i, j, 1, nx, ny);
//            int idx3 = idx(i, j, 3, nx, ny);
//            DTYPE tmp = f[idx1];
//            f[idx1] = f[idx3];
//            f[idx3] = tmp;
//            int idx2 = idx(i, j, 2, nx, ny);
//            int idx4 = idx(i, j, 4, nx, ny);
//            tmp = f[idx2];
//            f[idx2] = f[idx4];
//            f[idx4] = tmp;
//            int idx5 = idx(i, j, 5, nx, ny);
//            int idx7 = idx(i, j, 7, nx, ny);
//            tmp = f[idx5];
//            f[idx5] = f[idx7];
//            f[idx7] = tmp;
//            int idx6 = idx(i, j, 6, nx, ny);
//            int idx8 = idx(i, j, 8, nx, ny);
//            tmp = f[idx6];
//            f[idx6] = f[idx8];
//            f[idx8] = tmp;
//        }
//    }
//}
//
//__global__ void moving_lid_kernel(DTYPE* f, int nx, int ny, DTYPE U) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i < nx) {
//        int j = ny - 2;
//        f[idx(i, j, 4, nx, ny)] = f[idx(i, j, 2, nx, ny)];
//        f[idx(i, j, 7, nx, ny)] = f[idx(i, j, 5, nx, ny)] - DTYPE(1.0) / DTYPE(6.0) * U;
//        f[idx(i, j, 8, nx, ny)] = f[idx(i, j, 6, nx, ny)] + DTYPE(1.0) / DTYPE(6.0) * U;
//    }
//}
//
//template<typename T>
//__device__ __forceinline__ T device_sqrt(T x) {
//    if constexpr (std::is_same<T, float>::value) {
//        return sqrtf(x);
//    }
//    else {
//        return sqrt(x);
//    }
//}
//
//__global__ void compute_velocity_field_kernel(const DTYPE* f, DTYPE* ux, DTYPE* uy, DTYPE* velocity_mag, int nx, int ny) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//    if (i < nx && j < ny) {
//        DTYPE rho = 0.0;
//        DTYPE u_x = 0.0;
//        DTYPE u_y = 0.0;
//        for (int k = 0; k < numDirs; k++) {
//            DTYPE val = f[idx(i, j, k, nx, ny)];
//            rho += val;
//            u_x += val * cx_const[k];
//            u_y += val * cy_const[k];
//        }
//        if (rho > DTYPE(1e-12)) {
//            u_x /= rho;
//            u_y /= rho;
//        }
//        else {
//            u_x = 0.0;
//            u_y = 0.0;
//        }
//        ux[i + j * nx] = u_x;
//        uy[i + j * nx] = u_y;
//        DTYPE vel = device_sqrt(u_x * u_x + u_y * u_y);
//        velocity_mag[i + j * nx] = vel;
//    }
//}
//
//__global__ void compute_vorticity_kernel(const DTYPE* ux, const DTYPE* uy, DTYPE* vorticity, int nx, int ny) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1) {
//        DTYPE d_uy_dx = (uy[(i + 1) + j * nx] - uy[(i - 1) + j * nx]) / DTYPE(2.0);
//        DTYPE d_ux_dy = (ux[i + (j + 1) * nx] - ux[i + (j - 1) * nx]) / DTYPE(2.0);
//        vorticity[i + j * nx] = d_uy_dx - d_ux_dy;
//    }
//    else {
//        vorticity[i + j * nx] = DTYPE(0.0);
//    }
//}
//
//__global__ void fill_pbo_kernel(unsigned char* pbo_ptr, const DTYPE* data, int nx, int ny, float U, int mode, float min_vort, float max_vort) {
//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    int j = blockIdx.y * blockDim.y + threadIdx.y;
//    if (i < nx && j < ny) {
//        int idx_out = 4 * (i + j * nx);
//        float val = data[i + j * nx];
//        unsigned char r, g, b;
//
//        if (mode == 0) {
//            float t = fminf(val / U, 1.0f);
//            if (t < 0.5f) {
//                float s = t * 2.0f;
//                r = (unsigned char)(s * 255.0f);
//                g = (unsigned char)(s * 255.0f);
//                b = (unsigned char)((1.0f - s) * 255.0f);
//            }
//            else {
//                float s = (t - 0.5f) * 2.0f;
//                r = 255;
//                g = (unsigned char)((1.0f - s) * 255.0f);
//                b = 0;
//            }
//        }
//        else {
//
//			float vort_range = max_vort - min_vort;
//            if (vort_range > 1e-6f)
//            {
//				float t = (val - min_vort) / vort_range;
//                if (t < 0.5f) {
//                    float s = t * 2.0f;
//                    r = (unsigned char)(s * 255.0f);
//                    g = (unsigned char)(s * 255.0f);
//                    b = (unsigned char)((1.0f - s) * 255.0f);
//                }
//                else {
//                    float s = (t - 0.5f) * 2.0f;
//                    r = 255;
//                    g = (unsigned char)((1.0f - s) * 255.0f);
//                    b = 0;
//                }
//            }
//            //float vort_range = max_vort - min_vort;
//            //float normalized_vort;
//            //if (vort_range > 1e-6f) { // Avoid division by zero
//            //    normalized_vort = 2.0f * (val - min_vort) / vort_range - 1.0f;
//            //}
//            //else {
//            //    normalized_vort = 0.0f; // If range is too small, set to neutral
//            //}
//            //// Clamp to [-1, 1]
//            //normalized_vort = fminf(fmaxf(normalized_vort, -1.0f), 1.0f);
//
//            //// Diverging color map: blue (-1) -> white (0) -> red (1)
//            //if (normalized_vort < 0) {
//            //    float s = -normalized_vort; // Magnitude of negative vorticity
//            //    r = (unsigned char)(255.0f * s);
//            //    g = (unsigned char)(255.0f * s);
//            //    b = 255;
//            //}
//            //else {
//            //    float s = normalized_vort; // Magnitude of positive vorticity
//            //    r = 255;
//            //    g = (unsigned char)(255.0f * (1.0f - s));
//            //    b = (unsigned char)(255.0f * (1.0f - s));
//            //}
//        }
//        pbo_ptr[idx_out + 0] = r;
//        pbo_ptr[idx_out + 1] = g;
//        pbo_ptr[idx_out + 2] = b;
//        pbo_ptr[idx_out + 3] = 255;
//    }
//}
//
//void initialize_simulation(DTYPE rho0, DTYPE ux0, DTYPE uy0) {
//    cudaMalloc(&d_f, simSize);
//    cudaMalloc(&d_f_new, simSize);
//    cudaMalloc(&d_mask, maskSize);
//    cudaMalloc(&d_ux, velSize);
//    cudaMalloc(&d_uy, velSize);
//    cudaMalloc(&d_vorticity, velSize);
//
//    DTYPE* h_f = new DTYPE[nx * ny * numDirs];
//    for (int j = 0; j < ny; j++) {
//        for (int i = 0; i < nx; i++) {
//            DTYPE usq = ux0 * ux0 + uy0 * uy0;
//            for (int k = 0; k < numDirs; k++) {
//                DTYPE cu = DTYPE(3.0) * (cx[k] * ux0 + cy[k] * uy0);
//                h_f[idx_h(i, j, k, nx, ny)] = w[k] * rho0 * (DTYPE(1.0) + cu + DTYPE(0.5) * cu * cu - DTYPE(1.5) * usq);
//            }
//        }
//    }
//    cudaMemcpy(d_f, h_f, simSize, cudaMemcpyHostToDevice);
//    delete[] h_f;
//
//    char* h_mask = new char[nx * ny];
//    for (int j = 0; j < ny; j++) {
//        for (int i = 0; i < nx; i++) {
//            if (i == 1 || i == nx - 2 || j == 1)
//                h_mask[i + j * nx] = 1;
//            else
//                h_mask[i + j * nx] = 0;
//        }
//    }
//    cudaMemcpy(d_mask, h_mask, maskSize, cudaMemcpyHostToDevice);
//    delete[] h_mask;
//}
//
//void simulation_step() {
//    dim3 blockDim(16, 16);
//    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);
//
//    collision_kernel << <gridDim, blockDim >> > (d_f, omega, nx, ny);
//    streaming_kernel << <gridDim, blockDim >> > (d_f, d_f_new, nx, ny);
//    bounce_back_kernel << <gridDim, blockDim >> > (d_f_new, d_mask, nx, ny);
//
//    dim3 blockDim1(256);
//    dim3 gridDim1((nx + blockDim1.x - 1) / blockDim1.x);
//    moving_lid_kernel << <gridDim1, blockDim1 >> > (d_f_new, nx, ny, U);
//
//    DTYPE* temp = d_f;
//    d_f = d_f_new;
//    d_f_new = temp;
//}
//
//static GLuint pbo = 0;
//static struct cudaGraphicsResource* cuda_pbo = nullptr;
//static DTYPE* d_velocity = nullptr;
//static const int WIN_WIDTH = nx;
//static const int WIN_HEIGHT = ny;
//static int stepsPerFrame = 100;
//static bool displayVorticity = false;
//
//void create_pbo() {
//    glGenBuffers(1, &pbo);
//    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
//    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIN_WIDTH * WIN_HEIGHT * 4, NULL, GL_DYNAMIC_DRAW);
//    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//
//    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
//    cudaMalloc((void**)&d_velocity, nx * ny * sizeof(DTYPE));
//}
//
//void cleanup() {
//    if (cuda_pbo) {
//        cudaGraphicsUnregisterResource(cuda_pbo);
//        cuda_pbo = nullptr;
//    }
//    if (pbo) {
//        glDeleteBuffers(1, &pbo);
//        pbo = 0;
//    }
//    if (d_velocity) {
//        cudaFree(d_velocity);
//        d_velocity = nullptr;
//    }
//    if (d_f) cudaFree(d_f);
//    if (d_f_new) cudaFree(d_f_new);
//    if (d_mask) cudaFree(d_mask);
//    if (d_ux) cudaFree(d_ux);
//    if (d_uy) cudaFree(d_uy);
//    if (d_vorticity) cudaFree(d_vorticity);
//}
//
//void display() {
//    if (currentStep == 0) {
//        startTime = std::chrono::high_resolution_clock::now();
//    }
//    for (int s = 0; s < stepsPerFrame; s++) {
//        simulation_step();
//        currentStep++;
//    }
//    dim3 block(16, 16);
//    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
//
//    compute_velocity_field_kernel << <grid, block >> > (d_f, d_ux, d_uy, d_velocity, nx, ny);
//    cudaDeviceSynchronize();
//
//    compute_vorticity_kernel << <grid, block >> > (d_ux, d_uy, d_vorticity, nx, ny);
//    cudaDeviceSynchronize();
//
//    cudaGraphicsMapResources(1, &cuda_pbo, 0);
//    unsigned char* d_pbo_ptr = nullptr;
//    size_t num_bytes = 0;
//    cudaGraphicsResourceGetMappedPointer((void**)&d_pbo_ptr, &num_bytes, cuda_pbo);
//
//    if (displayVorticity) {
//        thrust::device_ptr<float> dev_vort_ptr(d_vorticity);
//		auto min_max = thrust::minmax_element(dev_vort_ptr, dev_vort_ptr + nx * ny);
//        float min_vort = *min_max.first;
//		float max_vort = *min_max.second;
//        fill_pbo_kernel << <grid, block >> > (d_pbo_ptr, d_vorticity, nx, ny, U, 1, min_vort, max_vort);
//    }
//    else {
//        fill_pbo_kernel << <grid, block >> > (d_pbo_ptr, d_velocity, nx, ny, U, 0, 0.0, 0.0);
//    }
//    cudaDeviceSynchronize();
//
//    cudaGraphicsUnmapResources(1, &cuda_pbo, 0);
//
//    glClear(GL_COLOR_BUFFER_BIT);
//    glRasterPos2f(-1, -1);
//    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
//    glDrawPixels(WIN_WIDTH, WIN_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
//    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//
//    auto currentTime = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<float> elapsedTime = currentTime - startTime;
//    updatesPerSecond = float(currentStep) / elapsedTime.count();
//    framesPerSecond = updatesPerSecond / float(stepsPerFrame);
//    mlups = (float(currentStep) * float(nx) * float(ny)) / (elapsedTime.count() * float(1e6));
//
//    std::ostringstream oss;
//    oss << "Time Step: " << currentStep
//        << "  UPS: " << std::fixed << std::setprecision(1) << updatesPerSecond
//        << "  MLUPS: " << std::fixed << std::setprecision(2) << mlups
//        << "  FPS: " << std::fixed << std::setprecision(1) << framesPerSecond
//        << "  Vis: " << (displayVorticity ? "Vorticity" : "Velocity");
//    std::string info = oss.str();
//
//    glColor3f(1.0f, 1.0f, 1.0f);
//    glRasterPos2f(-0.95f, 0.95f);
//    for (char c : info) {
//        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
//    }
//
//    glutSwapBuffers();
//}
//
//void keyboard(unsigned char key, int x, int y) {
//    if (key == 'v') {
//        displayVorticity = !displayVorticity;
//    }
//}
//
//void idle() {
//    glutPostRedisplay();
//}
//
//void initGL(int* argc, char** argv) {
//    glutInit(argc, argv);
//    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
//    glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);
//    glutCreateWindow("LBM + OpenGL Visualization");
//
//    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
//    printf("OpenGL vendor: %s\n", glGetString(GL_VENDOR));
//    printf("OpenGL renderer: %s\n", glGetString(GL_RENDERER));
//
//    GLenum err = glewInit();
//    if (GLEW_OK != err) {
//        fprintf(stderr, "Error initializing GLEW: %s\n", glewGetErrorString(err));
//        exit(1);
//    }
//
//    create_pbo();
//
//    glutDisplayFunc(display);
//    glutIdleFunc(idle);
//    glutKeyboardFunc(keyboard);
//
//    glDisable(GL_DEPTH_TEST);
//    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
//    glEnable(GL_BLEND);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//}
//
//int main(int argc, char** argv) {
//    nu = U * DTYPE(nx) / Re;
//    tao = DTYPE(3.0) * nu + DTYPE(0.5);
//    omega = DTYPE(1.0) / tao;
//    DTYPE rho0 = 1.0;
//    DTYPE ux0 = 0.0;
//    DTYPE uy0 = 0.0;
//    printf("Viscosity = %f, Relaxation time = %f, Omega = %f\n", nu, tao, omega);
//    printf("U = %f, Re = %f\n", U, Re);
//
//    initialize_simulation(rho0, ux0, uy0);
//    initGL(&argc, argv);
//    glutMainLoop();
//
//    cleanup();
//    return 0;
//}