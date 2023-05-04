#include <math.h>
#include <complex>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <chrono>
#include <limits>
#include <stack>
#include "MellowSim.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)


using namespace std;
using namespace cv;

// TODO-List: https://trello.com/b/37JofojU/mellowsim

//map<int, int> resolutions = { {1280,720}, {1920,1080}, {2048,1080}, {3840,2160}, {4096,2160} };

float zoom_factor = 0.2;
float zoom_change = 0.2;
float min_zoom = 0.05;
float max_zoom = 0.95;
unsigned long long magnification = 1;
float intensity = 2.;

const int hor_resolution = 1024;
const int ver_resolution = hor_resolution / aspect_ratio;

int prev_x = -1;
int prev_y = -1;
int prev_z = 0;

// Complex number: z = a + b*i

//typedef unsigned short T_IMG;
typedef unsigned char T_IMG;
stack<MandelArea<T_IMG>> st;


inline std::tm localtime_xp(std::time_t timer)
{
    std::tm bt{};
#if defined(__unix__)
    localtime_r(&timer, &bt);
#elif defined(_MSC_VER)
    localtime_s(&bt, &timer);
#else
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    bt = *std::localtime(&timer);
#endif
    return bt;
}

// default = "YYYY-MM-DD HH:MM:SS"
inline std::string time_stamp(const std::string& fmt = "%Y_%m_%d_%H_%M_%S") // "%F %T"
{
    auto bt = localtime_xp(std::time(0));
    char buf[64];
    return { buf, std::strftime(buf, sizeof(buf), fmt.c_str(), &bt) };
}

void show_progress_bar(float progress) {
    int barWidth = 70;
    if (progress <= 1.0) {
        std::cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << "X";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }
    if (progress == 1.0) {
        std::cout << std::endl;
    }
}


Mat showing;
bool showing_zoombox = true;
void onClick(int event, int x, int y, int z, void*) {
    MandelArea<T_IMG> area = st.top();

    int zoom_width = w_width * zoom_factor;
    int zoom_height = w_height * zoom_factor;
    int corrected_x = x - (zoom_width / 2);
    if (corrected_x < 0) corrected_x = 0;
    if (corrected_x + zoom_width + 1 > w_width) corrected_x = w_width - zoom_width;
    int corrected_y = y - (zoom_height / 2);
    if (corrected_y < 0) corrected_y = 0;
    if (corrected_y + zoom_height + 1 > w_height) corrected_y = w_height - zoom_height;

    if (event == EVENT_MBUTTONDOWN) {
        showing_zoombox = !showing_zoombox;
        if (!showing_zoombox) {
            imshow(w_name, area.img);
        }
    }

    if (event == EVENT_MOUSEWHEEL) {
        float new_zoom_factor = zoom_factor;
        if (z > 0) {
            new_zoom_factor = zoom_factor * (1 - zoom_change);
        }
        if (z < 0) {
            new_zoom_factor = zoom_factor * (1 + zoom_change);
        }
        zoom_factor = new_zoom_factor;
        if (new_zoom_factor < min_zoom) zoom_factor = min_zoom;
        if (new_zoom_factor > max_zoom) zoom_factor = max_zoom;
    }

    long double start_x, start_y;
    if (event == EVENT_LBUTTONDOWN) {
        MandelArea<T_IMG> area2 = area; // TODO: Remove - Only for debugging
        magnification /= zoom_factor;
        start_x = area.x_start + corrected_x * area.x_dist / w_width;
        start_y = area.y_start - corrected_y * area.y_dist / w_height;
        long double end_x = start_x + zoom_width * area.x_dist / w_width;
        long double end_y = start_y + zoom_height * area.y_dist / w_height;
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        st.push(MandelArea<T_IMG>(start_x, end_x, start_y, end_y, aspect_ratio, hor_resolution, intensity, magnification));
        chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        cout << "Time elapsed = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
        MandelArea<T_IMG> area = st.top();
        //blur(area.img, area.img, Size(3, 3), Point(-1,-1), 4);
        //GaussianBlur(area.img, area.img, Size(3, 3), 0.);
        //medianBlur(area.img, area.img, 3);
        cout << "Magnification = " << magnification << endl;
    }

    if (event == EVENT_RBUTTONDOWN && st.size() > 1) {
        st.pop();
        MandelArea<T_IMG> area = st.top();
        magnification = area.magnification;
        cout << "Magnification = " << magnification << endl;
    }

    if (showing_zoombox && event == EVENT_MOUSEMOVE) {
        Rect rect(corrected_x, corrected_y, zoom_width, zoom_height);
        area.img.copyTo(showing);

        rectangle(showing, rect, cv::Scalar(0, area.color_depth, 0));

        imshow(w_name, showing);
    }
    prev_x = x;
    prev_y = y;
    prev_z = z;
}

int startKernel() {
    printf("Started running\n");

    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int* A = (int*)malloc(sizeof(int) * LIST_SIZE);
    int* B = (int*)malloc(sizeof(int) * LIST_SIZE);
    for (i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    // Load the kernel source code into the array source_str
    FILE* fp;
    char* source_str;
    size_t source_size;

    fp = fopen("kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    printf("Kernel loading done\n\n");
    // Get platform and device information
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;


    cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    cl_platform_id* platforms = NULL;
    platforms = (cl_platform_id*)malloc(ret_num_platforms * sizeof(cl_platform_id));

    ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);
    //printf("ret at %d is %d\n", __LINE__, ret);

    ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
    //printf("ret at %d is %d\n", __LINE__, ret);

    char* device_name = new char[255];
    ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(string), device_name, NULL);
    cout << "Device name: " << device_name << endl;

    cl_bool available;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, NULL);
    string available_str = available ? "Yes" : "No";
    cout << "Device available: " << available_str << endl;

    // MaxComputeUnits * MaxClockFrequency
    cl_uint comp_units;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
    cout << "MaxComputeUnits: " << comp_units << endl;

    cl_uint clock_freq;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &clock_freq, NULL);
    cout << "MaxClockFrequency: " << clock_freq << endl;

    cout << "MaxComputeUnits * MaxClockFrequency: " << comp_units * clock_freq << endl;

    cl_ulong mem_size;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &mem_size, NULL);
    //printf("ret at %d is %d\n", __LINE__, ret);
    cout << "Mem size: " << mem_size << endl;

    size_t work_group_size;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &work_group_size, NULL);
    cout << "Work group size: " << work_group_size << endl;

    cout << "Compute units * Work group size (Threads): " << comp_units * work_group_size << endl;

    cl_uint work_item_dims;
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &work_item_dims, NULL);
    cout << "Work item dimensions: " << work_item_dims << endl;

    size_t* work_item_size = new size_t[work_item_dims];
    ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(work_item_size), &work_item_size, NULL);
    for (int i = 0; i < work_item_dims; i++) {
        printf("Work item size[%d]: %llu\n", i, work_item_size[i]);
    }

    // TODO: How many threads? https://stackoverflow.com/questions/5679726/how-many-threads-or-work-item-can-run-at-the-same-time
    

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    //printf("ret at %d is %d\n", __LINE__, ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    //printf("ret at %d is %d\n", __LINE__, ret);

    // Create memory buffers on the device for each vector 
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        LIST_SIZE * sizeof(int), NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
    //printf("ret at %d is %d\n", __LINE__, ret);

    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
        LIST_SIZE * sizeof(int), B, 0, NULL, NULL);
    //printf("ret at %d is %d\n", __LINE__, ret);

    //printf("before building\n");
    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&source_str, (const size_t*)&source_size, &ret);
    //printf("ret at %d is %d\n", __LINE__, ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    //printf("ret at %d is %d\n", __LINE__, ret);

    //printf("after building\n");
    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);
    //printf("ret at %d is %d\n", __LINE__, ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_mem_obj);
    //printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_mem_obj);
    //printf("ret at %d is %d\n", __LINE__, ret);

    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_mem_obj);
    //printf("ret at %d is %d\n", __LINE__, ret);

    //added this to fix garbage output problem
    //ret = clSetKernelArg(kernel, 3, sizeof(int), &LIST_SIZE);

    //printf("before execution\n");
    //// Execute the OpenCL kernel on the list
    //size_t global_item_size = LIST_SIZE; // Process the entire lists
    //size_t local_item_size = 64; // Divide work items into groups of 64
    //ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
    //    &global_item_size, &local_item_size, 0, NULL, NULL);
    //printf("after execution\n");
    //// Read the memory buffer C on the device to the local variable C
    //int* C = (int*)malloc(sizeof(int) * LIST_SIZE);
    //ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
    //    LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
    //printf("after copying\n");
    //// Display the result to the screen
    //for (i = 0; i < LIST_SIZE; i++)
    //    printf("%d + %d = %d\n", A[i], B[i], C[i]);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    //free(C);
    free(platforms);
    free(source_str);
    delete [] device_name;
    return 0;
}

int main() {
    startKernel();
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);
    cout << endl;

    //st.push(MandelArea<T_IMG>(first_start_x, first_end_x, first_start_y, first_end_y, aspect_ratio, hor_resolution, intensity, magnification));

    //namedWindow(w_name);

    //setMouseCallback(w_name, onClick, 0);

    //// Common resoltions: 1024, 2048, 4K: 4096, 8K: 7680, 16K: 15360

    //cout << endl;
    //while ((char)27 != (char)waitKey(0)) {
    //    if ((char)115 == (char)waitKey(0)) {
    //        MandelArea<T_IMG> area = st.top();
    //        cout << "Saving picture to " << area.filename << endl;
    //        imwrite(area.filename, area.img);
    //    }
    //}

    return 0;
}
