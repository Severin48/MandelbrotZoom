#pragma once
#include <iostream>
#include <Windows.h>
#include <limits.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <CL/opencl.hpp>

using namespace std;
using namespace cv;

const string w_name = "MellowSim";

const float dist_limit = 2.; //Arbitrary but has to be at least 2

const unsigned short block_size = 16192;

const unsigned short n_channels = 3;

unsigned int max_iter = 1000;

int sizes[] = { 255, 255, 255 };
typedef Point3_<uint8_t> Pixel;

const unsigned long magnification_cycle_value = 100000;

const float aspect_ratio = 16. / 9.;
const int w_width = 1024;
const int w_height = w_width / aspect_ratio;
const float first_start_x = -2.7;
const float first_end_x = 1.2;
const float first_start_y = 1.2;
const float first_end_y = -1.2;

template <typename T>
class MandelArea {
public:
    long double x_start;
    long double x_end;
    long double y_start;
    long double y_end;
    long double x_dist;
    long double y_dist;
    int px_count;
    int width;
    float ratio;
    int height;
    long double x_per_px;
    long double y_per_px;
    bool partial_write;
    string filename;
    unsigned int n_blocks;
    unsigned int left_over_pixels;
    float intensity;
    Mat img;
    const T color_depth = (T)-1;
    unsigned long long magnification;
    unsigned long long color_magnification;
    //cl::Context context;
    cl::Device device;

    MandelArea(long double x_start, long double x_end, long double y_start, long double y_end, float ratio, int width, float intensity, unsigned long long magnification) {
        //bool is_signed = false;
        //if (color_depth < 0) {
        //    is_signed = true;
        //}
        this->x_start = x_start;
        this->x_end = x_end;
        this->y_start = y_start;
        this->y_end = y_end;
        this->x_dist = x_start > x_end ? x_start - x_end : x_end - x_start;
        this->y_dist = y_start > y_end ? y_start - y_end : y_end - y_start;
        this->ratio = ratio;
        this->width = width;
        this->height = width / ratio;
        this->x_per_px = x_dist / width;
        this->y_per_px = y_dist / height;
        this->px_count = width * height;
        this->intensity = intensity;
        this->magnification = magnification;
        this->color_magnification = magnification % magnification_cycle_value;
        this->filename = get_filename();
        if (px_count > block_size) {
            partial_write = true;
        }
        else {
            partial_write = false;
        }
        this->n_blocks = px_count / block_size;
        this->left_over_pixels = px_count % block_size;
        getDevice(device);
        size_t mat_type = get_mat_type();
        if (mat_type == 0) return;
        this->img = Mat(height, width, mat_type);
        this->write_img(intensity, false);
        resize(img, img, Size(w_width, w_width / ratio), INTER_LINEAR_EXACT);
        imshow(w_name, img);
    }

    size_t get_mat_type() {
        const type_info& id = typeid(T);
        if (id == typeid(char)) return CV_8SC3;
        if (id == typeid(short)) return CV_16SC3;
        //if (id == typeid(int)) return CV_32SC3;
        if (id == typeid(float)) return CV_32FC3;
        if (id == typeid(double)) return CV_64FC3;
        if (id == typeid(unsigned char)) return CV_8UC3;
        if (id == typeid(unsigned short)) return CV_16UC3;
        return 0;
    }

    string get_filename() {
        string output_dir = "output/";
        string mkdir_str = "if not exist " + output_dir + " mkdir " + output_dir;
        CreateDirectory(output_dir.c_str(), NULL);
        string file_ending = ".png";
        string filename = output_dir + time_stamp() + file_ending;
        return filename;
    }

    void write_img(float intensity, bool save_img) {
        std::vector<double> real_vals(width);
        std::vector<double> imag_vals(height);

        for (unsigned int x = 0; x < width; x++) {
            real_vals[x] = x_start + x * x_per_px;
        }
        for (unsigned int y = 0; y < height; y++) {
            imag_vals[y] = y_start - y * y_per_px;
        }

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        startIterKernel(real_vals, imag_vals);
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Kernel (GPU) time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

        cout << endl << setprecision(numeric_limits<long double>::max_digits10) << "start_x=" << x_start << " start_y=" << y_start << endl;
        cvtColor(img, img, CV_HSV2BGR);
        if (save_img) imwrite(filename, img);
    }

    void getDevice(cl::Device& device) {
        cl::Context context(CL_DEVICE_TYPE_GPU);
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        cl_device_id device_id = NULL;
        cl_uint ret_num_devices;
        cl_uint ret_num_platforms;

        // cout << "Nr. devices: " << devices.size() << endl;

        device = devices[0];

        cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
        cl_platform_id* platforms = NULL;
        platforms = (cl_platform_id*)malloc(ret_num_platforms * sizeof(cl_platform_id));

        ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);

        if (platforms == NULL) {
            std::cerr << "No platforms detected." << std::endl;
            exit(1);
        }

        ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);

        int str_size = 255;
        char* device_name = new char[str_size];
        ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME, str_size, device_name, NULL);
        cout << "Device name: " << device_name << endl;

        delete[] device_name;
    }

    void startIterKernel(vector<double>& real_vals, vector<double>& imag_vals) {
        cl::Context context(CL_DEVICE_TYPE_GPU);
        cl::Buffer real_buf(context, CL_MEM_READ_ONLY, sizeof(double) * width);
        cl::Buffer imag_buf(context, CL_MEM_READ_ONLY, sizeof(double) * height);
        size_t output_size = sizeof(int) * width * height * n_channels;
        cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY, output_size);
        cl::CommandQueue queue(context, device);

        // Copy the input data to the input buffers
        queue.enqueueWriteBuffer(real_buf, CL_TRUE, 0, sizeof(double) * width, real_vals.data());
        queue.enqueueWriteBuffer(imag_buf, CL_TRUE, 0, sizeof(double) * height, imag_vals.data());

        // Create the kernel and set its arguments
        std::string kernel_file_path = "kernel.cl";
        std::ifstream kernel_file(kernel_file_path);
        if (!kernel_file.is_open()) {
            std::cerr << "Failed to open kernel file: " << kernel_file_path << std::endl;
            exit(1);
        }
        std::stringstream kernel_buffer;
        kernel_buffer << kernel_file.rdbuf();
        std::string kernel_source = kernel_buffer.str();

        cl::Program::Sources sources;
        sources.push_back({ kernel_source.c_str(), kernel_source.length() });
        cl::Program program(context, sources);
        program.build(device);

        // Check for build errors
        cl_int build_status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
        if (build_status != CL_SUCCESS) {
            std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cerr << "OpenCL build error:\n" << build_log << std::endl;
            exit(1);
        }
        cl::Kernel kernel(program, "mandel");
        int err;
        err = kernel.setArg(0, output_buf);
        cout << "Kernel::setArg()0 --> " << err << endl;
        err = kernel.setArg(1, real_buf);
        cout << "Kernel::setArg()1 --> " << err << endl;
        err = kernel.setArg(2, imag_buf);
        cout << "Kernel::setArg()2 --> " << err << endl;
        err = kernel.setArg(3, width);
        cout << "Kernel::setArg()3 --> " << err << endl;
        err = kernel.setArg(4, height);
        cout << "Kernel::setArg()4 --> " << err << endl;
        err = kernel.setArg(5, max_iter);
        cout << "Kernel::setArg()5 --> " << err << endl;
        err = kernel.setArg(6, (unsigned short)color_depth);
        cout << "Kernel::setArg()6 --> " << err << endl;
       
        // Enqueue the kernel for execution
        const cl::NDRange global_size(width, height);
        int ret;
        ret = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange);
        cout << "Kernel executed: " << ret << endl;

        cl_int kernel_error = queue.finish();
        if (kernel_error != CL_SUCCESS) {
            std::cerr << "Error running kernel: " << kernel_error << std::endl;
            exit(1);
        }
        vector<int> output_data(width * height * n_channels);
        queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, output_size, output_data.data());

        T* p = img.ptr<T>();
        for (int i = 0; i < output_data.size(); i++) {
            p[i] = (T)output_data[i];
        }
        //for (int i = 0; i < output_data.size(); i++) {
        //    p[i] = static_cast<unsigned char>(output_data[i]);
        //}
    }
    // Alternative:
    //    //for (Pixel& p : cv::Mat_<Pixel>(img)) {
    //    //    complex<double> c = scaled_coord(current_x, current_y, x_start, y_start);
    //    //    unsigned int iterations = get_iter_nr(c);
    //    //    p.x = bounded_color(b_factor * intensity * iterations * max_iter, b_factor); // B
    //    //    p.y = bounded_color(g_factor * intensity * iterations * max_iter, g_factor); // G
    //    //    p.z = bounded_color(r_factor * intensity * iterations * max_iter, r_factor); // R
};
