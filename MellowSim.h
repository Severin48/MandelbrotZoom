#pragma once
#include <iostream>
#include <Windows.h>
#include <limits.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <fstream>
#include <sstream>
#include <string>

#include <CL/opencl.hpp>

using namespace std;
using namespace cv;

const string w_name = "MellowSim";

const float dist_limit = 4.; //Arbitrary but has to be at least 2

const unsigned short n_channels = 3;

const unsigned int start_max_iter = 100;

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
    Mat img, full_res;
    const T color_depth = (T)-1;
    unsigned long long magnification;
    cl::Device device;
    size_t power_of_two_local_array_size;
    unsigned int max_iter;
    unsigned int prev_max_iter;
    bool stop_iterating;
    bool active;

    MandelArea(long double x_start, long double x_end, long double y_start, long double y_end, float ratio, int width, unsigned long long magnification) {
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
        this->magnification = magnification;
        this->filename = get_filename();
        this->prev_max_iter = magnification == 1 ? start_max_iter : max_iter;
        this->max_iter = start_max_iter * (log(magnification) * log(magnification) + 1);
        this->stop_iterating = false;
        this->active = true;
        cout << "Max_iter: " << max_iter << endl;
        getDevice(device, power_of_two_local_array_size);
        size_t mat_type = get_mat_type();
        if (mat_type == 0) return;
        this->img = Mat(height, width, mat_type);
        this->write_img(false);
        img.copyTo(full_res);
        if (w_width != width) resize(img, img, Size(w_width, w_width / ratio), INTER_LINEAR_EXACT);
        imshow(w_name, img);
        waitKey(1);
    }

    void set_stop_iterating(bool val) {
        stop_iterating = val;
    }

    bool isRendering() {
        return this->rendering;
    }

    void setRendering(bool val) {
        this->rendering = val;
    }

    size_t get_mat_type() {
        const type_info& id = typeid(T);
        if (id == typeid(char)) return CV_8SC3;
        if (id == typeid(short)) return CV_16SC3;
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

    void write_img(bool save_img) {
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
        //img.copyTo(full_res);
        //if (save_img) imwrite(filename, full_res);
    }

    void getDevice(cl::Device& device, size_t& power_of_two_local_array_size) {
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

        cl_ulong max_local_mem_size = 0;
        clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &max_local_mem_size, NULL);
        cout << "Local size: " << max_local_mem_size << endl;

        size_t max_work_group_size = 0;
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
        cout << "Max work group size: " << max_work_group_size << endl;
        cout << "Global size: " << width * height * n_channels << endl;

        size_t local_array_size = min(max_local_mem_size / sizeof(double), max_work_group_size);
        //cout << "Local array size: " << local_array_size << endl;

        power_of_two_local_array_size = 1;
        while (power_of_two_local_array_size <= local_array_size) {
            power_of_two_local_array_size <<= 1;
        }
        power_of_two_local_array_size >>= 1;

        //cout << "Power of 2 local array size: " << power_of_two_local_array_size << endl;
    }

    void startIterKernel(vector<double>& real_vals, vector<double>& imag_vals) {
        unsigned int current_iter = start_max_iter;
        cl::Context context(CL_DEVICE_TYPE_GPU);
        cl::Buffer real_buf(context, CL_MEM_READ_ONLY, sizeof(double) * width);
        cl::Buffer imag_buf(context, CL_MEM_READ_ONLY, sizeof(double) * height);
        size_t output_size = sizeof(int) * width * height * n_channels;
        size_t iter_size = sizeof(unsigned int) * width * height;
        size_t z_size = sizeof(cl_double2) * width * height;
        cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY, output_size);
        cl::Buffer end_iter_buf(context, CL_MEM_READ_WRITE, iter_size);
        cl::Buffer end_z_buf(context, CL_MEM_READ_WRITE, z_size);
        size_t counters_size = sizeof(unsigned int) * width * height;
        cl::Buffer pixel_counters_buf(context, CL_MEM_READ_WRITE, counters_size);
        vector<unsigned int> pixel_counters(width * height, 0);
        cl::CommandQueue queue(context, device);
        queue.enqueueWriteBuffer(pixel_counters_buf, CL_TRUE, 0, counters_size, pixel_counters.data());


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

        std::string macro_placeholder = "LOCAL_ARRAY_SIZE";
        std::string macro_value = to_string(power_of_two_local_array_size);
        size_t pos = 0;
        while ((pos = kernel_source.find(macro_placeholder, pos)) != std::string::npos) {
            kernel_source.replace(pos, macro_placeholder.length(), macro_value);
            pos += macro_value.length();
        }

        //cout << kernel_source << endl;

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
        //cout << "Kernel::setArg()0 --> " << err << endl;
        err = kernel.setArg(1, real_buf);
        //cout << "Kernel::setArg()1 --> " << err << endl;
        err = kernel.setArg(2, imag_buf);
        //cout << "Kernel::setArg()2 --> " << err << endl;
        err = kernel.setArg(3, width);
        //cout << "Kernel::setArg()3 --> " << err << endl;
        err = kernel.setArg(4, height);
        //cout << "Kernel::setArg()4 --> " << err << endl;
        err = kernel.setArg(5, start_max_iter);
        //cout << "Kernel::setArg()5 --> " << err << endl;
        err = kernel.setArg(6, (int)color_depth);
        //cout << "Kernel::setArg()6 --> " << err << endl;
        err = kernel.setArg(7, end_iter_buf);
        //cout << "Kernel::setArg()7 --> " << err << endl;
        err = kernel.setArg(8, end_z_buf);
        //cout << "Kernel::setArg()8 --> " << err << endl;
        err = kernel.setArg(9, pixel_counters_buf);
        //cout << "Kernel::setArg()9 --> " << err << endl;
       
        // Enqueue the kernel for execution
        const cl::NDRange global_size(width, height, n_channels);
        int ret;
        ret = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange);
        cout << "Kernel run 0 executed with code: " << ret << endl;

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
        Mat showing;
        img.copyTo(showing);
        if (w_width != width) resize(showing, showing, Size(w_width, w_width / ratio), INTER_LINEAR_EXACT);
        cvtColor(showing, showing, CV_HSV2BGR);
        imshow(w_name, showing);
        waitKey(1);

        // TODO: Abbrechen wenn geklickt wird
        // TODO: In einem eigenen Thread das ganze hier ausführen, damit das Handling noch funktioniert (Maus-Inputs etc.)
        int step_iter = 4 * start_max_iter;
        unsigned int rest = max_iter % step_iter;
        int loops = (max_iter / step_iter) - 1;
        if (rest > 0) loops++;
        for (int i=0; i < loops; i++) {
            if (stop_iterating) {
                break;
            }
            if (i == loops - 1 && rest != 0) {
                current_iter += rest;
            }
            else current_iter += step_iter;
            int status = queue.enqueueWriteBuffer(pixel_counters_buf, CL_TRUE, 0, counters_size, pixel_counters.data());
            if (status != CL_SUCCESS) {
                std::cerr << "Error: Failed to write to pixel_counters_buffer! Error code: " << status << std::endl;
                exit(1);
            }
            cl::Kernel continue_kernel(program, "continue_mandel");
            int err;
            err = continue_kernel.setArg(0, output_buf);
            //cout << "Kernel::setArg()0 --> " << err << endl;
            err = continue_kernel.setArg(1, real_buf);
            //cout << "Kernel::setArg()1 --> " << err << endl;
            err = continue_kernel.setArg(2, imag_buf);
            //cout << "Kernel::setArg()2 --> " << err << endl;
            err = continue_kernel.setArg(3, width);
            //cout << "Kernel::setArg()3 --> " << err << endl;
            err = continue_kernel.setArg(4, height);
            //cout << "Kernel::setArg()4 --> " << err << endl;
            err = continue_kernel.setArg(5, current_iter);
            //cout << "Kernel::setArg()5 --> " << err << endl;
            err = continue_kernel.setArg(6, (int)color_depth);
            //cout << "Kernel::setArg()6 --> " << err << endl;
            err = continue_kernel.setArg(7, end_iter_buf);
            //cout << "Kernel::setArg()6 --> " << err << endl;
            err = continue_kernel.setArg(8, end_z_buf);
            //cout << "Kernel::setArg()6 --> " << err << endl;
            err = continue_kernel.setArg(9, pixel_counters_buf);
            //cout << "Kernel::setArg()9 --> " << err << endl;
            ret = queue.enqueueNDRangeKernel(continue_kernel, cl::NullRange, global_size, cl::NullRange);
            cout << "Kernel run " << i + 1 << " executed with code: " << ret << endl;
            cout << "Current max_iter: " << current_iter << endl;

            cl_int kernel_error = queue.finish();
            if (kernel_error != CL_SUCCESS) {
                std::cerr << "Error running kernel: " << kernel_error << std::endl;
                exit(1);
            }
            output_data.clear();
            output_data.resize(width * height * n_channels);
            queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, output_size, output_data.data());

            T* p = img.ptr<T>();
            for (int j = 0; j < output_data.size(); j++) {
                p[j] = (T)output_data[j];
            }
            img.copyTo(showing);
            if (w_width != width) resize(showing, showing, Size(w_width, w_width / ratio), INTER_LINEAR_EXACT);
            cvtColor(showing, showing, CV_HSV2BGR);
            imshow(w_name, showing); // FIXME: Not showing gradual updates
            waitKey(1);
        }
    }
};
