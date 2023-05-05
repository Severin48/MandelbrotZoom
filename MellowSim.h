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
mutex img_data_mutex;
#include <mutex>

const double dist_limit = 4; //Arbitrary but has to be at least 2

const unsigned short block_size = 16192;

const unsigned short n_channels = 3;

unsigned int max_iter = 4000;

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

    //complex<long double> scaled_coord(int x, int y, float x_start, float y_start) {
    //    //Real axis ranges from -2.5 to 1
    //    //Imaginary axis ranges from -1 to 1 but is mirrored on the real axis
    //    return complex<double>(x_start + x * x_per_px, y_start - y * y_per_px);
    //}

    //unsigned int get_iter_nr(complex<long double> c) {
    //    unsigned int counter = 0;
    //    complex<long double> z = 0;
    //    double dist = abs(z);
    //    while (dist < dist_limit && counter < max_iter) {
    //        z = z * z + c;
    //        dist = abs(z);
    //        counter++;
    //    }
    //    if (counter == max_iter) {
    //        return 0;
    //    } //Complex number is in the set and therefore is colored black
    //    else {
    //        return counter;
    //    }
    //}

    void calculate_block(int current_block, float intensity, vector<int> iter_nrs) {
        // Gradual colors --> could be improved by weighting different colors to certain spans
        float r_factor = 0.0;
        float g_factor = 0.0;
        float b_factor = 0.0;
        int pixel_offset = current_block * block_size;
        unsigned int needed_pxs = current_block == n_blocks ? left_over_pixels : block_size;
        size_t data_size = needed_pxs * n_channels * sizeof(T);
        T* data_begin = (T*)malloc(data_size);
        T* data = data_begin;
        T* end = data + needed_pxs * n_channels;

        unsigned int current_x = pixel_offset % width;
        unsigned int current_y = pixel_offset / width;
        T* data_destination = img.ptr<T>() + pixel_offset * n_channels;
        unsigned char hue_depth = 180;
        unsigned char hue_shift = 120; // 120 for blue shift
        T hue, saturation, value;
        /*
            TODO:
            Max_iter muss je nach zoom faktor multipliziert werden aber vorsicht nicht zu schnell sonst dauert es ewig!-- > Kleiner Anstieg z.B. 0.1 * zoom_factor
            Also wenn 5x gezoomt wird soll max_iter um Faktor(1 + 0, 5) wachsen

            Helligkeitsfaktor und Intensitätsfaktor muss auch von max_iter abhängen!
        */

        int index = pixel_offset;
        for (; data != end; data++) {
            //complex<long double> c = scaled_coord(current_x, current_y, x_start, y_start);
            //unsigned int iterations = get_iter_nr(c);
            float iter_factor = (float)iter_nrs[index] / (float)max_iter;
            hue = (iter_factor * (hue_depth - 1)) + hue_shift;
            hue = hue < color_depth ? hue : hue_depth;
            *data = hue;
            data++;

            saturation = color_depth;
            *data = saturation;
            data++;

            value = 200*iter_factor*color_depth;
            //value = iter_nrs[index] == 0 ? 0 : color_depth/iter_nrs[index];
            //value = iter_nrs[index] == 0 ? 0 : color_depth /iter_nrs[index];
            value = value < color_depth ? value : color_depth;
            *data = value;

            if (current_x % (width - 1) == 0 && current_x != 0) {
                current_x = 0;
                current_y++;
            }
            else current_x++;

            index++;
        }
        img_data_mutex.lock();
        if (data_begin != nullptr) {
            memcpy(data_destination, data_begin, data_size);
            free(data_begin);
        }
        img_data_mutex.unlock();
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
        vector<int> iter_nrs = startIterKernel(real_vals, imag_vals);

        auto processor_count = thread::hardware_concurrency();
        processor_count = processor_count <= 0 ? 1 : processor_count;
        cout << endl << "Calculating Mandelbrot on " << processor_count << " cores." << endl;
        int remaining_blocks = n_blocks;
        int current_block = 0;
        while (remaining_blocks > 0) {
            float progress = n_blocks != 0 ? 1 - ((float)remaining_blocks / (float)n_blocks) : 1.;
            show_progress_bar(progress);
            int current_starting_block = n_blocks - remaining_blocks;
            vector<thread> threads(min((int)processor_count, remaining_blocks));
            for (size_t i = 0; i < threads.size(); ++i) {
                threads[i] = thread([this, current_block, intensity, iter_nrs] {calculate_block(current_block, intensity, iter_nrs); });
                current_block = i + 1 + current_starting_block;
            }
            for (size_t i = 0; i < threads.size(); ++i) {
                remaining_blocks--;
                threads[i].join();
            }
        }
        if (left_over_pixels > 0) {
            calculate_block(current_block, intensity, iter_nrs);
        }
        float progress = 1 - ((float)remaining_blocks / (float)n_blocks);
        show_progress_bar(progress);
        
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

    vector<int> startIterKernel(vector<double>& real_vals, vector<double>& imag_vals) {
        cl::Context context(CL_DEVICE_TYPE_GPU);
        cl::Buffer real_buf(context, CL_MEM_READ_ONLY, sizeof(double) * width);
        cl::Buffer imag_buf(context, CL_MEM_READ_ONLY, sizeof(double) * height);
        std::vector<int> output_data(width * height);
        cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY, sizeof(int) * width * height);
        cl::CommandQueue queue(context, device);

        // Copy the input data to the input buffers
        queue.enqueueWriteBuffer(real_buf, CL_TRUE, 0, sizeof(double) * width, real_vals.data());
        queue.enqueueWriteBuffer(imag_buf, CL_TRUE, 0, sizeof(double) * height, imag_vals.data());

        // Create the kernel and set its arguments
        std::string kernel_file_path = "iternr.cl";
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
        cl::Kernel kernel(program, "iternr");
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
        err = kernel.setArg(5, max_iter);
        //cout << "Kernel::setArg()5 --> " << err << endl;
        err = kernel.setArg(6, dist_limit);
        //cout << "Kernel::setArg()6 --> " << err << endl;

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

        queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, output_data.size() * sizeof(int), output_data.data());

        return output_data;
    }
    // Alternative:
    //    //for (Pixel& p : cv::Mat_<Pixel>(img)) {
    //    //    complex<double> c = scaled_coord(current_x, current_y, x_start, y_start);
    //    //    unsigned int iterations = get_iter_nr(c);
    //    //    p.x = bounded_color(b_factor * intensity * iterations * max_iter, b_factor); // B
    //    //    p.y = bounded_color(g_factor * intensity * iterations * max_iter, g_factor); // G
    //    //    p.z = bounded_color(r_factor * intensity * iterations * max_iter, r_factor); // R
};
