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
#include <filesystem>
#include <Windows.h>
#include <limits.h>
#include <opencv2/imgproc/types_c.h>
#include <sstream>
#include <string>

#include <CL/opencl.hpp>

using namespace std;
using namespace cv;

// TODO-List: https://trello.com/b/37JofojU/mellowsim

//map<int, int> resolutions = { {1280,720}, {1920,1080}, {2048,1080}, {3840,2160}, {4096,2160} };

float zoom_factor = 0.2;
float zoom_change = 0.2;
float min_zoom = 0.05;
float max_zoom = 0.95;
unsigned long long magnification = 1;

int prev_x = -1;
int prev_y = -1;
int prev_z = 0;

const string w_name = "MellowSim";

const float dist_limit = 4.; //Arbitrary but has to be at least 2

const unsigned short n_channels = 3;

const unsigned int start_max_iter = 100;

int sizes[] = { 255, 255, 255 };
typedef Point3_<uint8_t> Pixel;

const float aspect_ratio = 16. / 9.;
const int w_width = 960;
const int w_height = w_width / aspect_ratio;
const float first_start_x = -2.7;
const float first_end_x = 1.2;
const float first_start_y = 1.2;
const float first_end_y = -1.2;

const int hor_resolution = 2048;
const int ver_resolution = hor_resolution / aspect_ratio;

atomic<bool> ignore_callbacks(false);

// Complex number: z = a + b*i

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
        cout << "Global size: " << width * height << endl;

        size_t local_array_size = min(max_local_mem_size / sizeof(double), max_work_group_size);

        // Find fitting power of two
        power_of_two_local_array_size = 1;
        while (power_of_two_local_array_size <= local_array_size) {
            power_of_two_local_array_size <<= 1;
        }
        power_of_two_local_array_size >>= 1;
    }

    void startIterKernel(vector<double>& real_vals, vector<double>& imag_vals) {
        // Initialize buffers
        cl::Context context(CL_DEVICE_TYPE_GPU);
        cl::Buffer real_buf(context, CL_MEM_READ_ONLY, sizeof(double) * width);
        cl::Buffer imag_buf(context, CL_MEM_READ_ONLY, sizeof(double) * height);
        size_t output_size = sizeof(int) * width * height * n_channels;
        size_t iter_size = sizeof(unsigned int) * width * height;
        size_t z_size = sizeof(cl_double2) * width * height;
        cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY, output_size);
        cl::Buffer end_iter_buf(context, CL_MEM_READ_WRITE, iter_size);
        cl::Buffer end_z_buf(context, CL_MEM_READ_WRITE, z_size);
        cl::CommandQueue queue(context, device);

        // Copy input data to the input buffers
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
        // Setting kernel arguments
        kernel.setArg(0, output_buf);
        kernel.setArg(1, real_buf);
        kernel.setArg(2, imag_buf);
        kernel.setArg(3, width);
        kernel.setArg(4, height);
        kernel.setArg(5, start_max_iter);
        kernel.setArg(6, (int)color_depth);
        kernel.setArg(7, end_iter_buf);
        kernel.setArg(8, end_z_buf);

        // Enqueue the kernel for execution
        const cl::NDRange global_size(width, height);

        // Executing kernel the first time for initial result
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange);

        cl_int kernel_error = queue.finish();
        if (kernel_error != CL_SUCCESS) {
            std::cerr << "Error running kernel: " << kernel_error << std::endl;
            exit(1);
        }
        vector<int> output_data(width * height * n_channels);

        // Read resulting data
        queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, output_size, output_data.data());

        // Writing resulting color values (HSV) to image data pointer
        T* p = img.ptr<T>();
        for (int i = 0; i < output_data.size(); i++) {
            p[i] = (T)output_data[i];
        }

        // Show first resulting image
        Mat showing;
        img.copyTo(showing);
        if (w_width != width) resize(showing, showing, Size(w_width, w_width / ratio), INTER_LINEAR_EXACT);
        cvtColor(showing, showing, CV_HSV2BGR);
        imshow(w_name, showing);
        waitKey(1); // Fixes OpenCV bug of image not showing after imshow()

        unsigned int current_iter = start_max_iter;
        int step_iter = 4 * start_max_iter;
        unsigned int rest = max_iter % step_iter;
        int loops = (max_iter / step_iter) - 1;
        if (rest > 0) loops++;

        // Gradual iteration to show intermediate results
        for (int i = 0; i < loops; i++) {
            if (stop_iterating) {
                break;
            }
            if (i == loops - 1 && rest != 0) {
                current_iter += rest;
            }
            else current_iter += step_iter;

            cl::Kernel continue_kernel(program, "continue_mandel");
            // Setting kernel arguments
            continue_kernel.setArg(0, output_buf);
            continue_kernel.setArg(1, real_buf);
            continue_kernel.setArg(2, imag_buf);
            continue_kernel.setArg(3, width);
            continue_kernel.setArg(4, height);
            continue_kernel.setArg(5, current_iter);
            continue_kernel.setArg(6, (int)color_depth);
            continue_kernel.setArg(7, end_iter_buf);
            continue_kernel.setArg(8, end_z_buf);

            // Execute gradual kernel
            queue.enqueueNDRangeKernel(continue_kernel, cl::NullRange, global_size, cl::NullRange);
            cout << "Current max_iter: " << current_iter << endl;

            cl_int kernel_error = queue.finish();
            if (kernel_error != CL_SUCCESS) {
                std::cerr << "Error running kernel: " << kernel_error << std::endl;
                exit(1);
            }
            output_data.clear();
            output_data.resize(width * height * n_channels);

            // Read resulting data
            queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, output_size, output_data.data());

            // Write calculated color values (HSV) into image pointer
            T* p = img.ptr<T>();
            for (int j = 0; j < output_data.size(); j++) {
                p[j] = (T)output_data[j];
            }

            // Show result
            img.copyTo(showing);
            if (w_width != width) resize(showing, showing, Size(w_width, w_width / ratio), INTER_LINEAR_EXACT);
            cvtColor(showing, showing, CV_HSV2BGR);
            imshow(w_name, showing);
            waitKey(1);
        }
    }
};

typedef unsigned char T_IMG;
stack<MandelArea<T_IMG>> st;


inline tm localtime_xp(time_t timer)
{
    tm bt{};
#if defined(__unix__)
    localtime_r(&timer, &bt);
#elif defined(_MSC_VER)
    localtime_s(&bt, &timer);
#else
    static mutex mtx;
    lock_guard<mutex> lock(mtx);
    bt = *localtime(&timer);
#endif
    return bt;
}

// default = "YYYY-MM-DD HH:MM:SS"
inline string time_stamp(const string& fmt = "%Y_%m_%d_%H_%M_%S") // "%F %T"
{
    auto bt = localtime_xp(time(0));
    char buf[64];
    return { buf, strftime(buf, sizeof(buf), fmt.c_str(), &bt) };
}

void show_progress_bar(float progress) {
    int barWidth = 70;
    if (progress <= 1.0) {
        cout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) cout << "=";
            else if (i == pos) cout << "X";
            else cout << " ";
        }
        cout << "] " << int(progress * 100.0) << " %\r";
        cout.flush();
    }
    if (progress == 1.0) {
        cout << endl;
    }
}


Mat showing;
bool showing_zoombox = true;
void onChange(int event, int x, int y, int z, void*) {
    if (ignore_callbacks) {
        // cout << "Ignoring callback" << endl;
        return;
    }
    x = x > w_width ? w_width : x;
    y = y > w_height ? w_height : y;
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
        if (!showing_zoombox && area.active) {
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
        ignore_callbacks = true;

        cout << "Clicked x=" << x << " y=" << y << " z=" << z << endl;

        area.set_stop_iterating(true);
        area.active = false;
        magnification /= zoom_factor;
        start_x = area.x_start + corrected_x * area.x_dist / w_width;
        start_y = area.y_start - corrected_y * area.y_dist / w_height;
        long double end_x = start_x + zoom_width * area.x_dist / w_width;
        long double end_y = start_y + zoom_height * area.y_dist / w_height;
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        st.push(MandelArea<T_IMG>(start_x, end_x, start_y, end_y, aspect_ratio, hor_resolution, magnification));
        chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        cout << "Time elapsed = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
        MandelArea<T_IMG> area = st.top();
        cout << "Magnification = " << magnification << endl;

        ignore_callbacks = false;
    }

    if (event == EVENT_RBUTTONDOWN && st.size() > 1) {
        st.pop();
        MandelArea<T_IMG> area = st.top();
        area.active = true;
        magnification = area.magnification;
        cout << "Magnification = " << magnification << endl;
    }

    if (showing_zoombox && event == EVENT_MOUSEMOVE) {
        if (area.active) {
            Rect rect(corrected_x, corrected_y, zoom_width, zoom_height);
            area.img.copyTo(showing);

            rectangle(showing, rect, cv::Scalar(0, area.color_depth, 0));
            imshow(w_name, showing);
        }
    }
    prev_x = x;
    prev_y = y;
    prev_z = z;
}

std::string get_most_recent_file(const std::string& directory) {
    std::string latest_file_name;
    std::filesystem::file_time_type latest_time;
    cout << "All available files: " << endl;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            auto current_time = entry.last_write_time();
            if (latest_file_name.empty() || current_time > latest_time) {
                latest_file_name = entry.path().filename().string();
                latest_time = current_time;
            }
            cout << entry.path().filename().string() << endl;
        }
    }
    cout << endl;
    return latest_file_name;
}


void zoomOut() {
    while (st.size() > 1) {
        st.pop();
    }
    MandelArea<T_IMG> area = st.top();
    area.active = true;
    magnification = area.magnification;
    cout << "Magnification = " << magnification << endl;
}


void startZoom(string filename) {
    if (magnification > 1) {
        zoomOut();
    }
    ifstream file;
    string zoom_folder = "zooms/";
    if (filename == "") {
        bool accepted = false;
        string most_recent_file = get_most_recent_file(zoom_folder);
        while (!accepted) {
            if (most_recent_file == "") {
                cout << "No files in zooms folder. Exiting zoom selection..." << endl;
                return;
            }
            else {
                cout << "Most recent file is " << most_recent_file << " (to select press Enter)" << endl;
            }
            cout << "Enter the filename (including file ending): ";
            getline(std::cin, filename);

            if (filename.length() == 0) filename = most_recent_file;
            if (filename == "exit") {
                cout << "Exiting guided zoom selection..." << endl;
                return;
            }

            file = ifstream(zoom_folder + filename);
            if (file) {
                accepted = true;
            }
            else {
                cerr << "Error opening file: " << filename << endl;
                cout << "Try again or type in \"exit\" to exit the selection." << endl;
            }
        }
    } else { file = ifstream(zoom_folder + filename);
        if (!file) {
            cerr << "Error opening file: " << filename << endl;
            exit(1);
        }
    }
    
    int x, y, zoom_width, zoom_height;
    int zooms_count = 0;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    file >> zoom_width;
    zoom_height = zoom_width / aspect_ratio;
    float x_factor = (float)w_width / zoom_width;
    float y_factor = (float)w_height / zoom_height;
    if (x_factor - (int)x_factor != 0. || y_factor - (int)y_factor != 0.) {
        cerr << "Guided zoom was recorded in a non-compatible window-size. Exiting zoom selection..." << endl;
        return;
    }
    while (file >> x >> y) {
        onChange(1, (int)round(x*x_factor), (int)round(y*y_factor), 1, NULL);
        zooms_count++;
    }
    chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cout << endl << "Total time required for guided zoom: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    cout << "Number of zooms: " << zooms_count << endl;
}


int main() {
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);
    cout << endl;

    namedWindow(w_name);

    st.push(MandelArea<T_IMG>(first_start_x, first_end_x, first_start_y, first_end_y, aspect_ratio, hor_resolution, magnification));

    this_thread::sleep_for(std::chrono::milliseconds(1000));

    setMouseCallback(w_name, onChange, 0);

    //onChange(1, 785, 289, 1, NULL); // Test ride

    // Common resoltions: 1024, 2048, 4K: 4096, 8K: 7680, 16K: 15360

    cout << endl << "Press z to start a guided zoom" << endl << "Press s to save a picture" << endl << "Press Esc to exit" << endl;

    cout << endl;

    while (true) {
        char pressed_key = (char)waitKey(10);
        if ((char)27 == pressed_key) break;
        else if ((char)115 == pressed_key) {
            MandelArea<T_IMG> area = st.top();
            cout << "Saving picture to " << area.filename << endl;
            imwrite(area.filename, area.full_res);
        } else if ((char)122 == pressed_key) {
            cout << endl << "Starting guided zoom..." << endl;
            startZoom("");
        }
    }

    return 0;
}
