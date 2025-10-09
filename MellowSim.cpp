#include <algorithm>
#include <atomic>
#include <chrono>
#include <complex>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stack>
#include <fstream>

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/opencl.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/logger.hpp>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "DUMMY_PATH_FOR_IDE"
#endif

const std::filesystem::path project_root = PROJECT_SOURCE_DIR;
const std::filesystem::path kernel_path = project_root / "kernel.cl";
const std::filesystem::path zooms_dir   = project_root / "zooms";
const std::filesystem::path output_dir = project_root / "output";

float zoom_factor = 0.2f;
float zoom_change = 0.2f;
float min_zoom    = 0.05f;
float max_zoom    = 0.95f;
unsigned long long magnification = 1;

int prev_x = -1;
int prev_y = -1;
int prev_z = 0;

const std::string w_name = "MellowSim";

const float        dist_limit     = 4.f; // >= 2
const unsigned int start_max_iter = 100;

typedef cv::Point3_<uint8_t> Pixel;

const float aspect_ratio   = 16.f / 9.f;
const int   w_width        = 960;
const int   w_height       = static_cast<int>(w_width / aspect_ratio);
const float first_start_x  = -2.7f;
const float first_end_x    =  1.2f;
const float first_start_y  =  1.2f;
const float first_end_y    = -1.2f;

const int hor_resolution = 2048;
const int ver_resolution = static_cast<int>(hor_resolution / aspect_ratio);

std::atomic<bool> ignore_callbacks(false);

// ---------- Time helpers ----------
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

inline std::string time_stamp(const std::string& fmt = "%Y_%m_%d_%H_%M_%S")
{
    auto bt = localtime_xp(std::time(nullptr));
    char buf[64];
    return { buf, std::strftime(buf, sizeof(buf), fmt.c_str(), &bt) };
}

void show_progress_bar(float progress) {
    int barWidth = 70;
    if (progress <= 1.0f) {
        std::cout << "[";
        int pos = static_cast<int>(barWidth * progress);
        for (int i = 0; i < barWidth; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << "X";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0f) << " %\r";
        std::cout.flush();
    }
    if (progress >= 1.0f) std::cout << std::endl;
}

template <typename T>
class MandelArea {
public:
    long double x_start = 0, x_end = 0, y_start = 0, y_end = 0;
    long double x_dist = 0, y_dist = 0;
    int         px_count = 0;
    int         width = 0;
    float       ratio = 0.f;
    int         height = 0;
    long double x_per_px = 0, y_per_px = 0;

    bool        partial_write = false;
    std::string filename;

    cv::Mat     img, full_res;
    const T     color_depth = static_cast<T>(-1);
    unsigned long long magnification = 1;

    cl::Device  device{};
    size_t      power_of_two_local_array_size = 0;

    unsigned int max_iter = 0;
    unsigned int prev_max_iter = 0;

    bool        stop_iterating = false;
    bool        active = false;

    MandelArea(long double xs, long double xe, long double ys, long double ye,
               float r, int w, unsigned long long mag)
      : x_start(xs), x_end(xe), y_start(ys), y_end(ye), width(w), ratio(r),
        height(static_cast<int>(w / r)),
        magnification(mag)
    {
        x_dist   = fabsl(x_end - x_start);
        y_dist   = fabsl(y_start - y_end);
        x_per_px = x_dist / width;
        y_per_px = y_dist / height;
        px_count = width * height;

        filename      = get_filename();
        prev_max_iter = (magnification == 1) ? start_max_iter : max_iter;
        max_iter      = start_max_iter * (static_cast<unsigned int>(std::log(magnification) * std::log(magnification)) + 1);
        stop_iterating = false;
        active         = true;
        std::cout << "Max_iter: " << max_iter << std::endl;

        getDevice(device, power_of_two_local_array_size);

        const size_t mat_type = get_mat_type();
        if (mat_type == 0) return;
        img = cv::Mat(height, width, mat_type);

        write_img();
        img.copyTo(full_res);
        if (w_width != width) resize(img, img, cv::Size(w_width, static_cast<int>(w_width / ratio)), cv::INTER_LINEAR_EXACT);
        imshow(w_name, img);
        cv::waitKey(1);
    }

    size_t get_mat_type() {
        const std::type_info& id = typeid(T);
        if (id == typeid(char))           return CV_8SC3;
        if (id == typeid(short))          return CV_16SC3;
        if (id == typeid(float))          return CV_32FC3;
        if (id == typeid(double))         return CV_64FC3;
        if (id == typeid(unsigned char))  return CV_8UC3;
        if (id == typeid(unsigned short)) return CV_16UC3;
        return 0;
    }

    std::string get_filename() {
        std::error_code ec;
        std::filesystem::create_directories(output_dir, ec);
        return (output_dir / (time_stamp() + ".png")).string();
    }

    void write_img() {
        std::vector<double> real_vals(width);
        std::vector<double> imag_vals(height);

        for (int x = 0; x < width; ++x)  real_vals[x] = static_cast<double>(x_start + x * x_per_px);
        for (int y = 0; y < height; ++y) imag_vals[y] = static_cast<double>(y_start - y * y_per_px);

        auto begin = std::chrono::steady_clock::now();
        startIterKernel(real_vals, imag_vals);
        auto end   = std::chrono::steady_clock::now();
        std::cout << "Kernel (GPU) time = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
                  << " [ms]\n";

        std::cout << "\n" << std::setprecision(std::numeric_limits<long double>::max_digits10)
             << "start_x=" << x_start << " start_y=" << y_start << std::endl;

        cv::cvtColor(img, img, cv::COLOR_HSV2BGR);
    }

    void getDevice(cl::Device& deviceOut, size_t& pow2_local_arr_size) {
        const cl::Context context(CL_DEVICE_TYPE_DEFAULT);
        const std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) {
            std::cerr << "No OpenCL GPU devices found.\n";
            std::exit(1);
        }
        deviceOut = devices[0];

        cl_device_id device_id = nullptr;
        cl_uint ret_num_devices = 0, ret_num_platforms = 0;

        cl_int ret = clGetPlatformIDs(0, nullptr, &ret_num_platforms);
        if (ret != CL_SUCCESS || ret_num_platforms == 0) {
            std::cerr << "No OpenCL platforms detected.\n";
            std::exit(1);
        }

        std::vector<cl_platform_id> platforms(ret_num_platforms);
        ret = clGetPlatformIDs(ret_num_platforms, platforms.data(), nullptr);

        ret = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);
        if (ret != CL_SUCCESS || ret_num_devices == 0) {
            std::cerr << "No OpenCL devices on platform 0.\n";
            std::exit(1);
        }

        char device_name[256]{};
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
        std::cout << "Device name: " << device_name << std::endl;

        cl_ulong max_local_mem_size = 0;
        clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &max_local_mem_size, nullptr);
        std::cout << "Local size: " << max_local_mem_size << std::endl;

        size_t max_work_group_size = 0;
        clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
        std::cout << "Max work group size: " << max_work_group_size << std::endl;
        std::cout << "Global size: " << (static_cast<size_t>(width) * static_cast<size_t>(height)) << std::endl;

        size_t local_array_size = std::min(static_cast<size_t>(max_local_mem_size / sizeof(double)),
                                   static_cast<size_t>(max_work_group_size));


        // largest power of two <= local_array_size
        pow2_local_arr_size = 1;
        while ((pow2_local_arr_size << 1) <= local_array_size) {
            pow2_local_arr_size <<= 1;
        }
    }

    void startIterKernel(std::vector<double>& real_vals, std::vector<double>& imag_vals) {
        // Context / buffers
        cl::Context context(CL_DEVICE_TYPE_DEFAULT);
        cl::CommandQueue queue(context, device);

        cl::Buffer real_buf(context, CL_MEM_READ_ONLY,  sizeof(double) * width);
        cl::Buffer imag_buf(context, CL_MEM_READ_ONLY,  sizeof(double) * height);
        size_t output_size = sizeof(int) * static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
        size_t iter_size   = sizeof(unsigned int) * static_cast<size_t>(width) * static_cast<size_t>(height);
        size_t z_size      = sizeof(cl_double2)   * static_cast<size_t>(width) * static_cast<size_t>(height);
        cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY, output_size);
        cl::Buffer end_iter_buf(context, CL_MEM_READ_WRITE, iter_size);
        cl::Buffer end_z_buf(context, CL_MEM_READ_WRITE, z_size);

        // Upload inputs
        queue.enqueueWriteBuffer(real_buf, CL_TRUE, 0, sizeof(double) * width,  real_vals.data());
        queue.enqueueWriteBuffer(imag_buf, CL_TRUE, 0, sizeof(double) * height, imag_vals.data());

        // Load kernel file
        std::ifstream kernel_file(kernel_path);
        if (!kernel_file.is_open()) {
            std::cerr << "Failed to open kernel file: " << kernel_path << std::endl;
            std::exit(1);
        }
        std::stringstream kernel_buffer;
        kernel_buffer << kernel_file.rdbuf();
        std::string kernel_source = kernel_buffer.str();

        // Replace macro
        const std::string macro_placeholder = "LOCAL_ARRAY_SIZE";
        const std::string macro_value = std::to_string(power_of_two_local_array_size);
        size_t pos = 0;
        while ((pos = kernel_source.find(macro_placeholder, pos)) != std::string::npos) {
            kernel_source.replace(pos, macro_placeholder.length(), macro_value);
            pos += macro_value.length();
        }

        cl::Program::Sources sources;
        sources.push_back({ kernel_source.c_str(), kernel_source.length() });
        cl::Program program(context, sources);

        program.build(device);
        cl_int build_status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
        if (build_status != CL_SUCCESS) {
            std::string build_log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cerr << "OpenCL build error:\n" << build_log << std::endl;
            std::exit(1);
        }

        // First kernel
        cl::Kernel kernel(program, "mandel");
        kernel.setArg(0, output_buf);
        kernel.setArg(1, real_buf);
        kernel.setArg(2, imag_buf);
        kernel.setArg(3, width);
        kernel.setArg(4, height);
        kernel.setArg(5, start_max_iter);
        kernel.setArg(6, static_cast<int>(color_depth));
        kernel.setArg(7, end_iter_buf);
        kernel.setArg(8, end_z_buf);

        const cl::NDRange global_size(static_cast<size_t>(width), static_cast<size_t>(height));
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, global_size, cl::NullRange);
        cl_int kernel_error = queue.finish();
        if (kernel_error != CL_SUCCESS) {
            std::cerr << "Error running kernel: " << kernel_error << std::endl;
            std::exit(1);
        }

        std::vector<int> output_data(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, output_size, output_data.data());

        // Write HSV to Mat
        T* p = img.ptr<T>();
        for (size_t i = 0; i < output_data.size(); ++i) {
            p[i] = static_cast<T>(output_data[i]);
        }

        // Show first result
        cv::Mat showing;
        img.copyTo(showing);
        if (w_width != width) resize(showing, showing, cv::Size(w_width, static_cast<int>(w_width / ratio)), cv::INTER_LINEAR_EXACT);
        cv::cvtColor(showing, showing, cv::COLOR_HSV2BGR);
        imshow(w_name, showing);
        cv::waitKey(1);

        // Gradually increase iterations
        unsigned int current_iter = start_max_iter;
        int step_iter = 4 * static_cast<int>(start_max_iter);
        unsigned int rest  = max_iter % static_cast<unsigned int>(step_iter);
        int loops = (static_cast<int>(max_iter) / step_iter) - 1;
        if (rest > 0) ++loops;

        for (int i = 0; i < loops; ++i) {
            if (stop_iterating) break;

            if (i == loops - 1 && rest != 0) current_iter += rest;
            else                               current_iter += static_cast<unsigned int>(step_iter);

            cl::Kernel continue_kernel(program, "continue_mandel");
            continue_kernel.setArg(0, output_buf);
            continue_kernel.setArg(1, real_buf);
            continue_kernel.setArg(2, imag_buf);
            continue_kernel.setArg(3, width);
            continue_kernel.setArg(4, height);
            continue_kernel.setArg(5, current_iter);
            continue_kernel.setArg(6, static_cast<int>(color_depth));
            continue_kernel.setArg(7, end_iter_buf);
            continue_kernel.setArg(8, end_z_buf);

            queue.enqueueNDRangeKernel(continue_kernel, cl::NullRange, global_size, cl::NullRange);
            std::cout << "Current max_iter: " << current_iter << std::endl;

            kernel_error = queue.finish();
            if (kernel_error != CL_SUCCESS) {
                std::cerr << "Error running kernel: " << kernel_error << std::endl;
                std::exit(1);
            }
            output_data.assign(output_data.size(), 0);
            queue.enqueueReadBuffer(output_buf, CL_TRUE, 0, output_size, output_data.data());

            // Update HSV Mat
            T* p2 = img.ptr<T>();
            for (size_t j = 0; j < output_data.size(); ++j) {
                p2[j] = static_cast<T>(output_data[j]);
            }

            // Show incremental result
            img.copyTo(showing);
            if (w_width != width) resize(showing, showing, cv::Size(w_width, static_cast<int>(w_width / ratio)), cv::INTER_LINEAR_EXACT);
            cv::cvtColor(showing, showing, cv::COLOR_HSV2BGR);
            imshow(w_name, showing);
            cv::waitKey(1);
        }
    }

    void set_stop_iterating(bool v) { stop_iterating = v; }
};

// ---------- Globals that depend on MandelArea ----------
typedef unsigned char T_IMG;
std::stack<MandelArea<T_IMG>> st;

// ---------- Mouse handling ----------
cv::Mat showing;
bool showing_zoombox = true;

void onChange(int event, int x, int y, int z, void*) {
    if (ignore_callbacks) return;

    x = std::min(x, w_width);
    y = std::min(y, w_height);

    MandelArea<T_IMG> area = st.top();

    int zoom_width  = static_cast<int>(w_width  * zoom_factor);
    int zoom_height = static_cast<int>(w_height * zoom_factor);
    int corrected_x = std::max(0, x - (zoom_width  / 2));
    int corrected_y = std::max(0, y - (zoom_height / 2));
    if (corrected_x + zoom_width  + 1 > w_width)  corrected_x = w_width  - zoom_width;
    if (corrected_y + zoom_height + 1 > w_height) corrected_y = w_height - zoom_height;

    if (event == cv::EVENT_MBUTTONDOWN) {
        showing_zoombox = !showing_zoombox;
        if (!showing_zoombox && area.active) {
            imshow(w_name, area.img);
        }
    }

    if (event == cv::EVENT_MOUSEWHEEL) {
        float new_zoom_factor = zoom_factor;
        if (z > 0) new_zoom_factor = zoom_factor * (1 - zoom_change);
        if (z < 0) new_zoom_factor = zoom_factor * (1 + zoom_change);
        zoom_factor = std::clamp(new_zoom_factor, min_zoom, max_zoom);
    }

    if (event == cv::EVENT_LBUTTONDOWN) {
        ignore_callbacks = true;

        area.set_stop_iterating(true);
        area.active = false;
        magnification /= zoom_factor;

        long double start_x = area.x_start + corrected_x * area.x_dist / w_width;
        long double start_y = area.y_start - corrected_y * area.y_dist / w_height;
        long double end_x   = start_x + zoom_width  * area.x_dist / w_width;
        long double end_y   = start_y + zoom_height * area.y_dist / w_height;

        auto begin = std::chrono::steady_clock::now();
        st.push(MandelArea<T_IMG>(start_x, end_x, start_y, end_y, aspect_ratio, hor_resolution, magnification));
        auto end   = std::chrono::steady_clock::now();
        std::cout << "Time elapsed = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]\n";
        MandelArea<T_IMG> area2 = st.top();
        std::cout << "Magnification = " << magnification << std::endl;

        ignore_callbacks = false;
    }

    if (event == cv::EVENT_RBUTTONDOWN && st.size() > 1) {
        ignore_callbacks = true;
        st.pop();
        auto &area = st.top();
        area.active = true;
        magnification = area.magnification;

        cv::Mat disp = area.img;
        if (w_width != area.width) {
            cv::resize(disp, disp, cv::Size(w_width, w_width / area.ratio), cv::INTER_LINEAR_EXACT);
        }
        cv::Mat bgr;
        cv::cvtColor(disp, bgr, cv::COLOR_HSV2BGR);

        cv::imshow(w_name, bgr);
        cv::waitKey(1);

        std::cout << "Magnification = " << magnification << std::endl;
        ignore_callbacks = false;
    }

    if (showing_zoombox && event == cv::EVENT_MOUSEMOVE) {
        if (area.active) {
            cv::Rect rect(corrected_x, corrected_y, zoom_width, zoom_height);
            area.img.copyTo(showing);
            rectangle(showing, rect, cv::Scalar(0, area.color_depth, 0));
            imshow(w_name, showing);
        }
    }
    prev_x = x; prev_y = y; prev_z = z;
}

// ---------- Zoom file helpers ----------
std::string get_most_recent_file(const std::string& directory) {
    std::string latest_file_name;
    std::filesystem::file_time_type latest_time;
    std::cout << "All available files:\n";
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            auto current_time = entry.last_write_time();
            if (latest_file_name.empty() || current_time > latest_time) {
                latest_file_name = entry.path().filename().string();
                latest_time = current_time;
            }
            std::cout << entry.path().filename().string() << '\n';
        }
    }
    std::cout << std::endl;
    return latest_file_name;
}

void zoomOut() {
    while (st.size() > 1) st.pop();
    MandelArea<T_IMG> area = st.top();
    area.active = true;
    magnification = area.magnification;
    std::cout << "Magnification = " << magnification << std::endl;
}

void startZoom(std::string filename) {
    if (magnification > 1) zoomOut();

    std::ifstream file;
    if (filename.empty()) {
        bool accepted = false;
        std::string most_recent_file = get_most_recent_file(zooms_dir);
        while (!accepted) {
            if (most_recent_file.empty()) {
                std::cout << "No files in zooms folder. Exiting zoom selection...\n";
                return;
            } else {
                std::cout << "Most recent file is " << most_recent_file << " (Enter to select)\n";
            }
            std::cout << "Enter filename (with extension): ";
            getline(std::cin, filename);

            if (filename.empty()) filename = most_recent_file;
            if (filename == "exit") {
                std::cout << "Exiting guided zoom selection...\n";
                return;
            }
            file = std::ifstream(zooms_dir / filename);
            if (file) accepted = true;
            else {
                std::cerr << "Error opening file: " << filename << "\nTry again or type \"exit\".\n";
            }
        }
    } else {
        file = std::ifstream(zooms_dir / filename);
        if (!file) { std::cerr << "Error opening file: " << filename << std::endl; std::exit(1); }
    }

    bool original_zoombox_state = showing_zoombox;
    showing_zoombox = false;

    int x, y, zoom_width, zoom_height;
    int zooms_count = 0;
    auto begin = std::chrono::steady_clock::now();
    file >> zoom_width;
    zoom_height = static_cast<int>(zoom_width / aspect_ratio);
    float x_factor = static_cast<float>(w_width)  / zoom_width;
    float y_factor = static_cast<float>(w_height) / zoom_height;
    if (std::fmod(x_factor, 1.f) != 0.f || std::fmod(y_factor, 1.f) != 0.f) {
        std::cerr << "Guided zoom recorded at incompatible window size.\n";
        return;
    }
    while (file >> x >> y) {
        onChange(1, static_cast<int>(std::llround(x * x_factor)),
                    static_cast<int>(std::llround(y * y_factor)), 1, nullptr);
        ++zooms_count;
    }

    if (showing_zoombox && st.top().active) {
        onChange(cv::EVENT_MOUSEMOVE, prev_x, prev_y, 0, nullptr);
    }
    showing_zoombox = original_zoombox_state;

    auto end = std::chrono::steady_clock::now();
    std::cout << "\nTotal time for guided zoom: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " [ms]\n";
    std::cout << "Number of zooms: " << zooms_count << std::endl;
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
    std::cout << std::endl;

    cv::namedWindow(w_name, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);

    st.push(MandelArea<T_IMG>(first_start_x, first_end_x, first_start_y, first_end_y,
                              aspect_ratio, hor_resolution, magnification));

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    cv::setMouseCallback(w_name, onChange, nullptr);

    std::cout << "\nPress z to start a guided zoom\nPress s to save a picture\nPress Esc to exit\n\n";

    while (true) {
        char pressed_key = static_cast<char>(cv::waitKey(10));
        if (pressed_key == 27) break; // ESC
        else if (pressed_key == 's') {
            MandelArea<T_IMG> area = st.top();
            std::cout << "Saving picture to " << area.filename << std::endl;
            imwrite(area.filename, area.full_res.empty() ? area.img : area.full_res);
        } else if (pressed_key == 'z') {
            std::cout << "\nStarting guided zoom...\n";
            startZoom("");
        }
    }
    return 0;
}
