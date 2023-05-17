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
#include "MellowSim.h"


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

const int hor_resolution = 2048;
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
void onChange(int event, int x, int y, int z, void*) {
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

string get_most_recent_file(const string& directory) {
    string latest_file_name;
    filesystem::file_time_type latest_time;

    for (const auto& entry : filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            auto current_time = entry.last_write_time();
            if (latest_file_name.empty() || current_time > latest_time) {
                latest_file_name = entry.path().filename().string();
                latest_time = current_time;
            }
        }
    }

    return latest_file_name;
}


void zoomOut() {
    while (st.size() > 1) {
        st.pop();
    }
    MandelArea<T_IMG> area = st.top();
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
    }
    else {
        file = ifstream(zoom_folder + filename);
        if (!file) {
            cerr << "Error opening file: " << filename << endl;
            exit(1);
        }
    }

    int x, y;
    int zooms_count = 0;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    while (file >> x >> y) {
        onChange(1, x, y, 1, NULL);
        zooms_count++;
    }
    chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    cout << endl << "Total time required for guided zoom: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    cout << "Number of zooms: " << zooms_count << endl;
}

int main() {
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);
    cout << endl;

    st.push(MandelArea<T_IMG>(first_start_x, first_end_x, first_start_y, first_end_y, aspect_ratio, hor_resolution, intensity, magnification));

    namedWindow(w_name);

    setMouseCallback(w_name, onChange, 0);

    // Common resoltions: 1024, 2048, 4K: 4096, 8K: 7680, 16K: 15360

    cout << endl << "Press z to start a guided zoom" << endl << "Press s to save a picture" << endl << "Press Esc to exit" << endl;

    cout << endl;

    while (true) {
        char pressed_key = (char)waitKey(10);
        if ((char)27 == pressed_key) break;
        else if ((char)115 == pressed_key) {
            MandelArea<T_IMG> area = st.top();
            cout << "Saving picture to " << area.filename << endl;
            imwrite(area.filename, area.img);
        }
        else if ((char)122 == pressed_key) {
            cout << endl << "Starting guided zoom..." << endl;
            startZoom("");
        }
    }

    return 0;
}
