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

    if (event == EVENT_LBUTTONDOWN) {
        MandelArea<T_IMG> area2 = area;
        cout << area.filename << endl; // TODO: Remove
        magnification /= zoom_factor;
        long double start_x = area.x_start + corrected_x * area.x_dist / w_width;
        long double start_y = area.y_start - corrected_y * area.y_dist / w_height;
        long double end_x = start_x + zoom_width * area.x_dist / w_width;
        long double end_y = start_y + zoom_height * area.y_dist / w_height;
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        st.push(MandelArea<T_IMG>(start_x, end_x, start_y, end_y, aspect_ratio, hor_resolution, intensity, magnification));
        chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        cout << "Time difference = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
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

int main() {
    utils::logging::setLogLevel(utils::logging::LogLevel::LOG_LEVEL_SILENT);
    cout << endl;

    namedWindow(w_name);

    st.push(MandelArea<T_IMG>(first_start_x, first_end_x, first_start_y, first_end_y, aspect_ratio, hor_resolution, intensity, magnification));

    setMouseCallback(w_name, onClick, 0);

    // Common resoltions: 1024, 2048, 4K: 4096, 8K: 7680, 16K: 15360

    cout << endl;
    while ((char)27 != (char)waitKey(0)) {
        if ((char)115 == (char)waitKey(0)) {
            MandelArea<T_IMG> area = st.top();
            cout << "Saving picture to " << area.filename << endl;
            imwrite(area.filename, area.img);
        }
    }

    return 0;
}
