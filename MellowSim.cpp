#include <math.h>
#include <complex>
#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include <iomanip>
#include <time.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <Windows.h>
#include <limits>
#include <typeinfo>
#include <stack>
#include "MellowSim.h"


using namespace std;
using namespace cv;

typedef unsigned short T_IMG;

int sizes[] = { 255, 255, 255 };
typedef Point3_<uint8_t> Pixel;

// TODO-List: https://trello.com/b/37JofojU/mellowsim

const unsigned short max_iter = 2000;

const unsigned short dist_limit = 4; //Arbitrary but has to be at least 2

//const unsigned short color_depth = (1 << 16) - 1; // 2^16 - 1

const unsigned short block_size = 4096;

const unsigned short n_channels = 3;
map<int, int> resolutions = { {1280,720}, {1920,1080}, {2048,1080}, {3840,2160}, {4096,2160} };

const float aspect_ratio = 16. / 9.;
const int w_width = 1024;
const int w_height = w_width / aspect_ratio;
const float first_start_x = -2.7;
const float first_end_x = 1.2;
const float first_start_y = 1.2;
const float first_end_y = -1.2;

float zoom_factor = 0.2;
float zoom_change = 0.2;
float min_zoom = 0.05;
float max_zoom = 0.95;

int prev_x = -1;
int prev_y = -1;
int prev_z = 0;

// Complex number: z = a + b*i

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


void update_area() {
    // TODO: cleanFrame !!, MandelArea, coordinates
}


Mat showing;
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

    if (event == EVENT_MOUSEWHEEL) {
        cout << "Scrolled x: " << x << ", y: " << y << ", z: " << z << endl;
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

    // 1. Rechteck berechnen, in welches gezoomt wird
    // 2. Rechteck auf image zeichnen
    // 3. MandelArea dieses Rechtecks berechnen, vorherige noch in Memory halten vector<MandelArea> prev, bzw. Stack + mit Rechtsklick kann man rauszoomen
    // Durch Nutzen der alten MandelArea-Objekte muss das Bild nicht separat gespeichert werden, ist this->img, Koordinaten sind auch wieder richtig.
    // Es sollte eine Beschränkung der Menge an Zooms geben --> 20 Zooms werden in Memory gehalten, falls man 21 Mal zurückzoomt muss das herauszoomen
    // "manuell" berechnet werden.

    // TODO: Randerkennung --> Rechteck wird an nächstbeste Position platziert, in das es reinpasst, scroll verändert zoomgrad, mittlere Maustaste
    // resettet zoomgrad

    // Print center of MandelArea coordinates with std::fixed

    if (event == EVENT_LBUTTONDOWN) {
        cout << "Clicked on x: " << x << ", y: " << y << endl;
        // TODO: Archive area
        //double start_x = zoom_factor * area.x_dist;
        //double start_y = zoom_factor * area.y_dist;
        double start_x = area.x_start + corrected_x * area.x_per_px;
        double start_y = area.y_start - corrected_y * area.y_per_px;
        double end_x = start_x + zoom_width * area.x_per_px;
        double end_y = start_y + zoom_height * area.y_per_px;
        st.push(MandelArea<T_IMG>(start_x, end_x, start_y, end_y, aspect_ratio, w_width, 1.));
    }

    if (event == EVENT_RBUTTONDOWN) {
        if (st.size() > 1) st.pop();
    }

    if (prev_x != x || prev_y != y || prev_z != z) {
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
    cout << endl;

    namedWindow(w_name);

    st.push(MandelArea<T_IMG>(first_start_x, first_end_x, first_start_y, first_end_y, aspect_ratio, w_width, 1.));

    setMouseCallback(w_name, onClick, 0);

    //chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    
    // TODO: Ratio von (deltax/deltay) abhängig machen?
    //MandelArea m_area(-1.1, -0.9, 0.4, 0.2, ratio, 4096, 1.);
    //std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    //std::cout << "Time difference = " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    //cout << "Depth: " << m_area.color_depth << endl;

    // Common resoltions: 1024, 2048, 4K: 4096, 8K: 7680, 16K: 15360

    cout << endl;
    waitKey(0);

    return 0;
}
