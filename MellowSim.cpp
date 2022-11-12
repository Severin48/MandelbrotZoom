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


using namespace std;
using namespace cv;

struct BGR {
    uchar blue;
    uchar green;
    uchar red;
};

int sizes[] = { 255, 255, 255 };
typedef cv::Point3_<uint8_t> Pixel;

// TODO-List: https://trello.com/b/37JofojU/mellowsim

#define max_iter 2000

#define dist_limit 4 //Arbitrary but has to be at least 2

#define color_depth (1 << 8) - 1 // 2^8 - 1

#define block_size 4096

//unsigned int r[block_size];
//unsigned int g[block_size];
//unsigned int b[block_size];

// Complex number: z = a + b*i

map<int, int> resolutions = { {1280,720}, {1920,1080}, {2048,1080}, {3840,2160}, {4096,2160} };


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

class MandelArea {
public:
    double x_start;
    double x_end;
    double y_start;
    double y_end;
    double x_dist;
    double y_dist;
    int px_count;
    int width;
    float ratio;
    int height;
    double x_per_px;
    double y_per_px;
    bool partial_write;
    string filename;
    unsigned int current_block = 0;
    unsigned int last_block;
    unsigned int left_over_pixels;
    unsigned int current_x = 0;
    unsigned int current_y = 0;
    unsigned int current_px = 0;
    float intensity;
    Mat img;

    MandelArea(double x_start, double x_end, double y_start, double y_end, float ratio, int x_px, float intensity) {
        this->x_start = x_start;
        this->x_end = x_end;
        this->y_start = y_start;
        this->y_end = y_end;
        this->x_dist = x_start > x_end ? x_start - x_end : x_end - x_start;
        this->y_dist = y_start > y_end ? y_start - y_end : y_end - y_start;
        this->ratio = ratio;
        this->width = x_px;
        this->height = x_px / ratio;
        this->x_per_px = x_dist / x_px;
        this->y_per_px = y_dist / height;
        this->px_count = x_px * height;
        this->intensity = intensity;
        this->filename = get_filename();
        if (px_count > block_size) {
            partial_write = true;
        }
        else {
            partial_write = false;
        }
        this->last_block = px_count / block_size;
        this->left_over_pixels = px_count % block_size;
        img = Mat(height, width, CV_8UC3);
        this->calculate_block(intensity);
        imwrite(filename, img);
        imshow(filename, img);
        waitKey(0);
    }

    string get_filename() {
        string output_dir = "output/";
        string mkdir_str = "if not exist " + output_dir + " mkdir " + output_dir;
        system(mkdir_str.c_str());
        string file_ending = ".png";
        string filename = output_dir + time_stamp() + file_ending;

        //ofstream file;
        //file.open("output/" + filename + ".txt");
        //file << "P3\n" << x_px << " " << y_px << "\n" << color_depth << endl;
        //file.close();
        return filename;
    }


    // ~MandelArea(){
    // TODO - Aber schon in MyImage freigegeben?
    // }

    unsigned int test_color(unsigned int color, float factor) {
        if (color * factor > color_depth) {
            return color_depth;
        }
        else {
            return color;
        }
    }

    complex<double> scaled_coord(int x, int y, float x_start, float y_start) {
        //Real axis ranges from -2.5 to 1
        //Imaginary axis ranges from -1 to 1 but is mirrored on the real axis
        return complex<double>(x_start + x * x_per_px, y_start - y * y_per_px);    //-2.5 and 1 are the respective starting points.
    }

    unsigned int get_iter_nr(complex<double> c) {
        unsigned int counter = 0;
        complex<double> z = 0;
        double dist = abs(z);
        while (dist < dist_limit && counter < max_iter) {
            z = z * z + c;
            dist = abs(z);
            counter++;
        }
        //cout << "Counter: " << counter << "\tin_set: " << (counter == max_iter) << endl;
        if (counter == max_iter) {
            return 0;
        } //Complex number is in the set and therefore is colored black
        else {
            return counter;
        }
    }

    int calculate_block(float intensity) {
        // Gradual colors --> could be improved by weighting different colors to certain spans
        float r_factor = 1.; // 1.
        float g_factor = 0.5; // 0.5
        float b_factor = 0.2; // 0.2
        for (Pixel& p : cv::Mat_<Pixel>(img)) {
            complex<double> c = scaled_coord(current_x, current_y, x_start, y_start);
            unsigned int iterations = get_iter_nr(c);
            p.x = test_color(b_factor * intensity * iterations * max_iter, b_factor); // B
            p.y = test_color(g_factor * intensity * iterations * max_iter, g_factor); // G
            p.z = test_color(r_factor * intensity * iterations * max_iter, r_factor); // R
            if (current_x % (width - 1) == 0 && current_x != 0) {
                current_x = 0;
                current_y++;
            }
            else {
                current_x++;
            }
            //TODO: Show progress
        }


        //current_px += needed_pxs;
        //current_block++;
        return 0;
        // Maybe save the calculated iter_nrs into a file --> first line of file should contain the amount of pixels and therefore iter_nrs + maybe the date it was
        // calculated on or some other metadata
    }

    //void make_png(float intensity) {
    //    while (current_block <= last_block) {
    //        float progress = (float)current_block / (float)last_block;
    //        show_progress_bar(progress);
    //        //printf("Calculating block nr. %d / %d...\n", current_block, last_block);
    //        this->calculate_block(intensity);
    //        //printf("Partial calculation finished, writing %d pixels...\n", block_size);
    //        this->write_block(1., 1., 1.);
    //    }

    //    string command = "pnmtopng output/" + filename + ".txt > output/" + filename + ".png";
    //    system(command.c_str());
    //    string open_file = "eog output/" + filename + ".png";
    //    system(open_file.c_str());
    //}


};

int main() {
    try
    {
        cout << endl;
        //int ret;
        float ratio = 16. / 9.;
        const auto processor_count = std::thread::hardware_concurrency();
        MandelArea m_area(-2.7, 1.2, 1.2, -1.2, ratio, 4096, 1.); // TODO: Eingabe als Resolution level --> Ansonsten führt es auf Arrayzugriff mit falschem Index.

        // TODO: Achtung wenn processor_count = 0!

        // Common resoltions: 4K: 4096, 8K: 7680, 16K: 15360

        // ret = calculate_set();
        // if(ret == 0){
        //     save_png();
        // }
        cout << endl;
    }
    catch (cv::Exception& e)
    {
        cerr << e.msg << endl; // output exception message
    }


    return 0;
}


//#include <iostream>
//using namespace cv;
//using namespace std;
//int main(int argc, char** argv)
//{
//    Mat image;
//    image = imread(argv[1], IMREAD_COLOR); // Read the file
//    if (image.empty()) // Check for invalid input
//    {
//        cout << "Could not open or find the image" << std::endl;
//        return -1;
//    }
//    namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
//    imshow("Display window", image); // Show our image inside it.
//    waitKey(0); // Wait for a keystroke in the window
//    return 0;
//}
