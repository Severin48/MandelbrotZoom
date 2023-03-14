#pragma once
#include <iostream>
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace std;
using namespace cv;

const string w_name = "MellowSim";
mutex img_data_mutex;
#include <mutex>

const unsigned short dist_limit = 4; //Arbitrary but has to be at least 2

const unsigned short block_size = 16192;

const unsigned short n_channels = 3;

unsigned int max_iter = 10000;

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
        size_t mat_type = get_mat_type();
        if (mat_type == 0) return;
        this->img = Mat(height, width, mat_type);
        this->write_img(intensity, false);
        resize(img, img, Size(w_width, w_width / ratio), INTER_AREA);
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

    complex<long double> scaled_coord(int x, int y, float x_start, float y_start) {
        //Real axis ranges from -2.5 to 1
        //Imaginary axis ranges from -1 to 1 but is mirrored on the real axis
        return complex<double>(x_start + x * x_per_px, y_start - y * y_per_px);
    }

    unsigned int get_iter_nr(complex<long double> c) {
        unsigned int counter = 0;
        complex<long double> z = 0;
        double dist = abs(z);
        while (dist < dist_limit && counter < max_iter) {
            z = z * z + c;
            dist = abs(z);
            counter++;
        }
        if (counter == max_iter) {
            return 0;
        } //Complex number is in the set and therefore is colored black
        else {
            return counter;
        }
    }

    void calculate_block(int current_block, float intensity) {
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
        unsigned char hue_shift = 120;
        T hue, saturation, value;
        /*
            TODO:
            Max_iter muss je nach zoom faktor multipliziert werden aber vorsicht nicht zu schnell sonst dauert es ewig!-- > Kleiner Anstieg z.B. 0.1 * zoom_factor
            Also wenn 5x gezoomt wird soll max_iter um Faktor(1 + 0, 5) wachsen

            Helligkeitsfaktor und Intensitätsfaktor muss auch von max_iter abhängen!
        */
        for (; data != end; data++) {
            complex<long double> c = scaled_coord(current_x, current_y, x_start, y_start);
            unsigned int iterations = get_iter_nr(c);
            float iter_factor = (float)iterations / (float)max_iter;
            hue = (iter_factor * (hue_depth - 1)) + hue_shift;
            hue = hue < color_depth ? hue : hue_depth;
            *data = hue;
            data++;
            saturation = color_depth;
            *data = saturation;
            data++;
            value = 20*iter_factor*color_depth;
            value = value < color_depth ? value : color_depth;
            *data = value;
            if (current_x % (width - 1) == 0 && current_x != 0) {
                current_x = 0;
                current_y++;
            }
            else current_x++;
        }
        img_data_mutex.lock();
        if (data_begin != nullptr) {
            memcpy(data_destination, data_begin, data_size);
            free(data_begin);
        }
        img_data_mutex.unlock();
    }

    void write_img(float intensity, bool save_img) {
        auto processor_count = thread::hardware_concurrency();
        processor_count = processor_count <= 0 ? 1 : processor_count;
        cout << endl << "Calculating Mandelbrot with " << processor_count << " threads." << endl;
        int remaining_blocks = n_blocks;
        int current_block = 0;
        while (remaining_blocks > 0) {
            float progress = n_blocks != 0 ? 1 - ((float)remaining_blocks / (float)n_blocks) : 1.;
            show_progress_bar(progress);
            int current_starting_block = n_blocks - remaining_blocks;
            vector<thread> threads(min((int)processor_count, remaining_blocks));
            for (size_t i = 0; i < threads.size(); ++i) {
                threads[i] = thread([this, current_block, intensity] {calculate_block(current_block, intensity); });
                current_block = i + 1 + current_starting_block;
            }
            for (size_t i = 0; i < threads.size(); ++i) {
                remaining_blocks--;
                threads[i].join();
            }
        }
        if (left_over_pixels > 0) {
            calculate_block(current_block, intensity);
        }
        float progress = 1 - ((float)remaining_blocks / (float)n_blocks);
        show_progress_bar(progress);
        cvtColor(img, img, CV_HSV2BGR);
        if (save_img) imwrite(filename, img);
    } 
    // Alternative:
    //    //for (Pixel& p : cv::Mat_<Pixel>(img)) {
    //    //    complex<double> c = scaled_coord(current_x, current_y, x_start, y_start);
    //    //    unsigned int iterations = get_iter_nr(c);
    //    //    p.x = bounded_color(b_factor * intensity * iterations * max_iter, b_factor); // B
    //    //    p.y = bounded_color(g_factor * intensity * iterations * max_iter, g_factor); // G
    //    //    p.z = bounded_color(r_factor * intensity * iterations * max_iter, r_factor); // R
};
