#include <math.h>
#include <complex>
#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include  <iomanip>
#include <time.h>

using namespace std;

// TODO-List: https://trello.com/b/37JofojU/mellowsim

#define max_iter 2000

#define dist_limit 4 //Arbitrary but has to be at least 2

#define color_depth 65535 // 255 alternative

#define block_size 4096

unsigned int r[block_size];
unsigned int g[block_size];
unsigned int b[block_size];

// Complex number: z = a + b*i

map<int, int> resolutions = {{1280,720}, {1920,1080}, {2048,1080}, {3840,2160}, {4096,2160}};


const string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

void show_progress_bar(float progress){
    int barWidth = 70;
    if (progress <= 1.0){
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
    if (progress == 1.0){
        std::cout << std::endl;
    }
}

class MandelArea{
    public:
        double x_start;
        double x_end;
        double y_start;
        double y_end;
        double x_dist;
        double y_dist;
        unsigned long px_count;
        unsigned int x_px;
        float ratio;
        unsigned int y_px;
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
        
        void create_file(){
            system("mkdir -p output");
            ofstream file;
            file.open("output/" + filename + ".txt");
            file << "P3\n" << x_px << " " << y_px << "\n" << color_depth << endl;
            file.close();
        }

        MandelArea(double x_start, double x_end, double y_start, double y_end, float ratio, unsigned int x_px){
            this->x_start = x_start;
            this->x_end = x_end;
            this->y_start = y_start;
            this->y_end = y_end;
            this->x_dist = x_start > x_end ? x_start - x_end : x_end - x_start;
            this->y_dist = y_start > y_end ? y_start - y_end : y_end - y_start;
            this->x_px = x_px;
            this->ratio = ratio;
            this->y_px = x_px/ratio;
            this->x_per_px = x_dist/x_px;
            this->y_per_px = y_dist/y_px;
            this->px_count = x_px*y_px;
            this->filename = currentDateTime();
            if(px_count > block_size){
                partial_write = true;
            } else {
                partial_write = false;
            }
            this->last_block = px_count/block_size;
            this->left_over_pixels = px_count % block_size;
            create_file();
        }


        // ~MandelArea(){
        // TODO
        // }

        unsigned int test_color(unsigned int color, float factor){
            if(color*factor > color_depth){
                return color_depth;
            } else {
                return color;
            }
        }

        void write_block(float r_fact, float g_fact, float b_fact){
            ofstream file;
            file.open("output/" + filename + ".txt", ios::app);

            char s = ' ';
            unsigned int amount_px = current_block == last_block ? left_over_pixels : block_size;
            for(unsigned int i = 0; i < amount_px; i++){
                unsigned int r_val, g_val, b_val;
                //file << setfill('0') << setw(8) << right << hex << colors[i];
                r_val = test_color(r[i], r_fact);
                g_val = test_color(g[i], g_fact);
                b_val = test_color(b[i], b_fact);
                file << r_val*r_fact << s << g_val*g_fact << s << b_val*b_fact << s;
                if(i % x_px == 0 && i != 0){
                    file << "\n";
                }
            }
            file.close();
        }

        complex<double> scaled_coord(int x, int y, float x_start, float y_start){
            //Real axis ranges from -2.5 to 1
            //Imaginary axis ranges from -1 to 1 but is mirrored on the real axis
            return complex<double>(x_start + x*x_per_px, y_start - y*y_per_px);    //-2.5 and 1 are the respective starting points.
        }

        unsigned int get_iter_nr(complex<double> c){
            unsigned int counter = 0;
            complex<double> z = 0;
            double dist = abs(z);
            while(dist < dist_limit && counter < max_iter){
                z = z*z + c;
                dist = abs(z);
                counter++;
            }
            //cout << "Counter: " << counter << "\tin_set: " << (counter == max_iter) << endl;
            if(counter == max_iter){
                return 0;
            } //Complex number is in the set and therefore is colored black
            else {
                return counter;
            }
        }

        int calculate_block(float intensity){
            // Gradual colors --> could be improved by weighting different colors to certain spans
            float r_factor = 1.; // 1.
            float g_factor = 0.5; // 0.5
            float b_factor = 0.2; // 0.2
            unsigned int needed_pxs = current_block == last_block ? left_over_pixels : block_size;
            for(unsigned int i=0; i < needed_pxs; i++){
                complex<double> c = scaled_coord(current_x, current_y, x_start, y_start);
                unsigned int iterations = get_iter_nr(c);
                r[i] = r_factor*intensity*iterations*max_iter;
                g[i] = g_factor*intensity*iterations*max_iter;
                b[i] = b_factor*intensity*iterations*max_iter;
                if (current_x % (x_px - 1) == 0 && current_x != 0) {
                    current_x = 0;
                    current_y++;
                }
                else current_x++;
            }
            current_px += needed_pxs;
        current_block++;
        return 0;
        // Maybe save the calculated iter_nrs into a file --> first line of file should contain the amount of pixels and therefore iter_nrs + maybe the date it was
        // calculated on or some other metadata
    }

    void make_png(float intensity){
        while (current_block <= last_block){
            float progress = (float)current_block/(float)last_block;
            show_progress_bar(progress);
            //printf("Calculating block nr. %d / %d...\n", current_block, last_block);
            this->calculate_block(intensity);
            //printf("Partial calculation finished, writing %d pixels...\n", block_size);
            this->write_block(1., 1., 1.);
        }

        string command = "pnmtopng output/" + filename + ".txt > output/" + filename + ".png";
        system(command.c_str());
        string open_file = "eog output/" + filename + ".png";
        system(open_file.c_str());
    }


};

int main(){
    cout << endl;
    int ret;
    float ratio = 16./9.;
    MandelArea m1(-2.7, 1.2, 1.2, -1.2, ratio, 15360); // TODO: Eingabe als Resolution level --> Ansonsten führt es auf Arrayzugriff mit falschem Index.
    m1.make_png(1.);
    // Common resoltions: 4K: 4096, 8K: 7680, 16K: 15360

    // ret = calculate_set();
    // if(ret == 0){
    //     save_png();
    // }
    cout << endl;
    
    return 0;
}

