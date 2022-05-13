// TODO: Partial calculation in arbitrary locations, no mirroring or partial mirroring, better image quality (16 bit colors), border detection, GUI
#include <math.h>
#include <complex>
#include <iostream>
#include <map>
#include <vector>
#include <fstream>
#include  <iomanip>
#include <time.h>

using namespace std;

#define max_iter 2000

#define dist_limit 4 //Arbitrary but has to be at least 2

#define color_depth 65535 // 255 alternative


// TODO: Fix memory corruption, Logfile, Argparser, Refactor, Tests, linter

// Complex number: z = a + b*i

map<int, int> resolutions = {{1280,720}, {1920,1080}, {2048,1080}, {3840,2160}, {4096,2160}};
int x_px = 1280;
int y_px = resolutions[x_px];
int px_count = x_px*y_px;


const string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
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
        unsigned int y_px;
        double x_per_px;
        double y_per_px;
        bool partial_write;
        string filename;
        vector<unsigned int> r;
        vector<unsigned int> g;
        vector<unsigned int> b;
        //vector<unsigned long long> colors; // 64 bits --> 16 per color + opacity

        void create_file(){
            system("mkdir -p output");
            ofstream file;
            file.open("output/" + filename + ".txt");
            file << "P3\n" << x_px << " " << y_px << "\n" << color_depth << endl;
            file.close();
        }

        MandelArea(double x_start, double x_end, double y_start, double y_end, unsigned int x_px, unsigned int y_px){
            this->x_start = x_start;
            this->x_end = x_end;
            this->y_start = y_start;
            this->y_end = y_end;
            // if (x_start > x_end){
            //     this->x_dist = x_start - x_end; 
            // }
            // else {
            //     this->x_dist = x_end - x_start; 
            // }
            this->x_dist = x_start > x_end ? x_start - x_end : x_end - x_start;
            // if (y_start > y_end){
            //     this->y_dist = y_start - y_end; 
            // }
            // else {
            //     this->y_dist = y_end - y_start; 
            // }
            this->y_dist = y_start > y_end ? y_start - y_end : y_end - y_start;
            this->x_px = x_px;
            this->y_px = y_px;
            this->x_per_px = x_dist/x_px;
            this->y_per_px = y_dist/y_px;
            this->px_count = x_px*y_px;
            this->filename = currentDateTime();
            if(x_px*y_px > 1280){
                partial_write = true;
            } else {
                partial_write = false;
            }
            create_file();
            r.reserve(px_count);
            g.reserve(px_count);
            b.reserve(px_count);
            
        }


        ~MandelArea(){
            //vector<unsigned long long>().swap(colors); //This will create an empty vector with no memory allocated and swap it with colors, effectively deallocating the memory.
            r.erase(r.begin(), r.end());
            g.erase(g.begin(), g.end());
            b.erase(b.begin(), b.end());
        }

        unsigned int test_color(unsigned int color, float factor){
            if(color*factor > color_depth){
                return color_depth;
            } else {
                return color;
            }
        }

        void write(float r_fact, float g_fact, float b_fact){ //Sinnvolle Parameter? Filepath? Relative Position in Mandelbrotmenge? bool partial?
            // TODO: Muss noch abgeändert werden um an der passenden Stelle zu schreiben --> Oder beim Aufruf muss es an die richtige Stelle schreiben
            // anstatt die Textdatei zu überschreiben
            // siehe test.cpp --> Mit ios::app als parameter beim Konstruktor der ofstream file geht es leicht.
            
            ofstream file;
            file.open("output/" + filename + ".txt", ios::app);

            char s = ' ';
            for(unsigned long i = 0; i < px_count; i++){
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
            // string open_file = "eog output/" + filename + ".png";
            // system(open_file.c_str());
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

        int calculate_set(float intensity){
            // Gradual colors --> could be improved by weighting different colors to certain spans
            unsigned long counter = 0;
            for(unsigned int y=0; y <= y_px; y++){
                //cout << " y changes ";
                for(unsigned int x=0; x < x_px; x++){
                    complex<double> c = scaled_coord(x, y, x_start, y_start);
                    unsigned int iterations = get_iter_nr(c);
                    if(intensity*iterations > max_iter){
                        intensity = 1;
                    }
                    float r_factor = 1.;
                    float g_factor = 0.5;
                    float b_factor = 0.2;
                    r[counter] = r_factor*intensity*iterations*max_iter;
                    g[counter] = g_factor*intensity*iterations*max_iter;
                    b[counter] = b_factor*intensity*iterations*max_iter;
                    counter++;
                }
            }
        return 0;
        // Maybe save the calculated iter_nrs into a file --> first line of file should contain the amount of pixels and therefore iter_nrs + maybe the date it was
        // calculated on or some other metadata
    }

    void show(){
        float intensity = 1.;
        this->calculate_set(intensity);
        this->write(1., 1., 1.);

        string command = "pnmtopng output/" + filename + ".txt > output/" + filename + ".png";
        system(command.c_str());
        string open_file = "eog output/" + filename + ".png";
        system(open_file.c_str());
    }


};

int main(){
    cout << endl;
    int ret;
    MandelArea m1(-2.7, 1.2, 1.2, -1.2, 1280, resolutions[1280]);
    m1.show();

    // ret = calculate_set();
    // if(ret == 0){
    //     save_png();
    // }
    cout << endl;
    
    return 0;
}

