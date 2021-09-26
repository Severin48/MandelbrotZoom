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
        vector<unsigned long long> colors; // 64 bits --> 16 per color + opacity

        void create_file(){
            system("mkdir output");
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
            if(x_start < 0 && x_end < 0 ){
                this->x_dist = x_start - x_end;
            }
            else { //Dies beinhaltet auch den Fall x_start > 0 && x_end > 0 und setzt x_dist richtig.
                this->x_dist = x_end - x_start;
            }
            if(y_start < 0 && y_end < 0 ){
                this->y_dist = y_start - y_end;
            }
            else {
                this->y_dist = y_end - y_start;
            }
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
            colors.resize(px_count);
        }


        ~MandelArea(){
            vector<unsigned long long>().swap(colors); //This will create an empty vector with no memory allocated and swap it with colors, effectively deallocating the memory.
            //colors.erase(colors.begin(), colors.end());
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
                unsigned int r, g, b, a;
                r = (colors[i] & 0xFFFF000000000000) >> (6*8);
                g = (colors[i] & 0x0000FFFF00000000) >> (4*8);
                b = (colors[i] & 0x00000000FFFF0000) >> (2*8);
                a = (colors[i] & 0x000000000000FFFF);
                //file << setfill('0') << setw(8) << right << hex << colors[i];
                r = test_color(r, r_fact);
                g = test_color(g, g_fact);
                b = test_color(b, b_fact);
                file << r*r_fact << s << g*g_fact << s << b*b_fact << s;
                if(i % x_px == 0 && i != 0){
                    file << "\n";
                }
            }
            file.close();
            // string open_file = "eog output/" + filename + ".png";
            // system(open_file.c_str());
        }

        complex<double> scaled_coord(int x, int y){
            //Real axis ranges from -2.5 to 1
            //Imaginary axis ranges from -1 to 1 but is mirrored on the real axis
            return complex<double>((-2.5) + x*x_per_px, 1.- y*y_per_px);    //-2.5 and 1 are the respective starting points. 0.5*y_px because
                                                                            // we only need to calculate half of the y axis
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
            for(unsigned int y=0; y <= y_px; y++){ // Only iterate through to y_px/2 because we can mirror it
                //cout << " y changes ";
                for(unsigned int x=0; x < x_px; x++){
                    complex<double> c = scaled_coord(x, y);
                    unsigned int iterations = get_iter_nr(c);
                    if(intensity*iterations > max_iter){
                        intensity = 1;
                    }
                    //long long color = (intensity*iterations*max_iter)*0xFFFFFFFFFFFFFFFF;
                    long long color = (intensity*iterations*max_iter)*0xFFFFFFFF;
                    colors[counter] = color;
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
    MandelArea m1(-2.5, 1., -1., 1., 1280, resolutions[1280]);
    m1.show();

    // ret = calculate_set();
    // if(ret == 0){
    //     save_png();
    // }
    cout << endl;
    
    return 0;
}

