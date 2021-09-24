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

#define max_iter 100

#define dist_limit 4 //Arbitrary but has to be at least 2

#define color_precision 65535 // 255 alternative

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
    vector<unsigned long long> colors; // 64 bits --> 16 per color + opacity

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
        colors.resize(px_count);
    }



    ~MandelArea(){
        vector<unsigned long long>().swap(colors); //This will create an empty vector with no memory allocated and swap it with colors, effectively deallocating the memory.
        //colors.erase(colors.begin(), colors.end());
    }

    void write(){ //Sinnvolle Parameter? Filepath? Relative Position in Mandelbrotmenge? bool partial?

        // TODO: Muss noch abgeändert werden um an der passenden Stelle zu schreiben --> Oder beim Aufruf muss es an die richtige Stelle schreiben
        // anstatt die Textdatei zu überschreiben
        // siehe test.cpp --> Mit ios::app als parameter beim Konstruktor der ofstream file geht es leicht.

        system("mkdir output");
        string filename = currentDateTime();
        ofstream file;
        file.open("output/" + filename + ".txt");
        // Nur wenn die File noch leer ist bzw. nicht existiert
        //file << "P3\n" << x_px << " " << y_px << "\n" << color_precision << endl;
        char s = ' ';
        for(int i = 0; i < px_count; i++){
            int r, g, b, a;
            r = (colors[i] & 0xFF000000) >> (6*4);
            g = (colors[i] & 0x00FF0000) >> (4*4);
            b = (colors[i] & 0x0000FF00) >> (2*4);
            a = (colors[i] & 0x000000FF);
            //file << setfill('0') << setw(8) << right << hex << colors[i];
            file << r << s << g << s << b << s;
            if(i % x_px == 0 && i != 0){
                file << "\n";
            }
        }
        file.close();

        string command = "pnmtopng output/" + filename + ".txt > output/" + filename + ".png";
        system(command.c_str());
        string open_file = "eog output/" + filename + ".png";
        system(open_file.c_str());
    }
};

int main(){
    cout << endl;
    int ret;
    ret = calculate_set();
    if(ret == 0){
        save_png();
    }
    cout << endl;
    
    return 0;
}

