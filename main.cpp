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

#define max_iter 1000

#define dist_limit 4 //Arbitrary but has to be at least 2

#define color_precision 65535 // 255 alternative

map<int, int> resolutions = {{1280,720}, {1920,1080}, {2048,1080}, {3840,2160}, {4096,2160}};