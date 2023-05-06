#ifndef __OPENCL_C_VERSION__
#include <cmath>
#endif

const float dist_limit = 4.; // Has to be 4 because of squared dist
const short n_channels = 3;

__kernel void mandel(__global int* output, __global const double* real_vals, __global const double* imag_vals, const unsigned int width, const unsigned int height, const unsigned int max_iter, const unsigned short color_depth)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int idy = get_global_id(1);
    const unsigned int index = n_channels * (idy * width + idx);

    if (idx >= width || idy >= height) {
        return;
    }

    const double real_c = real_vals[idx];
    const double imag_c = imag_vals[idy];
    double real_z = 0;
    double imag_z = 0;
    double real_temp = 0;
    double imag_temp = 0;
    double dist_squared = 0;
    int iter_nr = 0;

    while (dist_squared < dist_limit * dist_limit && iter_nr < max_iter) {
        real_temp = real_z * real_z - imag_z * imag_z + real_c;
        imag_temp = 2 * real_z * imag_z + imag_c;
        real_z = real_temp;
        imag_z = imag_temp;
        dist_squared = real_z * real_z + imag_z * imag_z;
        iter_nr++;
    }

    if (iter_nr == max_iter) iter_nr = 0;

    unsigned short hue_depth = 180;
    unsigned short hue_shift = 120;

    float iter_factor = (float)iter_nr / (float)max_iter;
    int hue = (iter_factor * (hue_depth - 1)) + hue_shift;
    hue = hue < color_depth ? hue : hue_depth;
    output[index] = hue;

    output[index+1] = color_depth;

    int value = 200 * iter_factor * color_depth;
    value = value < color_depth ? value : color_depth;
    output[index+2] = value;
}
