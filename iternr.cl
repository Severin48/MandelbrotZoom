#ifndef __OPENCL_C_VERSION__
#include <cmath>
#endif

__kernel void iternr(__global int* output, __global const double* real_vals, __global const double* imag_vals, const unsigned int width, const unsigned int height, const unsigned int max_iter, const double dist_limit)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int idy = get_global_id(1);
    const unsigned int index = idy * width + idx;

    if (idx >= width || idy >= height) {
        return;
    }

    const double real_c = real_vals[idx];
    const double imag_c = imag_vals[idy];
    double real_z = 0;
    double imag_z = 0;
    double real_temp = 0;
    double imag_temp = 0;
    double dist = 0;
    unsigned int counter = 0;

    while (dist < dist_limit && counter < max_iter) {
        real_temp = real_z * real_z - imag_z * imag_z + real_c;
        imag_temp = 2 * real_z * imag_z + imag_c;
        real_z = real_temp;
        imag_z = imag_temp;
        dist = sqrt(real_z * real_z + imag_z * imag_z);
        counter++;
    }

    if (counter == max_iter) {
        output[index] = 2;
    }
    else {
        output[index] = counter;
    }
}
