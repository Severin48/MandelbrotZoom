#ifndef __OPENCL_C_VERSION__
#include <cmath>
#endif

__constant const float dist_limit_sq = 4.;
__constant const short n_channels = 3;

__kernel void mandel(__global int* output, __global const double* real_vals, __global const double* imag_vals, const unsigned int width, const unsigned int height, const unsigned int max_iter, const int color_depth)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int idy = get_global_id(1);

    if (idx >= width || idy >= height) {
        return;
    }

    const unsigned int index = n_channels * (idy * width + idx);

    __local double local_real_vals[LOCAL_ARRAY_SIZE];
    __local double local_imag_vals[LOCAL_ARRAY_SIZE];

    local_real_vals[idx] = real_vals[idx];
    local_imag_vals[idy] = imag_vals[idy];

    barrier(CLK_LOCAL_MEM_FENCE);

    const double real_c = real_vals[idx];
    const double imag_c = imag_vals[idy];
    double real_z = 0;
    double imag_z = 0;
    double dist_squared = 0;
    int iter_nr;

    for (iter_nr = 0; iter_nr < max_iter && dist_squared < dist_limit_sq; iter_nr++) {
        double real_temp = real_z * real_z - imag_z * imag_z + real_c;
        double imag_temp = 2 * real_z * imag_z + imag_c;
        real_z = real_temp;
        imag_z = imag_temp;
        dist_squared = real_z * real_z + imag_z * imag_z;
    }

    if (iter_nr == max_iter) iter_nr = 0;

    unsigned short hue_depth = 180;
    unsigned short hue_shift = 120;
    float iter_factor = (float)iter_nr / (float)max_iter;
    int hue = (iter_factor * (hue_depth - 1)) + hue_shift;
    hue = min(hue, (int)hue_depth);
    output[index] = hue;
    output[index + 1] = color_depth;
    int value = min((int)(200 * iter_factor * color_depth), color_depth);
    output[index + 2] = value;
}