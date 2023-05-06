#ifndef __OPENCL_C_VERSION__
#include <cmath>
#endif

__constant const float dist_limit = 4.;
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
    int iter_nr = 0;

    double2 z = (double2)(0, 0);
    double2 c = (double2)(real_c, imag_c);
    double2 temp;
    float dist_squared = 0;

    while (dist_squared < dist_limit && iter_nr < max_iter) {
        temp.x = z.x * z.x - z.y * z.y + c.x;
        temp.y = 2 * z.x * z.y + c.y;
        z = temp;
        dist_squared = dot(z, z);
        iter_nr++;
    }

    if (iter_nr == max_iter) iter_nr = 0;

    unsigned short hue_depth = 180;
    unsigned short hue_shift = 60;
    float iter_factor = (float)iter_nr / (float)max_iter;
    int hue = (iter_factor * (hue_depth - 1)) + hue_shift;
    hue = min(hue, (int)hue_depth);
    output[index] = hue;
    output[index + 1] = color_depth;
    int value = min((int)(200 * iter_factor * color_depth), color_depth);
    output[index + 2] = value;
}