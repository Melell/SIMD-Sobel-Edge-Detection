/**
* @file edge.cpp
* @author Miguel Echeverria , 540000918 , miguel.echeverria@digipen.edu
* @date 2021/03/05
* @brief Contains the SIMD implementation of a basic sobel edge detection algorithm.
*
* @copyright Copyright (C) 2020 DigiPen Institute of Technology .
*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "bmp.h"
#include "measure.h"
#include <iostream>

/* just used for time measurements */
#define REP 10

typedef void (*test_func) (unsigned char *, unsigned char *, unsigned,
                           unsigned);

// Stores the next 8 characters starting from originalStart onto copy, interleaving a char with value 0 in between each character.
void make_pixels_16_bits(const unsigned char* originalStart, unsigned char* const copy)
{
    for (unsigned i = 0, j = 0; i < 8; ++i, j += 2)
        copy[j] = originalStart[i];
}

// Stores the first 8 bits of each of the next 8 shorts starting from originalStart, onto copy.
void make_pixels_8_bits(const unsigned char* originalStart, unsigned char* const copy)
{
    for (unsigned i = 0, j = 0; i < 8; ++i, j += 2)
        copy[i] = originalStart[j];
}

void sse_sobel_edge_detection(unsigned char *data_out,
                              unsigned char *data_in, unsigned height,
                              unsigned width)
{
    unsigned size = width * height;
    const unsigned stride = 8;

    __m128i negationValue = _mm_set1_epi16(-1);
    __m128i leftShiftAmmount = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 1);

    for (unsigned col = 0; col < 3; ++col)
    {
        for (unsigned r = 1; r < height - 1; ++r)
        {
            for (unsigned c = 1; c < width - 1; c += stride)
            {
                // If the current 8 pixels to process exceed the width - 1, go back and process some old pixels again to fit
                if (c + stride >= width)
                    c -= c + stride - width + 1;

                const unsigned colorStart = col * size;

                const unsigned char* offsets[9] = {
                    data_in + colorStart + (r - 1) * width + (c - 1),
                    data_in + colorStart + (r - 1) * width + c,
                    data_in + colorStart + (r - 1) * width + (c + 1),
                    data_in + colorStart + r * width + (c - 1),
                    data_in + colorStart + r * width + c,
                    data_in + colorStart + r * width + (c + 1),
                    data_in + colorStart + (r + 1) * width + (c - 1),
                    data_in + colorStart + (r + 1) * width + c,
                    data_in + colorStart + (r + 1) * width + (c + 1)
                };

                // Store the pixel values that will be muliplied with the convolution "matrix" on 128 bit packed registers

                // Convolution matrix
                //  |a|b|c|
                //  |d|e|f|
                //  |g|h|i|

                // Top-left
                unsigned char a16bitPixels[16] = { 0 };
                make_pixels_16_bits(offsets[0], a16bitPixels);

                __m128i packedA = _mm_load_si128(reinterpret_cast<__m128i*>(a16bitPixels));


                // Top-middle
                unsigned char b16bitPixels[16] = { 0 };
                make_pixels_16_bits(offsets[1], b16bitPixels);

                __m128i packedB = _mm_load_si128(reinterpret_cast<__m128i*>(b16bitPixels));


                // Top-right
                unsigned char c16bitPixels[16] = { 0 };
                make_pixels_16_bits(offsets[2], c16bitPixels);

                __m128i packedC = _mm_load_si128(reinterpret_cast<__m128i*>(c16bitPixels));


                // Mid-left
                unsigned char d16bitPixels[16] = { 0 };
                make_pixels_16_bits(offsets[3], d16bitPixels);

                __m128i packedD = _mm_load_si128(reinterpret_cast<__m128i*>(d16bitPixels));


                // Mid
                unsigned char e16bitPixels[16] = { 0 };
                make_pixels_16_bits(offsets[4], e16bitPixels);

                __m128i packedE = _mm_load_si128(reinterpret_cast<__m128i*>(e16bitPixels));


                // Mid-right
                unsigned char f16bitPixels[16] = { 0 };
                make_pixels_16_bits(offsets[5], f16bitPixels);

                __m128i packedF = _mm_load_si128(reinterpret_cast<__m128i*>(f16bitPixels));


                // Bottom-left
                unsigned char g16bitPixels[16] = { 0 };
                make_pixels_16_bits(offsets[6], g16bitPixels);

                __m128i packedG = _mm_load_si128(reinterpret_cast<__m128i*>(g16bitPixels));


                // Bottom-mid
                unsigned char h16bitPixels[16] = { 0 };
                make_pixels_16_bits(offsets[7], h16bitPixels);

                __m128i packedH = _mm_load_si128(reinterpret_cast<__m128i*>(h16bitPixels));


                // Bottom-right
                unsigned char i16bitPixels[16] = { 0 };
                make_pixels_16_bits(offsets[8], i16bitPixels);

                __m128i packedI = _mm_load_si128(reinterpret_cast<__m128i*>(i16bitPixels));



                // Perform convolution with the vertical and horizontal matrices for the top-left element of the pixels (A)
                __m128i vertMultA = _mm_sign_epi16(packedA, negationValue);    // Negate (effectively multiply by -1)
                __m128i horMultA = _mm_sign_epi16(packedA, negationValue);     // Negate (effectively multiply by -1)

                // Perform convolution for B only with the vertical matrix since the result with the horizontal one will be 0
                __m128i vertMultB = _mm_sll_epi16(packedB, leftShiftAmmount);                     // Multiply by 2 shifting 1 to the left
                vertMultB = _mm_sign_epi16(vertMultB, negationValue);          // Negate (effectively multiply by -1)

                // Perform vertical convolution for C
                __m128i vertMultC = _mm_sign_epi16(packedC, negationValue);    // Negate (effectively multiply by -1)
                // We will use packedC instead of horizontal convolution since we are supposed to do a multiplication by 1

                // Perform convolution for D only with the horizontal matrix since the result with the vertical one will be 0
                __m128i horMultD = _mm_sll_epi16(packedD, leftShiftAmmount);                     // Multiply by 2 shifting 1 to the left
                horMultD = _mm_sign_epi16(horMultD, negationValue);          // Negate (effectively multiply by -1)

                // Skip convolution for E since both matrices will give 0

                // Perform convolution for F only with the horizontal matrix since the result with the vertical one will be 0
                __m128i horMultF = _mm_sll_epi16(packedF, leftShiftAmmount);                     // Multiply by 2 shifting 1 to the left

                // Perform horizontal convolution for G
                __m128i horMultG = _mm_sign_epi16(packedG, negationValue);    // Negate (effectively multiply by -1)
                // We will use packedG instead of vertical convolution since we are supposed to do a multiplication by 1

                // Perform convolution for H only with the vertical matrix since the result with the horizontal one will be 0
                __m128i vertMultH = _mm_sll_epi16(packedH, leftShiftAmmount);                     // Multiply by 2 shifting 1 to the left

                // We will use packedI for both vertical and horizontal since both are supposed to do a multiplication by 1


                // Add all of the convoluted values together for the vertical matrix convolution and the horizontal matrix convolution
                __m128i vertSum = _mm_add_epi16(packedI, _mm_add_epi16(vertMultH, _mm_add_epi16(packedG, _mm_add_epi16(vertMultC, _mm_add_epi16(vertMultB, vertMultA)))));
                __m128i horSum = _mm_add_epi16(packedI, _mm_add_epi16(horMultG, _mm_add_epi16(horMultF, _mm_add_epi16(horMultD, _mm_add_epi16(packedC, horMultA)))));

                // Get the absolute value of the vertical sum and the horizontal sum
                __m128i absVertSum = _mm_sign_epi16(vertSum, vertSum);
                __m128i absHorSum = _mm_sign_epi16(horSum, horSum);

                // Divide each 16 bit value by 8 by shifting to the right each value 3 times
                __m128i finalVertSum = _mm_srai_epi16(absVertSum, 3);
                __m128i finalHorSum = _mm_srai_epi16(absHorSum, 3);

                // Get the final value of each of the pixels
                __m128i newPixel = _mm_add_epi16(finalVertSum, finalHorSum);

                unsigned char* outStart = data_out + colorStart + r * width + c;
                unsigned char result16Bytes[16] = { 0 };
                _mm_store_si128(reinterpret_cast<__m128i*>(result16Bytes), newPixel);
                make_pixels_8_bits(result16Bytes, outStart);
            }
        }
    }
}


void basic_sobel_edge_detection(unsigned char *data_out,
                                unsigned char *data_in,
                                unsigned height, unsigned width)
{
    /* Sobel matrices for convolution */
    int sobelv[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };
    int sobelh[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    unsigned int size, i, j, lay;

    size = height * width;

    for (lay = 0; lay < 3; lay++) {
        for (i = 1; i < height - 1; ++i) {
            for (j = 1; j < width - 1; j++) {
                int sumh, sumv;
                sumh = 0;
                sumv = 0;

                /* Convolution part */
                for (int k = -1; k < 2; k++)
                    for (int l = -1; l < 2; l++) {
                        sumh =
                            sumh + sobelh[k + 1][l + 1] *
                            (int) data_in[lay * size +
                                          (i + k) * width + (j + l)];
                        sumv =
                            sumv + sobelv[k + 1][l + 1] *
                            (int) data_in[lay * size +
                                          (i + k) * width + (j + l)];
                    }
				        
                data_out[lay * size + i * width + j] =
                    abs(sumh / 8) + abs(sumv / 8);
            }
        }
    }
}


int main(int argc, char **argv)
{

    /* Some variables */
    bmp_header header;
    unsigned char *data_in, *data_out;
    unsigned int size;
    int rep;

    test_func functions[2] =
        { basic_sobel_edge_detection, sse_sobel_edge_detection };
    if (argc != 4) {
	std::cout << "Usage " << argv[0];
	std::cout << " <InFile> <OutFile> <0:basic_sobel or 1 :sse_sobel";
	std::cout << std::endl << std::endl;
        exit(0);
    }

    bmp_read(argv[1], &header, &data_in);
    
    size = header.height * header.width;
    data_out = new unsigned char[3 * size];
	memcpy_s(data_out, 3 * size, data_in, 3*size);
    printf("Resolution: (%d,%d) -> Size: %d\n", header.height,
           header.width, size);

    prof_time_t start = 0, end = 0;
    start_measure(start);
    end_measure(end);

    int which_func = std::atoi(argv[3]);
    prof_time_t base_time = end - start;
    start_measure(start);
    for (rep = 0; rep < REP; rep++) {
        functions[which_func] (data_out, data_in, header.height,
                               header.width);
    }
    end_measure(end);

    prof_time_t cycles_taken = end - start - base_time;

    std::cout << "Cycles taken: " << cycles_taken / REP << std::endl;


    bmp_write(argv[2], &header, data_out);

    return (0);
}
