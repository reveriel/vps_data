#include <stdio.h>
#include <stdlib.h>

#define BOOL char

FILE *fp;

/**
 * f(z) = z^2 + c
 * z = 0;
 * if  || f^(iter_times)(z) || < r , rerurn  1
 * else return 0
 * @param  cx         x of c
 * @param  cy         y of c
 * @param  iter_times iterater times
 * @param  r          radius
 * @return            1 or 0
 */
BOOL test(double cx, double cy, int iter_times, double r)
{
    double x, y;
    x = y = 0;
    int i;
    for (i = 0; i < iter_times && x * x + y * y < r * r; i++) {
        x = x * x - y * y + cy;
        y = 2 * x * y + cx;
    }
    if (i == iter_times && x * x + y * y < r * r) {
        return 1;
    } else {
        return 0;
    }
}

/**
 * mandelbrot set
 * f(z) = z^2 + c;
 * z = 0; f^(n)(z) is is still in radius r.
 * @param mat        the output matrix, caller allocated
 * @param cols       cols of mat
 * @param rows       rows of mat
 * @param x0         left bottom of the complex plane
 * @param y0         left bottom of the complex plane
 * @param dx         x0 + dx
 * @param dy         y0 + dy
 * @param iter_times
 * @param radius
 */
void mandlebrot(
    BOOL *mat,
    int cols, int rows,
    double x0, double y0,
    double dx, double dy,
    int iter_times, double radius)
{
    // x0  . . . x0 + dx
    double delta_x =  dx / (cols - 1);
    double delta_y = dy / (rows - 1);

    for (int i = 0; i < rows ; i++) {
        for (int j = 0; j <  cols; j++) {
            double x = j * delta_x + x0;
            double y = i * delta_y + y0;
            *(mat + i * cols + j) = test(x, y, iter_times, radius);
        }
    }
}

void pixel_write(int i, int j, BOOL is_black)
{
    static unsigned char color[3];
    if (is_black) {
        color[0] = 0;
        color[1] = 0;
        color[2] = 0;
    } else {
        color[0] = 255;
        color[1] = 255;
        color[2] = 255;
    }
    fwrite(color, 1, 3, fp);
}

void draw(BOOL *mat, int cols, int rows)
{
    fp = fopen("result.ppm", "wb");
    fprintf(fp, "P6\n%d %d\n255\n", rows, cols);
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++)  {
            pixel_write(i, j, *(mat + i * cols + j));
        }
    }
    fclose(fp);
}


int main(int argc, char ** argv)
{
    // usage : mand <clos> <rows> <x0> <y0> <dx> <dy>
    if (argc != 7) {
        printf("usage: mand <cols> <rows> <x0> <y0> <dx> <dy>\n");
        return -1;
    }
    int cols, rows;
    double x0, y0, dx, dy;
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    x0 = atof(argv[3]);
    y0 = atof(argv[4]);
    dx = atof(argv[5]);
    dy = atof(argv[6]);

    BOOL *mat = (BOOL *)malloc(sizeof(BOOL) * cols * rows);
    if (mat == NULL) {
        printf("malloc failed");
        exit(-1);
    }
    mandlebrot(mat, cols, rows, x0 ,y0, dx, dy, 150, 1.414);
    draw(mat, cols, rows);

    return 0;
}
