#include <assert.h>
#include <string.h>

#include <iostream>
#include <random>
#include <string>

#include "lib/tigr.h"

struct Coordinate {
    size_t x;
    size_t y;
};

struct Rectangle {
    Coordinate point1;
    Coordinate point2;
    TPixel color;
};

struct TigrRectangle {
    size_t x;
    size_t y;
    size_t w;
    size_t h;
    TPixel color;
};

enum ShapeType {
    rectangle,
    circle,
    triangle,
    bitmap,
};

inline TPixel rgb(unsigned long hex) {
    TPixel pixel;
    pixel.r = hex >> 16 & 0xff;
    pixel.g = hex >> 8  & 0xff;
    pixel.b = hex       & 0xff;
    pixel.a = 0xff;
    return pixel;
}

inline TPixel rgba(unsigned long hex) {
    TPixel pixel;
    pixel.r = hex >> 24 & 0xff;
    pixel.g = hex >> 16 & 0xff;
    pixel.b = hex >> 8  & 0xff;
    pixel.a = hex       & 0xff;
    return pixel;
}

bool quitKeyDown(Tigr *window) {
    return tigrKeyDown(window, TK_ESCAPE) || tigrKeyDown(window, 'Q');
}

TigrRectangle rectToTigr(Rectangle r) {
    return {
        r.point1.x,
        r.point1.y,
        r.point2.x - r.point1.x + 1,
        r.point2.y - r.point1.y + 1,
        r.color
    };
}

void fillRect(Tigr *window, Rectangle r) {
    TigrRectangle tr = rectToTigr(r);
    tigrFill(window, tr.x, tr.y, tr.w, tr.h, tr.color);
}

struct Config {
    size_t shape_limit;
    ShapeType shape_type;
    double difference_goal;
    bool use_alpha;
    double randomness_factor;

    size_t generation_size;
    double top_percent;
};

class Generator {
public:
    Config config;
    Generator(Tigr *img) {
        sourceImage = img;
        width = img->w;
        height = img->h;
    }
    Generator(const char *filename) {
        sourceImage = tigrLoadImage(filename);
        if (!sourceImage) {
            throw errno;
        }
        width = sourceImage->w;
        height = sourceImage->h;
    }
    ~Generator() {
        tigrFree(sourceImage);
        if (window) {
            tigrFree(window);
        }
    }
    void showImage() {
        if (!window) {
            window = tigrWindow(width, height, "Image", TIGR_AUTO);
        }
        tigrBlit(window, sourceImage, 0, 0, 0, 0, width, height);
        tigrUpdate(window);
    }

    void randomRectangles(size_t n, Rectangle arr[]) {
        for (size_t i = 0; i < n; i++) {
            arr[i] = randRect();
        }
    }

    void test() {
        if (!window) {
            window = tigrWindow(width, height, "Image", TIGR_FIXED);
        }
        tigrBlit(window, sourceImage, 0, 0, 0, 0, width, height);
        while (!tigrClosed(window) && !quitKeyDown(window)) {
            int x, y, buttons;
            tigrMouse(window, &x, &y, &buttons);
            if (buttons) {
                
            }
            tigrUpdate(window);
        }
    }
private:
    Tigr *sourceImage;
    Tigr *window;
    size_t width;
    size_t height;
    std::mt19937 rng{std::random_device{}()};

    // Random integer in [min, max)
    int randRange(int min, int max) {
        assert(min < max && "invalid range");
        return std::uniform_int_distribution<std::mt19937::result_type>(min, max - 1)(rng);
    }
    int randRange(int max) {
        assert(0 < max && "invalid range");
        return std::uniform_int_distribution<std::mt19937::result_type>(0, max - 1)(rng);
    }
    size_t randRange(size_t min, size_t max) {
        assert(min < max && "invalid range");
        return std::uniform_int_distribution<std::mt19937::result_type>(min, max - 1)(rng);
    }
    size_t randRange(size_t max) {
        assert(0 < max && "invalid range");
        return std::uniform_int_distribution<std::mt19937::result_type>(0, max - 1)(rng);
    }

    // Random coordinate in the image
    Coordinate randCoord() {
        return {randRange(width), randRange(height)};
    }

    Rectangle randRect() {
        Coordinate p1 = randCoord();
        Coordinate p2 = randCoord();
        return {
            {std::min(p1.x, p2.x), std::min(p1.y, p2.y)},
            {std::max(p1.x,p2.x), std::max(p1.y, p2.y)},
            rgba(0x00000000)
        };
    }

    Rectangle randRectColored() {
        Coordinate p1 = randCoord();
        Coordinate p2 = randCoord();
        Rectangle r = {
            {std::min(p1.x, p2.x), std::min(p1.y, p2.y)},
            {std::max(p1.x,p2.x), std::max(p1.y, p2.y)},
            rgba(0x00000000)
        };
        r.color = getAverage(r);
        return r;
    }

    void checkBounds(Rectangle r) {
        assert(r.point1.x >= 0);
        assert(r.point1.y >= 0);
        assert(r.point2.x < width);
        assert(r.point2.y < height);
        assert(r.point1.x <= r.point2.x);
        assert(r.point1.y <= r.point2.y);
    }

    void checkBounds(Rectangle r, Tigr *bmp) {
        assert(r.point1.x >= 0);
        assert(r.point1.y >= 0);
        assert(r.point2.x < bmp->w);
        assert(r.point2.y < bmp->h);
        assert(r.point1.x <= r.point2.x);
        assert(r.point1.y <= r.point2.y);
    }

    TPixel getAverage(Rectangle r) {
        checkBounds(r);
        TigrRectangle tr = rectToTigr(r);
        size_t nPixels = tr.w * tr.h;
        double sumR = 0;
        double sumG = 0;
        double sumB = 0;
        for (size_t y = r.point1.y; y < r.point2.y; y++) {
            for (size_t x = r.point1.x; x < r.point2.x; x++) {
                TPixel pix = sourceImage->pix[x + y * width];
                sumR += pix.r;
                sumG += pix.g;
                sumB += pix.b;
            }
        }
        return tigrRGB(sumR / nPixels, sumG / nPixels, sumB / nPixels);
    }

    unsigned long long getDifference(Tigr *bmp, TPixel color, Rectangle r) {
        checkBounds(r, bmp);
        unsigned long long sum = 0;
        for (size_t y = r.point1.y; y < r.point2.y; y++) {
            for (size_t x = r.point1.x; x < r.point2.x; x++) {
                TPixel pix = bmp->pix[x + y * width];
                sum += abs(pix.r - color.r);
                sum += abs(pix.g - color.g);
                sum += abs(pix.b - color.b);
            }
        }
        return sum;
    }

    unsigned long long getDifference(Tigr *bmp, Tigr *bmp2, Rectangle r) {
        assert(bmp->w == bmp2->w && bmp->h == bmp2->h);
        checkBounds(r, bmp);
        unsigned long long sum = 0;
        for (size_t y = r.point1.y; y < r.point2.y; y++) {
            for (size_t x = r.point1.x; x < r.point2.x; x++) {
                TPixel pix = bmp->pix[x + y * bmp->w];
                TPixel pix2 = bmp2->pix[x + y * bmp2->w];
                sum += abs(pix.r - pix2.r);
                sum += abs(pix.g - pix2.g);
                sum += abs(pix.b - pix2.b);
            }
        }
        return sum;
    }
};

int main(const int argc, const char *const argv[]) {
    if (argc < 2) {
        return 1;
    }
    const char *filename = argv[1];
    try {
        Generator gen(filename);
        gen.test();
        // gen.showImage();
    } catch (const int e) {
        std::cerr << strerror(e) << std::endl;
        return EXIT_FAILURE;
    }

    // Tigr *window = tigrWindow(400, 400, "hi", TIGR_FIXED);
    // Tigr *image = tigrLoadImage(filename);
    // Tigr *window = tigrWindow(image->w, image->h, "image", TIGR_FIXED);
    // tigrBlit(window, image, 0, 0, 0, 0, image->w, image->h);
    // std::random_device rdev;
    // std::mt19937 rng(rdev());
    // std::uniform_int_distribution<std::mt19937::result_type> distx(0, window->w - 1);
    // std::uniform_int_distribution<std::mt19937::result_type> disty(0, window->h - 1);
    // std::uniform_int_distribution<std::mt19937::result_type> distt(10, 9);
    // // tigrClear(window, rgb(0xffffff));
    // int x, y, b;
    // tigrPrint(window, tfont, 10, 10, rgb(0x000000), std::to_string(distt(rng)).c_str());
    // while (!tigrClosed(window) && !quitKeyDown(window)) {
    //     // if (float t = tigrTime()) {
    //         //     tigrFill(window, 10, 10 + (i * 20) % 400, 390, 20, rgb(0xffffff));
    //         //     tigrPrint(window, tfont, 10, 10 + (i * 20) % 400, rgb(0x000000), std::to_string(1/t).c_str());
    //         // }
    //     // tigrClear(window, rgb(0xffffff));
    //     tigrMouse(window, &x, &y, &b);
    //     if (b && x >= 0 && x < window->w && y >= 0 && y < window->h) {
    //         window->pix[x + y * window->w] = rgb(0x000000);
    //     }
    //     tigrUpdate(window);
    // }
    // tigrFree(window);
    return 0;
}