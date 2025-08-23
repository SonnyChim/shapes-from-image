CXX ?= g++
INCLUDES = src/lib/tigr.c
LDFLAGS = -lGLU -lGL -lX11
SRC = src/shapes_from_image.cpp

dev: sfi

sfi: $(SRC) $(INCLUDES)
	$(CXX) $< $(INCLUDES) $(LDFLAGS) -o $@

test: sfi
	./$< rcimg.png