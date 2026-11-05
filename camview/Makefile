# Build on the headset: make OPENXR_INC=/home/user/xr/build-openxr/include
OPENXR_INC ?= /usr/include
OPENXR_SRC_INC ?= $(OPENXR_INC)/../../OpenXR-SDK-Source-release-1.1.62/include
CFLAGS ?= -O2 -Wall -Wextra
pico4-camview: camview.c
	$(CC) $(CFLAGS) -I$(OPENXR_INC) -I$(OPENXR_SRC_INC) -o $@ $< -lvulkan -l:libopenxr_loader.so.1 -lpthread -lm
clean:
	rm -f pico4-camview
