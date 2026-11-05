// SPDX-License-Identifier: BSL-1.0
/*
 * pico4-camview: the PICO 4 tracking cameras in VR.
 *
 *   pico4-camview                 each camera on its own floating screen
 *   pico4-camview --passthrough   black-and-white passthrough
 *
 * Capture: each OV7251 stream (RAW10 Y10P 640x481 from msm_vfeN_video0, line
 * 481 is embedded data) is reduced to 8-bit grey by a thread per camera. The
 * camera pipeline and the Nordic MCU are set up by pico4-camview.sh.
 *
 * Screens: every camera is an OpenXR quad layer; Monado composites them.
 *
 * Passthrough: the cameras and the eyes are rigid on the headset, so for a
 * given scene depth the eye pixel -> camera pixel mapping is fixed. A coarse
 * mesh per eye stores that mapping for 8 candidate depths from the stock
 * calibration (persist pvr/mav0/camN/sensor.yaml: Kalibr omni + radtan, T_BS =
 * camera -> body). A worker thread picks the depth per mesh node where the
 * overlapping cameras agree, matches camera gains, and blends the cameras
 * into a stereo projection layer.
 *
 * Environment:
 *   CAMVIEW_GAIN=2.0            digital gain
 *   CAMVIEW_ROLL=r0,r1,r2,r3    screens: per-camera roll in degrees
 *   CAMVIEW_DEPTH=2.0           passthrough: one fixed scene distance in metres
 *                               (default: estimated per region, 0.35..6 m)
 *   CAMVIEW_MAP=0,3,2,1         passthrough: mav0 camN of capture 0..3
 *   CAMVIEW_HEAD=x,y,z          passthrough: head centre in the body frame
 *   CAMVIEW_EYE=960             passthrough: per-eye resolution
 *   CAMVIEW_CALIB=/var/lib/pico4/mav0
 *   CAMVIEW_DUMP=file.pgm       passthrough: save one stereo frame
 *   CAMVIEW_VIGNETTE=1.0        passthrough: vignetting compensation exponent
 *   CAMVIEW_BLEND_POWER=6       passthrough: cross-fade sharpness at seams
 *   CAMVIEW_NO_GAIN=1           passthrough: no per-camera gain matching
 */
#define _GNU_SOURCE // pthread_setaffinity_np
#define XR_USE_GRAPHICS_API_VULKAN
#define XR_USE_TIMESPEC
#include <time.h> // struct timespec for XR_USE_TIMESPEC
#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <errno.h>
#include <sched.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define CAM_COUNT 4
#define CAM_W 640
#define CAM_H 480 // 481 delivered; the last line is embedded data
#define CAM_LINES 481
#define CAM_STRIDE (CAM_W * 10 / 8)
#define CAM_PIXELS (CAM_W * CAM_H)
#define V4L2_BUFS 4
#define LUT_NONE UINT32_MAX

#define XR_CHECK(x)                                                                                                    \
	do {                                                                                                           \
		XrResult r_ = (x);                                                                                     \
		if (XR_FAILED(r_)) {                                                                                   \
			fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #x, r_);                         \
			exit(1);                                                                                       \
		}                                                                                                      \
	} while (0)
#define VK_CHECK(x)                                                                                                    \
	do {                                                                                                           \
		VkResult r_ = (x);                                                                                     \
		if (r_ != VK_SUCCESS) {                                                                                \
			fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #x, r_);                         \
			exit(1);                                                                                       \
		}                                                                                                      \
	} while (0)

//! Kalibr omni-radtan camera with its pose in the body frame.
struct calib
{
	double R[9], t[3]; //!< T_BS: x_body = R x_cam + t
	double xi, fu, fv, pu, pv;
	double k1, k2, p1, p2;
};

//! A GPU upload target: staging buffer + OpenXR swapchain.
struct target
{
	VkBuffer staging;
	VkDeviceMemory staging_mem;
	void *staging_ptr;
	XrSwapchain swapchain;
	VkImage images[8];
	uint32_t image_count;
	uint32_t width, height;
	bool has_image;
};

struct shm_hdr
{
	uint32_t magic, version, width, height, bytes_per_line;
	uint32_t seq;
	int64_t timestamp_ns;
	uint32_t is_head, frame_counter;
};

struct camera
{
	bool use_shm;
	int index;
	char dev[32];
	int fd;
	void *map[V4L2_BUFS];
	pthread_t thread;

	// Latest frame in 8-bit grey, written by the capture thread.
	pthread_mutex_t lock;
	uint8_t *gray;
	uint64_t seq;
	uint64_t frames;
	int64_t ts_ns; //!< V4L2 timestamp of the frame in @ref gray

	/*
	 * The station MCU interleaves two exposures (head frames for tracking,
	 * short hand/IR-LED frames) on every sensor, frame by frame. Keep the
	 * brighter half: running mean brightness per sequence parity.
	 */
	double parity_mean[2];
	uint64_t skipped;

	uint64_t uploaded_seq;
	struct target target; // screens mode
	float roll;
	struct calib cal; // passthrough mode
	bool has_cal;
};

static struct camera cams[CAM_COUNT];
static volatile sig_atomic_t quit_requested;
static uint8_t gain_lut[256];

static void
on_signal(int sig)
{
	(void)sig;
	quit_requested = 1;
}

static float
env_float(const char *name, float def)
{
	const char *v = getenv(name);
	return v ? strtof(v, NULL) : def;
}

/*
 *
 * V4L2 capture.
 *
 */

static int
find_video_node(int vfe, char *out, size_t len)
{
	char want[32], path[64], name[64];
	snprintf(want, sizeof(want), "msm_vfe%d_video0", vfe);
	for (int i = 0; i < 64; i++) {
		snprintf(path, sizeof(path), "/sys/class/video4linux/video%d/name", i);
		FILE *f = fopen(path, "r");
		if (!f) {
			continue;
		}
		bool match = fgets(name, sizeof(name), f) && strncmp(name, want, strlen(want)) == 0;
		fclose(f);
		if (match) {
			snprintf(out, len, "/dev/video%d", i);
			return 0;
		}
	}
	return -1;
}

static bool
shm_wanted(int index)
{
	const char *e = getenv("CAMVIEW_SHM");
	while (e && *e) {
		char *end;
		long v = strtol(e, &end, 10);
		if (end == e) {
			break;
		}
		if (v == index) {
			return true;
		}
		e = *end == ',' ? end + 1 : end;
	}
	return false;
}

static int
camera_open(struct camera *c)
{
	if (c->use_shm) {
		printf("camera %d: shared frames /dev/shm/pico4-cam-vfe%d\n", c->index, c->index);
		c->fd = -1;
		return 0;
	}
	if (find_video_node(c->index, c->dev, sizeof(c->dev)) < 0) {
		fprintf(stderr, "camera %d: no msm_vfe%d_video0\n", c->index, c->index);
		return -1;
	}
	c->fd = open(c->dev, O_RDWR);
	if (c->fd < 0) {
		perror(c->dev);
		return -1;
	}

	struct v4l2_format fmt = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE};
	fmt.fmt.pix_mp.width = CAM_W;
	fmt.fmt.pix_mp.height = CAM_LINES;
	fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_Y10P;
	fmt.fmt.pix_mp.num_planes = 1;
	if (ioctl(c->fd, VIDIOC_S_FMT, &fmt) < 0) {
		perror("VIDIOC_S_FMT");
		return -1;
	}

	struct v4l2_requestbuffers req = {
	    .count = V4L2_BUFS, .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, .memory = V4L2_MEMORY_MMAP};
	if (ioctl(c->fd, VIDIOC_REQBUFS, &req) < 0 || req.count < V4L2_BUFS) {
		perror("VIDIOC_REQBUFS");
		return -1;
	}
	for (int i = 0; i < V4L2_BUFS; i++) {
		struct v4l2_plane plane = {0};
		struct v4l2_buffer buf = {.index = i,
		                          .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
		                          .memory = V4L2_MEMORY_MMAP,
		                          .m.planes = &plane,
		                          .length = 1};
		if (ioctl(c->fd, VIDIOC_QUERYBUF, &buf) < 0) {
			perror("VIDIOC_QUERYBUF");
			return -1;
		}
		c->map[i] = mmap(NULL, plane.length, PROT_READ, MAP_SHARED, c->fd, plane.m.mem_offset);
		if (c->map[i] == MAP_FAILED) {
			perror("mmap");
			return -1;
		}
		if (ioctl(c->fd, VIDIOC_QBUF, &buf) < 0) {
			perror("VIDIOC_QBUF");
			return -1;
		}
	}
	enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
	if (ioctl(c->fd, VIDIOC_STREAMON, &type) < 0) {
		perror("VIDIOC_STREAMON");
		return -1;
	}
	printf("camera %d: %s streaming\n", c->index, c->dev);
	return 0;
}

//! MIPI RAW10 rows: four 8-bit MSBs, then one byte of the 2-bit LSBs.
static void
convert(const uint8_t *src, uint8_t *dst)
{
	for (int y = 0; y < CAM_H; y++) {
		const uint8_t *s = src + y * CAM_STRIDE;
		for (int x = 0; x < CAM_W; x += 4, s += 5) {
			*dst++ = gain_lut[s[0]];
			*dst++ = gain_lut[s[1]];
			*dst++ = gain_lut[s[2]];
			*dst++ = gain_lut[s[3]];
		}
	}
}

//! Frames published by pico4-slam (seqlock copy, head frames only).
static void *
shm_thread(void *arg)
{
	struct camera *c = arg;
	uint8_t *tmp = malloc(CAM_PIXELS);
	uint8_t *raw = malloc((size_t)CAM_STRIDE * CAM_LINES);
	char path[64];
	snprintf(path, sizeof(path), "/dev/shm/pico4-cam-vfe%d", c->index);
	struct shm_hdr *h = NULL;
	uint32_t last = 0;
	while (!quit_requested) {
		if (!h) {
			int f = open(path, O_RDONLY);
			if (f >= 0) {
				void *m = mmap(NULL, sizeof(*h) + (size_t)CAM_STRIDE * CAM_LINES, PROT_READ, MAP_SHARED, f, 0);
				close(f);
				if (m != MAP_FAILED) {
					h = m;
				}
			}
			if (!h || h->magic != 0x50344346 || h->bytes_per_line != CAM_STRIDE) {
				h = NULL;
				usleep(200000);
				continue;
			}
		}
		uint32_t s1 = __atomic_load_n(&h->seq, __ATOMIC_ACQUIRE);
		if ((s1 & 1) || h->frame_counter == last) {
			usleep(2000);
			continue;
		}
		memcpy(raw, h + 1, (size_t)CAM_STRIDE * CAM_H);
		int64_t ts = h->timestamp_ns;
		uint32_t head = h->is_head, fc = h->frame_counter;
		__atomic_thread_fence(__ATOMIC_SEQ_CST);
		if (__atomic_load_n(&h->seq, __ATOMIC_ACQUIRE) != s1) {
			continue;
		}
		last = fc;
		if (!head) {
			c->skipped++;
			continue;
		}
		convert(raw, tmp);
		pthread_mutex_lock(&c->lock);
		memcpy(c->gray, tmp, CAM_PIXELS);
		c->ts_ns = ts;
		c->seq++;
		c->frames++;
		pthread_mutex_unlock(&c->lock);
	}
	free(tmp);
	free(raw);
	return NULL;
}

static void *
camera_thread(void *arg)
{
	struct camera *c = arg;
	uint8_t *tmp = malloc(CAM_PIXELS);

	while (!quit_requested) {
		struct v4l2_plane plane = {0};
		struct v4l2_buffer buf = {.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
		                          .memory = V4L2_MEMORY_MMAP,
		                          .m.planes = &plane,
		                          .length = 1};
		if (ioctl(c->fd, VIDIOC_DQBUF, &buf) < 0) {
			if (errno == EINTR || errno == EAGAIN) {
				continue;
			}
			break;
		}
		if (plane.bytesused >= (size_t)CAM_STRIDE * CAM_H) {
			const uint8_t *raw = c->map[buf.index];
			uint32_t sum = 0, n = 0;
			for (uint32_t i = 0; i < CAM_STRIDE * CAM_H; i += 97, n++) {
				sum += raw[i];
			}
			int parity = buf.sequence & 1;
			double mean = (double)sum / n;
			double *pm = c->parity_mean;
			pm[parity] = pm[parity] > 0.0 ? pm[parity] * 0.9 + mean * 0.1 : mean;
			bool primed = pm[0] > 0.0 && pm[1] > 0.0;
			if (primed && pm[parity] < pm[!parity]) {
				c->skipped++;
			} else {
				convert(raw, tmp);
				pthread_mutex_lock(&c->lock);
				memcpy(c->gray, tmp, CAM_PIXELS);
				c->ts_ns = buf.timestamp.tv_sec * 1000000000LL + buf.timestamp.tv_usec * 1000LL;
				c->seq++;
				c->frames++;
				pthread_mutex_unlock(&c->lock);
			}
		}
		ioctl(c->fd, VIDIOC_QBUF, &buf);
	}
	free(tmp);
	return NULL;
}

/*
 *
 * Calibration (Kalibr sensor.yaml) and the passthrough lookup table.
 *
 */

static bool
yaml_numbers(const char *text, const char *key, double *out, int count)
{
	const char *p = strstr(text, key);
	if (!p) {
		return false;
	}
	p = strchr(p, '[');
	if (!p) {
		return false;
	}
	p++;
	for (int i = 0; i < count; i++) {
		char *end;
		out[i] = strtod(p, &end);
		if (end == p) {
			return false;
		}
		p = end;
		while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
			p++;
		}
	}
	return true;
}

static bool
calib_load(int mav_index, struct calib *cal)
{
	const char *dir = getenv("CAMVIEW_CALIB") ? getenv("CAMVIEW_CALIB") : "/var/lib/pico4/mav0";
	char path[256];
	snprintf(path, sizeof(path), "%s/cam%d/sensor.yaml", dir, mav_index);
	FILE *f = fopen(path, "r");
	if (!f) {
		perror(path);
		return false;
	}
	char text[4096];
	size_t n = fread(text, 1, sizeof(text) - 1, f);
	fclose(f);
	text[n] = 0;

	double T[16], in[5], d[4];
	const char *tbs = strstr(text, "T_BS");
	if (!tbs || !yaml_numbers(tbs, "data:", T, 16) || !yaml_numbers(text, "intrinsics:", in, 5) ||
	    !yaml_numbers(text, "distortion_coefficients:", d, 4) || !strstr(text, "camera_model: omni")) {
		fprintf(stderr, "%s: not an omni-radtan Kalibr camera\n", path);
		return false;
	}
	for (int r = 0; r < 3; r++) {
		for (int c = 0; c < 3; c++) {
			cal->R[r * 3 + c] = T[r * 4 + c];
		}
		cal->t[r] = T[r * 4 + 3];
	}
	cal->xi = in[0], cal->fu = in[1], cal->fv = in[2], cal->pu = in[3], cal->pv = in[4];
	cal->k1 = d[0], cal->k2 = d[1], cal->p1 = d[2], cal->p2 = d[3];
	printf("calibration %s: xi %.3f f %.1f c %.1f,%.1f\n", path, cal->xi, cal->fu, cal->pu, cal->pv);
	return true;
}

//! Project a body-frame point; returns false outside the useful image.
static bool
calib_project(const struct calib *cal, const double pb[3], double *u, double *v, double *cos_theta)
{
	double q[3] = {pb[0] - cal->t[0], pb[1] - cal->t[1], pb[2] - cal->t[2]};
	double pc[3];
	for (int i = 0; i < 3; i++) {
		pc[i] = cal->R[0 * 3 + i] * q[0] + cal->R[1 * 3 + i] * q[1] + cal->R[2 * 3 + i] * q[2];
	}
	double norm = sqrt(pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]);
	double xs = pc[0] / norm, ys = pc[1] / norm, zs = pc[2] / norm;
	// Stay within ~80 degrees of the optical axis: past that the fisheye
	// image circle and the housing take over.
	if (zs < 0.17) {
		return false;
	}
	double den = zs + cal->xi;
	double mx = xs / den, my = ys / den;
	double r2 = mx * mx + my * my;
	double rad = 1.0 + cal->k1 * r2 + cal->k2 * r2 * r2;
	double dx = 2.0 * cal->p1 * mx * my + cal->p2 * (r2 + 2.0 * mx * mx);
	double dy = cal->p1 * (r2 + 2.0 * my * my) + 2.0 * cal->p2 * mx * my;
	*u = cal->fu * (mx * rad + dx) + cal->pu;
	*v = cal->fv * (my * rad + dy) + cal->pv;
	*cos_theta = zs;
	return *u >= 0.0 && *u <= CAM_W - 2 && *v >= 0.0 && *v <= CAM_H - 2;
}

/*
 * Passthrough warp on a coarse mesh: a node every GRID_STEP eye pixels holds,
 * per candidate depth and camera, where that camera sees the node's ray at
 * that depth. Per camera set:
 *   1. depth: per node, the candidate where its two overlapping cameras agree
 *      best (SAD over a small patch), biased to the previous choice and kept
 *      in textureless areas, then a 3x3 median;
 *   2. warp: per GRID_STEP block, camera coordinates and blend weights are
 *      interpolated from the four corner nodes and the cameras blended.
 */
#define GRID_STEP 16
#define DEPTHS 8

struct node_cam
{
	float u, v; //!< camera pixel
	float w;    //!< blend weight, 0 when the camera does not see the point
	float vig;  //!< vignetting gain
};

struct eye
{
	XrPosef pose; //!< in VIEW space (relative to the head)
	XrFovf fov;
	uint32_t gn;           //!< nodes per axis
	struct node_cam *pre;  //!< [node][DEPTHS][CAM_COUNT]
	float *depth;          //!< [node] smoothed depth, fractional candidate index
	float *depth_tmp;      //!< [node] scratch for filtering
	struct node_cam *cur;  //!< [node][CAM_COUNT] at the chosen depth
};

static uint32_t eye_size;
static int depth_count = DEPTHS;
static double depth_m[DEPTHS];

static void
quat_rotate(XrQuaternionf q, const double v[3], double out[3])
{
	// v' = v + 2 w (u x v) + 2 u x (u x v)
	double u[3] = {q.x, q.y, q.z}, w = q.w;
	double c1[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
	double c2[3] = {u[1] * c1[2] - u[2] * c1[1], u[2] * c1[0] - u[0] * c1[2], u[0] * c1[1] - u[1] * c1[0]};
	for (int i = 0; i < 3; i++) {
		out[i] = v[i] + 2.0 * (w * c1[i] + c2[i]);
	}
}

/*
 * Where every camera sees the eye ray through eye pixel (px, py) at @p depth.
 * Blend weight (1 - theta / theta_max)^power by the angle from the camera's
 * optical axis (a high power keeps the cross-fade narrow); vignetting gain
 * 1 / cos(theta)^k, capped.
 */
static void
node_project(const struct eye *e, double px, double py, double depth, const double head_in_body[3],
             struct node_cam out[CAM_COUNT])
{
	static double theta_max, power, vignette;
	if (theta_max == 0.0) {
		theta_max = acos(0.17);
		power = env_float("CAMVIEW_BLEND_POWER", 6.0f);
		vignette = env_float("CAMVIEW_VIGNETTE", 1.0f);
	}
	double n = eye_size;
	double tl = tan(e->fov.angleLeft), tr = tan(e->fov.angleRight);
	double tu = tan(e->fov.angleUp), td = tan(e->fov.angleDown);
	double d[3] = {tl + px / n * (tr - tl), tu - py / n * (tu - td), -1.0};
	double len = sqrt(d[0] * d[0] + d[1] * d[1] + 1.0);
	double de[3] = {d[0] / len * depth, d[1] / len * depth, d[2] / len * depth};
	// Eye frame -> head (OpenXR VIEW space).
	double ph[3];
	quat_rotate(e->pose.orientation, de, ph);
	ph[0] += e->pose.position.x, ph[1] += e->pose.position.y, ph[2] += e->pose.position.z;
	// Head (x right, y up, z back) -> calibration body frame (x up, y left, z back).
	double pb[3] = {ph[1] + head_in_body[0], -ph[0] + head_in_body[1], ph[2] + head_in_body[2]};

	for (int c = 0; c < CAM_COUNT; c++) {
		double u, v, ct;
		out[c] = (struct node_cam){0};
		if (!cams[c].has_cal || cams[c].fd < 0 || !calib_project(&cams[c].cal, pb, &u, &v, &ct)) {
			continue;
		}
		double f = 1.0 - acos(ct) / theta_max;
		double vig = pow(ct, -vignette);
		out[c] = (struct node_cam){(float)u, (float)v, (float)pow(f, power), (float)(vig > 2.0 ? 2.0 : vig)};
	}
}

static void
eye_build_grid(struct eye *e, const double head_in_body[3])
{
	uint32_t gn = eye_size / GRID_STEP + 1, nodes = gn * gn;
	e->gn = gn;
	e->pre = calloc((size_t)nodes * DEPTHS * CAM_COUNT, sizeof(struct node_cam));
	e->cur = calloc((size_t)nodes * CAM_COUNT, sizeof(struct node_cam));
	e->depth = calloc(nodes, sizeof(float));
	e->depth_tmp = calloc(nodes, sizeof(float));

	// Start every node at the candidate nearest 2 m.
	int k0 = 0;
	for (int k = 0; k < depth_count; k++) {
		if (fabs(depth_m[k] - 2.0) < fabs(depth_m[k0] - 2.0)) {
			k0 = k;
		}
	}
	uint32_t overlap = 0;
	for (uint32_t gy = 0; gy < gn; gy++) {
		for (uint32_t gx = 0; gx < gn; gx++) {
			uint32_t i = gy * gn + gx;
			for (int k = 0; k < depth_count; k++) {
				node_project(e, gx * GRID_STEP, gy * GRID_STEP, depth_m[k], head_in_body,
				             &e->pre[((size_t)i * DEPTHS + k) * CAM_COUNT]);
			}
			e->depth[i] = (float)k0;
			memcpy(&e->cur[i * CAM_COUNT], &e->pre[((size_t)i * DEPTHS + k0) * CAM_COUNT],
			       sizeof(struct node_cam) * CAM_COUNT);
			int seen = 0;
			for (int c = 0; c < CAM_COUNT; c++) {
				seen += e->cur[i * CAM_COUNT + c].w > 0.0f;
			}
			overlap += seen >= 2;
		}
	}
	printf("eye grid %ux%u nodes, %d depths %.2f..%.2f m, %.0f%% of nodes seen by 2+ cameras\n", gn, gn,
	       depth_count, depth_m[0], depth_m[depth_count - 1], 100.0 * overlap / nodes);
}

static inline float
sample(const uint8_t *f, float u, float v)
{
	u = u < 0.0f ? 0.0f : u > CAM_W - 2.001f ? CAM_W - 2.001f : u;
	v = v < 0.0f ? 0.0f : v > CAM_H - 2.001f ? CAM_H - 2.001f : v;
	int x = (int)u, y = (int)v;
	float fx = u - x, fy = v - y;
	const uint8_t *p = f + y * CAM_W + x;
	float a = p[0] + fx * (p[1] - p[0]);
	float b = p[CAM_W] + fx * (p[CAM_W + 1] - p[CAM_W]);
	return a + fy * (b - a);
}

//! The two strongest cameras of a node (b = -1 if there is no overlap).
static void
node_pair(const struct node_cam *nc, int *a, int *b)
{
	*a = -1, *b = -1;
	for (int c = 0; c < CAM_COUNT; c++) {
		if (nc[c].w <= 0.0f) {
			continue;
		}
		if (*a < 0 || nc[c].w > nc[*a].w) {
			*b = *a;
			*a = c;
		} else if (*b < 0 || nc[c].w > nc[*b].w) {
			*b = c;
		}
	}
	if (*b >= 0 && nc[*b].w < 0.05f * nc[*a].w) {
		*b = -1;
	}
}

//! Node camera data at fractional depth index @p d (linear between candidates).
static void
node_at_depth(const struct eye *e, uint32_t i, float d, struct node_cam out[CAM_COUNT])
{
	int k0 = (int)d, k1 = k0 + 1 < depth_count ? k0 + 1 : k0;
	float t = d - k0;
	const struct node_cam *a = &e->pre[((size_t)i * DEPTHS + k0) * CAM_COUNT];
	const struct node_cam *b = &e->pre[((size_t)i * DEPTHS + k1) * CAM_COUNT];
	for (int c = 0; c < CAM_COUNT; c++) {
		if (a[c].w > 0.0f && b[c].w > 0.0f) {
			out[c].u = a[c].u + (b[c].u - a[c].u) * t;
			out[c].v = a[c].v + (b[c].v - a[c].v) * t;
			out[c].w = a[c].w + (b[c].w - a[c].w) * t;
			out[c].vig = a[c].vig + (b[c].vig - a[c].vig) * t;
		} else {
			out[c] = t < 0.5f ? a[c] : b[c];
		}
	}
}

/*
 * Per node: the depth where its two overlapping cameras agree best (SAD over
 * a 3x3 patch, 2 px apart), refined with a parabola through the neighbouring
 * candidates, biased to the previous estimate, ignored where the patch has
 * no texture. Then smoothed in time and space so the warp stays continuous
 * (a slanted floor spans many depths).
 */
static void
eye_estimate_depth(struct eye *e, uint8_t *const frames[CAM_COUNT])
{
	const float lambda = 6.0f, texture_min = 60.0f, temporal = 0.35f;
	uint32_t gn = e->gn, nodes = gn * gn;
	int kmid = depth_count / 2;

	for (uint32_t i = 0; i < nodes; i++) {
		float prev = e->depth[i];
		e->depth_tmp[i] = prev;
		int a, b;
		if (depth_count == 1) {
			continue;
		}
		node_pair(&e->pre[((size_t)i * DEPTHS + kmid) * CAM_COUNT], &a, &b);
		if (b < 0) {
			continue; // one camera only: no depth information
		}
		float sad[DEPTHS], smin = 1e30f, smax = 0.0f, best = 1e30f;
		int kbest = -1;
		for (int k = 0; k < depth_count; k++) {
			const struct node_cam *nc = &e->pre[((size_t)i * DEPTHS + k) * CAM_COUNT];
			sad[k] = 1e30f;
			if (nc[a].w <= 0.0f || nc[b].w <= 0.0f) {
				continue;
			}
			float acc = 0.0f;
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					float va = sample(frames[a], nc[a].u + dx * 2, nc[a].v + dy * 2) * nc[a].vig;
					float vb = sample(frames[b], nc[b].u + dx * 2, nc[b].v + dy * 2) * nc[b].vig;
					acc += fabsf(va - vb);
				}
			}
			sad[k] = acc;
			smin = acc < smin ? acc : smin;
			smax = acc > smax ? acc : smax;
			float cost = acc + lambda * fabsf(k - prev);
			if (cost < best) {
				best = cost;
				kbest = k;
			}
		}
		if (kbest < 0 || smax - smin < texture_min) {
			continue;
		}
		float d = kbest;
		if (kbest > 0 && kbest + 1 < depth_count && sad[kbest - 1] < 1e29f && sad[kbest + 1] < 1e29f) {
			float l = sad[kbest - 1], c = sad[kbest], r = sad[kbest + 1];
			float den = l - 2.0f * c + r;
			if (den > 1e-3f) {
				float off = 0.5f * (l - r) / den;
				d += off < -0.5f ? -0.5f : off > 0.5f ? 0.5f : off;
			}
		}
		e->depth_tmp[i] = prev + (d - prev) * temporal;
	}

	// 3x3 median against outliers, then two 3x3 box blurs.
	for (uint32_t gy = 0; gy < gn; gy++) {
		for (uint32_t gx = 0; gx < gn; gx++) {
			float v[9];
			int n = 0;
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					int x = (int)gx + dx, y = (int)gy + dy;
					if (x >= 0 && y >= 0 && x < (int)gn && y < (int)gn) {
						v[n++] = e->depth_tmp[y * gn + x];
					}
				}
			}
			for (int p = 1; p < n; p++) {
				for (int q = p; q > 0 && v[q - 1] > v[q]; q--) {
					float t = v[q];
					v[q] = v[q - 1];
					v[q - 1] = t;
				}
			}
			e->depth[gy * gn + gx] = v[n / 2];
		}
	}
	for (int pass = 0; pass < 2; pass++) {
		for (uint32_t gy = 0; gy < gn; gy++) {
			for (uint32_t gx = 0; gx < gn; gx++) {
				float sum = 0.0f;
				int n = 0;
				for (int dy = -1; dy <= 1; dy++) {
					for (int dx = -1; dx <= 1; dx++) {
						int x = (int)gx + dx, y = (int)gy + dy;
						if (x >= 0 && y >= 0 && x < (int)gn && y < (int)gn) {
							sum += e->depth[y * gn + x];
							n++;
						}
					}
				}
				e->depth_tmp[gy * gn + gx] = sum / n;
			}
		}
		memcpy(e->depth, e->depth_tmp, sizeof(float) * nodes);
	}
	for (uint32_t i = 0; i < nodes; i++) {
		node_at_depth(e, i, e->depth[i], &e->cur[i * CAM_COUNT]);
	}
}

//! Run the calling thread on a big core (SM8250: CPUs 4-6 A77, 7 prime; 0-3 are A55).
static void
pin_big_core(int index)
{
	cpu_set_t set;
	CPU_ZERO(&set);
	CPU_SET(4 + index % 4, &set);
	pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

struct render_job
{
	const struct eye *eye;
	uint8_t *const *frames;
	uint32_t *dst; //!< this eye's top-left pixel; rows of 2 * eye_size
	uint32_t by0, by1;
	int core;
};

static void *
render_blocks(void *arg)
{
	struct render_job *j = arg;
	pin_big_core(j->core);
	const struct eye *e = j->eye;
	const uint32_t gn = e->gn, B = GRID_STEP, stride = eye_size * 2;

	for (uint32_t by = j->by0; by < j->by1; by++) {
		for (uint32_t bx = 0; bx + 1 < gn; bx++) {
			const struct node_cam *n00 = &e->cur[(by * gn + bx) * CAM_COUNT];
			const struct node_cam *n10 = n00 + CAM_COUNT;
			const struct node_cam *n01 = n00 + gn * CAM_COUNT;
			const struct node_cam *n11 = n01 + CAM_COUNT;

			// The two strongest cameras seen at all four corners (coordinates
			// can only be interpolated then); drop one carrying under 2%.
			int act[2] = {-1, -1};
			float peak[2] = {0, 0};
			for (int c = 0; c < CAM_COUNT; c++) {
				if (n00[c].w <= 0 || n10[c].w <= 0 || n01[c].w <= 0 || n11[c].w <= 0) {
					continue;
				}
				float m = fmaxf(fmaxf(n00[c].w, n10[c].w), fmaxf(n01[c].w, n11[c].w));
				if (m > peak[0]) {
					act[1] = act[0], peak[1] = peak[0];
					act[0] = c, peak[0] = m;
				} else if (m > peak[1]) {
					act[1] = c, peak[1] = m;
				}
			}
			int na = act[0] < 0 ? 0 : act[1] < 0 || peak[1] < 0.02f * peak[0] ? 1 : 2;

			for (uint32_t yy = 0; yy < B; yy++) {
				uint32_t *row = j->dst + (size_t)(by * B + yy) * stride + bx * B;
				if (na == 0) {
					memset(row, 0, B * 4);
					continue;
				}
				float fy = (yy + 0.5f) / B;
				// 16.16 fixed-point camera coordinates; float weights.
				int32_t u[2], v[2], du[2], dv[2];
				float w[2], g[2], dw[2], dg[2];
				for (int k = 0; k < na; k++) {
					int c = act[k];
#define LERP2(f) float l##f = n00[c].f + (n01[c].f - n00[c].f) * fy, r##f = n10[c].f + (n11[c].f - n10[c].f) * fy
					LERP2(u);
					LERP2(v);
					LERP2(w);
					LERP2(vig);
#undef LERP2
					float su = (ru - lu) / B, sv = (rv - lv) / B;
					u[k] = (int32_t)((lu + su * 0.5f) * 65536.0f), du[k] = (int32_t)(su * 65536.0f);
					v[k] = (int32_t)((lv + sv * 0.5f) * 65536.0f), dv[k] = (int32_t)(sv * 65536.0f);
					dw[k] = (rw - lw) / B, w[k] = lw + dw[k] * 0.5f;
					dg[k] = (rvig - lvig) / B, g[k] = lvig + dg[k] * 0.5f;
				}
				for (uint32_t xx = 0; xx < B; xx++) {
					float acc = 0.0f, wsum = 0.0f;
					for (int k = 0; k < na; k++) {
						const uint8_t *p = j->frames[act[k]] + (v[k] >> 16) * CAM_W + (u[k] >> 16);
						uint32_t fx = (u[k] >> 8) & 0xff, fyy = (v[k] >> 8) & 0xff;
						uint32_t a = p[0] * (256 - fx) + p[1] * fx;
						uint32_t b = p[CAM_W] * (256 - fx) + p[CAM_W + 1] * fx;
						float val = (float)(a * (256 - fyy) + b * fyy) * (1.0f / 65536.0f);
						acc += w[k] * g[k] * val;
						wsum += w[k];
						u[k] += du[k], v[k] += dv[k], w[k] += dw[k], g[k] += dg[k];
					}
					uint32_t px = wsum > 0.0f ? (uint32_t)(acc / wsum + 0.5f) : 0;
					px = px > 255 ? 255 : px;
					row[xx] = px | px << 8 | px << 16 | 0xff000000u;
				}
			}
		}
	}
	return NULL;
}

/*
 * Gain compensation (Brown & Lowe): per-camera gains g minimising
 * sum over overlapping pairs N_ab ((g_a I_ab - g_b I_ba)^2 / sn^2 + (1 - g_a)^2 / sg^2),
 * from the overlap nodes of both eyes at their chosen depth. Smoothed over time.
 */
static double cam_gain[CAM_COUNT] = {1, 1, 1, 1};

static void
gain_update(const struct eye *eyes_, int eye_count, uint8_t *const frames[CAM_COUNT])
{
	const double sn = 10.0, sg = 0.1;
	double N[CAM_COUNT][CAM_COUNT] = {{0}}, I[CAM_COUNT][CAM_COUNT] = {{0}};
	for (int ei = 0; ei < eye_count; ei++) {
		const struct eye *e = &eyes_[ei];
		for (uint32_t i = 0; i < e->gn * e->gn; i++) {
			const struct node_cam *nc = &e->cur[i * CAM_COUNT];
			int a, b;
			node_pair(nc, &a, &b);
			if (b < 0) {
				continue;
			}
			double ra = sample(frames[a], nc[a].u, nc[a].v), rb = sample(frames[b], nc[b].u, nc[b].v);
			if (ra < 4 || rb < 4 || ra > 250 || rb > 250) {
				continue; // black or clipped: no information
			}
			N[a][b] += 1, N[b][a] += 1;
			I[a][b] += ra * nc[a].vig, I[b][a] += rb * nc[b].vig;
		}
	}

	double A[CAM_COUNT][CAM_COUNT + 1] = {{0}};
	for (int a = 0; a < CAM_COUNT; a++) {
		A[a][a] += 1.0 / (sg * sg);
		A[a][CAM_COUNT] += 1.0 / (sg * sg);
		for (int b = 0; b < CAM_COUNT; b++) {
			if (b == a || N[a][b] < 20) {
				continue;
			}
			double iab = I[a][b] / N[a][b], iba = I[b][a] / N[a][b];
			A[a][a] += N[a][b] * (iab * iab / (sn * sn) + 1.0 / (sg * sg));
			A[a][b] -= N[a][b] * iab * iba / (sn * sn);
			A[a][CAM_COUNT] += N[a][b] / (sg * sg);
		}
	}
	// Gauss-Jordan with partial pivoting.
	for (int c = 0; c < CAM_COUNT; c++) {
		int piv = c;
		for (int r = c + 1; r < CAM_COUNT; r++) {
			if (fabs(A[r][c]) > fabs(A[piv][c])) {
				piv = r;
			}
		}
		for (int k = 0; k <= CAM_COUNT; k++) {
			double tmp = A[c][k];
			A[c][k] = A[piv][k];
			A[piv][k] = tmp;
		}
		if (fabs(A[c][c]) < 1e-12) {
			return;
		}
		for (int r = 0; r < CAM_COUNT; r++) {
			if (r == c) {
				continue;
			}
			double f = A[r][c] / A[c][c];
			for (int k = c; k <= CAM_COUNT; k++) {
				A[r][k] -= f * A[c][k];
			}
		}
	}
	for (int c = 0; c < CAM_COUNT; c++) {
		double g = A[c][CAM_COUNT] / A[c][c];
		g = g < 0.5 ? 0.5 : g > 2.5 ? 2.5 : g;
		cam_gain[c] = cam_gain[c] * 0.9 + g * 0.1;
	}
}

static void
gain_apply(uint8_t *const frames[CAM_COUNT])
{
	for (int c = 0; c < CAM_COUNT; c++) {
		uint8_t lut[256];
		for (int v = 0; v < 256; v++) {
			double g = v * cam_gain[c];
			lut[v] = g > 255.0 ? 255 : (uint8_t)g;
		}
		uint8_t *p = frames[c];
		for (int i = 0; i < CAM_PIXELS; i++) {
			p[i] = lut[p[i]];
		}
	}
}

/*
 *
 * OpenXR + Vulkan.
 *
 */

static XrInstance xr_instance;
static XrSystemId xr_system;
static XrSession xr_session;
static XrSpace xr_local, xr_view;
static PFN_xrConvertTimespecTimeToTimeKHR xr_timespec_to_time;
static VkInstance vk_instance;
static VkPhysicalDevice vk_phys;
static VkDevice vk_device;
static VkQueue vk_queue;
static uint32_t vk_queue_family;
static VkCommandPool vk_pool;
static VkCommandBuffer vk_cmd;
static VkFence vk_fence;
static int64_t swapchain_format;

static uint32_t
find_memory(uint32_t bits, VkMemoryPropertyFlags flags)
{
	VkPhysicalDeviceMemoryProperties props;
	vkGetPhysicalDeviceMemoryProperties(vk_phys, &props);
	for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
		if ((bits & (1u << i)) && (props.memoryTypes[i].propertyFlags & flags) == flags) {
			return i;
		}
	}
	fprintf(stderr, "no memory type\n");
	exit(1);
}

static void
xr_init(void)
{
	const char *exts[] = {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME, XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME};
	XrInstanceCreateInfo ici = {.type = XR_TYPE_INSTANCE_CREATE_INFO,
	                            .enabledExtensionCount = 2,
	                            .enabledExtensionNames = exts};
	strcpy(ici.applicationInfo.applicationName, "pico4-camview");
	ici.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
	XR_CHECK(xrCreateInstance(&ici, &xr_instance));

	XR_CHECK(xrGetInstanceProcAddr(xr_instance, "xrConvertTimespecTimeToTimeKHR",
	                               (PFN_xrVoidFunction *)&xr_timespec_to_time));

	XrSystemGetInfo sgi = {.type = XR_TYPE_SYSTEM_GET_INFO, .formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY};
	XR_CHECK(xrGetSystem(xr_instance, &sgi, &xr_system));

	PFN_xrGetVulkanGraphicsRequirements2KHR get_reqs;
	PFN_xrCreateVulkanInstanceKHR create_instance;
	PFN_xrGetVulkanGraphicsDevice2KHR get_device;
	PFN_xrCreateVulkanDeviceKHR create_device;
	XR_CHECK(xrGetInstanceProcAddr(xr_instance, "xrGetVulkanGraphicsRequirements2KHR",
	                               (PFN_xrVoidFunction *)&get_reqs));
	XR_CHECK(xrGetInstanceProcAddr(xr_instance, "xrCreateVulkanInstanceKHR", (PFN_xrVoidFunction *)&create_instance));
	XR_CHECK(xrGetInstanceProcAddr(xr_instance, "xrGetVulkanGraphicsDevice2KHR", (PFN_xrVoidFunction *)&get_device));
	XR_CHECK(xrGetInstanceProcAddr(xr_instance, "xrCreateVulkanDeviceKHR", (PFN_xrVoidFunction *)&create_device));

	XrGraphicsRequirementsVulkan2KHR reqs = {.type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
	XR_CHECK(get_reqs(xr_instance, xr_system, &reqs));

	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	                         .pApplicationName = "pico4-camview",
	                         .apiVersion = VK_API_VERSION_1_1};
	VkInstanceCreateInfo vici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
	XrVulkanInstanceCreateInfoKHR xvici = {.type = XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR,
	                                       .systemId = xr_system,
	                                       .pfnGetInstanceProcAddr = vkGetInstanceProcAddr,
	                                       .vulkanCreateInfo = &vici};
	VkResult vr;
	XR_CHECK(create_instance(xr_instance, &xvici, &vk_instance, &vr));
	VK_CHECK(vr);

	XrVulkanGraphicsDeviceGetInfoKHR dgi = {.type = XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR,
	                                        .systemId = xr_system,
	                                        .vulkanInstance = vk_instance};
	XR_CHECK(get_device(xr_instance, &dgi, &vk_phys));

	uint32_t qcount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(vk_phys, &qcount, NULL);
	VkQueueFamilyProperties qprops[16];
	if (qcount > 16) {
		qcount = 16;
	}
	vkGetPhysicalDeviceQueueFamilyProperties(vk_phys, &qcount, qprops);
	vk_queue_family = UINT32_MAX;
	for (uint32_t i = 0; i < qcount; i++) {
		if (qprops[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			vk_queue_family = i;
			break;
		}
	}
	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	                               .queueFamilyIndex = vk_queue_family,
	                               .queueCount = 1,
	                               .pQueuePriorities = &prio};
	VkDeviceCreateInfo dci = {
	    .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
	XrVulkanDeviceCreateInfoKHR xdci = {.type = XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR,
	                                    .systemId = xr_system,
	                                    .pfnGetInstanceProcAddr = vkGetInstanceProcAddr,
	                                    .vulkanPhysicalDevice = vk_phys,
	                                    .vulkanCreateInfo = &dci};
	XR_CHECK(create_device(xr_instance, &xdci, &vk_device, &vr));
	VK_CHECK(vr);
	vkGetDeviceQueue(vk_device, vk_queue_family, 0, &vk_queue);

	XrGraphicsBindingVulkan2KHR binding = {.type = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR,
	                                       .instance = vk_instance,
	                                       .physicalDevice = vk_phys,
	                                       .device = vk_device,
	                                       .queueFamilyIndex = vk_queue_family,
	                                       .queueIndex = 0};
	XrSessionCreateInfo sci = {.type = XR_TYPE_SESSION_CREATE_INFO, .next = &binding, .systemId = xr_system};
	XR_CHECK(xrCreateSession(xr_instance, &sci, &xr_session));

	XrReferenceSpaceCreateInfo rsci = {.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
	                                   .referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL,
	                                   .poseInReferenceSpace = {.orientation = {0, 0, 0, 1}}};
	XR_CHECK(xrCreateReferenceSpace(xr_session, &rsci, &xr_local));
	rsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
	XR_CHECK(xrCreateReferenceSpace(xr_session, &rsci, &xr_view));

	VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	                               .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
	                               .queueFamilyIndex = vk_queue_family};
	VK_CHECK(vkCreateCommandPool(vk_device, &pci, NULL, &vk_pool));
	VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
	                                   .commandPool = vk_pool,
	                                   .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
	                                   .commandBufferCount = 1};
	VK_CHECK(vkAllocateCommandBuffers(vk_device, &cai, &vk_cmd));
	VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VK_CHECK(vkCreateFence(vk_device, &fci, NULL, &vk_fence));

	uint32_t n = 0;
	XR_CHECK(xrEnumerateSwapchainFormats(xr_session, 0, &n, NULL));
	int64_t formats[64];
	if (n > 64) {
		n = 64;
	}
	XR_CHECK(xrEnumerateSwapchainFormats(xr_session, n, &n, formats));
	const int64_t want[] = {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB,
	                        VK_FORMAT_B8G8R8A8_UNORM};
	for (size_t w = 0; w < sizeof(want) / sizeof(want[0]) && !swapchain_format; w++) {
		for (uint32_t i = 0; i < n; i++) {
			if (formats[i] == want[w]) {
				swapchain_format = want[w];
			}
		}
	}
	if (!swapchain_format) {
		fprintf(stderr, "no RGBA8 swapchain format\n");
		exit(1);
	}
}

static void
target_init(struct target *t, uint32_t width, uint32_t height)
{
	t->width = width;
	t->height = height;
	VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
	                          .size = (VkDeviceSize)width * height * 4,
	                          .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
	VK_CHECK(vkCreateBuffer(vk_device, &bci, NULL, &t->staging));
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(vk_device, t->staging, &mr);
	VkMemoryAllocateInfo mai = {
	    .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
	    .allocationSize = mr.size,
	    .memoryTypeIndex =
	        find_memory(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
	VK_CHECK(vkAllocateMemory(vk_device, &mai, NULL, &t->staging_mem));
	VK_CHECK(vkBindBufferMemory(vk_device, t->staging, t->staging_mem, 0));
	VK_CHECK(vkMapMemory(vk_device, t->staging_mem, 0, VK_WHOLE_SIZE, 0, &t->staging_ptr));

	XrSwapchainCreateInfo sci = {.type = XR_TYPE_SWAPCHAIN_CREATE_INFO,
	                             .usageFlags = XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT |
	                                           XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT,
	                             .format = swapchain_format,
	                             .sampleCount = 1,
	                             .width = width,
	                             .height = height,
	                             .faceCount = 1,
	                             .arraySize = 1,
	                             .mipCount = 1};
	XR_CHECK(xrCreateSwapchain(xr_session, &sci, &t->swapchain));
	XrSwapchainImageVulkan2KHR imgs[8];
	for (int i = 0; i < 8; i++) {
		imgs[i] = (XrSwapchainImageVulkan2KHR){.type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR};
	}
	XR_CHECK(xrEnumerateSwapchainImages(t->swapchain, 8, &t->image_count, (XrSwapchainImageBaseHeader *)imgs));
	for (uint32_t i = 0; i < t->image_count; i++) {
		t->images[i] = imgs[i].image;
	}
}

static void
barrier(VkImage image, VkImageLayout from, VkImageLayout to, VkAccessFlags src, VkAccessFlags dst)
{
	VkImageMemoryBarrier b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
	                          .srcAccessMask = src,
	                          .dstAccessMask = dst,
	                          .oldLayout = from,
	                          .newLayout = to,
	                          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	                          .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
	                          .image = image,
	                          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
	vkCmdPipelineBarrier(vk_cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL,
	                     0, NULL, 1, &b);
}

//! Copy the staging buffer of @p t into its next swapchain image.
static void
target_submit(struct target *t)
{
	uint32_t idx;
	XrSwapchainImageAcquireInfo ai = {.type = XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
	XR_CHECK(xrAcquireSwapchainImage(t->swapchain, &ai, &idx));
	XrSwapchainImageWaitInfo wi = {.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO, .timeout = XR_INFINITE_DURATION};
	XR_CHECK(xrWaitSwapchainImage(t->swapchain, &wi));

	VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	                               .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
	VK_CHECK(vkResetCommandBuffer(vk_cmd, 0));
	VK_CHECK(vkBeginCommandBuffer(vk_cmd, &bi));
	barrier(t->images[idx], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
	        VK_ACCESS_TRANSFER_WRITE_BIT);
	VkBufferImageCopy region = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
	                            .imageExtent = {t->width, t->height, 1}};
	vkCmdCopyBufferToImage(vk_cmd, t->staging, t->images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
	// OpenXR wants colour swapchain images back in COLOR_ATTACHMENT_OPTIMAL.
	barrier(t->images[idx], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
	        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
	VK_CHECK(vkEndCommandBuffer(vk_cmd));
	VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &vk_cmd};
	VK_CHECK(vkQueueSubmit(vk_queue, 1, &si, vk_fence));
	VK_CHECK(vkWaitForFences(vk_device, 1, &vk_fence, VK_TRUE, UINT64_MAX));
	VK_CHECK(vkResetFences(vk_device, 1, &vk_fence));

	XrSwapchainImageReleaseInfo ri = {.type = XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
	XR_CHECK(xrReleaseSwapchainImage(t->swapchain, &ri));
	t->has_image = true;
}

//! Screens mode: upload the newest frame of @p c if there is one.
static void
camera_upload(struct camera *c)
{
	struct target *t = &c->target;
	pthread_mutex_lock(&c->lock);
	bool fresh = c->seq != c->uploaded_seq;
	if (fresh || !t->has_image) {
		uint32_t *dst = t->staging_ptr;
		for (int i = 0; i < CAM_PIXELS; i++) {
			uint32_t g = c->gray[i];
			dst[i] = g | g << 8 | g << 16 | 0xff000000u;
		}
		c->uploaded_seq = c->seq;
	}
	pthread_mutex_unlock(&c->lock);
	if (fresh || !t->has_image) {
		target_submit(t);
	}
}

static XrQuaternionf
yaw_roll(float yaw, float roll)
{
	// q = q_yaw(Y) * q_roll(Z)
	float cy = cosf(yaw / 2), sy = sinf(yaw / 2), cr = cosf(roll / 2), sr = sinf(roll / 2);
	return (XrQuaternionf){.x = sy * sr, .y = sy * cr, .z = cy * sr, .w = cy * cr};
}

/*
 *
 * Passthrough.
 *
 */

static struct eye eyes[2];
//! Head-relative views at the capture time of the camera set on screen.
static XrView set_views[2];
static bool set_views_valid;
static struct target stereo;
static uint8_t *snapshot[CAM_COUNT];
static double pt_ms;      //!< smoothed CPU time of one passthrough set
static double pt_stage_ms[3]; //!< smoothed: gains, depth, render
static int64_t pt_set_ns; //!< timestamp of the last camera set taken
static uint64_t pt_sets;  //!< camera sets processed

/*
 * The worker renders into the back image and swaps it to the front; the
 * OpenXR loop copies the front image into the swapchain when it is new.
 */
static pthread_t pt_thread;
static bool pt_thread_started;
static pthread_mutex_t pt_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t *pt_front, *pt_back;
static bool pt_new;
static int64_t pt_front_ns;

//! Copy one frame per camera if all are from the same trigger (within 5 ms).
static bool
passthrough_take_set(void)
{
	const int64_t window = 5000000;
	int64_t lo = INT64_MAX, hi = 0;
	for (int c = 0; c < CAM_COUNT; c++) {
		if (cams[c].fd < 0) {
			continue;
		}
		pthread_mutex_lock(&cams[c].lock);
		int64_t t = cams[c].ts_ns;
		pthread_mutex_unlock(&cams[c].lock);
		lo = t < lo ? t : lo;
		hi = t > hi ? t : hi;
	}
	if (hi == 0 || lo == 0 || hi - lo > window || hi <= pt_set_ns) {
		return false;
	}
	for (int c = 0; c < CAM_COUNT; c++) {
		if (cams[c].fd < 0) {
			continue;
		}
		pthread_mutex_lock(&cams[c].lock);
		bool same = cams[c].ts_ns >= lo && cams[c].ts_ns <= hi;
		if (same) {
			memcpy(snapshot[c], cams[c].gray, CAM_PIXELS);
		}
		pthread_mutex_unlock(&cams[c].lock);
		if (!same) {
			return false; // a newer frame arrived meanwhile; next round
		}
	}
	pt_set_ns = hi;
	pt_sets++;
	return true;
}

struct depth_job
{
	struct eye *eye;
	int index;
};

static void *
depth_thread(void *arg)
{
	struct depth_job *j = arg;
	pin_big_core(j->index);
	eye_estimate_depth(j->eye, snapshot);
	return NULL;
}

static void *
passthrough_worker(void *arg)
{
	(void)arg;
	const uint32_t blocks = eyes[0].gn - 1;
	pin_big_core(3); // prime core; waits while the helpers render
	while (!quit_requested) {
		if (!passthrough_take_set()) {
			usleep(1000);
			continue;
		}
		struct timespec t0, t1, ta, tb;
		clock_gettime(CLOCK_MONOTONIC, &t0);
		if (!getenv("CAMVIEW_NO_GAIN")) {
			gain_update(eyes, 2, snapshot);
			gain_apply(snapshot);
		}
		clock_gettime(CLOCK_MONOTONIC, &ta);
		struct depth_job djobs[2] = {{&eyes[0], 1}, {&eyes[1], 2}};
		pthread_t dthreads[2];
		for (int i = 0; i < 2; i++) {
			pthread_create(&dthreads[i], NULL, depth_thread, &djobs[i]);
		}
		for (int i = 0; i < 2; i++) {
			pthread_join(dthreads[i], NULL);
		}
		clock_gettime(CLOCK_MONOTONIC, &tb);

		struct render_job jobs[4];
		pthread_t threads[4];
		for (int i = 0; i < 4; i++) {
			jobs[i] = (struct render_job){.eye = &eyes[i / 2],
			                              .frames = snapshot,
			                              .dst = pt_back + (i / 2) * eye_size,
			                              .by0 = (i % 2) * blocks / 2,
			                              .by1 = (i % 2 + 1) * blocks / 2,
			                              .core = i};
			pthread_create(&threads[i], NULL, render_blocks, &jobs[i]);
		}
		for (int i = 0; i < 4; i++) {
			pthread_join(threads[i], NULL);
		}
		clock_gettime(CLOCK_MONOTONIC, &t1);
		double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
		pt_ms = pt_ms ? pt_ms * 0.95 + ms * 0.05 : ms;
		double st[3] = {(ta.tv_sec - t0.tv_sec) * 1e3 + (ta.tv_nsec - t0.tv_nsec) / 1e6,
		                (tb.tv_sec - ta.tv_sec) * 1e3 + (tb.tv_nsec - ta.tv_nsec) / 1e6,
		                (t1.tv_sec - tb.tv_sec) * 1e3 + (t1.tv_nsec - tb.tv_nsec) / 1e6};
		for (int i = 0; i < 3; i++) {
			pt_stage_ms[i] = pt_stage_ms[i] ? pt_stage_ms[i] * 0.95 + st[i] * 0.05 : st[i];
		}

		pthread_mutex_lock(&pt_lock);
		uint32_t *t = pt_front;
		pt_front = pt_back;
		pt_back = t;
		pt_front_ns = pt_set_ns;
		pt_new = true;
		pthread_mutex_unlock(&pt_lock);
	}
	return NULL;
}

static bool
passthrough_init(XrTime time)
{
	XrViewLocateInfo vli = {.type = XR_TYPE_VIEW_LOCATE_INFO,
	                        .viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
	                        .displayTime = time,
	                        .space = xr_view};
	XrViewState vs = {.type = XR_TYPE_VIEW_STATE};
	XrView views[2] = {{.type = XR_TYPE_VIEW}, {.type = XR_TYPE_VIEW}};
	uint32_t count = 0;
	XR_CHECK(xrLocateViews(xr_session, &vli, &vs, 2, &count, views));
	if (count != 2) {
		return false;
	}

	// Candidate depths, uniform in inverse depth; CAMVIEW_DEPTH fixes one.
	if (getenv("CAMVIEW_DEPTH")) {
		depth_count = 1;
		depth_m[0] = env_float("CAMVIEW_DEPTH", 2.0f);
	} else {
		const double near = 0.35, far = 6.0;
		for (int k = 0; k < DEPTHS; k++) {
			depth_m[k] = 1.0 / (1.0 / near + k * (1.0 / far - 1.0 / near) / (DEPTHS - 1));
		}
	}
	// Head centre (between the eyes) in the calibration body frame: midway
	// between the cameras, a few centimetres behind them.
	double head[3] = {-0.024, 0.051, 0.07};
	if (getenv("CAMVIEW_HEAD")) {
		sscanf(getenv("CAMVIEW_HEAD"), "%lf,%lf,%lf", &head[0], &head[1], &head[2]);
	}
	for (int i = 0; i < 2; i++) {
		eyes[i].pose = views[i].pose;
		eyes[i].fov = views[i].fov;
		eye_build_grid(&eyes[i], head);
	}
	target_init(&stereo, eye_size * 2, eye_size);
	for (int c = 0; c < CAM_COUNT; c++) {
		snapshot[c] = calloc(1, CAM_PIXELS);
	}
	pt_front = calloc((size_t)eye_size * 2 * eye_size, 4);
	pt_back = calloc((size_t)eye_size * 2 * eye_size, 4);
	pthread_create(&pt_thread, NULL, passthrough_worker, NULL);
	pt_thread_started = true;
	return true;
}

//! Fill the projection layer; false until the first camera set is rendered.
static bool
passthrough_frame(XrTime time, XrCompositionLayerProjection *layer, XrCompositionLayerProjectionView pv[2])
{
	bool fresh = false;
	int64_t capture_ns = 0;
	pthread_mutex_lock(&pt_lock);
	if (pt_new) {
		memcpy(stereo.staging_ptr, pt_front, (size_t)stereo.width * stereo.height * 4);
		capture_ns = pt_front_ns;
		pt_new = false;
		fresh = true;
	}
	pthread_mutex_unlock(&pt_lock);
	if (fresh) {
		// Monado keeps showing the last released image between sets.
		target_submit(&stereo);
	}
	if (!stereo.has_image) {
		return false;
	}

	// CAMVIEW_DUMP=file.pgm: save the stereo image once, after warm-up.
	static int dump_countdown = 150;
	if (fresh && getenv("CAMVIEW_DUMP") && dump_countdown > 0 && --dump_countdown == 0) {
		FILE *f = fopen(getenv("CAMVIEW_DUMP"), "wb");
		if (f) {
			fprintf(f, "P5 %u %u 255\n", stereo.width, stereo.height);
			const uint32_t *px = stereo.staging_ptr;
			for (uint32_t i = 0; i < stereo.width * stereo.height; i++) {
				fputc(px[i] & 0xff, f);
			}
			fclose(f);
			printf("dumped %s\n", getenv("CAMVIEW_DUMP"));
		}
	}

	/*
	 * Tag the image with the head pose of its capture time: Monado then
	 * reprojects it to the current pose every refresh, so the world stays
	 * put between the ~30 Hz camera sets instead of moving with the head.
	 */
	if (fresh || !set_views_valid) {
		XrTime capture = time;
		struct timespec ts = {.tv_sec = capture_ns / 1000000000LL, .tv_nsec = capture_ns % 1000000000LL};
		if (capture_ns && xr_timespec_to_time(xr_instance, &ts, &capture) != XR_SUCCESS) {
			capture = time;
		}
		XrViewLocateInfo vli = {.type = XR_TYPE_VIEW_LOCATE_INFO,
		                        .viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
		                        .displayTime = capture,
		                        .space = xr_local};
		XrViewState vs = {.type = XR_TYPE_VIEW_STATE};
		uint32_t count = 0;
		set_views[0] = (XrView){.type = XR_TYPE_VIEW};
		set_views[1] = (XrView){.type = XR_TYPE_VIEW};
		XR_CHECK(xrLocateViews(xr_session, &vli, &vs, 2, &count, set_views));
		set_views_valid = true;
	}
	for (int i = 0; i < 2; i++) {
		pv[i] = (XrCompositionLayerProjectionView){
		    .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW,
		    .pose = set_views[i].pose,
		    .fov = eyes[i].fov,
		    .subImage = {.swapchain = stereo.swapchain,
		                 .imageRect = {.offset = {(int32_t)(i * eye_size), 0},
		                               .extent = {(int32_t)eye_size, (int32_t)eye_size}}}};
	}
	*layer = (XrCompositionLayerProjection){
	    .type = XR_TYPE_COMPOSITION_LAYER_PROJECTION, .space = xr_local, .viewCount = 2, .views = pv};
	return true;
}

int
main(int argc, char **argv)
{
	bool passthrough = argc > 1 && strcmp(argv[1], "--passthrough") == 0;
	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);

	float gain = env_float("CAMVIEW_GAIN", 1.0f);
	for (int v = 0; v < 256; v++) {
		float g = v * gain;
		gain_lut[v] = g > 255.0f ? 255 : (uint8_t)g;
	}
	float rolls[CAM_COUNT] = {0};
	if (getenv("CAMVIEW_ROLL")) {
		sscanf(getenv("CAMVIEW_ROLL"), "%f,%f,%f,%f", &rolls[0], &rolls[1], &rolls[2], &rolls[3]);
	}
	/*
	 * Captures 0..3 are stock slots 0, 3, 1, 2 (CSIPHY5, 4, 3, 3-combo).
	 * Slots 0/3 are mav0 cam0/cam3, but the right pair is crossed: slot 1
	 * (CSIPHY3 A) is cam2 bottom-right, slot 2 (combo B) is cam1 top-right
	 * (checked by eye in passthrough, 2026-10-21).
	 */
	int map[CAM_COUNT] = {0, 3, 2, 1};
	if (getenv("CAMVIEW_MAP")) {
		sscanf(getenv("CAMVIEW_MAP"), "%d,%d,%d,%d", &map[0], &map[1], &map[2], &map[3]);
	}
	eye_size = (uint32_t)env_float("CAMVIEW_EYE", 960) / GRID_STEP * GRID_STEP;

	int open_count = 0;
	for (int i = 0; i < CAM_COUNT; i++) {
		struct camera *c = &cams[i];
		c->index = i;
		c->roll = rolls[i] * (float)M_PI / 180.0f;
		pthread_mutex_init(&c->lock, NULL);
		c->gray = calloc(1, CAM_PIXELS);
		if (passthrough) {
			c->has_cal = calib_load(map[i], &c->cal);
		}
		c->use_shm = shm_wanted(i);
		if (camera_open(c) == 0) {
			pthread_create(&c->thread, NULL, c->use_shm ? shm_thread : camera_thread, c);
			open_count++;
		} else {
			c->fd = -1;
		}
	}
	if (open_count == 0) {
		fprintf(stderr, "no camera could be opened\n");
		return 1;
	}

	xr_init();
	if (!passthrough) {
		for (int i = 0; i < CAM_COUNT; i++) {
			target_init(&cams[i].target, CAM_W, CAM_H);
		}
	}

	bool running = false, exiting = false, pt_ready = false;
	uint64_t frame_count = 0;
	while (!exiting) {
		if (quit_requested && running) {
			xrRequestExitSession(xr_session);
			quit_requested = 0;
		}

		XrEventDataBuffer ev = {.type = XR_TYPE_EVENT_DATA_BUFFER};
		while (xrPollEvent(xr_instance, &ev) == XR_SUCCESS) {
			if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
				XrSessionState s = ((XrEventDataSessionStateChanged *)&ev)->state;
				if (s == XR_SESSION_STATE_READY) {
					XrSessionBeginInfo sbi = {.type = XR_TYPE_SESSION_BEGIN_INFO,
					                          .primaryViewConfigurationType =
					                              XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO};
					XR_CHECK(xrBeginSession(xr_session, &sbi));
					running = true;
					printf("session running\n");
				} else if (s == XR_SESSION_STATE_STOPPING) {
					xrEndSession(xr_session);
					running = false;
				} else if (s == XR_SESSION_STATE_EXITING || s == XR_SESSION_STATE_LOSS_PENDING) {
					exiting = true;
				}
			} else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
				exiting = true;
			}
			ev = (XrEventDataBuffer){.type = XR_TYPE_EVENT_DATA_BUFFER};
		}
		if (!running) {
			usleep(10000);
			continue;
		}

		XrFrameState fs = {.type = XR_TYPE_FRAME_STATE};
		XrFrameWaitInfo fwi = {.type = XR_TYPE_FRAME_WAIT_INFO};
		XR_CHECK(xrWaitFrame(xr_session, &fwi, &fs));
		XrFrameBeginInfo fbi = {.type = XR_TYPE_FRAME_BEGIN_INFO};
		XR_CHECK(xrBeginFrame(xr_session, &fbi));

		XrCompositionLayerQuad quads[CAM_COUNT];
		XrCompositionLayerProjection proj;
		XrCompositionLayerProjectionView pv[2];
		const XrCompositionLayerBaseHeader *layers[CAM_COUNT];
		uint32_t layer_count = 0;
		if (fs.shouldRender && passthrough) {
			if (!pt_ready) {
				pt_ready = passthrough_init(fs.predictedDisplayTime);
			}
			if (pt_ready && passthrough_frame(fs.predictedDisplayTime, &proj, pv)) {
				layers[layer_count++] = (const XrCompositionLayerBaseHeader *)&proj;
			}
		} else if (fs.shouldRender) {
			// 4:3 screens on an arc 1.5 m in front of the start pose.
			const float dist = 1.5f, width = 0.7f, step = 0.48f;
			for (int i = 0; i < CAM_COUNT; i++) {
				struct camera *c = &cams[i];
				camera_upload(c);
				float yaw = ((CAM_COUNT - 1) / 2.0f - i) * step;
				quads[layer_count] = (XrCompositionLayerQuad){
				    .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
				    .space = xr_local,
				    .eyeVisibility = XR_EYE_VISIBILITY_BOTH,
				    .subImage = {.swapchain = c->target.swapchain,
				                 .imageRect = {.offset = {0, 0}, .extent = {CAM_W, CAM_H}}},
				    .pose = {.orientation = yaw_roll(yaw, c->roll),
				             .position = {-sinf(yaw) * dist, 0.0f, -cosf(yaw) * dist}},
				    .size = {width, width * CAM_H / CAM_W}};
				layers[layer_count] = (const XrCompositionLayerBaseHeader *)&quads[layer_count];
				layer_count++;
			}
		}

		XrFrameEndInfo fei = {.type = XR_TYPE_FRAME_END_INFO,
		                      .displayTime = fs.predictedDisplayTime,
		                      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
		                      .layerCount = layer_count,
		                      .layers = layers};
		XR_CHECK(xrEndFrame(xr_session, &fei));

		if (++frame_count % 360 == 0) {
			printf("frame %lu, camera frames %lu %lu %lu %lu (skipped %lu %lu %lu %lu), sets %lu, "
			       "passthrough %.1f ms (gain %.1f depth %.1f render %.1f), gains %.2f %.2f %.2f %.2f\n",
			       (unsigned long)frame_count, (unsigned long)cams[0].frames, (unsigned long)cams[1].frames,
			       (unsigned long)cams[2].frames, (unsigned long)cams[3].frames,
			       (unsigned long)cams[0].skipped, (unsigned long)cams[1].skipped,
			       (unsigned long)cams[2].skipped, (unsigned long)cams[3].skipped, (unsigned long)pt_sets,
			       pt_ms, pt_stage_ms[0], pt_stage_ms[1], pt_stage_ms[2], cam_gain[0], cam_gain[1],
			       cam_gain[2], cam_gain[3]);
			fflush(stdout);
		}
	}

	quit_requested = 1;
	if (pt_thread_started) {
		pthread_join(pt_thread, NULL);
	}
	for (int i = 0; i < CAM_COUNT; i++) {
		if (cams[i].fd >= 0) {
			enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
			ioctl(cams[i].fd, VIDIOC_STREAMOFF, &type);
			pthread_join(cams[i].thread, NULL);
		}
	}
	xrDestroySession(xr_session);
	xrDestroyInstance(xr_instance);
	return 0;
}
