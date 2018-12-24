#ifndef IMAGE_HANDLING_H
#define IMAGE_HANDLING_H

struct ImageSplit
{
	int X_offset;
	int Y_offset;
	cv::Mat image;

	ImageSplit(int x_off, int y_off, cv::Mat img)
	{
		X_offset = x_off;
		Y_offset = y_off;
		image = img;
	}
};

struct thread_data
{
	dlib::matrix<dlib::rgb_pixel> *target;
	dlib::cv_image<dlib::bgr_pixel> source;
	int id;
	dlib::mutex *mutex;
	dlib::signaler *count_signaler;//(count_mutex);
	int *thread_count;
};

void get_splits(cv::Mat original, std::vector<ImageSplit> &splits, const int maxH, const int maxW);

void cv2Dlib(const cv::Mat source, dlib::matrix<dlib::rgb_pixel> *target);

void threaded_copy(void *arg);

#endif