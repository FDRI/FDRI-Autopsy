
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <dlib/image_io.h>
#include <dlib/opencv.h>

#include <FDRI/image_handling.hpp>

void get_splits(cv::Mat original, std::vector<ImageSplit> &splits, const int maxH, const int maxW)
{
	int height = original.rows, width = original.cols;
	int num_divisionsH = 0, num_divisionsW = 0;

	while (height > maxH)
	{
		num_divisionsH++;
		height /= 2;
	}
	while (width > maxW)
	{
		num_divisionsW++;
		width /= 2;
	}
	height = height % 2 == 0 ? height : height - (height % 2);
	width = width % 2 == 0 ? width : width - (width % 2);

	//cout << "NumDW - " << num_divisionsW * 2 << " NumDH - " << num_divisionsH * 2 << endl;
	if (num_divisionsH == 0 && num_divisionsW == 0)
	{
		splits.push_back(ImageSplit(0, 0, original));
	}
	else if (num_divisionsH && num_divisionsW)
	{
		int y = 0;
		for (int i = 0; i < num_divisionsH * 2; i++)
		{
			int x = 0;
			for (int j = 0; j < num_divisionsW * 2; j++)
			{
				splits.push_back(ImageSplit(width * j, height * i, original(cv::Rect(x, y, width, height))));
				x += width;
			}
			y += height;
		}
	}
	else if (num_divisionsH)
	{
		int y = 0;
		for (int i = 0; i < num_divisionsH * 2; i++)
		{
			splits.push_back(ImageSplit(0, height * i, original(cv::Rect(0, y, width, height))));
			y += height;
		}
	}
	else if (num_divisionsW)
	{
		int x = 0;
		for (int j = 0; j < num_divisionsW * 2; j++)
		{
			splits.push_back(ImageSplit(width * j, 0, original(cv::Rect(x, 0, width, height))));

			x += width;
		}
	}
}

void cv2Dlib(const cv::Mat source, dlib::matrix<dlib::rgb_pixel> *target)
{
	dlib::cv_image<dlib::bgr_pixel> image(source);

	target->set_size(image.nr(), image.nc());
	//cout << "Starting convertion" << endl;
	std::vector<thread_data> t_data;
	//cout << "Creating threads" << endl;
	/*
		struct thread_data
		{
			dlib::matrix<dlib::rgb_pixel> *target;
			dlib::cv_image<dlib::bgr_pixel> *source;
			int id;
			dlib::mutex *count_mutex;
			dlib::signaler *count_signaler;//(count_mutex);
			int thread_count;
		};
	*/
	int thread_ammount = std::thread::hardware_concurrency() > 4 ? 4 : std::thread::hardware_concurrency();
	dlib::mutex t_mutex;
	dlib::signaler count_signaler(t_mutex);
	for (int t_id = 0; t_id < thread_ammount; t_id++)
	{
		thread_data data;
		data.target = target;
		data.source = image;
		data.id = t_id;
		data.thread_count = &thread_ammount;
		data.mutex = &t_mutex;
		data.count_signaler = &count_signaler;

		t_data.push_back(data);
		dlib::create_new_thread(threaded_copy, (void *)&t_data[t_id]);
	}

	dlib::auto_mutex abcd(t_mutex);
	while (thread_ammount > 0)
	{
		count_signaler.wait();
	}
}

void threaded_copy(void *arg)
{
	thread_data *data = (thread_data *)arg;

	int start = data->id;
	int jump = std::thread::hardware_concurrency() > 4 ? 4 : std::thread::hardware_concurrency();
	//std::cout << start << std::endl;
	//std::cout << "Jump " << jump << std::endl;
	for (long r = start; r < data->source.nr(); r += jump)
	{
		for (long c = 0; c < data->source.nc(); c++)
		{
			dlib::assign_pixel((*data->target)(r, c), data->source(r, c));
		}
	}
	//cout << "Done my job -- closing" << endl;
	dlib::auto_mutex locker(*data->mutex);
	*data->thread_count = *data->thread_count - 1;
	// Now we signal this change.  This will cause one thread that is currently waiting
	// on a call to count_signaler.wait() to unblock.
	data->count_signaler->signal();
}