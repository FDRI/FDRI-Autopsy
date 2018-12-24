

#include "helper_functions.hpp"

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/logger.h>
#include <dlib/opencv.h>

#include <time.h>
#include <ctime>
#include <chrono>

#include <boost/filesystem.hpp>

using namespace boost;
using namespace dlib;

namespace po = boost::program_options;

void arg_parser(const int argc, const char *const argv[], po::variables_map args, po::options_description desc)
{

    try
    {
        po::store(
            po::parse_command_line(argc, argv, desc),
            args);
    }
    catch (po::error const &e)
    {
        std::cerr << e.what() << '\n';
        exit(EXIT_FAILURE);
    }

    po::notify(args);

    if (args.count("version"))
    {
        std::cout << "Compilation date: " __DATE__ << " " << __TIME__ << std::endl;
        exit(0);
    }
    if (args.count("list_devices"))
    {
        int num_devices = cuda::get_num_devices();
        std::cout << "Found " << num_devices << " CUDA supported devices" << std::endl;
        for (int i = 0; i < num_devices; i++)
        {
            std::cout << "[" << i << "] => " << cuda::get_device_name(i) << std::endl;
        }
        exit(0);
    }

    if (args.count("help") || argc < 2 || !args.count("params"))
    {
        std::cout << desc << std::endl;
        exit(2);
    }

    if (args.count("set_device"))
    {
        int selected_device = args["set_device"].as<int>();
        std::cout << "Device [" << selected_device << "] " << cuda::get_device_name(selected_device) << " selected" << std::endl;
    }

    /*
	if (!args.count("debug"))
	{
#undef DEBUG
#define DEBUG 0
	}
	*/
}

std::vector<filesystem::path> fileSearch(std::string path)
{
    std::cout << "Searching files::filesearch" << std::endl;
    std::vector<filesystem::path> images_path;
    std::vector<std::string> targetExtensions;

    targetExtensions.push_back(".JPG");
    targetExtensions.push_back(".BMP");
    targetExtensions.push_back(".GIF");
    targetExtensions.push_back(".PNG");

    if (!filesystem::exists(path))
    {
        std::cout << "Path doesn't exist" << std::endl;
        exit(4);
    }

    for (filesystem::recursive_directory_iterator end, dir(path); dir != end; ++dir)
    {
        std::string extension = filesystem::path(*dir).extension().generic_string();
        transform(extension.begin(), extension.end(), extension.begin(), ::toupper);
        if (std::find(targetExtensions.begin(), targetExtensions.end(), extension) != targetExtensions.end())
        {
            images_path.push_back(filesystem::path(*dir));
        }
    }
    return images_path;
}

Log_handler::Log_handler(std::string filename)
{
    out_file.open(filename);
}

void Log_handler::
    log(const std::string &logger_name, const dlib::log_level &ll, const dlib::uint64 thread_id, const char *message_to_log)
{

    std::chrono::system_clock::time_point p = std::chrono::system_clock::now();
    time_t t = std::chrono::system_clock::to_time_t(p);

    char buffer[20];
    strftime(buffer, 20, "%Y-%m-%d %H:%M:%S", localtime(&t));

    // We print allways for log and stdout
    out_file << buffer << " " << ll << " [" << thread_id << "] " << logger_name << ": " << message_to_log << std::endl;

    // But only log messages that are of LINFO priority or higher to the console.
    if (ll < LERROR)
    {
        std::cout << ctime(&t) << "\n"
                  << ll << " [" << thread_id << "] " << logger_name << ": " << message_to_log << std::endl;
    }
    else
    {
        std::cerr << ctime(&t) << "\n"
                  << ll << " [" << thread_id << "] " << logger_name << ": " << message_to_log << std::endl;
    }
}

Log_handler::~Log_handler()
{
    out_file.close();
}
