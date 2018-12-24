
#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/filesystem.hpp>

#include <dlib/logger.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing.h>

#include <iostream>

namespace po = boost::program_options;

std::vector<boost::filesystem::path> fileSearch(std::string path);
void arg_parser(const int argc, const char *const argv[], po::variables_map args, po::options_description desc);
class Log_handler
{
  private:
	std::ofstream out_file;

  public:
	~Log_handler();
	Log_handler(std::string filename);

	void log(const std::string &logger_name, const dlib::log_level &ll, const dlib::uint64 thread_id, const char *message_to_log);
};

#endif