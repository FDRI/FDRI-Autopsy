#ifndef DFXML_CREATOR_H
#define DFXML_CREATOR_H

#include <pugixml.hpp>

#include <dlib/image_processing.h>

// Function declaration (like before main)
class DFXMLCreator
{
  private:
	int openssl_sha1(char *name, unsigned char *out);

  public:

	void add_DFXML_creator(pugi::xml_node &parent,
						   const char *program_name,
						   const char *program_version);

	void add_fileobject(pugi::xml_node &parent,
						const char *file_path,
						const int number_faces,
						const long original_width,
						const long original_height,
						const long working_width,
						const long working_height,
						const std::vector<dlib::mmod_rect, std::allocator<dlib::mmod_rect>> detected_faces);

	pugi::xml_document create_document();
};

#endif