

#include <chrono>
#include <ctime>
#include <Lmcons.h>

#include <boost/filesystem.hpp>
#include <boost/version.hpp>
#include <boost/format.hpp>

#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/string.h>
#include <dlib/image_io.h>


#include <openssl/sha.h>

#include <FDRI/DFXML_creator.hpp>

using namespace std;
using namespace boost;
using namespace dlib;

using std::ofstream;

// PRIVATE FUNCTION

int DFXMLCreator::openssl_sha1(char *name, unsigned char *out)
{
	FILE *f;
	unsigned char buf[8192];
	SHA_CTX sc;
	int err;

	f = fopen(name, "rb");
	if (f == NULL)
	{
		cout << "Couldn't open file" << endl;
		return -1;
	}
	SHA1_Init(&sc);
	for (;;)
	{
		size_t len;

		len = fread(buf, 1, sizeof buf, f);
		if (len == 0)
			break;
		SHA1_Update(&sc, buf, len);
	}
	err = ferror(f);
	fclose(f);
	if (err)
	{
		cout << "Error hashing file" << endl;
		return -1;
	}
	SHA1_Final(out, &sc);
	return 0;
}


// CONSTRUCTOR
// NONE

pugi::xml_document DFXMLCreator::create_document()
{
	pugi::xml_document doc;
	doc.load_string("<?xml version='1.0' encoding='UTF-8'?>\n");
	pugi::xml_node dfxml_node = doc.append_child("dfxml");
	dfxml_node.append_attribute("xmlns") = "http://www.forensicswiki.org/wiki/Category:Digital_Forensics_XML";
	dfxml_node.append_attribute("xmlns:dc") = "http://purl.org/dc/elements/1.1/";
	dfxml_node.append_attribute("xmlns:xsi") = "http://www.w3.org/2001/XMLSchema-instance";
	dfxml_node.append_attribute("version") = "1.1.1";

	return doc;
}

void DFXMLCreator::add_DFXML_creator(pugi::xml_node &parent,
								const char *program_name,
								const char *program_version)
{
	pugi::xml_node creator_node = parent.append_child("creator");
	creator_node.append_attribute("version") = "1.0";
	creator_node.append_child("program").text().set(program_name);
	creator_node.append_child("version").text().set(program_version);
	pugi::xml_node build_node = creator_node.append_child("build_environment");

#ifdef BOOST_VERSION
	{
		char buf[64];
		snprintf(buf, sizeof(buf), "%d", BOOST_VERSION);
		pugi::xml_node lib_node = build_node.append_child("library");
		lib_node.append_attribute("name") = "boost";
		lib_node.append_attribute("version") = buf;
	}
#endif
	pugi::xml_node lib_node = build_node.append_child("library");
	lib_node.append_attribute("name") = "pugixml";
	lib_node.append_attribute("version") = "1.9";

	lib_node = build_node.append_child("library");
	lib_node.append_attribute("name") = "dlib";
	lib_node.append_attribute("version") = "19.10";

	pugi::xml_node exe_node = creator_node.append_child("execution_environment");

	chrono::system_clock::time_point p = chrono::system_clock::now();
	time_t t = chrono::system_clock::to_time_t(p);
	exe_node.append_child("start_date").text().set(ctime(&t));

	char username[UNLEN + 1];
	DWORD username_len = UNLEN + 1;
	GetUserName(username, &username_len);
	exe_node.append_child("username").text().set(username);
}

void DFXMLCreator::add_fileobject(pugi::xml_node &parent,
							 const char *file_path,
							 const int number_faces,
							 const long original_width,
							 const long original_height,
							 const long working_width,
							 const long working_height,
							 const std::vector<dlib::mmod_rect, std::allocator<dlib::mmod_rect>> detected_faces)
{
	// TODO: Check why is this giving error
	filesystem::path p(file_path);
	uintmax_t f_size =  filesystem::file_size(p);
	pugi::xml_node file_obj = parent.append_child("fileobject");
	file_obj.append_child("filesize").text().set(f_size);

	string delimiter = "__id__";
	string aux_name(p.filename().string());
	string img_n = aux_name.substr(0, aux_name.find(delimiter));
	file_obj.append_child("filename").text().set(img_n.c_str());
	// incluir as hashs
	unsigned char hash_buff[SHA_DIGEST_LENGTH];
	if (openssl_sha1((char *)file_path, hash_buff))
	{
		cout << "Error getting file hash" << endl;//dlog << LWARN << "Error getting file hash";
	}
	else
	{
		pugi::xml_node hash_nodeMD5 = file_obj.append_child("hashdigest");
		hash_nodeMD5.append_attribute("type") = "sha1";
		char tmphash[SHA_DIGEST_LENGTH];

		for (size_t i = 0; i < SHA_DIGEST_LENGTH; i++)
		{
			sprintf((char *)&(tmphash[i * 2]), "%02x", hash_buff[i]);
		}

		hash_nodeMD5.text().set(tmphash);
	}


	pugi::xml_node detection_node = file_obj.append_child("facialdetection");
	detection_node.append_child("number_faces").text().set(number_faces);
	std::stringstream ss;
	ss << original_width << "x" << original_height;
	detection_node.append_child("original_size").text().set(ss.str().c_str());
	ss.clear();
	ss.str("");
	ss << working_width << "x" << working_height;
	detection_node.append_child("working_size").text().set(ss.str().c_str());

	for (int i = 1; i <= detected_faces.size(); i++)
	{
		std::stringstream ss;
		rectangle rec = detected_faces[i - 1];
		ss << rec.left() << " "
		   << rec.top() << " "
		   << rec.right() << " "
		   << rec.bottom();

		pugi::xml_node face_node = detection_node.append_child("face");
		face_node.text().set(ss.str().c_str());

		pugi::xml_node score = face_node.append_child("confidence_score");
		// TODO:: Converter / arredondar
		score.text().set(detected_faces[i].detection_confidence);
	}
}

