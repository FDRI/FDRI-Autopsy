// Native
#include <iostream>
#include <windows.h>
#include <streambuf>
#include <typeinfo>
#include <time.h>
#include <ctime>
#include <Lmcons.h>

// DLIB
#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/string.h>
#include <dlib/image_io.h>

// Boost
#include <boost/filesystem.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/version.hpp>

// PUGI XML Parsing
#include <pugixml.hpp>

/**
 * Sparing some text
*/
using namespace std;
using namespace dlib;
using namespace boost;
using std::ofstream;
namespace po = boost::program_options;

/**
 *  Recognition network definition
*/
template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
												  alevel0<
													  alevel1<
														  alevel2<
															  alevel3<
																  alevel4<
																	  max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2, input_rgb_image_sized<150>>>>>>>>>>>>>;

/**
 *  Detection network definition
*/
template <long num_filters, typename SUBNET>
using con5d = con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET>
using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = relu<affine<con5<45, SUBNET>>>;

using net_type = loss_mmod<con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

/**
 *  Function prototypes
*/
std::vector<filesystem::path> fileSearch(string path);

int findFaces(const std::vector<filesystem::path> path_to_images,
			  net_type detector,
			  const shape_predictor sp,
			  std::vector<matrix<rgb_pixel>> &faces,
			  std::list<std::pair<filesystem::path, int>> &mapping,
			  ofstream &output_file,
			  const long min_size,
			  const long max_size,
			  const bool mark,
			  pugi::xml_document &doc);

void add_DFXML_creator(pugi::xml_node &parent,
					   const char *program_name,
					   const char *program_version);

std::string xmlescape(const string &xml);

void add_fileobject(pugi::xml_node &parent,
					const char *file_path,
					const int number_faces,
					const long original_width,
					const long original_height,
					const long working_width,
					const long working_height,
					const std::vector<dlib::mmod_rect, std::allocator<dlib::mmod_rect>> detected_faces);

pugi::xml_document create_document();
void arg_parser(const int argc, const char *const argv[], po::variables_map args, po::options_description desc);

static string xml_lt("&lt;");
static string xml_gt("&gt;");
static string xml_am("&amp;");
static string xml_ap("&apos;");
static string xml_qu("&quot;");

// % encodings
static string encoding_null("%00");
static string encoding_r("%0D");
static string encoding_n("%0A");
static string encoding_t("%09");

/**
 *  DEBUGG toggling
*/
#define DEBUG 0

int main(int argc, char **argv)
{
	po::variables_map args;
	long min_size, max_size;
	string params_path;

	po::options_description desc("Allowed options");
	desc.add_options()("help", "")
	("params", po::value(&params_path), "Path to configuration file")
	("min", po::value(&min_size)->default_value(1200 * 1200), "Minimum image size (long - default: 1200x1200)")
	("max", po::value(&max_size)->default_value(2500 * 2500), "Maximum image size (long - default: 2500x2500)");

	arg_parser(argc, argv, &args, desc);

	if (max_size < 0 || min_size < 0 || max_size < min_size) {
		cout << "Invalid image size" << endl;
		exit(1);
	}
	try
	{
		property_tree::ptree json_tree;
		try
		{
#if DEBUG
			cout << "Loading json configuration" << endl;
#endif
			read_json(params_path, json_tree);
		}
		catch (std::exception &e)
		{
			cout << "Error parsing json" << endl;
			exit(2);
		}
		
		string outputPath, outputPositivePath,
			imagesToFindPath, positiveImgPath,
			detectorPath, recognitionPath,
			shapePredictorPath, dfxmlPath;

		bool doRecognition;
		try
		{
#if DEBUG
			cout << "Parsing json variables" << endl;
#endif
			outputPath = json_tree.get<string>("outputPath");
			outputPositivePath = json_tree.get<string>("outputPositivePath");
			imagesToFindPath = json_tree.get<string>("imagesPath");
			positiveImgPath = json_tree.get_child("paths.").get<string>("1");
			detectorPath = json_tree.get_child("paths.").get<string>("2");
			recognitionPath = json_tree.get_child("paths.").get<string>("3");
			shapePredictorPath = json_tree.get_child("paths.").get<string>("4");
			dfxmlPath = json_tree.get<string>("dfxml_out_path");
			doRecognition = json_tree.get<bool>("doRecognition");
		}
		catch (std::exception &e)
		{
			cout << "Error Initiating variables" << endl;
			exit(3);
		}
		std::vector<filesystem::path> positivePath, imagesPath;
		try
		{
#if DEBUG
			cout << "Searching files for detection" << endl;
#endif
			if (doRecognition)
			{
				positivePath = fileSearch(positiveImgPath);
			}
			imagesPath = fileSearch(imagesToFindPath);
		}
		catch (std::exception &e)
		{
			cout << "Erro ao carregar imagens" << endl;
			exit(4);
		}
		net_type detector_dnn;
		anet_type recognition_dnn;
		shape_predictor sp;
		cout << "A iniciar detectores" << endl;
		if (doRecognition)
		{
			try
			{
#if DEBUG
				cout << "Initializing recognition network" << endl;
#endif
				deserialize(recognitionPath) >> recognition_dnn;
			}
			catch (std::exception &e)
			{
				cout << "Error loading: dlib_face_recognition_resnet_model_v1.dat";
				exit(5);
			}
		}

		try
		{
#if DEBUG
			cout << "Initializing shape predictor" << endl;
#endif
			deserialize(shapePredictorPath) >> sp;
		}
		catch (std::exception &e)
		{
			cout << "Error loading: shape_predictor_5_face_landmarks.dat";
			exit(6);
		}

		try
		{
#if DEBUG
			cout << "Initializing face detector" << endl;
#endif
			deserialize(detectorPath) >> detector_dnn;
		}
		catch (std::exception &e)
		{
			cout << "Erro ao carregar detector" << endl;
			exit(7);
		}

		std::vector<matrix<rgb_pixel>> faces;
		std::list<std::pair<filesystem::path, int>> imgToFaces;
		ofstream output_file;
#if DEBUG
		cout << "Creating file to store output" << endl;
#endif

		output_file.open(outputPath);
		int num_positive_faces = 0;
		if (doRecognition)
		{
#if DEBUG
			cout << "Searching faces in positive images" << endl;
#endif
			num_positive_faces = findFaces(positivePath, detector_dnn, sp, faces, imgToFaces, output_file, min_size, max_size, false, pugi::xml_document());
			if (!num_positive_faces)
			{
				cout << "ERROR => Didn't find any positive images in provided folder" << endl;
				return 8;
			}
		}

		pugi::xml_document dfxml_doc = create_document();
		pugi::xml_node dfxml_node = dfxml_doc.child("dfxml");

		char filename[MAX_PATH];
		DWORD size = GetModuleFileNameA(NULL, filename, MAX_PATH);
		if (size)
		{
			add_DFXML_creator(dfxml_node, filename, "1.0");
		}
#if DEBUG
		cout << "Searching faces in target images" << endl;
#endif
		int num_faces_found = findFaces(imagesPath, detector_dnn, sp, faces, imgToFaces, output_file, min_size, max_size, true, dfxml_doc);
		if (num_faces_found == 0)
		{
			cout << "ERROR => No faces found in provided images" << endl;
			return 9;
		}
		output_file.close();

		if (doRecognition)
		{
#if DEBUG
			cout << "Maching people with positive images" << endl;
#endif
			std::vector<matrix<float, 0, 1>> face_descriptors = recognition_dnn(faces);

			int num_img_positivas = positivePath.size();

			std::list<std::pair<filesystem::path, int>>::iterator it = imgToFaces.begin();
			std::advance(it, num_img_positivas);

			int counter = 0;
#if DEBUG
			cout << "Opening file to store images with target people" << endl;
#endif
			output_file.open(outputPositivePath);
			for (it; it != imgToFaces.end(); it++)
			{
				int num_faces_in_image = ((std::pair<filesystem::path, int>)*it).second;
				int delta = num_positive_faces + counter;
				for (int i = 0; i < num_faces_in_image; i++)
				{
					for (int j = 0; j < num_positive_faces; j++)
					{
						if (length(face_descriptors[i + delta] - face_descriptors[j]) < 0.6)
						{
							output_file << ((std::pair<filesystem::path, int>)*it).first.filename().string() << endl;
						}
					}
				}
				counter += num_faces_in_image;
			}
			output_file.close();
		}

#if DEBUG
		cout << "Storing dfxml" << endl;
#endif
		ofstream os;
		os.open(dfxmlPath);
		dfxml_doc.save(os);
#if DEBUG
		cout << "Closing files" << endl;
#endif
		os.close();
		cout << "Fim" << endl;
	}
	catch (std::exception &e)
	{
		cout << "Erro: " << e.what() << endl;
		return 11;
	}
	return 0;
}

int findFaces(const std::vector<filesystem::path> path_to_images,
			  net_type detector,
			  const shape_predictor sp,
			  std::vector<matrix<rgb_pixel>> &faces,
			  std::list<std::pair<filesystem::path, int>> &mapping,
			  ofstream &output_file,
			  const long min_size,
			  const long max_size,
			  const bool mark,
			  pugi::xml_document &doc)
{
	int num_faces = 0;
	pugi::xml_node dfxml_node = doc.child("dfxml");
	for (size_t i = 0; i < path_to_images.size(); i++)
	{
		matrix<rgb_pixel> img, img2;
		load_image(img, path_to_images[i].generic_string());
		long original_width = img.nc(), original_height = img.nr();

		while (img.size() > max_size)
		{
			pyramid_down<2> down_scaler;
			down_scaler(img, img2);
			img = img2;
		}

		while (img.size() < min_size)
		{
			pyramid_up(img);
		}

		long working_width = img.nc(), working_height = img.nr();
		int num_faces_in_image = 0;
		std::vector<dlib::mmod_rect, std::allocator<dlib::mmod_rect>> detected_faces;

		try
		{
			detected_faces = detector(img);
		}
		catch (std::exception &e)
		{
			cout << "Cuda out of memory! Try lowering the image resolution" << endl;
			exit(10);
		}

		for (dlib::mmod_rect face : detected_faces)
		{
			if (mark)
			{
				dlib::draw_rectangle(img, face, dlib::rgb_pixel(255, 0, 0), 3);
			}
			auto shape = sp(img, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(std::move(face_chip));
			num_faces++;
			num_faces_in_image++;
		}
		if (mark)
		{
			if (num_faces_in_image != 0)
			{
				output_file << path_to_images[i].filename().string() + "\n";
				dlib::save_png(img, path_to_images[i].generic_string());
			}
			add_fileobject(dfxml_node, path_to_images[i].generic_string().c_str(),
						   num_faces_in_image, original_width,
						   original_height, working_width,
						   working_height, detected_faces);
		}
		mapping.push_back(std::make_pair(path_to_images[i], num_faces_in_image));
	}
	return num_faces;
}

std::vector<filesystem::path> fileSearch(string path)
{
	std::vector<filesystem::path> images_path;
	std::vector<string> targetExtensions;

	targetExtensions.push_back(".JPG");
	targetExtensions.push_back(".BMP");
	targetExtensions.push_back(".GIF");
	targetExtensions.push_back(".PNG");

	if (!filesystem::exists(path))
	{
		exit(4);
	}

	for (filesystem::recursive_directory_iterator end, dir(path); dir != end; ++dir)
	{
		string extension = filesystem::path(*dir).extension().generic_string();
		transform(extension.begin(), extension.end(), extension.begin(), ::toupper);
		if (std::find(targetExtensions.begin(), targetExtensions.end(), extension) != targetExtensions.end())
		{
			images_path.push_back(filesystem::path(*dir));
		}
	}
	return images_path;
}

pugi::xml_document create_document()
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

void add_DFXML_creator(pugi::xml_node &parent,
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
	struct tm *tm;
	time_t tim;
	time(&tim);
	tm = localtime(&tim);
	char buf[64];
	snprintf(buf, sizeof(buf), "%4d-%02d-%02dT%s", tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday, __TIME__);
	exe_node.append_child("start_date").text().set(buf);

	char username[UNLEN + 1];
	DWORD username_len = UNLEN + 1;
	GetUserName(username, &username_len);
	exe_node.append_child("username").text().set(username);
}

void add_fileobject(pugi::xml_node &parent,
					const char *file_path,
					const int number_faces,
					const long original_width,
					const long original_height,
					const long working_width,
					const long working_height,
					const std::vector<dlib::mmod_rect, std::allocator<dlib::mmod_rect>> detected_faces)
{
	boost::filesystem::path p(file_path);
	uintmax_t f_size = boost::filesystem::file_size(p);
	pugi::xml_node file_obj = parent.append_child("fileobject");
	file_obj.append_child("filesize").text().set(f_size);

	string delimiter = "__id__";
	string aux_name(p.filename().string());
	string img_n = aux_name.substr(0, aux_name.find(delimiter));
	file_obj.append_child("filename").text().set(img_n.c_str());
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
		/*
		face_node.append_child("confidence_score")
			.text()
			.set(detected_faces[i].detection_confidence);
		*/
	}
}

std::string xmlescape(const string &xml)
{
	string ret;
	for (string::const_iterator i = xml.begin(); i != xml.end(); i++)
	{
		switch (*i)
		{
		// XML escapes
		case '>':
			ret += xml_gt;
			break;
		case '<':
			ret += xml_lt;
			break;
		case '&':
			ret += xml_am;
			break;
		case '\'':
			ret += xml_ap;
			break;
		case '"':
			ret += xml_qu;
			break;

		// % encodings
		case '\000':
			ret += encoding_null;
			break; // retain encoded nulls
		case '\r':
			ret += encoding_r;
			break;
		case '\n':
			ret += encoding_n;
			break;
		case '\t':
			ret += encoding_t;
			break;
		default:
			ret += *i;
		}
	}
	return ret;
}

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

	if (args.count("help") || argc < 2 || !args.count("params"))
	{
		std::cout << desc << std::endl;
		exit(2);
	}
}