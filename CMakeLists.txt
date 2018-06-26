# Comandos a executar para fazer build do projeto
#cmake -G "Visual Studio 14 2015 Win64" -T host=x64 .. -DBOOST_ROOT=D:\boost_1_66_0 -DBOOST_LIBRARYDIR=D:\boost_1_66_0\boost_1_66_0\stage\lib
#cmake --build . --config Release


# Versao minima recomendada pelo Dlib -> 2.8.12
cmake_minimum_required(VERSION 2.8.12)
# Nome do projeto
project(FDRI)

# Incluir Dlib no projeto
add_subdirectory(./dlib dlib_build)
add_subdirectory(./pugixml-1.9 pugi_build)

#set(Boost_DEBUG ON)
set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost 1.66.0 REQUIRED  filesystem date_time program_options)


# Executaveis e codigo
add_executable(FDRI main.cpp)

# Bibliotecas adicionais
target_link_libraries(FDRI dlib::dlib)
target_link_libraries(FDRI Boost::filesystem Boost::date_time Boost::program_options)
target_link_libraries(FDRI pugixml)
