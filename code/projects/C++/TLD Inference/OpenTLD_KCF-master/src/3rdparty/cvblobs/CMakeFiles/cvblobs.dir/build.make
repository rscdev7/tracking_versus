# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/rosario/programs/anaconda3/bin/cmake

# The command to remove a file.
RM = /home/rosario/programs/anaconda3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/rosario/run_folder/OpenTLD_KCF-master

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/rosario/run_folder/OpenTLD_KCF-master

# Include any dependencies generated for this target.
include src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/depend.make

# Include the progress variables for this target.
include src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/progress.make

# Include the compile flags for this target's objects.
include src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/flags.make

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/blob.cpp.o: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/flags.make
src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/blob.cpp.o: src/3rdparty/cvblobs/blob.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rosario/run_folder/OpenTLD_KCF-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/blob.cpp.o"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvblobs.dir/blob.cpp.o -c /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/blob.cpp

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/blob.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvblobs.dir/blob.cpp.i"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/blob.cpp > CMakeFiles/cvblobs.dir/blob.cpp.i

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/blob.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvblobs.dir/blob.cpp.s"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/blob.cpp -o CMakeFiles/cvblobs.dir/blob.cpp.s

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobContour.cpp.o: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/flags.make
src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobContour.cpp.o: src/3rdparty/cvblobs/BlobContour.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rosario/run_folder/OpenTLD_KCF-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobContour.cpp.o"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvblobs.dir/BlobContour.cpp.o -c /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobContour.cpp

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobContour.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvblobs.dir/BlobContour.cpp.i"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobContour.cpp > CMakeFiles/cvblobs.dir/BlobContour.cpp.i

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobContour.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvblobs.dir/BlobContour.cpp.s"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobContour.cpp -o CMakeFiles/cvblobs.dir/BlobContour.cpp.s

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobOperators.cpp.o: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/flags.make
src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobOperators.cpp.o: src/3rdparty/cvblobs/BlobOperators.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rosario/run_folder/OpenTLD_KCF-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobOperators.cpp.o"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvblobs.dir/BlobOperators.cpp.o -c /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobOperators.cpp

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobOperators.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvblobs.dir/BlobOperators.cpp.i"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobOperators.cpp > CMakeFiles/cvblobs.dir/BlobOperators.cpp.i

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobOperators.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvblobs.dir/BlobOperators.cpp.s"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobOperators.cpp -o CMakeFiles/cvblobs.dir/BlobOperators.cpp.s

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobProperties.cpp.o: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/flags.make
src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobProperties.cpp.o: src/3rdparty/cvblobs/BlobProperties.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rosario/run_folder/OpenTLD_KCF-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobProperties.cpp.o"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvblobs.dir/BlobProperties.cpp.o -c /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobProperties.cpp

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobProperties.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvblobs.dir/BlobProperties.cpp.i"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobProperties.cpp > CMakeFiles/cvblobs.dir/BlobProperties.cpp.i

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobProperties.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvblobs.dir/BlobProperties.cpp.s"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobProperties.cpp -o CMakeFiles/cvblobs.dir/BlobProperties.cpp.s

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobResult.cpp.o: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/flags.make
src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobResult.cpp.o: src/3rdparty/cvblobs/BlobResult.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rosario/run_folder/OpenTLD_KCF-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobResult.cpp.o"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvblobs.dir/BlobResult.cpp.o -c /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobResult.cpp

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobResult.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvblobs.dir/BlobResult.cpp.i"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobResult.cpp > CMakeFiles/cvblobs.dir/BlobResult.cpp.i

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobResult.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvblobs.dir/BlobResult.cpp.s"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/BlobResult.cpp -o CMakeFiles/cvblobs.dir/BlobResult.cpp.s

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.o: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/flags.make
src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.o: src/3rdparty/cvblobs/ComponentLabeling.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/rosario/run_folder/OpenTLD_KCF-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.o"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.o -c /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/ComponentLabeling.cpp

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.i"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/ComponentLabeling.cpp > CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.i

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.s"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/ComponentLabeling.cpp -o CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.s

# Object files for target cvblobs
cvblobs_OBJECTS = \
"CMakeFiles/cvblobs.dir/blob.cpp.o" \
"CMakeFiles/cvblobs.dir/BlobContour.cpp.o" \
"CMakeFiles/cvblobs.dir/BlobOperators.cpp.o" \
"CMakeFiles/cvblobs.dir/BlobProperties.cpp.o" \
"CMakeFiles/cvblobs.dir/BlobResult.cpp.o" \
"CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.o"

# External object files for target cvblobs
cvblobs_EXTERNAL_OBJECTS =

lib/libcvblobs.a: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/blob.cpp.o
lib/libcvblobs.a: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobContour.cpp.o
lib/libcvblobs.a: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobOperators.cpp.o
lib/libcvblobs.a: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobProperties.cpp.o
lib/libcvblobs.a: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/BlobResult.cpp.o
lib/libcvblobs.a: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/ComponentLabeling.cpp.o
lib/libcvblobs.a: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/build.make
lib/libcvblobs.a: src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/rosario/run_folder/OpenTLD_KCF-master/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX static library ../../../lib/libcvblobs.a"
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && $(CMAKE_COMMAND) -P CMakeFiles/cvblobs.dir/cmake_clean_target.cmake
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cvblobs.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/build: lib/libcvblobs.a

.PHONY : src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/build

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/clean:
	cd /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs && $(CMAKE_COMMAND) -P CMakeFiles/cvblobs.dir/cmake_clean.cmake
.PHONY : src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/clean

src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/depend:
	cd /home/rosario/run_folder/OpenTLD_KCF-master && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/rosario/run_folder/OpenTLD_KCF-master /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs /home/rosario/run_folder/OpenTLD_KCF-master /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs /home/rosario/run_folder/OpenTLD_KCF-master/src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/3rdparty/cvblobs/CMakeFiles/cvblobs.dir/depend

