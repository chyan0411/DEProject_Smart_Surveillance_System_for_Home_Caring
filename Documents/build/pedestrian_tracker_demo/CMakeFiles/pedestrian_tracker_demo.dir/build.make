# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /opt/intel/openvino/deployment_tools/inference_engine/samples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/Documents/build

# Include any dependencies generated for this target.
include pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/depend.make

# Include the progress variables for this target.
include pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/progress.make

# Include the compile flags for this target's objects.
include pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/flags.make

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/flags.make
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/cnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Documents/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/cnn.cpp

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.i"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/cnn.cpp > CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.i

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.s"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/cnn.cpp -o CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.s

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o.requires:

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o.requires

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o.provides: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o.requires
	$(MAKE) -f pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build.make pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o.provides.build
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o.provides

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o.provides.build: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o


pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/flags.make
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/core.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Documents/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/core.cpp

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.i"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/core.cpp > CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.i

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.s"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/core.cpp -o CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.s

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o.requires:

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o.requires

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o.provides: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o.requires
	$(MAKE) -f pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build.make pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o.provides.build
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o.provides

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o.provides.build: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o


pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/flags.make
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/detector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Documents/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/detector.cpp

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.i"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/detector.cpp > CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.i

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.s"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/detector.cpp -o CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.s

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o.requires:

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o.requires

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o.provides: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o.requires
	$(MAKE) -f pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build.make pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o.provides.build
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o.provides

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o.provides.build: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o


pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/flags.make
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/distance.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Documents/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/distance.cpp

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.i"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/distance.cpp > CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.i

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.s"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/distance.cpp -o CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.s

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o.requires:

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o.requires

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o.provides: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o.requires
	$(MAKE) -f pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build.make pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o.provides.build
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o.provides

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o.provides.build: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o


pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/flags.make
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/image_reader.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Documents/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/image_reader.cpp

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.i"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/image_reader.cpp > CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.i

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.s"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/image_reader.cpp -o CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.s

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o.requires:

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o.requires

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o.provides: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o.requires
	$(MAKE) -f pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build.make pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o.provides.build
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o.provides

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o.provides.build: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o


pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/flags.make
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/kuhn_munkres.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Documents/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/kuhn_munkres.cpp

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.i"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/kuhn_munkres.cpp > CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.i

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.s"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/kuhn_munkres.cpp -o CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.s

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o.requires:

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o.requires

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o.provides: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o.requires
	$(MAKE) -f pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build.make pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o.provides.build
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o.provides

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o.provides.build: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o


pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/flags.make
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/tracker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Documents/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/tracker.cpp

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.i"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/tracker.cpp > CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.i

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.s"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/tracker.cpp -o CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.s

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o.requires:

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o.requires

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o.provides: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o.requires
	$(MAKE) -f pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build.make pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o.provides.build
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o.provides

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o.provides.build: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o


pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/flags.make
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Documents/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/utils.cpp

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.i"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/utils.cpp > CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.i

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.s"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/src/utils.cpp -o CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.s

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o.requires:

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o.requires

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o.provides: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o.requires
	$(MAKE) -f pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build.make pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o.provides.build
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o.provides

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o.provides.build: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o


pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/flags.make
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o: /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/Documents/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o -c /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/main.cpp

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.i"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/main.cpp > CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.i

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.s"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo/main.cpp -o CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.s

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o.requires:

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o.requires

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o.provides: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o.requires
	$(MAKE) -f pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build.make pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o.provides.build
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o.provides

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o.provides.build: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o


# Object files for target pedestrian_tracker_demo
pedestrian_tracker_demo_OBJECTS = \
"CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o" \
"CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o" \
"CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o" \
"CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o" \
"CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o" \
"CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o" \
"CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o" \
"CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o" \
"CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o"

# External object files for target pedestrian_tracker_demo
pedestrian_tracker_demo_EXTERNAL_OBJECTS =

armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o
armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o
armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o
armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o
armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o
armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o
armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o
armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o
armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o
armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build.make
armv7l/Release/pedestrian_tracker_demo: armv7l/Release/lib/libcpu_extension.so
armv7l/Release/pedestrian_tracker_demo: /opt/intel/openvino/deployment_tools/inference_engine/lib/armv7l/libinference_engine.so
armv7l/Release/pedestrian_tracker_demo: /opt/intel/openvino/opencv/lib/libopencv_highgui.so.4.1.0
armv7l/Release/pedestrian_tracker_demo: armv7l/Release/lib/libgflags_nothreads.a
armv7l/Release/pedestrian_tracker_demo: /opt/intel/openvino/opencv/lib/libopencv_videoio.so.4.1.0
armv7l/Release/pedestrian_tracker_demo: /opt/intel/openvino/opencv/lib/libopencv_imgcodecs.so.4.1.0
armv7l/Release/pedestrian_tracker_demo: /opt/intel/openvino/opencv/lib/libopencv_imgproc.so.4.1.0
armv7l/Release/pedestrian_tracker_demo: /opt/intel/openvino/opencv/lib/libopencv_core.so.4.1.0
armv7l/Release/pedestrian_tracker_demo: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/Documents/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable ../armv7l/Release/pedestrian_tracker_demo"
	cd /home/pi/Documents/build/pedestrian_tracker_demo && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pedestrian_tracker_demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build: armv7l/Release/pedestrian_tracker_demo

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/build

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/requires: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/cnn.cpp.o.requires
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/requires: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/core.cpp.o.requires
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/requires: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/detector.cpp.o.requires
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/requires: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/distance.cpp.o.requires
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/requires: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/image_reader.cpp.o.requires
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/requires: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/kuhn_munkres.cpp.o.requires
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/requires: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/tracker.cpp.o.requires
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/requires: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/src/utils.cpp.o.requires
pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/requires: pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/main.cpp.o.requires

.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/requires

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/clean:
	cd /home/pi/Documents/build/pedestrian_tracker_demo && $(CMAKE_COMMAND) -P CMakeFiles/pedestrian_tracker_demo.dir/cmake_clean.cmake
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/clean

pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/depend:
	cd /home/pi/Documents/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /opt/intel/openvino/deployment_tools/inference_engine/samples /opt/intel/openvino/deployment_tools/inference_engine/samples/pedestrian_tracker_demo /home/pi/Documents/build /home/pi/Documents/build/pedestrian_tracker_demo /home/pi/Documents/build/pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : pedestrian_tracker_demo/CMakeFiles/pedestrian_tracker_demo.dir/depend

