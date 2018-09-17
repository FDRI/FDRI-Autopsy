# -*- coding: utf-8 -*-

#
# Date: 17 September 2018
# Author: Alexandre Frazao Rosario
#         Patricio Domingues
#
# Module Full Description: 
# FDRI is a image analysis module that focus in finding human faces in images, 
# as well finding images that contain a specific person. It provides this functionality’s appealing to AI Convolutional Neural Networks.
# The executable is a implementation of facial detection and recognition with Dlib DNN(http://dlib.net/).
# 
# The facial recognition element is activated when selecting a folder with images from the person that 
# the program should look for, it will look for the person and if it finds, marks it as interesting file hit.
#
# All the detectors used can be found at: https://github.com/davisking/dlib-models
#
#====================================================================
# License Apache 2.0
#====================================================================
# Copyright 2018 Alexandre Frazão Rosário
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import json
from datetime import datetime
import os  # file checking
import shutil  # file copy
import subprocess  # .exe calling
import time
import signal
import hashlib
import xml.dom.minidom as m_dom
import jarray
from java.awt import BorderLayout, GridLayout, FlowLayout, Dimension
from java.awt.event import KeyAdapter, KeyEvent, KeyListener
from threading import Thread
from distutils.dir_util import copy_tree

# Java librarys
from java.io import File
from java.lang import System
from java.lang import Thread as JThread
from java.util.logging import Level


# UI librarys
from javax.swing import (BorderFactory, BoxLayout, JButton, JCheckBox,
                         JComponent, JFileChooser, JFrame, JLabel, JPanel,
                         JScrollPane, JTextField, JToolBar)
from javax.swing.event import DocumentEvent, DocumentListener
from org.sleuthkit.autopsy.casemodule import Case
from org.sleuthkit.autopsy.casemodule.services import (Blackboard, FileManager,
                                                       Services)
from org.sleuthkit.autopsy.coreutils import Logger
from org.sleuthkit.autopsy.datamodel import ContentUtils
# sleuthkit librarys
from org.sleuthkit.autopsy.ingest import (DataSourceIngestModule,
                                          FileIngestModule, IngestMessage,
                                          IngestModule,
                                          IngestModuleFactoryAdapter,
                                          IngestModuleIngestJobSettings,
                                          IngestModuleIngestJobSettingsPanel,
                                          IngestModuleGlobalSettingsPanel,
                                          IngestServices, ModuleDataEvent)
from org.sleuthkit.autopsy.ingest.IngestModule import IngestModuleException
from org.sleuthkit.datamodel import (AbstractFile, BlackboardArtifact,
                                     BlackboardAttribute, SleuthkitCase,
                                     TskData, ReadContentInputStream)

#====================================================================
# Configuration
#====================================================================
# Version of the code. We simply put the date and a letter.
C_VERSION = "2018-09-17" 

# Module configurations for multiple cases
GLOBAL_CONFIGURATION_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "configuration.json")

CONFIGURATION_PATH = ""
# Number of bytes to read on hashing function
BLOCKSIZE = 5120

# Minimum size for an image file to be processed (in bytes=
C_FILE_MIN_SIZE = 1025

# Name of file to hold the filenames where faces were detected
C_FACES_FOUND_FNAME = "FDRI_faces_found.txt"

# Name of file to hold the files where recognition occurred 
C_FDRI_WANTED_FNAME = "FDRI_wanted.txt"

# Name of created DFXML file
C_DFXML_FNAME = "dfxml.xml"

# Name of file to register filenames and size
C_FILE_WITH_FNAMES_AND_SIZES = "FDRI_filenames+size.log.txt"

# Name of file to get the list of repeated files
C_REPEATED_FILES_LOG = "FDRI_repeated_files.log.txt"

# Name of file holding JSON parameters
C_PARAMS_JSON_FNAME="params.json"

# Label for an annotated file
C_ANNOTATED_LABEL="Annotated_"

# Represent annotated when pathname is being built
C_ANNOTATED_DIR="annotated" 

# name of FDRI included in path
C_FDRI_DIR="FDRI"

# Label for GUI configuration
C_LABEL_INFO_AUTOPSY_TEMP = "Save copied images outside of Autopsy's /Temp"

# Create DFXML (internal use in this script)
C_CREATE_DFXML = True

# Compute hashes for DFXML
C_COMPUTE_HASHES = True

# Row separator
C_SEP_S = "#---------------------------------------------------------\n"

#====================================================================
# Code
#====================================================================
# Factory that defines the name and details of the module and allows Autopsy
# to create instances of the modules that will do the analysis.
class FDRIModuleFactory(IngestModuleFactoryAdapter):

    moduleName = "FDRI"
    moduleVersion = "V1.0"

    #--------------------------------------------
    # Class variables
    # The variables are shared among the various
    # threads that might run the module.
    # (Autopsy creates several threads to process
    # data sources with a FileIngest module)
    #--------------------------------------------
    # Register start time
    g_start_time = time.time()
    g_elapsed_time_secs = -1 # (impossible value)

    def getModuleDisplayName(self):
        return self.moduleName

    def getModuleDescription(self):
        return "Facial Detection and Recognition in Images"

    def getModuleVersionNumber(self):
        return self.moduleVersion

    def isDataSourceIngestModuleFactory(self):
        return True

    def createDataSourceIngestModule(self, ingestOptions):
        return FDRIModule(self.settings)

    def hasIngestJobSettingsPanel(self):
        return True
    
    def hasGlobalSettingsPanel(self):
        return True

    def getGlobalSettingsPanel(self):
        return UIGlobalSettingsPanel()

    def getDefaultIngestJobSettings(self):
        return UISettings()

    def getIngestJobSettingsPanel(self, settings):
        self.settings = settings
        return UISettingsPanel(self.settings)

# Data Source-level ingest module.  One gets created per data source.
class FDRIModule(DataSourceIngestModule):

    _logger = Logger.getLogger(FDRIModuleFactory.moduleName)

    def log(self, level, msg):
        self._logger.logp(level, self.__class__.__name__,
                          inspect.stack()[1][3], msg)

    def __init__(self, settings):
        self.context = None
        self.localSettings = settings
        self.extensions = []
        self.deleteAfter = False
        self.doRecognition = True
        self.userPaths = {
            "0": "", 
            "1": "", 
            "2": ""
        }

        # True to create the DFXML file
        self.createDFXML    = C_CREATE_DFXML

        # Time acumulators to determine the 
        # cumulative time needed to compute
        # MD5, SHA1 and SHA256
        self.needed_time_MD5    = 0.0
        self.needed_time_SHA1   = 0.0
        self.needed_time_SHA256 = 0.0

        #CONFIGURATION_PATH = Case.getCurrentCase().getModuleDirectory() + "\\FDRI.json"
        
        # Error list in acordance with .exe code
        # for unknow errors please run executable via command line
        self.errorList = {
            1: ' FDRI.exe Parameters error',
            2: ' Error loading parameter file ',
            3: ' Error parsing parameter file ',
            4: ' Error finding image directory ',
            5: ' Error initializing recognition network ',
            6: ' Error initializing shape predictor ',
            7: ' Error initializing detection network ',
            8: ' Didn\'t find any positive faces ',
            9: ' Didn\'t find any target faces ',
            10: ' CUDA out of memory ',
            11: ' Didn\'t find any usable CUDA devices '
        }

    # Where any setup and configuration is done
    # 'context' is an instance of org.sleuthkit.autopsy.ingest.IngestJobContext.
    # See: http://sleuthkit.org/autopsy/docs/api-docs/4.4/classorg_1_1sleuthkit_1_1autopsy_1_1ingest_1_1_ingest_job_context.html
    def startUp(self, context):
        # Supported file format from dlib
        acceptedFiles = ['.jpg', '.jpeg', '.png']
        i = 0
        for ext in acceptedFiles:
            if self.localSettings.getFlag(i):
                self.extensions.append(ext)
            i += 1

        if not self.extensions:
            raise IngestModuleException(
                "Need to select at least one type of file!")

        # self.generate_hash controls whether MD5,SHA1 and SHA256 hashes
        # are added to the DFXML generated file.
        if self.localSettings.getFlag(3):
            self.generate_hash = True
        else:
            self.generate_hash = False

        #
        # Checking for default detectors and auxiliary files
        #
        self.pathToExe = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "FDRI.exe")

        self.defaultPaths = {
            '0': os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "mmod_human_face_detector.dat"),
            '1': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            "dlib_face_recognition_resnet_model_v1.dat"),
            '2': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                            "shape_predictor_5_face_landmarks.dat")
        }

        save_file = False
        if os.path.exists(GLOBAL_CONFIGURATION_PATH):
            with open(GLOBAL_CONFIGURATION_PATH, "r") as out:
                content = json.load(out)
                save_file = content['save_files']
                self.userPaths = content['paths']

        for code in self.userPaths:
            if not self.userPaths[code]:
                    # user didn't set models path, we assume module location
                    self.userPaths[code] = self.defaultPaths[code]
        
        # Update global config file
        #self.log(Level.INFO, GLOBAL_CONFIGURATION_PATH)
        with open(GLOBAL_CONFIGURATION_PATH, "w") as out:
            json.dump({"save_files": save_file,
                       "paths": self.userPaths}, out)

            folder_positive_photos = self.localSettings.getPath("1")

            # No folder for positive photos was given: recognition OFF
            if len(folder_positive_photos) == 0:
                self.doRecognition = False
                Msg_S = "Face recognition OFF (no folder with positive photo(s) given)"
                self.log(Level.INFO,Msg_S)
            elif not os.path.exists(folder_positive_photos):
                # The folder with positive photo doesn' exist: recognition will
                # be OFF
                self.doRecognition = False
                Msg_S = "Folder with positive photos NOT found: '%s'" %\
                                                (folder_positive_photos)
                self.log(Level.WARNING,Msg_S)
            else:
                # Ok, recognition is ON
                self.doRecognition = True
                Msg_S = "Face recognition ON (folder positive photo(s):'%s')"%\
                                                (folder_positive_photos)
                self.log(Level.INFO,Msg_S)

                
                with open(Case.getCurrentCase().getModuleDirectory() + "\\config.json", 'w') as safe_file:
                    json.dump({"flags": self.localSettings.getAllFlags(), "wanted_folder": folder_positive_photos}, safe_file)

        # Activate for DEBUG
        #with open(CONFIGURATION_PATH, "w") as out:
        #    json.dump({"flags": self.localSettings.getAllFlags(),
        #               "paths": self.localSettings.getAllPaths()}, out)
        self.context = context


    #--------------------------------------------------------------------
    # Added by Patricio 
    # 2018-07-21
    #--------------------------------------------------------------------
    def shutDown(self):
        """shutdown code"""
        # Elaspsed time
        FDRIModuleFactory.g_elapsed_time_secs = time.time() -\
                                        FDRIModuleFactory.g_start_time

        Log_S = "Total elapsed time: %f secs" %\
                (FDRIModuleFactory.g_elapsed_time_secs)
        self.log(Level.INFO, Log_S)


    # Where the analysis is done.
    # The 'dataSource' object being passed in is of type org.sleuthkit.datamodel.Content.
    # See: http://www.sleuthkit.org/sleuthkit/docs/jni-docs/4.4.1/interfaceorg_1_1sleuthkit_1_1datamodel_1_1_content.html
    # 'progressBar' is of type org.sleuthkit.autopsy.ingest.DataSourceIngestModuleProgress
    # See: http://sleuthkit.org/autopsy/docs/api-docs/4.4/classorg_1_1sleuthkit_1_1autopsy_1_1ingest_1_1_data_source_ingest_module_progress.html
    def process(self, dataSource, progressBar):

        # we don't know how much work there is yet
        progressBar.switchToIndeterminate()

        # Start timer for file copy operation
        start_copy_time = time.time()

        # case insensitive SQL LIKE clause is used to query the case database
        # FileManager API: http://sleuthkit.org/autopsy/docs/api-docs/4.4.1/classorg_1_1sleuthkit_1_1autopsy_1_1casemodule_1_1services_1_1_file_manager.html
        fileManager = Case.getCurrentCase().getServices().getFileManager()

        files = []
        for extension in self.extensions:
            try:
                files.extend(fileManager.findFiles(
                    dataSource, "%" + extension))
            except TskCoreException:
                self.log(Level.INFO, "Error getting files from: '" +
                         extension + "'")

        numFiles = len(files)
        if not numFiles:
            self.log(Level.WARNING, "Didn't find any usable files!")
            return IngestModule.ProcessResult.OK

        # Check if the user pressed cancel while we were busy
        if self.context.isJobCancelled():
            return IngestModule.ProcessResult.OK

        output_dir = Case.getCurrentCase().getModuleDirectory()
        module_dir = os.path.join(output_dir,dataSource.getName(),C_FDRI_DIR)
        
        # Create top-level DIR to save FDIR's created files
        full_dirname_dataSource = os.path.join(output_dir,dataSource.getName())
        if not os.path.exists(full_dirname_dataSource):
            os.mkdir(full_dirname_dataSource)

        # TEMP is needed by Autopsy
        temp_dir = os.path.join(Case.getCurrentCase().getTempDirectory(),
                                                        dataSource.getName())
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        temp_dir = os.path.join(temp_dir, C_FDRI_DIR)
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)

        # We always copy the files (except if a copy already exists)
        # as we will want to change them.
        # We detect the existence of a previous copy if the creation of the dir
        # 'module_dir' triggers an exception
        try:
            os.mkdir(module_dir)
        except:
            self.log(Level.INFO, "Directory already exists for this module")


        #----------------------------------------
        # Init file which holds filenames + size 
        #----------------------------------------
        file_path = os.path.join(module_dir,C_FILE_WITH_FNAMES_AND_SIZES)
        fnames_and_sizes_F = open(file_path,"w") 
        fnames_and_sizes_F.write(C_SEP_S)
        fnames_and_sizes_F.write("# Filename:size (bytes)\n") 
        timestamp_S = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        fnames_and_sizes_F.write("# START: %s\n" % (timestamp_S))
        fnames_and_sizes_F.write(C_SEP_S)

        # Dict to detect identical files
        files_hash_D = {}

        # Flag to record whether files were copied or not
        were_files_copied = True

        # Minimum size (in bytes) for an image file to be processed
        total_files = 0 
        total_small_files = 0

        # A initial version mispelled 'Annotated"...
        avoid_prefix_1 = "Anotated_"
        avoid_prefix_2 = "Annotated_"
        try:
            dir_img = os.path.join(module_dir,"img") 
            os.mkdir(dir_img)
            
            dir_small_files = os.path.join(module_dir,"small_files") + "\\"
            os.mkdir(dir_small_files)

            for file in files:
                total_files = total_files + 1

                filename_S = file.getName()
                Log_S = ""
                if filename_S.find(avoid_prefix_1) is 0:
                    Log_S = "%s file found '%s': skipping" %\
                                    (avoid_prefix_1,filename_S)
                elif filename_S.find(avoid_prefix_2) is 0:
                    Log_S = "%s file found '%s': skipping" %\
                                    (avoid_prefix_2,filename_S)
                if len(Log_S):
                    # Annotated_ found
                    # Log and skip this file
                    self.log(Level.INFO, Log_S)
                    continue

                file_size = file.getSize()
                filename, file_extension = os.path.splitext(file.getName())
                # Record filename and file size in C_FILE_WITH_FNAMES_AND_SIZES
                fnames_and_sizes_F.write("%s:%d\n" %(file.getName(),file_size))

                # If file size is more than C_FILE_MIN_SIZE
                # TODO:: User Choice as option
                if file_size >= C_FILE_MIN_SIZE:
                    new_fname = "%s__id__%s%s" %\
                            (filename,str(file.getId()),file_extension)
                    fullpath_dest = os.path.join(dir_img,new_fname)
                    ContentUtils.writeToFile(file, File(fullpath_dest))


                # We copy small files to a different DIR, so that we
                # can look at them, if needed
                if file_size < C_FILE_MIN_SIZE:
                    total_small_files = total_small_files + 1

                    dest_filename = "%s%s__id__%d%s" %\
                      (dir_small_files,filename,file.getId(),file_extension)
                    ContentUtils.writeToFile(file, File(dest_filename))

                    Log_S = "Skipping file: %s (%d bytes)" %\
                                        (file.getName(),file.getSize())
                    # LOG
                    self.log(Level.INFO, Log_S)

                #--------------------------------
                # Code to detect repeated files
                # We simply use a dictionary 
                # keyed by the MD5 of the file
                # Patricio
                #--------------------------------
                if file_size > 0:
                    md5_hash = self.create_hash(file, "md5")
                    if md5_hash in files_hash_D:
                        # hash already exists: repetition
                        files_hash_D[md5_hash].append(file.getName())
                    else:
                        # hash doesn't yet exist in dictionary: 1st time
                        files_hash_D[md5_hash] = [file_size,file.getName()]
                        
        ##except:
        except Exception, e:            
            were_files_copied = False
            self.log(Level.INFO,"Image folder already exists, skiping file copy")
            self.log(Level.INFO,"Exception: " + str(e))


        #----------------------------------------
        # Close filename+size file
        # Patricio
        #----------------------------------------
        timestamp_S = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        fnames_and_sizes_F.write("# DONE: %s\n" % (timestamp_S))
        if were_files_copied is False:
            Msg_S = "# Exception occurred\n"
            fnames_and_sizes_F.write(Msg_S)
        fnames_and_sizes_F.close()

        #----------------------------------------
        # Dump hash with repeated files
        # (only if files were copied)
        #----------------------------------------
        if were_files_copied is True:
            file_path = os.path.join(module_dir,C_REPEATED_FILES_LOG)
            repeated_files_log_F = open(file_path,"w") 
            repeated_files_log_F.write(C_SEP_S)
            repeated_files_log_F.write("# Repeated files\n")
            timestamp_S = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
            repeated_files_log_F.write("# %s\n" % (timestamp_S))
            repeated_files_log_F.write(C_SEP_S)


            for key, info_L in files_hash_D.iteritems():
                if len(info_L) > 2:
                    # only list with more than 2 entries 
                    # (one entry is the file size)
                    S = ""
                    for datum in info_L:
                        S = "%s%s:" % (S,datum)
                    repeated_files_log_F.write("%s\n" %(S))

            repeated_files_log_F.write(C_SEP_S)
            repeated_files_log_F.write("# DONE: %s\n" % (timestamp_S))
            repeated_files_log_F.write(C_SEP_S)
            repeated_files_log_F.close()

        #----------------------------------------
        # Log stats
        #----------------------------------------
        # shutdown copy file timer
        elapsed_copy_time_secs = time.time() - start_copy_time

        Log_S = "%d image files (%d of these were left out -- size <= "\
                "%d bytes)" % (total_files, total_small_files,C_FILE_MIN_SIZE)
        self.log(Level.INFO, Log_S)
        total_copied_files = total_files - total_small_files
        Log_S = "Files copy operation (%d files) took %f secs" %\
                (total_copied_files,elapsed_copy_time_secs)
        self.log(Level.INFO, Log_S)

        #----------------------------------------
        # Start processing timer
        #----------------------------------------
        start_FDRIexe_time = time.time()

        # Location where the output of executable will appear
        timestamp = datetime.now().strftime('%Y-%m-%d_%Hh%Mm%Ss')
        workspace = os.path.join(module_dir,timestamp)
        configFilePath = os.path.join(workspace,C_PARAMS_JSON_FNAME)

        os.mkdir(workspace)

        with open(configFilePath, "w") as out:
            json.dump({
                "paths": self.userPaths,#self.localSettings.getAllPaths(),
                "wanted_faces" : self.localSettings.getPath("1"),
                "imagesPath": os.path.join(module_dir,"img"),
                "doRecognition": self.doRecognition,
                "workspace": workspace,
            }, out)

        #
        # Different calls can also be provided to specify the image size
        #
        # Note that 2GB of GPU memory handle around 2000*2000 images
        # Note that 4GB of GPU memory handle around 3500*3500 images
        # Note that 8GB of GPU memory handle around 6000*6000 images
        #
        # Example:
        #                                                   Required    Minimum size Maximum size
        # target=lambda: self.thread_work(self.pathToExe, configFilePath, 1200*1200, 2000*2000))
        # target=lambda: self.thread_work(self.pathToExe, configFilePath, 1200*1200))
        executable_thread = Thread(
            target=lambda: self.thread_work(self.pathToExe, configFilePath))
        executable_thread.start()

        while(executable_thread.isAlive()):
            if self.context.isJobCancelled():
                self.log(Level.INFO, "User cancelled job! Terminating thread")
                JThread.interrupt(executable_thread)
                self.log(Level.INFO, "Thread terminated")
                self.deleteFiles(module_dir)
                return IngestModule.ProcessResult.OK
            time.sleep(1)

        # Checking if cancel was pressed before starting another job
        if self.context.isJobCancelled():
            return IngestModule.ProcessResult.OK

        #----------------------------------------
        # Compute time takne by FDRI.exe
        #----------------------------------------
        elapsed_FDRIexe_time_secs = time.time() - start_FDRIexe_time
        Log_S = "Process of image files by FDRI.exe took %f secs" %\
                (elapsed_FDRIexe_time_secs)
        self.log(Level.INFO, Log_S)

        #----------------------------------------
        # Start timer of last stage
        #----------------------------------------
        self.log(Level.INFO, "START of last stage")
        start_last_stage_time = time.time()

 
        # Use blackboard class to index blackboard artifacts for keyword search
        blackboard = Case.getCurrentCase().getServices().getBlackboard()

        # Tag files with faces
        artifact_type = BlackboardArtifact.ARTIFACT_TYPE.TSK_INTERESTING_FILE_HIT

        # Copy files from workspace to temp_dir
        tree_source      = os.path.join(workspace, C_ANNOTATED_DIR)
        tree_destination = os.path.join(temp_dir, C_ANNOTATED_DIR)
        copy_tree(tree_source,tree_destination)

        # Add images with the wanted faces to blackboard
        outPositiveFile = os.path.join(workspace,C_FDRI_WANTED_FNAME)
        if os.path.exists(outPositiveFile):
            with open(outPositiveFile, "r") as out:
                for line in out:
                    file_id = line.split('__id__')[1].split('.')
                    interestingFile = self.findByID(files, file_id[0])

                    if interestingFile == None:
                        continue

                    # Creating new artifacts with faces found
                    artifactList = interestingFile.getArtifacts(artifact_type)
                    if artifactList:
                        self.log(
                            Level.INFO, "Artifact already exists! ignoring")
                    else:
                        art = interestingFile.newArtifact(artifact_type)
                        att = BlackboardAttribute(BlackboardAttribute.ATTRIBUTE_TYPE.TSK_SET_NAME.getTypeID(),
                                                  FDRIModuleFactory.moduleName, dataSource.getName() + "/Wanted faces")
                        art.addAttribute(att)
                        try:
                            # index the artifact for keyword search
                            blackboard.indexArtifact(art)
                        except Blackboard.BlackboardException as e:
                            self.log(
                                Level.SEVERE, "Error indexing artifact " + art.getDisplayName())

                        # Adding derivated files to case
                        # These are files with borders on the found faces
                        # Code to deal with filenames with multiple "."
                        # Patricio, 2018.08.09
                        interestingFName = interestingFile.getName()
                        try:
                            name, extension = self.split_fname(interestingFName)
                        except Exception, e:
                            Err_S = "Error in splitting name/extension of '%s' (skipping file)" % (interestingFName)
                            self.log(Level.SEVERE,Err_S)
                            self.log(Level.SEVERE,"Exception: " + str(e))
                            continue

                        # Still here? Good.
                        f_path = "%s__id__%s.%s" %\
                                (name,str(interestingFile.getId()),extension)

                        # We need path relative to temp folder for Autopsy API
                        f_temp_path = os.path.join("Temp",dataSource.getName(),
                                C_FDRI_DIR, C_ANNOTATED_DIR, f_path)
                        f_abs_path = os.path.join(workspace,
                                            C_ANNOTATED_DIR, f_path)

                        # Temporary fix
                        if os.path.exists(f_abs_path):
                            f_size = os.path.getsize(f_abs_path)
                            case = Case.getCurrentCase().getSleuthkitCase()

                            try:
                                abstract_f = case.getAbstractFileById(
                                    interestingFile.getId())

                                # https://sleuthkit.org/autopsy/docs/api-docs/4.4/classorg_1_1sleuthkit_1_1autopsy_1_1casemodule_1_1services_1_1_file_manager.html
                                label_S = C_ANNOTATED_LABEL + interestingFName
                                case.addDerivedFile(label_S, f_temp_path, 
                                       f_size, 0, 0, 0, 0, True, abstract_f,
                                       "", FDRIModuleFactory.moduleName, 
                                       FDRIModuleFactory.moduleVersion, 
                                       "Image with faces",
                                       TskData.EncodingType.NONE)
                            except:
                                self.log(Level.SEVERE,"Error getting abs file")

                    if self.generate_hash:
                        dfxml_path = os.path.join(workspace,C_DFXML_FNAME)
                        self.complete_dfxml( dfxml_path, interestingFile)

        # Name of file that holds the data regarding detected faces
        # Each row corresponds to a detected face
        outPositiveFile = os.path.join(workspace,C_FACES_FOUND_FNAME)
        self.log(Level.INFO,"File with found faces from FDRI.exe:'%s'" %\
                (outPositiveFile))

        # Count the number of images where at least one face was detected
        images_with_faces_count = 0

        if os.path.exists(outPositiveFile):
            with open(outPositiveFile, "r") as out:
                for line in out:
                    # Another file with at least one face
                    images_with_faces_count += 1
                    file_id = line.split('__id__')[1].split('.')
                    interestingFile = self.findByID(files, file_id[0])

                    if interestingFile == None:
                        continue

                    # Creating new artifacts with faces found
                    artifactList = interestingFile.getArtifacts(artifact_type)
                    if artifactList:
                        self.log(Level.INFO,"Artifact already exists! ignoring")
                    else:
                        art = interestingFile.newArtifact(artifact_type)
                        att = BlackboardAttribute(BlackboardAttribute.ATTRIBUTE_TYPE.TSK_SET_NAME.getTypeID(),
                                                  FDRIModuleFactory.moduleName, dataSource.getName() + "/Images with faces")
                        art.addAttribute(att)
                        try:
                            # index the artifact for keyword search
                            blackboard.indexArtifact(art)
                        except Blackboard.BlackboardException as e:
                            self.log(Level.SEVERE, 
                             "Error indexing artifact " + art.getDisplayName())

                        # Adding derivated files to case
                        # These are files with borders on the found faces

                        # Code to deal with filenames with multiple "."
                        # Patricio, 2018.08.09
                        interestingFName = interestingFile.getName()
                        try:
                            name, extension = self.split_fname(interestingFName)
                        except Exception, e:
                            Err_S = "Error in splitting name/extension of '%s' (skipping file)" % (interestingFName)
                            self.log(Level.SEVERE,Err_S)
                            self.log(Level.SEVERE,"Exception: " + str(e))
                            continue

                        # Still here? Good.
                        f_path = "%s__id__%s.%s" %\
                                (name,str(interestingFile.getId()),extension)

                        # We need path relative to temp folder since the 
                        # Autopsy's API requires files in the case's 
                        # TEMP folder
                        f_temp_path = os.path.join("Temp",dataSource.getName(),
                                C_FDRI_DIR, C_ANNOTATED_DIR, f_path)
                        f_abs_path = os.path.join(workspace, 
                                        C_ANNOTATED_DIR, f_path)

                        # Temporary fix
                        if os.path.exists(f_abs_path):
                            f_size = os.path.getsize(f_abs_path)
                            case = Case.getCurrentCase().getSleuthkitCase()

                            try:
                                abstract_f = case.getAbstractFileById(
                                                    interestingFile.getId())

                                # https://sleuthkit.org/autopsy/docs/api-docs/4.4/classorg_1_1sleuthkit_1_1autopsy_1_1casemodule_1_1services_1_1_file_manager.html
                                label_S = C_ANNOTATED_LABEL + interestingFName
                                case.addDerivedFile(label_S, f_temp_path, 
                                        f_size, 0, 0, 0, 0, True, abstract_f, 
                                        "", FDRIModuleFactory.moduleName, 
                                        FDRIModuleFactory.moduleVersion, 
                                        "Image with faces",
                                        TskData.EncodingType.NONE)
                            except:
                                self.log(Level.SEVERE,
                                         "Error getting abs file")
                    if self.generate_hash:
                        dfxml_path = workspace + "\\" + C_DFXML_FNAME
                        self.complete_dfxml(dfxml_path, interestingFile)

        #----------------------------------------
        # End timer of last stage
        #----------------------------------------
        last_stage_time = time.time() - start_last_stage_time
        Log_S = "Last stage took %f secs" % (last_stage_time)
        self.log(Level.INFO, Log_S)

        if C_COMPUTE_HASHES:
            Log_S = "hashes took: MD5=%f secs; SHA1=%f secs; SHA256=%f secs" %\
             (self.needed_time_MD5, self.needed_time_SHA1, self.needed_time_SHA256)
        else:
            Log_S = "hashes NOT computed"
        self.log(Level.INFO, Log_S)

        IngestServices.getInstance().fireModuleDataEvent(
            ModuleDataEvent(FDRIModuleFactory.moduleName,
             BlackboardArtifact.ARTIFACT_TYPE.TSK_INTERESTING_FILE_HIT, None))

        # Should we delete the IMG files? (user's configuration)
        if self.deleteAfter:
            Msg_S = "Going to delete image files (as required by the user)"
            self.log(Level.INFO,Msg_S)

            dir_to_del = os.path.join(output_dir,dataSource.getName())
            self.deleteFiles(dir_to_del)


        # End time measurement
        FDRIModuleFactory.g_elapsed_time_secs = time.time() -\
                                        FDRIModuleFactory.g_start_time

        #------------------------------------------------------------
        # Format message to be shown at "Ingest messages"
        #------------------------------------------------------------
        if self.doRecognition:
            recognition_S = "ON"
        else:
            recognition_S = "OFF"
            
        ingest_msg_S = "Found %d images with faces: %f secs (FDRI.exe:%f secs). Recognition:%s" %\
                (images_with_faces_count, FDRIModuleFactory.g_elapsed_time_secs,
                        elapsed_FDRIexe_time_secs, recognition_S)

        message = IngestMessage.createMessage( IngestMessage.MessageType.DATA,
                FDRIModuleFactory.moduleName, ingest_msg_S)
        IngestServices.getInstance().postMessage(message)

        return IngestModule.ProcessResult.OK

    #==========================================================================
    # Helper functions
    #==========================================================================
    # File cleanup
    def deleteFiles(self, path):
        # ignoring the error if the directory is empty
        shutil.rmtree(path, ignore_errors=True)

    # Subprocess initiator
    def thread_work(self, path, param_path, min_size=0, max_size=0):

        sub_args = [path, "--params", param_path]
        if min_size > 0:
            sub_args.extend(["--min", str(min_size)])

        if max_size > 0:
            sub_args.extend(["--max", str(max_size)])

        returnCode = subprocess.call(sub_args)
        if returnCode:
            Err_S = "Error in executable: got '%s'" % (str(returnCode))
            self.log(Level.SEVERE,Err_S)
            if returnCode <= len(self.errorList) and returnCode > 0:
                self.log(Level.SEVERE, self.errorList[returnCode])
        else:
            msg_S = "Child process FDRI.exe terminated with no problems"
            self.log(Level.INFO, msg_S)

    #----------------------------------------------------------------
    # File mapping from temp folder to Autopsy files
    #----------------------------------------------------------------
    def findByID(self, file_list, id):
        for file in file_list:
            if file.getId() == int(id):
                return file

        return None

    #----------------------------------------------------------------
    # Complete the DFMXL file, adding the hashes 
    # (MD5, SHA1 and SHA256) of each individual file.
    #----------------------------------------------------------------
    def complete_dfxml(self, dfxml_path, file):
        xml_doc = None

        # Should we compute hashes? 
        do_compute_hashes = C_COMPUTE_HASHES

        with open(dfxml_path, "r") as dfxml:
            xml_doc = m_dom.parse(dfxml)
            file_elements = xml_doc.getElementsByTagName("fileobject")

            if file.isFile() and file.canRead():
                #----------------------
                # Append file hashes
                #----------------------
                # SHA1
                hash_nodeSHA1 = xml_doc.createElement("hashdigest")
                if do_compute_hashes:
                    (sha1_hash,time_used) = self.create_hash(file, "sha1")
                    self.needed_time_SHA1 = self.needed_time_SHA1 + time_used
                else:
                    sha1_hash = "0"
                hash_nodeSHA1.setAttribute("type", "sha1")
                hash_nodeSHA1.appendChild(xml_doc.createTextNode(sha1_hash))

                # SHA256
                hash_nodeSHA256 = xml_doc.createElement("hashdigest")
                if do_compute_hashes:
                    (sha256_hash,time_used) = self.create_hash(file, "sha256")
                    self.needed_time_SHA256 = self.needed_time_SHA256+time_used
                else:
                    sha256_hash = "0"
                hash_nodeSHA256.setAttribute("type", "sha256")
                hash_nodeSHA256.appendChild(xml_doc.createTextNode(sha256_hash))

                # MD5
                hash_nodeMD5 = xml_doc.createElement("hashdigest")
                if do_compute_hashes:
                    (md5_hash,time_used) = self.create_hash(file, "md5")
                    self.needed_time_MD5 = self.needed_time_MD5 + time_used
                else:
                    md5_hash = "0"
                hash_nodeMD5.setAttribute("type", "md5")
                hash_nodeMD5.appendChild(xml_doc.createTextNode(md5_hash))

                for element in file_elements:
                    file_name_node=element.getElementsByTagName("filename")[0]
                    if file_name_node.firstChild.nodeValue == file.getName().split(".")[0]:
                        element.appendChild(hash_nodeSHA1)
                        element.appendChild(hash_nodeSHA256)
                        element.appendChild(hash_nodeMD5)

        with open(dfxml_path, "w") as out:
            xml_doc.writexml(out, encoding="utf-8")

    #----------------------------------------------------------------
    # Hash calculation, Autopsy seems to not provide these
    #----------------------------------------------------------------
    def create_hash(self, f_target, algorithm):
        time_start = time.time()

        hash_creator = hashlib.new(algorithm)

        inputStream = ReadContentInputStream(f_target)
        buffer = jarray.zeros(BLOCKSIZE, "b")
        len = inputStream.read(buffer)

        while (len != -1):
            hash_creator.update(buffer)
            len = inputStream.read(buffer)

        time_consumed = time.time()-time_start
        return (hash_creator.hexdigest(),time_consumed)




    #--------------------------------------------------------------------
    # Split filename into basename and extension
    #--------------------------------------------------------------------
    def split_fname(self, fname):
        if fname is None or len(fname)==0:
            return "",""
        fname_L = fname.split(".")
        fname_L_len = len(fname_L)
        if fname_L_len == 1:
            # No extension
            return fname,""
        # Still here?
        if fname_L_len == 2:
            # simple: name + extension
            return fname_L[0],fname_L[1]
        else:
            name_S = ""
            for elm in fname_L[:-1]:
                name_S = name_S + "." + elm
            return name_S,fname_L[fname_L_len-1]


#----------------------------------------------------------------------
# Global settings UI class, responsible for AI models weights location
# This is case independent
#----------------------------------------------------------------------
class UIGlobalSettingsPanel(IngestModuleGlobalSettingsPanel):

    def __init__(self):
        self.save_file_cbox = JCheckBox(C_LABEL_INFO_AUTOPSY_TEMP)
        ## "Save copied images outside of AUTOPSY's temp"

        self.textInputs = {
            '0': JTextField('', 30),
            '1': JTextField('', 30),
            '2': JTextField('', 30)
        }

        self.buttons = {
            '0': JButton("Choose file", actionPerformed=self.chooseFolder),
            '1': JButton("Choose file", actionPerformed=self.chooseFolder),
            '2': JButton("Choose file", actionPerformed=self.chooseFolder)
        }
        
        self.initComponents()
        self.load()

    def checkBoxEvent(self, event):
        self.save_files = self.save_file_cbox.isSelected()

    def saveSettings(self):
        all_paths = {}
        for code in self.textInputs:
            all_paths[code] = self.textInputs[code].text
        
        with open(GLOBAL_CONFIGURATION_PATH, "w") as out:
            json.dump({
                "save_files": self.save_file_cbox.isSelected(),
                "paths": all_paths
            }, out)

    def load(self):
        # Load settings from file
        if os.path.exists(GLOBAL_CONFIGURATION_PATH):
            with open(GLOBAL_CONFIGURATION_PATH, "r") as out:
                content = json.load(out)
                # self.save_file_cbox.setSelected(content['save_files'])
                self.textInputs['0'].text = content['paths']['0']
                self.textInputs['1'].text = content['paths']['1']
                self.textInputs['2'].text = content['paths']['2']
                

    def chooseFolder(self, e):
        button = e.getSource()
        code = button.getActionCommand()
        fileChooser = JFileChooser()
        fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY)

        ret = fileChooser.showDialog(self, "Choose folder")
        if ret == JFileChooser.APPROVE_OPTION:
            ff = fileChooser.getSelectedFile()
            path = ff.getCanonicalPath()
            self.textInputs[code].text = path

    def initComponents(self):
        self.setLayout(None)
        self.setPreferredSize(Dimension(500, 400))

        lblNewLabel = JLabel("Detector model path:")
        lblNewLabel.setBounds(45, 144, 227, 16)
        self.add(lblNewLabel)

        lblNewLabel_1 = JLabel("Recognition model path:")
        lblNewLabel_1.setBounds(44, 210, 228, 16)
        self.add(lblNewLabel_1)

        lblNewLabel_2 = JLabel("Shape predictor model path:")
        lblNewLabel_2.setBounds(45, 275, 227, 16)
        self.add(lblNewLabel_2)

        self.textInputs['0'].setBounds(44, 173, 228, 22)
        self.textInputs['0'].setColumns(30)
        self.add(self.textInputs['0'])
        
        self.textInputs['1'].setColumns(30)
        self.textInputs['1'].setBounds(44, 238, 228, 22)
        self.add(self.textInputs['1'])

        self.textInputs['2'].setColumns(30)
        self.textInputs['2'].setBounds(45, 304, 228, 22)
        self.add(self.textInputs['2'])

        self.buttons['0'].setBounds(284, 172, 97, 25)
        self.buttons['0'].setActionCommand("0")
        self.add(self.buttons['0'])

        self.buttons['1'].setBounds(284, 237, 97, 25)
        self.buttons['1'].setActionCommand("1")
        self.add(self.buttons['1'])

        self.buttons['2'].setBounds(284, 303, 97, 25)
        self.buttons['2'].setActionCommand("2")
        self.add(self.buttons['2'])

        self.save_file_cbox = JCheckBox(C_LABEL_INFO_AUTOPSY_TEMP)
        self.save_file_cbox.setBounds(45, 98, 300, 25)
        self.add(self.save_file_cbox)


#----------------------------------------------------------------
# Case level settings object class
#----------------------------------------------------------------
class UISettings(IngestModuleIngestJobSettings):
    serialVersionUID = 1L

    def __init__(self):
        #             JPG   JPEG  PNG   Delete file after
        self.flags = [True, True, True, True]
        self.paths = {
            "1": ""
        }

    def getVersionNumber(self):
        return self.serialVersionUID

    def getFlag(self, pos):
        return self.flags[pos]

    def setFlag(self, flag, pos):
        self.flags[pos] = flag

    def setPath(self, code, path):
        self.paths[code] = path

    def getAllPaths(self):
        return self.paths

    def getAllFlags(self):
        return self.flags

    def getPath(self, code):
        return self.paths[code]

    def loadConfig(self):
        CONFIGURATION_PATH = Case.getCurrentCase().getModuleDirectory() + "\\config.json"
        if os.path.exists(CONFIGURATION_PATH):
            with open(CONFIGURATION_PATH, "r") as out:
                content = json.load(out)
                self.flags = content['flags']
                self.paths['1'] = content['wanted_folder']

#-------------------------------------------------------------
# Case level settings UI class
#-------------------------------------------------------------
class UISettingsPanel(IngestModuleIngestJobSettingsPanel):

    def __init__(self, settings):
        self.localSettings = settings
        self.buttons = {
            '1': JButton("Choose", actionPerformed=self.chooseFolder),
        }

        self.textInputs = {
            "1": JTextField('', 5),
        }

        self.initComponents()
        self.customizeComponents()

    def checkBoxEvent(self, event):
        self.localSettings.setFlag(self.checkboxJPG.isSelected(), 0)
        self.localSettings.setFlag(self.checkboxJPEG.isSelected(), 1)
        self.localSettings.setFlag(self.checkboxPNG.isSelected(), 2)
        self.localSettings.setFlag(self.chckbxGenerateImageHash.isSelected(), 3)

    def clear(self, e):
        button = e.getSource()
        code = button.getActionCommand()
        self.localSettings.setPath(code, "")
        self.textInputs[code].text = ""

    def chooseFolder(self, e):
        button = e.getSource()
        code = button.getActionCommand()
        fileChooser = JFileChooser()
        fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY)
        text = "Choose folder"

        ret = fileChooser.showDialog(self, text)
        if ret == JFileChooser.APPROVE_OPTION:
            ff = fileChooser.getSelectedFile()
            path = ff.getCanonicalPath()
            self.localSettings.setPath(code, path)
            self.textInputs[code].text = path

    def initComponents(self):
        self.setLayout(None)
		
        lblFileExtensionsTo = JLabel("File extensions to look for:")
        lblFileExtensionsTo.setBounds(43, 37, 161, 16)
        self.add(lblFileExtensionsTo)

        self.checkboxPNG = JCheckBox(".PNG", actionPerformed=self.checkBoxEvent)
        self.checkboxPNG.setBounds(43, 62, 72, 25)
        self.add(self.checkboxPNG)

        self.checkboxJPG = JCheckBox(".JPG", actionPerformed=self.checkBoxEvent)
        self.checkboxJPG.setBounds(43, 92, 72, 25)
        self.add(self.checkboxJPG)

        self.checkboxJPEG = JCheckBox(".JPEG", actionPerformed=self.checkBoxEvent)
        self.checkboxJPEG.setBounds(118, 62, 113, 25)
        self.add(self.checkboxJPEG)

        lblNewLabel = JLabel("Folder with images of person to find:")
        lblNewLabel.setBounds(43, 145, 223, 16)
        self.add(lblNewLabel)

        textField = self.textInputs['1']
        textField.setBounds(43, 167, 223, 22)
        self.add(textField)
        textField.setColumns(30)

        self.buttons['1'].setActionCommand("1")
        self.buttons['1'].setBounds(43, 195, 113, 25)
        self.add(self.buttons['1'])


        #TODO:: This no longer is required, it's done within the executable
        self.chckbxGenerateImageHash = JCheckBox("Generate image hash for DFXML")
        self.chckbxGenerateImageHash.setBounds(43, 239, 223, 25)
        self.add(self.chckbxGenerateImageHash)

    def customizeComponents(self):
        self.localSettings.loadConfig()
        self.checkboxJPG.setSelected(self.localSettings.getFlag(0))
        self.checkboxJPEG.setSelected(self.localSettings.getFlag(1))
        self.checkboxPNG.setSelected(self.localSettings.getFlag(2))
        self.chckbxGenerateImageHash.setSelected(self.localSettings.getFlag(3))

        for code in self.textInputs:
            self.textInputs[code].text = self.localSettings.getPath(code)

    def getSettings(self):
        return self.localSettings



