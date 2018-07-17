# -*- coding: utf-8 -*-

import inspect
import json
import os  # file checking
import shutil  # file copy
import subprocess  # .exe calling
import time
import signal
import hashlib
import xml.dom.minidom as m_dom
import jarray
from java.awt import BorderLayout, GridLayout, FlowLayout
from java.awt.event import KeyAdapter, KeyEvent, KeyListener
from threading import Thread

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
                                          IngestServices, ModuleDataEvent)
from org.sleuthkit.autopsy.ingest.IngestModule import IngestModuleException
from org.sleuthkit.datamodel import (AbstractFile, BlackboardArtifact,
                                     BlackboardAttribute, SleuthkitCase,
                                     TskData, ReadContentInputStream)

# Module conf file location
CONFIGURATION_PATH = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "configuration.json")

# Number of bytes to read on hashing function
BLOCKSIZE = 5120

# Factory that defines the name and details of the module and allows Autopsy
# to create instances of the modules that will do the analysis.


class FDRIModuleFactory(IngestModuleFactoryAdapter):

    moduleName = "FDRI"
    moduleVersion = "V1.0"

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
            11: ' Unknown error - run in command line for additional info '
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

        if self.localSettings.getFlag(3):
            self.deleteAfter = True

        #
        # Checking for default detectors and auxiliary files
        #
        self.pathToExe = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "FDRI.exe")

        self.defaultPaths = {
            "2": os.path.join(os.path.dirname(os.path.abspath(__file__)), "mmod_human_face_detector.dat"),
            "3": os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlib_face_recognition_resnet_model_v1.dat"),
            "4": os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_predictor_5_face_landmarks.dat")
        }

        userPaths = self.localSettings.getAllPaths()
        for code in userPaths:
            if not userPaths[code]:
                if code == "1":
                    self.doRecognition = False
                else:
                    # user didn't set models path, we assume module location
                    self.localSettings.setPath(code, self.defaultPaths[code])

            if code != "1" or self.doRecognition:
                # checking if any files are missing
                if not os.path.exists(userPaths[code]):
                    raise IngestModuleException(
                        "File not found: " + userPaths[code])

        with open(CONFIGURATION_PATH, "w") as out:
            json.dump({"flags": self.localSettings.getAllFlags(),
                       "paths": self.localSettings.getAllPaths()}, out)

        self.context = context

    # Where the analysis is done.
    # The 'dataSource' object being passed in is of type org.sleuthkit.datamodel.Content.
    # See: http://www.sleuthkit.org/sleuthkit/docs/jni-docs/4.4.1/interfaceorg_1_1sleuthkit_1_1datamodel_1_1_content.html
    # 'progressBar' is of type org.sleuthkit.autopsy.ingest.DataSourceIngestModuleProgress
    # See: http://sleuthkit.org/autopsy/docs/api-docs/4.4/classorg_1_1sleuthkit_1_1autopsy_1_1ingest_1_1_data_source_ingest_module_progress.html
    def process(self, dataSource, progressBar):

        # we don't know how much work there is yet
        progressBar.switchToIndeterminate()

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

        tempDir = Case.getCurrentCase().getTempDirectory()
        module_dir = tempDir + "\\" + dataSource.getName() + "\\FDRIV"
        if not os.path.exists(tempDir + "\\" + dataSource.getName()):
            os.mkdir(tempDir + "\\" + dataSource.getName())

        # We allways copy the files, as we will want to change them
        try:
            os.mkdir(module_dir)
            for file in files:
                #
                # If file size is more than 0
                # TODO:: User Choice as option
                #
                if file.getSize() > 1:
                    filename, file_extension = os.path.splitext(file.getName())
                    ContentUtils.writeToFile(file, File(
                        module_dir + "\\" + filename + "__id__" + str(file.getId()) + file_extension))
                else:
                    self.log(Level.INFO, "Skiping file: " + file.getName())
        except:
            self.log(
                Level.INFO, "Directory already exists for this data source skipping file copy")

        # Location where the output of executable will appear
        outFile = module_dir + "\\facesFound.txt"
        outDFxml = module_dir + "\\dfxml_" + FDRIModuleFactory.moduleName + ".xml"
        outPositiveFile = module_dir + "\\wanted.txt"

        # Some file cleanup before executing
        if os.path.exists(outFile):
            os.remove(outFile)

        if os.path.exists(outDFxml):
            os.remove(outDFxml)

        if os.path.exists(outPositiveFile):
            os.remove(outPositiveFile)

        configFilePath = module_dir + "\\params.json"

        with open(configFilePath, "w") as out:
            json.dump({
                "flags": self.localSettings.getAllFlags(),
                "paths": self.localSettings.getAllPaths(),
                "imagesPath": module_dir,
                "doRecognition": self.doRecognition,
                "outputPath": outFile,
                "outputPositivePath": outPositiveFile,
                "dfxml_out_path": outDFxml
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
        #  target=lambda: self.thread_work(self.pathToExe, configFilePath, 1200*1200))
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

        # Use blackboard class to index blackboard artifacts for keyword search
        blackboard = Case.getCurrentCase().getServices().getBlackboard()

        # Tag files with faces
        artifact_type = BlackboardArtifact.ARTIFACT_TYPE.TSK_INTERESTING_FILE_HIT

        count = 0
        # Add images with the wanted faces to blackboard
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
                        f_path = "Temp\\" + dataSource.getName() + "\\FDRIV\\" + \
                            interestingFile.getName().split(
                                ".")[0] + "__id__" + str(interestingFile.getId()) + ".jpg"

                        f_size = os.path.getsize(module_dir + "\\" + interestingFile.getName().split(
                            '.')[0] + "__id__" + str(interestingFile.getId()) + ".jpg")
                        case = Case.getCurrentCase().getSleuthkitCase()
                        try:
                            abstract_f = case.getAbstractFileById(
                                interestingFile.getId())

                            # https://sleuthkit.org/autopsy/docs/api-docs/4.4/classorg_1_1sleuthkit_1_1autopsy_1_1casemodule_1_1services_1_1_file_manager.html
                            case.addDerivedFile("Anotated_" + interestingFile.getName(),
                                                f_path, f_size, 0, 0, 0, 0, True, abstract_f, "",
                                                FDRIModuleFactory.moduleName, FDRIModuleFactory.moduleVersion, "Image with faces",
                                                TskData.EncodingType.NONE)
                            self.complete_dfxml(outDFxml, interestingFile)
                        except:
                            self.log(Level.SEVERE, "Error getting abs file")

                    self.complete_dfxml(outDFxml, interestingFile)

        if os.path.exists(outFile):
            with open(outFile, "r") as out:
                for line in out:
                    count += 1
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
                                                  FDRIModuleFactory.moduleName, dataSource.getName() + "/Images with faces")
                        art.addAttribute(att)
                        try:
                            # index the artifact for keyword search
                            blackboard.indexArtifact(art)
                        except Blackboard.BlackboardException as e:
                            self.log(
                                Level.SEVERE, "Error indexing artifact " + art.getDisplayName())

                        # Adding derivated files to case
                        # These are files with borders on the found faces
                        f_path = "Temp\\" + dataSource.getName() + "\\FDRIV\\" + \
                            interestingFile.getName().split(
                                ".")[0] + "__id__" + str(interestingFile.getId()) + ".jpg"

                        f_size = os.path.getsize(module_dir + "\\" + interestingFile.getName().split(
                            '.')[0] + "__id__" + str(interestingFile.getId()) + ".jpg")
                        case = Case.getCurrentCase().getSleuthkitCase()
                        try:
                            abstract_f = case.getAbstractFileById(
                                interestingFile.getId())

                            # https://sleuthkit.org/autopsy/docs/api-docs/4.4/classorg_1_1sleuthkit_1_1autopsy_1_1casemodule_1_1services_1_1_file_manager.html
                            case.addDerivedFile("Anotated_" + interestingFile.getName(),
                                                f_path, f_size, 0, 0, 0, 0, True, abstract_f, "",
                                                FDRIModuleFactory.moduleName, FDRIModuleFactory.moduleVersion, "Image with faces",
                                                TskData.EncodingType.NONE)

                        except:
                            self.log(Level.SEVERE, "Error getting abs file")
                    self.complete_dfxml(outDFxml, interestingFile)

        IngestServices.getInstance().fireModuleDataEvent(
            ModuleDataEvent(FDRIModuleFactory.moduleName, BlackboardArtifact.ARTIFACT_TYPE.TSK_INTERESTING_FILE_HIT, None))

        # User doesn't want to keep files
        if self.deleteAfter:
            self.deleteFiles(tempDir + "\\" + dataSource.getName())

        message = IngestMessage.createMessage(
            IngestMessage.MessageType.DATA, FDRIModuleFactory.moduleName, "Found %d images with faces" % count)

        IngestServices.getInstance().postMessage(message)

        return IngestModule.ProcessResult.OK

    #
    # Helper functions
    #

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
            self.log(Level.SEVERE, "Error in executable: got " + str(returnCode))
            if returnCode <= len(self.errorList) and returnCode > 0:
                self.log(Level.SEVERE, self.errorList[returnCode])
        else:
            self.log(Level.INFO, "Child process terminated with no problems")

    # File mapping from temp folder to Autopsy files

    def findByID(self, file_list, id):
        for file in file_list:
            if file.getId() == int(id):
                return file

        return None

    def complete_dfxml(self, dfxml_path, file):
        xml_doc = None
        with open(dfxml_path, "r") as dfxml:
            xml_doc = m_dom.parse(dfxml)
            file_elements = xml_doc.getElementsByTagName("fileobject")

            if file.isFile() and file.canRead():
                # Append file hash
                hash_nodeSHA1 = xml_doc.createElement("hashdigest")
                sha1_hash = self.create_hash(file, "sha1")
                hash_nodeSHA1.setAttribute("type", "sha1")
                hash_nodeSHA1.appendChild(xml_doc.createTextNode(sha1_hash))

                hash_nodeSHA256 = xml_doc.createElement("hashdigest")
                sha256_hash = self.create_hash(file, "sha256")
                hash_nodeSHA256.setAttribute("type", "sha256")
                hash_nodeSHA256.appendChild(
                    xml_doc.createTextNode(sha256_hash))

                hash_nodeMD5 = xml_doc.createElement("hashdigest")
                md5_hash = self.create_hash(file, "md5")
                hash_nodeMD5.setAttribute("type", "md5")
                hash_nodeMD5.appendChild(xml_doc.createTextNode(md5_hash))

                for element in file_elements:
                    file_name_node = element.getElementsByTagName("filename")[
                        0]
                    if file_name_node.firstChild.nodeValue == file.getName().split(".")[0]:
                        element.appendChild(hash_nodeSHA1)
                        element.appendChild(hash_nodeSHA256)
                        element.appendChild(hash_nodeMD5)

        with open(dfxml_path, "w") as out:
            xml_doc.writexml(out, encoding="utf-8")

    # Hash calculation, Autopsy seems to not provide these

    def create_hash(self, f_target, algorithm):
        hash_creator = hashlib.new(algorithm)

        inputStream = ReadContentInputStream(f_target)
        buffer = jarray.zeros(BLOCKSIZE, "b")
        len = inputStream.read(buffer)

        while (len != -1):
            hash_creator.update(buffer)
            len = inputStream.read(buffer)

        return hash_creator.hexdigest()

# UI class representation


class UISettings(IngestModuleIngestJobSettings):
    serialVersionUID = 1L

    def __init__(self):
        #             JPG   JPEG  PNG   Delete file after
        self.flags = [True, True, True, False]
        self.paths = {
            "1": "",
            "2": "",
            "3": "",
            "4": ""
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

    def loadConfig(self, path):
        if os.path.exists(path):
            with open(path, "r") as out:
                content = json.load(out)
                self.flags = content['flags']
                self.paths = content['paths']

# UI objects


class UISettingsPanel(IngestModuleIngestJobSettingsPanel):

    def __init__(self, settings):
        self.localSettings = settings
        self.modes = {
            '1': JFileChooser.DIRECTORIES_ONLY,
            '2': JFileChooser.FILES_ONLY,
            '3': JFileChooser.FILES_ONLY,
            '4': JFileChooser.FILES_ONLY
        }

        self.buttons = {
            '1': JButton("Choose", actionPerformed=self.chooseFolder),
            '2': JButton("Choose", actionPerformed=self.chooseFolder),
            '3': JButton("Choose", actionPerformed=self.chooseFolder),
            '4': JButton("Choose", actionPerformed=self.chooseFolder),
        }

        self.clear_buttons = {
            '1': JButton("Clear", actionPerformed=self.clear),
            '2': JButton("Clear", actionPerformed=self.clear),
            '3': JButton("Clear", actionPerformed=self.clear),
            '4': JButton("Clear", actionPerformed=self.clear),
        }

        self.textInputs = {
            "1": JTextField('', 5),
            "2": JTextField('', 5),
            "3": JTextField('', 5),
            "4": JTextField('', 5)
        }

        self.initComponents()
        self.customizeComponents()

    def checkBoxEvent(self, event):
        self.localSettings.setFlag(self.checkboxJPG.isSelected(), 0)
        self.localSettings.setFlag(self.checkboxJPEG.isSelected(), 1)
        self.localSettings.setFlag(self.checkboxPNG.isSelected(), 2)
        self.localSettings.setFlag(self.checkboxDelete.isSelected(), 3)

    def clear(self, e):
        button = e.getSource()
        code = button.getActionCommand()
        self.localSettings.setPath(code, "")
        self.textInputs[code].text = ""

    def chooseFolder(self, e):
        button = e.getSource()
        code = button.getActionCommand()
        fileChooser = JFileChooser()
        fileChooser.setFileSelectionMode(self.modes[code])
        text = "Choose folder"
        if code != "1":
            text = "Choose file"

        ret = fileChooser.showDialog(self.panel, text)
        if ret == JFileChooser.APPROVE_OPTION:
            ff = fileChooser.getSelectedFile()
            path = ff.getCanonicalPath()
            self.localSettings.setPath(code, path)
            self.textInputs[code].text = path

    def initComponents(self):
        self.setLayout(BoxLayout(self, BoxLayout.Y_AXIS))
        self.setAlignmentX(JComponent.LEFT_ALIGNMENT)
        self.panel = JPanel()
        self.panel.setLayout(BoxLayout(self.panel, BoxLayout.Y_AXIS))
        self.panel.setAlignmentY(JComponent.LEFT_ALIGNMENT)  

        self.labelTop = JLabel("Choose file extensions to look for:")
        self.labelTop.setAlignmentX(JComponent.LEFT_ALIGNMENT)
        self.checkboxJPG = JCheckBox(
            ".jpg", actionPerformed=self.checkBoxEvent)
        self.checkboxJPEG = JCheckBox(
            ".jpeg", actionPerformed=self.checkBoxEvent)
        self.checkboxPNG = JCheckBox(
            ".png", actionPerformed=self.checkBoxEvent)

        fl = JPanel()
        fl.setLayout(FlowLayout(FlowLayout.LEFT))
        fl.add(self.checkboxJPG)
        fl.add(self.checkboxJPEG)
        fl.add(self.checkboxPNG)
        self.panel.add(fl)

        self.labelBlank = JLabel(" ")
        self.panel.add(self.labelBlank)
        self.checkboxDelete = JCheckBox(
            "Delete files after use", actionPerformed=self.checkBoxEvent)
        fl = JPanel()
        fl.setLayout(FlowLayout(FlowLayout.LEFT))
        fl.add(self.checkboxDelete)
        self.panel.add(fl)

        self.panel.add(JLabel("Path to folder with faces to find"))
        self.panel.add(self.textInputs['1'])
        self.buttons['1'].setActionCommand("1")
        self.textInputs['1'].setEnabled(False)
        self.clear_buttons['1'].setActionCommand("1")
        fl = JPanel()
        fl.setLayout(FlowLayout(FlowLayout.LEFT))
        fl.add(self.buttons['1'])
        fl.add(self.clear_buttons['1'])
        self.panel.add(fl)


        self.panel.add(
            JLabel("Path to detector - Defaults to module path"))
                                       
        self.panel.add(self.textInputs['2'])
        self.buttons['2'].setActionCommand("2")
        self.textInputs['2'].setEnabled(False)
        self.clear_buttons['2'].setActionCommand("2")
        fl = JPanel()
        fl.setLayout(FlowLayout(FlowLayout.LEFT))
        fl.add(self.buttons['2'])
        fl.add(self.clear_buttons['2'])
        self.panel.add(fl)

        self.panel.add(
            JLabel("Path to recognitor - Defaults to module path"))
        self.panel.add(self.textInputs['3'])
        self.buttons['3'].setActionCommand("3")
        self.textInputs['3'].setEnabled(False)
        self.clear_buttons['3'].setActionCommand("3")
        fl = JPanel()
        fl.setLayout(FlowLayout(FlowLayout.LEFT))
        fl.add(self.buttons['3'])
        fl.add(self.clear_buttons['3'])
        self.panel.add(fl)

        self.panel.add(
            JLabel("Path to shape predictor - Defaults to module path"))
        self.panel.add(self.textInputs['4'])
        self.buttons['4'].setActionCommand("4")
        self.textInputs['4'].setEnabled(False)
        self.clear_buttons['4'].setActionCommand("4")
        fl = JPanel()
        fl.setLayout(FlowLayout(FlowLayout.LEFT))
        fl.add(self.buttons['4'])
        fl.add(self.clear_buttons['4'])
        self.panel.add(fl)

        
        self.add(self.panel)

    def customizeComponents(self):
        self.localSettings.loadConfig(CONFIGURATION_PATH)
        self.checkboxJPG.setSelected(self.localSettings.getFlag(0))
        self.checkboxJPEG.setSelected(self.localSettings.getFlag(1))
        self.checkboxPNG.setSelected(self.localSettings.getFlag(2))
        self.checkboxDelete.setSelected(self.localSettings.getFlag(3))

        for code in self.textInputs:
            self.textInputs[code].text = self.localSettings.getPath(code)

    def getSettings(self):
        return self.localSettings
