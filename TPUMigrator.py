import sys
import ast
from refactor import Session, Rule, common
from refactor.actions import Replace
from refactor.actions import InsertAfter
from pathlib import Path
import os

import logging

from GlobalLogger import initializeLogger,logger

from Rules import *
from Utils import ChangesCounter, CommentsDict, optimization_rules

default_logging_level = logging.INFO

class TPUMigrator:
  target_file = ""
  _path = ""
  def __init__(self, path, level = default_logging_level):
    self.target_file = Path(path)
    self._path = path
    initializeLogger(level)
    logger.info('Logger Initialized with level : ' + str(logger.level))

  def __str__(self):
    return f"Migration object for file {self.target_file}"

  def print(self):
    print("Migration object for file ", self.target_file)

  def setLoggingLevel(self, level = default_logging_level):
    initializeLogger(level)
    logger.info('Logger Initialized with level : ' + str(logger.level))

  #### Command for testcase generation######
  ##python C:\Users\kundu\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\pynguin\__main__.py --project-path ./Project --output-path ./Project --module-name example -v
  def generateTestCases(self, PYNGUINPATH):
    logger.info("Invoking Dynamosa to generate test cases...")
    if PYNGUINPATH == "":
      logger.error("PYNGUINPATH needs to be given")
      return
    command = "python " + PYNGUINPATH + '__main__.py '
    command = command + "--project-path ./Project --output-path ./Project --module-name "
    command = command + self._path.split(".")[0] + " -v --maximum-iterations 500"
    os.system(command)
  def execute(self, comments = True, reformatter = True, optimize = True, genTestCases = False, PYNGUINPATH=""):
    logger.info('Execution Started...')
    if genTestCases:
      self.generateTestCases(PYNGUINPATH)
    session = Session(rules=[CheckBaseTorchlibraryAlias, ImportBaseLibRule, ImportCoreLibRule, AddBaseRule, ModifyTensor_cudaSupported,
                             ModifyTensor_random, ModifyTensor_creation, ModifyTensor_generator, ModifyTensor_spectralOps,
                             ModifyTensor_indices, CheckNNlibraryAlias, StoreNNModuleClass, ModifyNNModuleClass,
                             ModifyTensor_nn_cnvLayer, ModifyTensor_nn_normalizationLayer, ModifyTensor_nn_rnnLayer,
                             ModifyTensor_nn_linearLayer, ModifyTensor_nn_embbLayer])
    # session = Session(rules=[CheckBaseTorchlibraryAlias, ImportBaseLibRule, ImportCoreLibRule, AddBaseRule, AddPerformanceOptimizationRule])

    change = session.run_file(self.target_file)
    if change != None:
      #print(change.compute_diff())
      logger.info('Applying changes...')
      change.apply_diff()
    else :
      logger.info('No changes to be applied...')
    logger.info('Code Migration complete...')

    if comments:
      self.addComments()

    if optimize:
      self.addOptimization()
      ChangesCounter().increment(3)

    if reformatter:
      self.runFormatter()
    logger.info("Total lines changed : " + str(ChangesCounter().getCount()))


  def addComments(self):
    commentsDict = CommentsDict()
    counter = ChangesCounter()
    if not commentsDict.hasComments():
      return False
    # with is like your try .. finally block in this case
    with open(self.target_file, 'r') as file:
      # read a list of lines into data
      data = file.readlines()

    for index, comment in commentsDict.getCommentList():
      index = index - 1
      comment = "# " + comment + "\n"
      counter.increment()
      data.insert(index, comment)

    # and write everything back
    with open(self.target_file, 'w') as file:
      file.writelines(data)
    return True

  def runFormatter(self):
    logger.info("Starting formatter...")
    os.system("black " + str(self.target_file))

  def addOptimization(self):
    # with is like your try .. finally block in this case
    with open(self.target_file, 'r') as file:
      # read a list of lines into data
      data = file.readlines()

    data.insert(0, optimization_rules)

    # and write everything back
    with open(self.target_file, 'w') as file:
      file.writelines(data)
    return True
