# PyTPU
PyTPU is an automated tool that takes a regular PyTorch program as input and pro-
duces its TPU executable counterpart without involving any human developer in the loop.

On one hand, PyTPU is a migrator that performs behavior-
preserving source-to-source translation from PyTorch to
PyTorch-XLA by comparing automatically generated test-
cases. On the other hand, it is an optimizer that checks whether
the updated code has superior performance than the original
version by using optimization rules and executing it on TPUs.


Heterogeneous hardware timeline.

| Hardware        | Public Release |
| ------------- |:-------------:|
| ASIC      | 1967 |
| FPGA      | 1983      |  
| GPU | 1970      |  
| TPU | 2018      |  


## Sample Usage
```python
import logging
from TPUMigrator import TPUMigrator
migrator = TPUMigrator("kernel_code.py", level = logging.INFO)
migrator.execute(comments=True, reformatter=True, optimize = True, genTestCases = False, PYNGUINPATH="")
```
    
 Logging level supported are,
* logging.CRITICAL: For critical errors
* logging.FATAL: For fatal errors
* logging.ERROR: For general errors
* logging.WARNING: For application warnings
* logging.INFO: For information, verbose. This is default.
* logging.DEBUG: For debugging
* logging.NOTSET: Logger not set
    
##  Availability of TPU
At the time of writing of this paper, TPUs (except for Edge
TPUs) are available through various cloud configurations. The
popular being :
- TPU Research Cloud
* Google Cloud TPU platform
+ Google Colab (free)

Whereas, Edge TPU hardware can be used through Google’s
Pixel 4 smartphone and its prototyping kit known as Coral.

## Sample Migration
![This is an image](/images/sample.png)

 In the figure the left side code represents input kernel source and the right side depicts the migrated code for the same. The yellow colored labels and green colored labels represents added comments and modified/added source codes respectively. It can be noted from the figure, that seven comments were added and 9 code changes were made for XLA integration and TPU support

