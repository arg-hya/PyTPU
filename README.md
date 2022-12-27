# PyTPU
PyTPU is an automated tool that takes a regular PyTorch program as input and pro-
duces its TPU executable counterpart without involving any human developer in the loop.

On one hand, PyTPU is a migrator that performs behavior-
preserving source-to-source translation from PyTorch to
PyTorch-XLA by comparing automatically generated test-
cases. On the other hand, it is an optimizer that checks whether
the updated code has superior performance than the original
version by using optimization rules and executing it on TPUs.


## Sample Usage
    import logging
    from TPUMigrator import TPUMigrator
    migrator = TPUMigrator("kernel_code.py", level = logging.INFO)
    migrator.execute(comments=True, reformatter=True, optimize = True, genTestCases = False, PYNGUINPATH="")
    
##  Availability of TPU
At the time of writing of this paper, TPUs (except for Edge
TPUs) are available through various cloud configurations. The
popular being :
- TPU Research Cloud
* Google Cloud TPU platform
+ Google Colab (free)

Whereas, Edge TPU hardware can be used through Google’s
Pixel 4 smartphone and its prototyping kit known as Coral.

