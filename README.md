# PyTPU
PyTPU is an automated tool that takes a regular PyTorch program as input and pro-
duces its TPU executable counterpart without involving any human developer in the loop.

On one hand, PyTPU is a migrator that performs behavior-
preserving source-to-source translation from PyTorch to
PyTorch-XLA by comparing automatically generated test-
cases. On the other hand, it is an optimizer that checks whether
the updated code has superior performance than the original
version by using optimization rules and executing it on TPUs.


