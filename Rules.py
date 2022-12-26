import ast
import logging

from refactor import Session, Rule, common
from refactor.actions import Replace
from refactor.actions import InsertAfter

from GlobalLogger import logger
from Utils import ChangesCounter, CommentsDict


###################### Torch Libs ###########################
library_base_added = False
library_core_added = False
base_rule_added = False
base_rule_cuda = False
# imports the torch_xla package

baseTorchLibName ='torch'
baseTorchLibAlias = 'torch'
class CheckBaseTorchlibraryAlias(Rule):

    def match(self, node):
        global baseTorchLibName
        global baseTorchLibAlias
        assert isinstance(node, ast.Import)
        #print(ast.dump(node))
        assert node.names[0].name == baseTorchLibName
        logger.debug("NN module found")
        if node.names[0].asname :
            baseTorchLibAlias = node.names[0].asname
            logger.debug("NN module found by alias " + baseTorchLibAlias)
        return None

class ImportBaseLibRule(Rule):

    def match(self, node):
        global library_base_added
        if library_base_added is True:
            return None
        assert isinstance(node, ast.Import)
        logger.debug("Adding Base lib after lib : " + str(node.names[0].name) + "in line " + str(node.lineno + 1))
        import_expr = ast.Import(names=[ast.alias(name='torch_xla', asname=None)])
        library_base_added = True
        CommentsDict().addComment(node.lineno + 1, "Base lib for torch xla.")
        ChangesCounter().increment()
        return InsertAfter(node, import_expr)

class ImportCoreLibRule(Rule):

    def match(self, node):
        global library_core_added
        if library_core_added is True:
            return None
        assert isinstance(node, ast.Import)
        assert str(node.names[0].name) == "torch_xla"
        logger.debug("Adding Core lib after lib : " + str(node.names[0].name) + "in line " + str(node.lineno + 1))
        import_expr = ast.Import(names=[ast.alias(name='torch_xla.core.xla_model', asname="xm")])
        library_core_added = True
        CommentsDict().addComment(node.lineno + 1, "Core lib for torch xla.")
        ChangesCounter().increment()
        return InsertAfter(node, import_expr)

###################### TPU device base rule ######################################
#dev = xm.xla_device()
class AddBaseRule(Rule):
    def match(self, node):
        global base_rule_added
        if base_rule_added is True:
            return None
        assert isinstance(node, ast.Import)
        assert node.names[0].name == 'torch_xla.core.xla_model'

        logger.debug("Adding Base Rule in line " + str(node.lineno + 1))
        base_rule_expr = ast.Assign(targets=[ast.Name(id='dev', ctx=ast.Store())],
                              value=ast.Name(id='xm.xla_device()', ctx=ast.Load()),
                              lineno=node.lineno+1)

        #base_rule_expr = ast.Import(names=[ast.alias(name='torch_xla.core.xla_model', asname="xm")])
        base_rule_added = True
        CommentsDict().addComment(node.lineno + 1, "Gets a xla device.")
        ChangesCounter().increment()
        return InsertAfter(node, base_rule_expr)

# id device is already defined
class ModifyTensor_cudaSupported(Rule):

    def match(self, node):
        global base_rule_cuda
        if base_rule_cuda is True:
            return None
        global baseTorchLibAlias
        #logger.info(str(ast.dump(node)))
        assert isinstance(node, ast.Assign)
        assert isinstance(node.value, ast.Call)
        assert (isinstance(node.value.func, ast.Name) or
                isinstance(node.value.func, ast.Attribute))

        #print(ast.dump(node))
        if isinstance(node.value.func, ast.Name):
            assert node.value.func.id == baseTorchLibAlias
        if isinstance(node.value.func, ast.Attribute):
            assert isinstance(node.value.func.value, ast.Name)
            assert node.value.func.value.id == baseTorchLibAlias

        #logger.debug(ast.dump(node))
        logger.debug("Torch callee module found : " + str(node.value.func.value.id))

        assert node.value.func.attr == 'device'
        print("CIDA enabled")
        logger.info(ast.dump(node))
        logger.debug("Adding Base Rule in line " + str(node.lineno + 1))
        print("Var name ", node.targets[0].id)
        base_rule_expr = ast.Assign(targets=[ast.Name(id=node.targets[0].id, ctx=ast.Store())],
                                    value=ast.Name(id='dev', ctx=ast.Load()),
                                    lineno=node.lineno + 1)
        #Assign(targets=[Name(id='variable', ctx=Store())], value=Name(id='thisvalue', ctx=Load()))
        base_rule_cuda = True
        CommentsDict().addComment(node.lineno, "Converts from CUDA support")
        ChangesCounter().increment()
        return Replace(node, base_rule_expr)


######################## Tensors #####################################

random_funcs = ["rand", "rand_like", "randn", "randn_like", "randint", "randint_like", "randperm"]
# Creates a random tensor on xla:1 (a Cloud TPU core)
# dev = xm.xla_device()
# t1 = torch.ones(3, 3, device = dev)
class ModifyTensor_random(Rule):

    def match(self, node):
        global baseTorchLibAlias
        assert isinstance(node, ast.Assign)
        assert isinstance(node.value, ast.Call)
        assert (isinstance(node.value.func, ast.Name) or
                isinstance(node.value.func, ast.Attribute))

        #print(ast.dump(node))
        if isinstance(node.value.func, ast.Name):
            assert node.value.func.id == baseTorchLibAlias
        if isinstance(node.value.func, ast.Attribute):
            assert isinstance(node.value.func.value, ast.Name)
            assert node.value.func.value.id == baseTorchLibAlias

        logger.debug(ast.dump(node))
        logger.debug("Torch callee module found : " + str(node.value.func.value.id))

        assert node.value.func.attr in random_funcs
        logger.debug(" Submodule found : " + str(node.value.func.attr))
        assert len(node.value.args) != 0
        logger.debug("Attributes found :")
        for attr in node.value.args:
            #logger.debug(attr._fields[0])
            funcc = getattr(attr, attr._fields[0])
            logger.debug(funcc)
        logger.debug("Check if support already added...")
        for keyword in node.value.keywords:
            if keyword.arg == "device":
                logger.debug(keyword.arg + " already added.")
                return None
        logger.debug("Modifying expression...")
        new_expr = node
        new_expr.value.keywords.append(ast.keyword(arg='device', value=ast.Name(id='dev', ctx=ast.Load())))
        CommentsDict().addComment(node.lineno, "Execute " + str(node.value.func.attr) + " to TPU")
        ChangesCounter().increment()
        return Replace(node, new_expr)


creation_funcs = ["tensor", "sparse_coo_tensor", "asarray", "as_tensor", "zeros", "zeros_like", "ones",
                  "ones_like", "arange", "range", "linspace", "logspace", "eye", "empty", "empty_like",
                  "empty_strided", "full", "full_like"]
# Creates a random tensor on xla:1 (a Cloud TPU core)
# dev = xm.xla_device()
# torch.tensor([0, 1], device= dev)
class ModifyTensor_creation(Rule):

    def match(self, node):
        global baseTorchLibAlias
        assert isinstance(node, ast.Assign)
        assert isinstance(node.value, ast.Call)
        assert (isinstance(node.value.func, ast.Name) or
                isinstance(node.value.func, ast.Attribute))

        if isinstance(node.value.func, ast.Name):
            assert node.value.func.id == baseTorchLibAlias
        if isinstance(node.value.func, ast.Attribute):
            assert isinstance(node.value.func.value, ast.Name)
            assert node.value.func.value.id == baseTorchLibAlias

        logger.debug(ast.dump(node))
        logger.debug("Torch callee module found : " + str(node.value.func.value.id))

        assert node.value.func.attr in creation_funcs
        logger.debug(" Submodule found : " + str(node.value.func.attr))
        assert len(node.value.args) != 0
        logger.debug("Attributes found :")
        for attr in node.value.args:
            #logger.debug(attr._fields[0])
            funcc = getattr(attr, attr._fields[0])
            logger.debug(funcc)
        logger.debug("Check if support already added...")
        for keyword in node.value.keywords:
            if keyword.arg == "device":
                logger.debug(keyword.arg + " already added.")
                return None
        logger.debug("Modifying expression...")
        new_expr = node
        new_expr.value.keywords.append(ast.keyword(arg='device', value=ast.Name(id='dev', ctx=ast.Load())))
        CommentsDict().addComment(node.lineno, "Moving " + str(node.value.func.attr) + " to TPU")
        ChangesCounter().increment()
        return Replace(node, new_expr)

generator_funcs = ["Generator"]
# Creates a random tensor on xla:1 (a Cloud TPU core)
# dev = xm.xla_device()
# g_cuda = torch.Generator(device=dev)
class ModifyTensor_generator(Rule):

    def match(self, node):
        global baseTorchLibAlias
        assert isinstance(node, ast.Assign)
        assert isinstance(node.value, ast.Call)
        assert (isinstance(node.value.func, ast.Name) or
                isinstance(node.value.func, ast.Attribute))

        if isinstance(node.value.func, ast.Name):
            assert node.value.func.id == baseTorchLibAlias
        if isinstance(node.value.func, ast.Attribute):
            assert isinstance(node.value.func.value, ast.Name)
            assert node.value.func.value.id == baseTorchLibAlias

        logger.debug(ast.dump(node))
        logger.debug("Torch callee module found : " + str(node.value.func.value.id))

        assert node.value.func.attr in generator_funcs
        logger.debug(" Submodule found : " + str(node.value.func.attr))
        assert len(node.value.args) == 0
        # logger.debug("Attributes found :")
        # for attr in node.value.args:
        #     #logger.debug(attr._fields[0])
        #     funcc = getattr(attr, attr._fields[0])
        #     logger.debug(funcc)
        logger.debug("Check if support already added...")
        for keyword in node.value.keywords:
            if keyword.arg == "device":
                logger.debug(keyword.arg + " already added.")
                return None
        logger.debug("Modifying expression...")
        new_expr = node
        new_expr.value.keywords.append(ast.keyword(arg='device', value=ast.Name(id='dev', ctx=ast.Load())))
        CommentsDict().addComment(node.lineno, "Run Generator to TPU")
        ChangesCounter().increment()
        return Replace(node, new_expr)


spectral_funcs = ["bartlett_window", "blackman_window", "hamming_window", "hann_window", "kaiser_window"]
# Creates a random tensor on xla:1 (a Cloud TPU core)
# dev = xm.xla_device()
# torch.hann_window(window_size, periodic=False, dtype=dtype)
class ModifyTensor_spectralOps(Rule):

    def match(self, node):
        global baseTorchLibAlias
        assert isinstance(node, ast.Assign)
        assert isinstance(node.value, ast.Call)
        assert (isinstance(node.value.func, ast.Name) or
                isinstance(node.value.func, ast.Attribute))

        if isinstance(node.value.func, ast.Name):
            assert node.value.func.id == baseTorchLibAlias
        if isinstance(node.value.func, ast.Attribute):
            assert isinstance(node.value.func.value, ast.Name)
            assert node.value.func.value.id == baseTorchLibAlias

        logger.debug(ast.dump(node))
        logger.debug("Torch callee module found : " + str(node.value.func.value.id))

        assert node.value.func.attr in spectral_funcs
        logger.debug(" Submodule found : " + str(node.value.func.attr))
        assert len(node.value.args) != 0
        logger.debug("Attributes found :")
        if logger.level == logging.DEBUG :
            for attr in node.value.args:
                #logger.debug(attr._fields[0])
                funcc = getattr(attr, attr._fields[0])
                logger.debug(funcc)
        logger.debug("Check if support already added...")
        for keyword in node.value.keywords:
            if keyword.arg == "device":
                logger.debug(keyword.arg + " already added.")
                return None
        logger.debug("Modifying expression...")
        new_expr = node
        new_expr.value.keywords.append(ast.keyword(arg='device', value=ast.Name(id='dev', ctx=ast.Load())))
        CommentsDict().addComment(node.lineno, "Execute " + str(node.value.func.attr) + " to TPU")
        ChangesCounter().increment()
        return Replace(node, new_expr)


indices_funcs = ["tril_indices", "triu_indices"]
# Creates a random tensor on xla:1 (a Cloud TPU core)
# dev = xm.xla_device()
# a = torch.triu_indices(4, 3, 1)
class ModifyTensor_indices(Rule):

    def match(self, node):
        global baseTorchLibAlias
        assert isinstance(node, ast.Assign)
        assert isinstance(node.value, ast.Call)
        assert (isinstance(node.value.func, ast.Name) or
                isinstance(node.value.func, ast.Attribute))

        if isinstance(node.value.func, ast.Name):
            assert node.value.func.id == baseTorchLibAlias
        if isinstance(node.value.func, ast.Attribute):
            assert isinstance(node.value.func.value, ast.Name)
            assert node.value.func.value.id == baseTorchLibAlias

        logger.debug(ast.dump(node))
        logger.debug("Torch callee module found : " + str(node.value.func.value.id))

        assert node.value.func.attr in indices_funcs
        logger.debug(" Submodule found : " + str(node.value.func.attr))
        assert len(node.value.args) != 0
        logger.debug("Attributes found :")
        if logger.level == logging.DEBUG :
            for attr in node.value.args:
                #logger.debug(attr._fields[0])
                funcc = getattr(attr, attr._fields[0])
                logger.debug(funcc)
        logger.debug("Check if support already added...")
        for keyword in node.value.keywords:
            if keyword.arg == "device":
                logger.debug(keyword.arg + " already added.")
                return None
        logger.debug("Modifying expression...")
        new_expr = node
        new_expr.value.keywords.append(ast.keyword(arg='device', value=ast.Name(id='dev', ctx=ast.Load())))
        CommentsDict().addComment(node.lineno, "Run " + str(node.value.func.attr) + " to TPU")
        ChangesCounter().increment()
        return Replace(node, new_expr)

################ Modules ########################

baseNNModuleName ='torch.nn'
baseNNModuleAlias = 'torch.nn'
class CheckNNlibraryAlias(Rule):

    def match(self, node):
        global baseNNModuleName
        global baseNNModuleAlias
        assert isinstance(node, ast.Import)
        #print(ast.dump(node))
        assert node.names[0].name == baseNNModuleName
        logger.debug("NN module found")
        if node.names[0].asname :
            baseNNModuleAlias = node.names[0].asname
            logger.debug("NN module found by alias " + baseNNModuleAlias)
        return None

nn_module_classes = []
class StoreNNModuleClass(Rule):

    def match(self, node):
        global baseNNModuleAlias
        global nn_module_classes
        assert isinstance(node, ast.ClassDef)
        #print(ast.dump(node))
        assert len(node.bases) > 0
        for attr in node.bases:
            assert isinstance(attr, ast.Attribute)
            assert isinstance(name := attr.value, ast.Name)
            #print(name.id)
            #print(baseNNModuleAlias)
            assert name.id is baseNNModuleAlias
            nn_module_classes.append(node.name)
        logger.debug("Module class found : " + str(nn_module_classes))
        #print("Module class found : ", nn_module_classes)
        # assert node.names[0].name == baseNNModuleName
        # if node.names[0].asname :
        #     baseNNModuleAlias = node.names[0].asname
        return None

class ModifyNNModuleClass(Rule):

    def match(self, node):
        global nn_module_classes
        assert isinstance(node, ast.Assign)
        assert isinstance(node.value, ast.Call)
        #print(ast.dump(node))
        assert isinstance(name := node.value.func, ast.Name)
        # assert isinstance(callee := node.value.func.value, ast.Call)
        # assert isinstance(callee.func, ast.Name)
        if name.id in nn_module_classes:
            new_expr = node
            new_expr.value = ast.Call(func = ast.Attribute(value = new_expr.value, attr='to', ctx=ast.Load()), args=[ast.Name(id='dev', ctx=ast.Load())], keywords=[])
            logger.debug("Migrating module " + str(name.id) + " to TPU")
            CommentsDict().addComment(node.lineno, "Migrating module " + str(name.id) + " to TPU")
            ChangesCounter().increment()
            return Replace(node, new_expr)
        return None

######################## tensor.nn library ##############################

cnvLayer_funcs = ["Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "LazyConv1d",
                  "LazyConv2d", "LazyConv3d", "LazyConvTranspose1d", "LazyConvTranspose2d", "LazyConvTranspose3d"]
# Creates a random tensor on xla:1 (a Cloud TPU core)
# dev = xm.xla_device()
# m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
class ModifyTensor_nn_cnvLayer(Rule):

    def match(self, node):
        global baseNNModuleAlias
        assert isinstance(node, ast.Assign)
        assert isinstance(node.targets[0], ast.Name)
        #print(ast.dump(node))
        assert isinstance(node.value, ast.Call)
        assert isinstance(node.value.func, ast.Attribute)
        assert isinstance(node.value.func.value, ast.Name)
        # print(node.value.func.value)
        # print("baseNNModuleAlias ", baseNNModuleAlias)
        assert node.value.func.value.id == baseNNModuleAlias
        # print(ast.dump(node))
        logger.debug(ast.dump(node))
        logger.debug("Torch.nn callee module found : " + str(node.value.func.value.id))
        assert node.value.func.attr in cnvLayer_funcs
        logger.debug(" Submodule found : " + str(node.value.func.attr))
        assert len(node.value.args) != 0
        logger.debug("Attributes found :")
        if logger.level == logging.DEBUG :
            for attr in node.value.args:
                #logger.debug(attr._fields[0])
                funcc = getattr(attr, attr._fields[0])
                logger.debug(funcc)
        logger.debug("Check if support already added...")
        for keyword in node.value.keywords:
            if keyword.arg == "device":
                logger.debug(keyword.arg + " already added.")
                return None
        logger.debug("Modifying expression...")
        new_expr = node
        new_expr.value.keywords.append(ast.keyword(arg='device', value=ast.Name(id='dev', ctx=ast.Load())))
        CommentsDict().addComment(node.lineno, "Run " + str(node.value.func.attr) + " to TPU")
        ChangesCounter().increment()
        return Replace(node, new_expr)


normalizationLayer_funcs = ["BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LazyBatchNorm1d", "LazyBatchNorm2d", "LazyBatchNorm3d", "GroupNorm",
                            "SyncBatchNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "LazyInstanceNorm1d",
                            "LazyInstanceNorm2d", "LazyInstanceNorm3d", "LayerNorm"]
# Creates a random tensor on xla:1 (a Cloud TPU core)
# dev = xm.xla_device()
# m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
class ModifyTensor_nn_normalizationLayer(Rule):

    def match(self, node):
        global baseNNModuleAlias
        assert isinstance(node, ast.Assign)
        assert isinstance(node.targets[0], ast.Name)
        #print(ast.dump(node))
        assert isinstance(node.value, ast.Call)
        assert isinstance(node.value.func, ast.Attribute)
        assert isinstance(node.value.func.value, ast.Name)
        # print(node.value.func.value)
        # print("baseNNModuleAlias ", baseNNModuleAlias)
        assert node.value.func.value.id == baseNNModuleAlias
        # print(ast.dump(node))
        logger.debug(ast.dump(node))
        logger.debug("Torch.nn callee module found : " + str(node.value.func.value.id))
        assert node.value.func.attr in normalizationLayer_funcs
        logger.debug(" Submodule found : " + str(node.value.func.attr))
        assert len(node.value.args) != 0
        logger.debug("Attributes found :")
        if logger.level == logging.DEBUG :
            for attr in node.value.args:
                #logger.debug(attr._fields[0])
                funcc = getattr(attr, attr._fields[0])
                logger.debug(funcc)
        logger.debug("Check if support already added...")
        for keyword in node.value.keywords:
            if keyword.arg == "device":
                logger.debug(keyword.arg + " already added.")
                return None
        logger.debug("Modifying expression...")
        new_expr = node
        new_expr.value.keywords.append(ast.keyword(arg='device', value=ast.Name(id='dev', ctx=ast.Load())))
        CommentsDict().addComment(node.lineno, "Run " + str(node.value.func.attr) + " to TPU")
        ChangesCounter().increment()
        return Replace(node, new_expr)



rnn_funcs = ["RNNBase", "RNNCell", "LSTMCell", "GRUCell"]
# Creates a random tensor on xla:1 (a Cloud TPU core)
# dev = xm.xla_device()
# m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
class ModifyTensor_nn_rnnLayer(Rule):

    def match(self, node):
        global baseNNModuleAlias
        assert isinstance(node, ast.Assign)
        assert isinstance(node.targets[0], ast.Name)
        #print(ast.dump(node))
        assert isinstance(node.value, ast.Call)
        assert isinstance(node.value.func, ast.Attribute)
        assert isinstance(node.value.func.value, ast.Name)
        # print(node.value.func.value)
        # print("baseNNModuleAlias ", baseNNModuleAlias)
        assert node.value.func.value.id == baseNNModuleAlias
        # print(ast.dump(node))
        logger.debug(ast.dump(node))
        logger.debug("Torch.nn callee module found : " + str(node.value.func.value.id))
        assert node.value.func.attr in rnn_funcs
        logger.debug(" Submodule found : " + str(node.value.func.attr))
        assert len(node.value.args) != 0
        logger.debug("Attributes found :")
        if logger.level == logging.DEBUG :
            for attr in node.value.args:
                #logger.debug(attr._fields[0])
                funcc = getattr(attr, attr._fields[0])
                logger.debug(funcc)
        logger.debug("Check if support already added...")
        for keyword in node.value.keywords:
            if keyword.arg == "device":
                logger.debug(keyword.arg + " already added.")
                return None
        logger.debug("Modifying expression...")
        new_expr = node
        new_expr.value.keywords.append(ast.keyword(arg='device', value=ast.Name(id='dev', ctx=ast.Load())))
        CommentsDict().addComment(node.lineno, "Run " + str(node.value.func.attr) + " to TPU")
        ChangesCounter().increment()
        return Replace(node, new_expr)



linear_funcs = ["Linear", "Bilinear", "LazyLinear"]
# Creates a random tensor on xla:1 (a Cloud TPU core)
# dev = xm.xla_device()
# m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
class ModifyTensor_nn_linearLayer(Rule):

    def match(self, node):
        global baseNNModuleAlias
        assert isinstance(node, ast.Assign)
        assert isinstance(node.targets[0], ast.Name)
        #print(ast.dump(node))
        assert isinstance(node.value, ast.Call)
        assert isinstance(node.value.func, ast.Attribute)
        assert isinstance(node.value.func.value, ast.Name)
        # print(node.value.func.value)
        # print("baseNNModuleAlias ", baseNNModuleAlias)
        assert node.value.func.value.id == baseNNModuleAlias
        # print(ast.dump(node))
        logger.debug(ast.dump(node))
        logger.debug("Torch.nn callee module found : " + str(node.value.func.value.id))
        assert node.value.func.attr in linear_funcs
        logger.debug(" Submodule found : " + str(node.value.func.attr))
        assert len(node.value.args) != 0
        logger.debug("Attributes found :")
        if logger.level == logging.DEBUG :
            for attr in node.value.args:
                #logger.debug(attr._fields[0])
                funcc = getattr(attr, attr._fields[0])
                logger.debug(funcc)
        logger.debug("Check if support already added...")
        for keyword in node.value.keywords:
            if keyword.arg == "device":
                logger.debug(keyword.arg + " already added.")
                return None
        logger.debug("Modifying expression...")
        new_expr = node
        new_expr.value.keywords.append(ast.keyword(arg='device', value=ast.Name(id='dev', ctx=ast.Load())))
        CommentsDict().addComment(node.lineno, "Run " + str(node.value.func.attr) + " to TPU")
        ChangesCounter().increment()
        return Replace(node, new_expr)


embb_funcs = ["Embedding", "EmbeddingBag"]
# Creates a random tensor on xla:1 (a Cloud TPU core)
# dev = xm.xla_device()
# m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
class ModifyTensor_nn_embbLayer(Rule):

    def match(self, node):
        global baseNNModuleAlias
        assert isinstance(node, ast.Assign)
        assert isinstance(node.targets[0], ast.Name)
        #print(ast.dump(node))
        assert isinstance(node.value, ast.Call)
        assert isinstance(node.value.func, ast.Attribute)
        assert isinstance(node.value.func.value, ast.Name)
        # print(node.value.func.value)
        # print("baseNNModuleAlias ", baseNNModuleAlias)
        assert node.value.func.value.id == baseNNModuleAlias
        # print(ast.dump(node))
        logger.debug(ast.dump(node))
        logger.debug("Torch.nn callee module found : " + str(node.value.func.value.id))
        assert node.value.func.attr in embb_funcs
        logger.debug(" Submodule found : " + str(node.value.func.attr))
        assert len(node.value.args) != 0
        logger.debug("Attributes found :")
        if logger.level == logging.DEBUG :
            for attr in node.value.args:
                #logger.debug(attr._fields[0])
                funcc = getattr(attr, attr._fields[0])
                logger.debug(funcc)
        logger.debug("Check if support already added...")
        for keyword in node.value.keywords:
            if keyword.arg == "device":
                logger.debug(keyword.arg + " already added.")
                return None
        logger.debug("Modifying expression...")
        new_expr = node
        new_expr.value.keywords.append(ast.keyword(arg='device', value=ast.Name(id='dev', ctx=ast.Load())))
        CommentsDict().addComment(node.lineno, "Run " + str(node.value.func.attr) + " to TPU")
        ChangesCounter().increment()
        return Replace(node, new_expr)