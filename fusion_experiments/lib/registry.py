#
#
#


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


__all__ = [
    'Registry',
]


class Registry(dict):

    def register(self, module_name, module=None):
        """


        """
        if module is not None:
            self[module_name] = module
            return

        def _register(fn):
            self[module_name] = fn
            return fn

        return _register

