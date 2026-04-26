"""Tool implementations registered as Unity Catalog Python functions.

Every public function in this package is the *body* of a UC function — it
must be self-contained: only stdlib + a tiny Spark/SDK surface, with all
imports inside the function (UC requires this).
"""
