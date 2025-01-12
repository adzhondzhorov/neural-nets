def pprint(l, s="", level=0):
    indent = "  " * level
    if isinstance(l, list) and not isinstance(l[0], list):
        nest = ", ".join([str(i) for i in l])
        s += "{indent}[{nest}]".format(indent=indent, nest=nest)
        return s
    else:
        nest=",\n".join([pprint(i, s, level + 1) for i in l])
        s += "{indent}[\n{nest}\n{indent}]".format(indent=indent, nest=nest)
        return s