
def parse_discrete(coded_values: str):
    """
    Return a dict specifying a mapping between integer values and their coded label for items in the COPDGene data
    dict

    Parameters
    ----------
    coded_values: CodedValues cell in COPDGene data dict for a single row

    Returns
    -------
    mapping

    """
    pairs = coded_values.split(" | ")
    mapping = {int(s[0]): s[1] for s in [pair.split("=") for pair in pairs]}
    return mapping
