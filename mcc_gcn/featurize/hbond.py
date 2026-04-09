DONOR_CHANGE_LIST = [
    ('sp C', True),
    ('sp2 C', True),
    ('sp3 C', True),
    ('aromatic C', True),
    ('carboncationic C', True),
]

ACCEPTOR_CHANGE_LIST = []


def change_hbond_criterion(
    criterion=None,
    donor_types_to_change=DONOR_CHANGE_LIST,
    acceptor_types_to_change=ACCEPTOR_CHANGE_LIST,
):
    import ccdc
    if criterion is None:
        criterion = ccdc.molecule.Molecule.HBondCriterion()
    if donor_types_to_change:
        for name, value in donor_types_to_change:
            criterion.donor_types[name] = value
    if acceptor_types_to_change:
        for name, value in acceptor_types_to_change:
            criterion.acceptor_types[name] = value
    return criterion
