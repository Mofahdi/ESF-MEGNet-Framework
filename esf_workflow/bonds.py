"""Align electronic descriptors by directed bond and periodic image."""


def align_bond_descriptors(icohp_records, icobi_records):
    def index(records, name):
        result = {}
        for record in records:
            key = (tuple(record[0]), tuple(record[2]))
            if key in result:
                raise ValueError(f"Duplicate directed bond in {name}: {key}")
            result[key] = record
        return result

    hp = index(icohp_records, "ICOHP")
    bi = index(icobi_records, "ICOBI")
    if hp.keys() != bi.keys():
        missing = sorted(hp.keys() - bi.keys())
        extra = sorted(bi.keys() - hp.keys())
        raise ValueError(
            "ICOHP/ICOBI bond coverage differs: "
            f"{len(missing)} missing ICOBI bonds, {len(extra)} extra ICOBI bonds; "
            f"examples: missing={missing[:3]}, extra={extra[:3]}"
        )
    # Preserve ICOHP's existing graph order, matching each ICOBI value by identity.
    return [(record, bi[key][-1]) for key, record in hp.items()]
